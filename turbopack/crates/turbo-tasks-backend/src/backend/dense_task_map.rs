//! Dense, independently locked task storage.
//!
//! `boxcar::Vec` is used only as an append-only directory of `OnceLock` chunk pointers. This fits
//! the task-ID use case because chunks are published in index order, boxcar keeps their addresses
//! stable during concurrent growth, and removal/reuse happens inside each chunk rather than in the
//! directory. Missing intermediate chunks cost one uninitialized `OnceLock`, not 1024 task slots.

use std::{
    ops::{Deref, DerefMut},
    sync::{
        OnceLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use turbo_tasks::{TRANSIENT_TASK_BIT, TaskId, parallel};

pub(crate) const CHUNK_SHIFT: usize = 10;
pub(crate) const CHUNK_SIZE: usize = 1 << CHUNK_SHIFT;
const CHUNK_MASK: usize = CHUNK_SIZE - 1;

type Slot<T> = Mutex<Option<T>>;

pub(crate) struct TaskChunk<T> {
    pub(crate) modified_count: AtomicU64,
    slots: Box<[Slot<T>; CHUNK_SIZE]>,
}

impl<T> TaskChunk<T> {
    fn new() -> Self {
        Self {
            modified_count: AtomicU64::new(0),
            slots: Box::new(std::array::from_fn(|_| Mutex::new(None))),
        }
    }

    pub(crate) fn lock(&self, offset: usize) -> MutexGuard<'_, Option<T>> {
        self.slots[offset].lock()
    }

    pub(crate) fn slots(&self) -> &[Slot<T>; CHUNK_SIZE] {
        &self.slots
    }
}

struct ChunkedVec<T> {
    chunks: boxcar::Vec<OnceLock<Box<TaskChunk<T>>>>,
    len: AtomicUsize,
}

impl<T> ChunkedVec<T> {
    fn with_chunk_capacity(chunk_capacity: usize) -> Self {
        Self {
            chunks: boxcar::Vec::with_capacity(chunk_capacity),
            len: AtomicUsize::new(0),
        }
    }

    fn chunk(&self, index: usize) -> Option<&TaskChunk<T>> {
        self.chunks
            .get(index >> CHUNK_SHIFT)?
            .get()
            .map(Box::as_ref)
    }

    fn get_or_create_chunk(&self, index: usize) -> &TaskChunk<T> {
        let chunk_index = index >> CHUNK_SHIFT;
        loop {
            if let Some(chunk) = self.chunks.get(chunk_index) {
                return chunk.get_or_init(|| Box::new(TaskChunk::new()));
            }

            // `count` includes indices reserved by in-progress pushes. If our index has already
            // been reserved, wait for its `OnceLock` to become visible rather than over-growing
            // the directory. Otherwise help extend the append-only directory toward it.
            if self.chunks.count() <= chunk_index {
                self.chunks.push(OnceLock::new());
            } else {
                std::hint::spin_loop();
            }
        }
    }

    fn get_or_insert_with(
        &self,
        index: usize,
        create: impl FnOnce() -> T,
    ) -> (MappedMutexGuard<'_, T>, &AtomicU64) {
        let chunk = self.get_or_create_chunk(index);
        let mut slot = chunk.lock(index & CHUNK_MASK);
        if slot.is_none() {
            *slot = Some(create());
            self.len.fetch_add(1, Ordering::Relaxed);
        }
        (
            MutexGuard::map(slot, |slot| slot.as_mut().unwrap()),
            &chunk.modified_count,
        )
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    fn chunks(&self) -> impl Iterator<Item = (usize, &TaskChunk<T>)> {
        self.chunks
            .iter()
            .filter_map(|(index, chunk)| Some((index, chunk.get()?.as_ref())))
    }

    fn clear(&self)
    where
        T: Send,
    {
        let chunks: Vec<_> = self.chunks().map(|(_, chunk)| chunk).collect();
        parallel::for_each(&chunks, |chunk| {
            for slot in chunk.slots() {
                slot.lock().take();
            }
            chunk.modified_count.store(0, Ordering::Relaxed);
        });
        self.len.store(0, Ordering::Relaxed);
    }
}

pub(crate) struct TaskMap<T> {
    persistent: ChunkedVec<T>,
    transient: ChunkedVec<T>,
}

impl<T> TaskMap<T> {
    pub(crate) fn new(small_preallocation: bool) -> Self {
        let persistent_chunk_capacity = if small_preallocation {
            1
        } else {
            (1024 * 1024) / CHUNK_SIZE
        };
        Self {
            persistent: ChunkedVec::with_chunk_capacity(persistent_chunk_capacity),
            transient: ChunkedVec::with_chunk_capacity(1),
        }
    }

    fn namespace_and_index(&self, key: TaskId) -> (&ChunkedVec<T>, usize) {
        let raw = *key as usize;
        let index = raw & !(TRANSIENT_TASK_BIT as usize);
        if key.is_transient() {
            (&self.transient, index)
        } else {
            (&self.persistent, index)
        }
    }

    pub(crate) fn get(&self, key: TaskId) -> Option<TaskMapGuard<'_, T>> {
        let (namespace, index) = self.namespace_and_index(key);
        let chunk = namespace.chunk(index)?;
        let inner =
            MutexGuard::try_map(chunk.lock(index & CHUNK_MASK), |slot| slot.as_mut()).ok()?;
        Some(TaskMapGuard {
            key,
            modified_count: &chunk.modified_count,
            inner,
        })
    }

    pub(crate) fn get_or_insert_with(
        &self,
        key: TaskId,
        create: impl FnOnce() -> T,
    ) -> TaskMapGuard<'_, T> {
        let (namespace, index) = self.namespace_and_index(key);
        let (inner, modified_count) = namespace.get_or_insert_with(index, create);
        TaskMapGuard {
            key,
            modified_count,
            inner,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn remove(&self, key: TaskId) -> Option<T> {
        let (namespace, index) = self.namespace_and_index(key);
        let chunk = namespace.chunk(index)?;
        let value = chunk.lock(index & CHUNK_MASK).take();
        if value.is_some() {
            namespace.len.fetch_sub(1, Ordering::Relaxed);
        }
        value
    }

    pub(crate) fn len(&self) -> usize {
        self.persistent.len() + self.transient.len()
    }

    pub(crate) fn clear(&self)
    where
        T: Send,
    {
        self.persistent.clear();
        self.transient.clear();
    }

    pub(crate) fn chunks(&self) -> Vec<TaskChunkRef<'_, T>> {
        self.persistent
            .chunks()
            .map(|(index, chunk)| TaskChunkRef {
                base_id: index << CHUNK_SHIFT,
                transient: false,
                chunk,
                len: &self.persistent.len,
            })
            .chain(self.transient.chunks().map(|(index, chunk)| TaskChunkRef {
                base_id: index << CHUNK_SHIFT,
                transient: true,
                chunk,
                len: &self.transient.len,
            }))
            .collect()
    }
}

pub(crate) struct TaskChunkRef<'a, T> {
    pub(crate) base_id: usize,
    pub(crate) transient: bool,
    pub(crate) chunk: &'a TaskChunk<T>,
    len: &'a AtomicUsize,
}

impl<T> Copy for TaskChunkRef<'_, T> {}

impl<T> Clone for TaskChunkRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> TaskChunkRef<'_, T> {
    pub(crate) fn update_slot<R>(
        &self,
        offset: usize,
        update: impl FnOnce(&mut Option<T>) -> R,
    ) -> R {
        let mut slot = self.chunk.lock(offset);
        let occupied_before = slot.is_some();
        let result = update(&mut slot);
        match (occupied_before, slot.is_some()) {
            (false, true) => {
                self.len.fetch_add(1, Ordering::Relaxed);
            }
            (true, false) => {
                self.len.fetch_sub(1, Ordering::Relaxed);
            }
            _ => {}
        }
        result
    }

    pub(crate) fn task_id(&self, offset: usize) -> TaskId {
        let mut raw = (self.base_id + offset) as u32;
        if self.transient {
            raw |= TRANSIENT_TASK_BIT;
        }
        TaskId::try_from(raw).expect("occupied task slots always have valid task IDs")
    }
}

pub(crate) struct TaskMapGuard<'a, T> {
    key: TaskId,
    pub(crate) modified_count: &'a AtomicU64,
    inner: MappedMutexGuard<'a, T>,
}

impl<T> TaskMapGuard<'_, T> {
    pub(crate) fn key(&self) -> &TaskId {
        &self.key
    }
}

impl<T> Deref for TaskMapGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for TaskMapGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
    };

    use turbo_tasks::TRANSIENT_TASK_BIT;

    use super::*;

    fn task_id(raw: u32) -> TaskId {
        TaskId::try_from(raw).unwrap()
    }

    #[test]
    fn grows_across_chunk_boundaries_and_reuses_slots() {
        let map = TaskMap::new(true);
        for raw in [
            1,
            CHUNK_SIZE as u32 - 1,
            CHUNK_SIZE as u32,
            CHUNK_SIZE as u32 + 1,
        ] {
            *map.get_or_insert_with(task_id(raw), || raw) = raw;
        }
        assert_eq!(map.len(), 4);
        assert_eq!(
            *map.get(task_id(CHUNK_SIZE as u32)).unwrap(),
            CHUNK_SIZE as u32
        );
        assert_eq!(
            map.remove(task_id(CHUNK_SIZE as u32)),
            Some(CHUNK_SIZE as u32)
        );
        assert!(map.get(task_id(CHUNK_SIZE as u32)).is_none());
        assert_eq!(*map.get_or_insert_with(task_id(CHUNK_SIZE as u32), || 7), 7);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn clear_drops_all_namespaces_in_parallel() {
        let map = TaskMap::new(true);
        for raw in [1, CHUNK_SIZE as u32 + 1, 7 | TRANSIENT_TASK_BIT] {
            map.get_or_insert_with(task_id(raw), || raw);
        }
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.get(task_id(1)).is_none());
        assert!(map.get(task_id(CHUNK_SIZE as u32 + 1)).is_none());
        assert!(map.get(task_id(7 | TRANSIENT_TASK_BIT)).is_none());
    }

    #[test]
    fn persistent_and_transient_namespaces_do_not_alias() {
        let map = TaskMap::new(true);
        *map.get_or_insert_with(task_id(7), || "persistent") = "persistent";
        *map.get_or_insert_with(task_id(7 | TRANSIENT_TASK_BIT), || "transient") = "transient";
        assert_eq!(*map.get(task_id(7)).unwrap(), "persistent");
        assert_eq!(
            *map.get(task_id(7 | TRANSIENT_TASK_BIT)).unwrap(),
            "transient"
        );
    }

    #[test]
    fn concurrent_first_access_initializes_once() {
        let map = Arc::new(TaskMap::new(true));
        let barrier = Arc::new(Barrier::new(9));
        let initializations = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let map = map.clone();
                let barrier = barrier.clone();
                let initializations = initializations.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let value = map.get_or_insert_with(task_id(1), || {
                        initializations.fetch_add(1, Ordering::Relaxed);
                        42
                    });
                    assert_eq!(*value, 42);
                })
            })
            .collect();
        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(initializations.load(Ordering::Relaxed), 1);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn concurrent_growth_preserves_existing_entries() {
        let map = Arc::new(TaskMap::new(true));
        *map.get_or_insert_with(task_id(1), || 1) = 1;
        let barrier = Arc::new(Barrier::new(9));
        let threads: Vec<_> = (0..8)
            .map(|thread_index| {
                let map = map.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for chunk in 1..32 {
                        let raw = (chunk * CHUNK_SIZE + thread_index + 1) as u32;
                        *map.get_or_insert_with(task_id(raw), || raw) = raw;
                        assert_eq!(*map.get(task_id(1)).unwrap(), 1);
                    }
                })
            })
            .collect();
        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(map.len(), 1 + 8 * 31);
    }

    #[test]
    fn chunks_cover_dense_and_sparse_entries_once() {
        let map = TaskMap::new(true);
        let ids = [1, 2, CHUNK_SIZE as u32 + 3, (CHUNK_SIZE * 4) as u32 + 5];
        for raw in ids {
            map.get_or_insert_with(task_id(raw), || raw);
        }
        let mut seen = Vec::new();
        for chunk in map.chunks() {
            for (offset, slot) in chunk.chunk.slots().iter().enumerate() {
                if slot.lock().is_some() {
                    seen.push(*chunk.task_id(offset));
                }
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, ids);
    }
}
