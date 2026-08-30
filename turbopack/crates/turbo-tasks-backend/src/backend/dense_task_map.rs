//! Dense, independently locked task storage.
//!
//! `boxcar::Vec` is used only as an append-only directory of `OnceLock` chunk pointers. This fits
//! the task-ID use case because chunks are published in index order, boxcar keeps their addresses
//! stable during concurrent growth, and removal/reuse happens inside each chunk rather than in the
//! directory. Missing intermediate chunks cost one uninitialized `OnceLock`, not 1024 task slots.

use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{
        OnceLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use turbo_tasks::{TRANSIENT_TASK_BIT, TaskId, parallel};

pub(crate) const CHUNK_SHIFT: usize = 10;
pub(crate) const CHUNK_SIZE: usize = 1 << CHUNK_SHIFT;
const CHUNK_MASK: usize = CHUNK_SIZE - 1;
const BITMAP_WORD_BITS: usize = u64::BITS as usize;
pub(crate) const BITMAP_WORDS: usize = CHUNK_SIZE / BITMAP_WORD_BITS;

/// Value stored in an always-initialized intrusive task slot.
///
/// # Safety
///
/// Implementations must keep the lock at a stable address, initialize it in `EMPTY`, require the
/// lock for every payload/presence access, and leave the source lock untouched when vacating.
pub(crate) unsafe trait TaskSlotValue: Sized {
    const EMPTY: Self;

    fn lock(&self);

    /// # Safety
    ///
    /// The current thread must own this value's lock and perform no protected access afterward.
    unsafe fn unlock(&self);

    fn is_occupied(&self) -> bool;
    fn occupy(&mut self);
    fn take_and_vacate(&mut self) -> Self;
    fn vacate_in_place(&mut self);
}

/// Stable storage for a value whose mutex is embedded inside the value itself.
#[repr(transparent)]
pub(crate) struct TaskSlot<T: TaskSlotValue>(UnsafeCell<T>);

impl<T: TaskSlotValue> TaskSlot<T> {
    pub(crate) const fn empty() -> Self {
        Self(UnsafeCell::new(T::EMPTY))
    }

    fn lock(&self) -> TaskSlotGuard<'_, T> {
        // SAFETY: The value is initialized by `empty` and never moved after its chunk is published.
        // Calling `lock` only reads/mutates the embedded raw mutex, whose implementation provides
        // the synchronization for all subsequent accesses through the returned guard.
        unsafe { &*self.0.get() }.lock();
        TaskSlotGuard {
            slot: self,
            _not_send: PhantomData,
        }
    }
}

// SAFETY: `TaskSlotValue` requires all shared payload access to be protected by its embedded lock.
// `T: Send` allows ownership of protected values to move between threads when detached.
unsafe impl<T: TaskSlotValue + Send> Sync for TaskSlot<T> {}

struct TaskSlotGuard<'a, T: TaskSlotValue> {
    slot: &'a TaskSlot<T>,
    // parking_lot guards are !Send by default. Preserve that property for the custom raw guard so
    // a task lock cannot be held across `.await` in a sendable future.
    _not_send: PhantomData<*const ()>,
}

impl<T: TaskSlotValue> Deref for TaskSlotGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: This guard owns the slot's intrusive lock.
        unsafe { &*self.slot.0.get() }
    }
}

impl<T: TaskSlotValue> DerefMut for TaskSlotGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: This guard exclusively owns the slot's intrusive lock.
        unsafe { &mut *self.slot.0.get() }
    }
}

impl<T: TaskSlotValue> Drop for TaskSlotGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: The guard owns the lock and performs no protected access after this call.
        unsafe { (&*self.slot.0.get()).unlock() };
    }
}

pub(crate) struct TaskChunk<T: TaskSlotValue> {
    pub(crate) modified_count: AtomicU64,
    probably_occupied: [AtomicU64; BITMAP_WORDS],
    slots: Box<[TaskSlot<T>; CHUNK_SIZE]>,
}

impl<T: TaskSlotValue> TaskChunk<T> {
    fn new() -> Self {
        Self {
            modified_count: AtomicU64::new(0),
            // Start conservatively set so chunk construction and dense first insertion need no
            // atomic read-modify-write per slot. The first scan locks/rechecks vacant slots and
            // clears their stale hints; subsequent sparse scans skip them.
            probably_occupied: [const { AtomicU64::new(u64::MAX) }; BITMAP_WORDS],
            slots: Box::new([const { TaskSlot::empty() }; CHUNK_SIZE]),
        }
    }

    fn word_and_mask(offset: usize) -> (usize, u64) {
        (offset / BITMAP_WORD_BITS, 1 << (offset % BITMAP_WORD_BITS))
    }

    fn mark_probably_occupied(&self, offset: usize) {
        let (word, mask) = Self::word_and_mask(offset);
        self.probably_occupied[word].fetch_or(mask, Ordering::Release);
    }

    fn clear_probably_occupied(&self, offset: usize) {
        let (word, mask) = Self::word_and_mask(offset);
        self.probably_occupied[word].fetch_and(!mask, Ordering::Release);
    }

    pub(crate) fn is_probably_occupied(&self, offset: usize) -> bool {
        let (word, mask) = Self::word_and_mask(offset);
        self.probably_occupied[word].load(Ordering::Acquire) & mask != 0
    }

    fn lock(&self, offset: usize) -> TaskSlotGuard<'_, T> {
        self.slots[offset].lock()
    }

    pub(crate) fn probably_occupied_offsets(&self) -> ProbablyOccupiedOffsets<'_> {
        ProbablyOccupiedOffsets {
            words: &self.probably_occupied,
            word_index: 0,
            bits: 0,
        }
    }
}

pub(crate) struct ProbablyOccupiedOffsets<'a> {
    words: &'a [AtomicU64; BITMAP_WORDS],
    word_index: usize,
    bits: u64,
}

impl Iterator for ProbablyOccupiedOffsets<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.bits != 0 {
                let bit = self.bits.trailing_zeros() as usize;
                self.bits &= self.bits - 1;
                return Some((self.word_index - 1) * BITMAP_WORD_BITS + bit);
            }
            let word = self.words.get(self.word_index)?;
            self.word_index += 1;
            self.bits = word.load(Ordering::Acquire);
        }
    }
}

struct ChunkedVec<T: TaskSlotValue> {
    chunks: boxcar::Vec<OnceLock<Box<TaskChunk<T>>>>,
    len: AtomicUsize,
}

impl<T: TaskSlotValue> ChunkedVec<T> {
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
            for offset in chunk.probably_occupied_offsets() {
                let mut value = chunk.lock(offset);
                if value.is_occupied() {
                    value.vacate_in_place();
                }
                chunk.clear_probably_occupied(offset);
            }
            chunk.modified_count.store(0, Ordering::Relaxed);
        });
        self.len.store(0, Ordering::Relaxed);
    }
}

pub(crate) struct TaskMap<T: TaskSlotValue> {
    persistent: ChunkedVec<T>,
    transient: ChunkedVec<T>,
}

impl<T: TaskSlotValue> TaskMap<T> {
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
        let offset = index & CHUNK_MASK;
        if !chunk.is_probably_occupied(offset) {
            return None;
        }
        let inner = chunk.lock(offset);
        if !inner.is_occupied() {
            // Point misses leave stale hints alone. Bulk chunk scans clean them while holding this
            // same slot lock; avoiding a clear here prevents miss-then-insert bitmap churn.
            return None;
        }
        Some(TaskMapGuard::new(key, inner, chunk, offset, &namespace.len))
    }

    pub(crate) fn get_or_insert(&self, key: TaskId) -> TaskMapGuard<'_, T> {
        let (namespace, index) = self.namespace_and_index(key);
        let chunk = namespace
            .chunk(index)
            .unwrap_or_else(|| namespace.get_or_create_chunk(index));
        let offset = index & CHUNK_MASK;
        let mut inner = chunk.lock(offset);
        if !inner.is_occupied() {
            // Publish the advisory bit first. A racing scan may observe a stale set bit and recheck
            // under this lock, but an occupied slot is never intentionally hidden by a clear bit.
            if !chunk.is_probably_occupied(offset) {
                chunk.mark_probably_occupied(offset);
            }
            inner.occupy();
            namespace.len.fetch_add(1, Ordering::Relaxed);
        }
        TaskMapGuard::new(key, inner, chunk, offset, &namespace.len)
    }

    #[allow(dead_code)]
    pub(crate) fn remove(&self, key: TaskId) -> Option<T> {
        let guard = self.get(key)?;
        Some(guard.take_and_vacate())
    }

    #[allow(dead_code)]
    pub(crate) fn remove_discard(&self, key: TaskId) -> bool {
        let Some(guard) = self.get(key) else {
            return false;
        };
        guard.vacate();
        true
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

pub(crate) struct TaskChunkRef<'a, T: TaskSlotValue> {
    pub(crate) base_id: usize,
    pub(crate) transient: bool,
    pub(crate) chunk: &'a TaskChunk<T>,
    len: &'a AtomicUsize,
}

impl<T: TaskSlotValue> Copy for TaskChunkRef<'_, T> {}

impl<T: TaskSlotValue> Clone for TaskChunkRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: TaskSlotValue> TaskChunkRef<'_, T> {
    pub(crate) fn get(&self, offset: usize) -> Option<TaskMapGuard<'_, T>> {
        if !self.chunk.is_probably_occupied(offset) {
            return None;
        }
        let inner = self.chunk.lock(offset);
        if !inner.is_occupied() {
            self.chunk.clear_probably_occupied(offset);
            return None;
        }
        Some(TaskMapGuard::new(
            self.task_id(offset),
            inner,
            self.chunk,
            offset,
            self.len,
        ))
    }

    pub(crate) fn task_id(&self, offset: usize) -> TaskId {
        let mut raw = (self.base_id + offset) as u32;
        if self.transient {
            raw |= TRANSIENT_TASK_BIT;
        }
        TaskId::try_from(raw).expect("occupied task slots always have valid task IDs")
    }
}

pub(crate) struct TaskMapGuard<'a, T: TaskSlotValue> {
    key: TaskId,
    pub(crate) modified_count: &'a AtomicU64,
    inner: TaskSlotGuard<'a, T>,
    len: &'a AtomicUsize,
}

impl<'a, T: TaskSlotValue> TaskMapGuard<'a, T> {
    fn new(
        key: TaskId,
        inner: TaskSlotGuard<'a, T>,
        chunk: &'a TaskChunk<T>,
        _offset: usize,
        len: &'a AtomicUsize,
    ) -> Self {
        Self {
            key,
            modified_count: &chunk.modified_count,
            inner,
            len,
        }
    }

    pub(crate) fn key(&self) -> &TaskId {
        &self.key
    }

    pub(crate) fn take_and_vacate(mut self) -> T {
        let detached = self.inner.take_and_vacate();
        // Leave the advisory bit set. A later scan locks, observes authoritative vacancy, and
        // clears it. Immediate ID reuse therefore needs no bitmap clear/set round trip.
        self.len.fetch_sub(1, Ordering::Relaxed);
        detached
    }

    pub(crate) fn vacate(mut self) {
        self.inner.vacate_in_place();
        // As above, stale true is intentional and is cleaned by the next scan.
        self.len.fetch_sub(1, Ordering::Relaxed);
    }
}

impl<T: TaskSlotValue> Deref for TaskMapGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: TaskSlotValue> DerefMut for TaskMapGuard<'_, T> {
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

    use parking_lot::{RawMutex, lock_api::RawMutex as RawMutexTrait};
    use turbo_tasks::TRANSIENT_TASK_BIT;

    use super::*;

    struct TestValue {
        lock: RawMutex,
        occupied: bool,
        value: usize,
        dropped: Option<Arc<AtomicUsize>>,
    }

    impl TestValue {
        fn set(&mut self, value: usize) {
            self.value = value;
        }
    }

    impl Drop for TestValue {
        fn drop(&mut self) {
            if let Some(dropped) = &self.dropped {
                dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    // SAFETY: The raw lock is always initialized, is never moved after publication, and all
    // payload/presence access in TaskMap goes through its guard.
    unsafe impl TaskSlotValue for TestValue {
        const EMPTY: Self = Self {
            lock: <RawMutex as RawMutexTrait>::INIT,
            occupied: false,
            value: 0,
            dropped: None,
        };

        fn lock(&self) {
            self.lock.lock();
        }

        unsafe fn unlock(&self) {
            // SAFETY: Forwarded from the guard that owns this lock.
            unsafe { self.lock.unlock() };
        }

        fn is_occupied(&self) -> bool {
            self.occupied
        }

        fn occupy(&mut self) {
            assert!(!self.occupied);
            self.occupied = true;
        }

        fn take_and_vacate(&mut self) -> Self {
            assert!(self.occupied);
            let value = self.value;
            let dropped = self.dropped.take();
            self.value = 0;
            self.occupied = false;
            Self {
                lock: <RawMutex as RawMutexTrait>::INIT,
                occupied: false,
                value,
                dropped,
            }
        }

        fn vacate_in_place(&mut self) {
            assert!(self.occupied);
            self.value = 0;
            self.dropped = None;
            self.occupied = false;
        }
    }

    fn task_id(raw: u32) -> TaskId {
        TaskId::try_from(raw).unwrap()
    }

    #[test]
    fn grows_across_chunk_boundaries_and_reuses_slots() {
        let map = TaskMap::<TestValue>::new(true);
        for raw in [
            1,
            CHUNK_SIZE as u32 - 1,
            CHUNK_SIZE as u32,
            CHUNK_SIZE as u32 + 1,
        ] {
            map.get_or_insert(task_id(raw)).set(raw as usize);
        }
        assert_eq!(map.len(), 4);
        assert_eq!(
            map.get(task_id(CHUNK_SIZE as u32)).unwrap().value,
            CHUNK_SIZE
        );
        assert_eq!(
            map.remove(task_id(CHUNK_SIZE as u32)).unwrap().value,
            CHUNK_SIZE
        );
        assert!(map.get(task_id(CHUNK_SIZE as u32)).is_none());
        map.get_or_insert(task_id(CHUNK_SIZE as u32)).set(7);
        assert_eq!(map.get(task_id(CHUNK_SIZE as u32)).unwrap().value, 7);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn clear_drops_all_namespaces_in_parallel() {
        let map = TaskMap::<TestValue>::new(true);
        for raw in [1, CHUNK_SIZE as u32 + 1, 7 | TRANSIENT_TASK_BIT] {
            map.get_or_insert(task_id(raw)).set(raw as usize);
        }
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.get(task_id(1)).is_none());
        assert!(map.get(task_id(CHUNK_SIZE as u32 + 1)).is_none());
        assert!(map.get(task_id(7 | TRANSIENT_TASK_BIT)).is_none());
    }

    #[test]
    fn persistent_and_transient_namespaces_do_not_alias() {
        let map = TaskMap::<TestValue>::new(true);
        map.get_or_insert(task_id(7)).set(1);
        map.get_or_insert(task_id(7 | TRANSIENT_TASK_BIT)).set(2);
        assert_eq!(map.get(task_id(7)).unwrap().value, 1);
        assert_eq!(map.get(task_id(7 | TRANSIENT_TASK_BIT)).unwrap().value, 2);
    }

    #[test]
    fn concurrent_first_access_initializes_once() {
        let map = Arc::new(TaskMap::<TestValue>::new(true));
        let barrier = Arc::new(Barrier::new(9));
        let initializations = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let map = map.clone();
                let barrier = barrier.clone();
                let initializations = initializations.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let mut value = map.get_or_insert(task_id(1));
                    if value.value == 0 {
                        initializations.fetch_add(1, Ordering::Relaxed);
                        value.value = 42;
                    }
                    assert_eq!(value.value, 42);
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
        let map = Arc::new(TaskMap::<TestValue>::new(true));
        map.get_or_insert(task_id(1)).set(1);
        let barrier = Arc::new(Barrier::new(9));
        let threads: Vec<_> = (0..8)
            .map(|thread_index| {
                let map = map.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for chunk in 1..32 {
                        let raw = (chunk * CHUNK_SIZE + thread_index + 1) as u32;
                        map.get_or_insert(task_id(raw)).set(raw as usize);
                        assert_eq!(map.get(task_id(1)).unwrap().value, 1);
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
    fn stale_bitmap_bit_is_safely_rechecked() {
        let map = TaskMap::<TestValue>::new(true);
        let id = task_id(1);
        map.get_or_insert(id);
        let chunk = map.chunks()[0];
        {
            let mut value = chunk.get(1).unwrap();
            // Deliberately vacate without clearing the advisory bit to model the handoff window.
            drop(value.inner.take_and_vacate());
            value.len.fetch_sub(1, Ordering::Relaxed);
        }
        assert!(chunk.chunk.is_probably_occupied(1));
        assert!(map.get(id).is_none());
        assert!(
            chunk.chunk.is_probably_occupied(1),
            "point misses leave hint cleanup to bulk scans"
        );
        assert!(chunk.get(1).is_none());
        assert!(!chunk.chunk.is_probably_occupied(1));
    }

    #[test]
    fn chunks_visit_dense_and_sparse_entries_once() {
        let map = TaskMap::<TestValue>::new(true);
        let ids = [1, 2, CHUNK_SIZE as u32 + 3, (CHUNK_SIZE * 4) as u32 + 5];
        for raw in ids {
            map.get_or_insert(task_id(raw)).set(raw as usize);
        }
        let mut seen = Vec::new();
        for chunk in map.chunks() {
            for offset in chunk.chunk.probably_occupied_offsets() {
                if chunk.get(offset).is_some() {
                    seen.push(*chunk.task_id(offset));
                }
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, ids);
    }
}
