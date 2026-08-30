use std::{hint::black_box, num::NonZeroU64, sync::Arc, thread, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use turbo_tasks::{FxDashMap, TaskId};
use turbo_tasks_malloc::TurboMalloc;

#[allow(dead_code, unused_imports)]
#[path = "../src/backend/dense_task_map.rs"]
mod dense_task_map;

use dense_task_map::TaskMap;

const LOOKUP_TASKS: u32 = 64 * 1024;
const BUILD_TASKS: u32 = 8 * 1024;

#[derive(Clone, Copy)]
struct Payload {
    niche: NonZeroU64,
    data: [u64; 15],
}

impl Payload {
    fn new(id: u32) -> Self {
        Self {
            niche: NonZeroU64::new(id as u64).unwrap(),
            data: [id as u64; 15],
        }
    }

    fn value(&self) -> u64 {
        self.niche.get() ^ self.data[0]
    }
}

fn task_id(raw: u32) -> TaskId {
    TaskId::try_from(raw).unwrap()
}

fn dash_map(count: u32, stride: u32) -> FxDashMap<TaskId, Box<Payload>> {
    let map = FxDashMap::with_capacity_and_hasher(count as usize, Default::default());
    for i in 1..=count {
        let raw = i * stride;
        map.insert(task_id(raw), Box::new(Payload::new(raw)));
    }
    map
}

fn dense_map(count: u32, stride: u32) -> TaskMap<Payload> {
    let map = TaskMap::new(true);
    for i in 1..=count {
        let raw = i * stride;
        map.get_or_insert_with(task_id(raw), || Payload::new(raw));
    }
    map
}

fn report_memory(count: u32, stride: u32, shape: &str) {
    TurboMalloc::collect(true);
    let before = TurboMalloc::memory_usage();
    let before_allocations = TurboMalloc::allocation_counters().allocation_count;
    let dash = dash_map(count, stride);
    let dash_bytes = TurboMalloc::memory_usage().saturating_sub(before);
    let dash_allocations = TurboMalloc::allocation_counters()
        .allocation_count
        .saturating_sub(before_allocations);
    black_box(&dash);
    drop(dash);

    TurboMalloc::collect(true);
    let before = TurboMalloc::memory_usage();
    let before_allocations = TurboMalloc::allocation_counters().allocation_count;
    let dense = dense_map(count, stride);
    let dense_bytes = TurboMalloc::memory_usage().saturating_sub(before);
    let dense_allocations = TurboMalloc::allocation_counters()
        .allocation_count
        .saturating_sub(before_allocations);
    black_box(&dense);
    drop(dense);

    println!(
        "resident_storage_memory shape={shape} residents={count} high_water={} \
         dash_bytes={dash_bytes} dash_bytes_per_task={:.2} dash_allocations={dash_allocations} \
         dense_bytes={dense_bytes} dense_bytes_per_task={:.2} \
         dense_allocations={dense_allocations}",
        count * stride,
        dash_bytes as f64 / count as f64,
        dense_bytes as f64 / count as f64,
    );
}

pub fn resident_storage(c: &mut Criterion) {
    report_memory(100_000, 1, "dense");
    report_memory(10_000, 10, "warm_sparse");

    let dash = dash_map(LOOKUP_TASKS, 1);
    let dense = dense_map(LOOKUP_TASKS, 1);

    let mut lookup = c.benchmark_group("resident_storage_lookup");
    lookup.throughput(Throughput::Elements(1));
    let mut next = 1_u32;
    lookup.bench_function("dash_map", |b| {
        b.iter(|| {
            next = next % LOOKUP_TASKS + 1;
            let item = dash.get(&task_id(next)).unwrap();
            black_box(item.value().as_ref().value())
        })
    });
    let mut next = 1_u32;
    lookup.bench_function("dense", |b| {
        b.iter(|| {
            next = next % LOOKUP_TASKS + 1;
            black_box(dense.get(task_id(next)).unwrap().value())
        })
    });
    let mut next = 1_u32;
    lookup.bench_function("dash_map_mut", |b| {
        b.iter(|| {
            next = next % LOOKUP_TASKS + 1;
            let item = dash
                .entry(task_id(next))
                .or_insert_with(|| Box::new(Payload::new(next)));
            black_box(item.value().as_ref().value())
        })
    });
    let mut next = 1_u32;
    lookup.bench_function("dense_mut", |b| {
        b.iter(|| {
            next = next % LOOKUP_TASKS + 1;
            let id = task_id(next);
            let item = dense
                .get(id)
                .unwrap_or_else(|| dense.get_or_insert_with(id, || Payload::new(next)));
            black_box(item.value())
        })
    });
    lookup.finish();

    let mut build = c.benchmark_group("resident_storage_build");
    build.sample_size(20);
    build.throughput(Throughput::Elements(BUILD_TASKS as u64));
    build.bench_with_input(
        BenchmarkId::new("dash_map", BUILD_TASKS),
        &BUILD_TASKS,
        |b, &count| b.iter(|| black_box(dash_map(count, 1))),
    );
    build.bench_with_input(
        BenchmarkId::new("dense", BUILD_TASKS),
        &BUILD_TASKS,
        |b, &count| b.iter(|| black_box(dense_map(count, 1))),
    );
    build.finish();

    let dash = dash_map(LOOKUP_TASKS, 1);
    let dense = dense_map(LOOKUP_TASKS, 1);
    let mut reuse = c.benchmark_group("resident_storage_remove_reuse");
    let mut next = 1_u32;
    reuse.bench_function("dash_map", |b| {
        b.iter(|| {
            next = next % LOOKUP_TASKS + 1;
            let id = task_id(next);
            black_box(dash.remove(&id));
            black_box(dash.insert(id, Box::new(Payload::new(next))));
        })
    });
    let mut next = 1_u32;
    reuse.bench_function("dense", |b| {
        b.iter(|| {
            next = next % LOOKUP_TASKS + 1;
            let id = task_id(next);
            black_box(dense.remove(id));
            black_box(dense.get_or_insert_with(id, || Payload::new(next)));
        })
    });
    reuse.finish();

    let collision_probe = dash_map(0, 1);
    let first = task_id(1);
    let target_shard = collision_probe.determine_shard(collision_probe.hash_usize(&first));
    let colliding_ids: Vec<_> = (1..)
        .map(task_id)
        .filter(|id| {
            collision_probe.determine_shard(collision_probe.hash_usize(id)) == target_shard
        })
        .take(4)
        .collect();
    let dash = Arc::new(FxDashMap::default());
    let dense = Arc::new(TaskMap::new(true));
    for &id in &colliding_ids {
        dash.insert(id, Box::new(Payload::new(*id)));
        dense.get_or_insert_with(id, || Payload::new(*id));
    }
    let mut contention = c.benchmark_group("resident_storage_independent_task_contention");
    contention.bench_function("dash_map_same_shard", |b| {
        b.iter_custom(|iterations| {
            let per_thread = iterations.div_ceil(colliding_ids.len() as u64);
            let start = Instant::now();
            thread::scope(|scope| {
                for &id in &colliding_ids {
                    let dash = dash.clone();
                    scope.spawn(move || {
                        for _ in 0..per_thread {
                            dash.get_mut(&id).unwrap().data[0] ^= 1;
                        }
                    });
                }
            });
            start.elapsed()
        })
    });
    contention.bench_function("dense_independent_locks", |b| {
        b.iter_custom(|iterations| {
            let per_thread = iterations.div_ceil(colliding_ids.len() as u64);
            let start = Instant::now();
            thread::scope(|scope| {
                for &id in &colliding_ids {
                    let dense = dense.clone();
                    scope.spawn(move || {
                        for _ in 0..per_thread {
                            dense.get(id).unwrap().data[0] ^= 1;
                        }
                    });
                }
            });
            start.elapsed()
        })
    });
    contention.finish();

    let dash = dash_map(LOOKUP_TASKS, 1);
    let dense = dense_map(LOOKUP_TASKS, 1);
    let mut iteration = c.benchmark_group("resident_storage_iteration");
    iteration.throughput(Throughput::Elements(LOOKUP_TASKS as u64));
    iteration.bench_function("dash_map", |b| {
        b.iter(|| {
            let sum = dash
                .iter()
                .fold(0_u64, |sum, item| sum ^ item.value().value());
            black_box(sum)
        })
    });
    iteration.bench_function("dense", |b| {
        b.iter(|| {
            let mut sum = 0_u64;
            for chunk in dense.chunks() {
                for slot in chunk.chunk.slots() {
                    if let Some(value) = slot.lock().as_ref() {
                        sum ^= value.value();
                    }
                }
            }
            black_box(sum)
        })
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    iteration.bench_function("dash_map_parallel", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let sums: Vec<u64> = turbo_tasks::parallel::map_collect(dash.shards(), |shard| {
                    shard
                        .read()
                        .iter()
                        .fold(0_u64, |sum, (_, value)| sum ^ value.value())
                });
                black_box(sums.into_iter().fold(0_u64, |sum, value| sum ^ value))
            })
        })
    });
    iteration.bench_function("dense_parallel", |b| {
        // Keep 1024-entry allocation chunks, but schedule 64-entry regions independently so a
        // scan exposes as much parallel work as the highly-sharded DashMap baseline.
        let regions: Vec<_> = dense
            .chunks()
            .into_iter()
            .flat_map(|chunk| {
                (0..dense_task_map::CHUNK_SIZE)
                    .step_by(64)
                    .map(move |start| (chunk, start))
            })
            .collect();
        b.iter(|| {
            runtime.block_on(async {
                let sums: Vec<u64> =
                    turbo_tasks::parallel::map_collect(&regions, |&(chunk, start)| {
                        chunk.chunk.slots()[start..start + 64]
                            .iter()
                            .fold(0_u64, |sum, slot| {
                                sum ^ slot.lock().as_ref().map_or(0, Payload::value)
                            })
                    });
                black_box(sums.into_iter().fold(0_u64, |sum, value| sum ^ value))
            })
        })
    });
    iteration.finish();
}
