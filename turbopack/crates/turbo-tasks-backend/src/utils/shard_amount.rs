use turbo_tasks::parallel::available_parallelism;

/// Computes the shard count for the small map of tasks modified while a snapshot is running.
///
/// Resident tasks no longer use these shards for locking, so the old quadratic collision-avoidance
/// formula would only preallocate thousands of mostly empty snapshot shards. One shard per worker
/// (rounded to a power of two for DashMap) is enough for this rare, short-lived path.
pub fn compute_snapshot_shard_amount(
    num_workers: Option<usize>,
    small_preallocation: bool,
) -> usize {
    if small_preallocation {
        return 4;
    }
    num_workers
        .unwrap_or_else(|| available_parallelism().map_or(4, |v| v.get()))
        .clamp(4, 256)
        .next_power_of_two()
}
