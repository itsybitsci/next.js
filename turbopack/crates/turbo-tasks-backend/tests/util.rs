//! Shared setup for tests that need a real on-disk persistent backend and direct access to
//! [`TurboTasksBackend`] test hooks (`snapshot_and_evict_for_testing`, `parent_count_for_testing`,
//! …).
//!
//! These tests can't use `turbo_tasks_testing::register!`: that harness hands back an
//! `Arc<dyn TurboTasksApi>`, which erases the concrete backend type and makes `tt.backend()`
//! unreachable. It also owns the session loop (re-running the body and comparing results), whereas
//! these tests need to drive snapshots — and sometimes a DB reopen — on their own schedule.
//!
//! This module is compiled into each test binary separately, so any helper a given binary doesn't
//! call reads as dead code there.
#![allow(dead_code)]

use std::{path::Path, sync::Arc, time::Duration};

use turbo_tasks::TurboTasks;
use turbo_tasks_backend::{
    BackendOptions, BackingStorageOptions, EvictionMode, GitVersionInfo, TurboTasksBackend,
};

/// Opens a backend rooted at `path`.
///
/// Reusing the same `path` (after the previous backend has been stopped) reopens the persisted
/// database, which is how a test can assert that state survives a restart.
fn open_tt_at(path: &Path, num_workers: usize) -> Arc<TurboTasks<TurboTasksBackend>> {
    // GC on by default: these are the GC tests, and `gc_for_testing` asserts the backend is
    // GC-enabled. Per-backend rather than via `TURBO_ENGINE_GC`, which every test in the binary
    // would share.
    open_tt_at_with_gc(path, num_workers, Some(true), None)
}

/// Like [`open_tt_at`], but forces the GC on or off for this backend instead of deriving it from
/// the `TURBO_ENGINE_GC` env var.
///
/// A test that depends on the **persisted GC roots map** must force it on: the map is only written
/// by the GC branch of `snapshot_and_persist`, so with GC off a session persists an empty root set
/// and the cross-session behaviour under test silently never engages. Forcing it per-backend keeps
/// it off the process environment, which every test in the binary shares.
fn open_tt_at_with_gc(
    path: &Path,
    num_workers: usize,
    gc: Option<bool>,
    gc_root_ttl: Option<Duration>,
) -> Arc<TurboTasks<TurboTasksBackend>> {
    TurboTasks::new(TurboTasksBackend::new(
        BackendOptions {
            num_workers: Some(num_workers),
            small_preallocation: true,
            // Avoid racing with the background snapshot loop; these tests drive
            // snapshot_and_evict_for_testing manually.
            storage_mode: Some(turbo_tasks_backend::StorageMode::ReadWriteOnShutdown),
            eviction_mode: EvictionMode::Full,
            gc,
            gc_root_ttl,
            ..Default::default()
        },
        turbo_tasks_backend::turbo_backing_storage(
            path,
            &GitVersionInfo {
                describe: "test-unversioned",
                dirty: false,
            },
            BackingStorageOptions {
                is_short_session: true,
                skip_compaction: true,
                ..Default::default()
            },
        )
        .unwrap()
        .0,
    ))
}

/// A persistence directory that outlives the backends opened on it, so a test can stop one backend
/// and open another on the same path to simulate a restart. Pair with [`reopen_tt_with_gc`].
pub fn create_persistence_dir(name: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("{name}-"))
        .tempdir()
        .unwrap()
}

/// Opens a backend on an existing persistence directory — a new *session* over the same database —
/// with the GC forced on, for tests that assert on cross-session root behaviour. The previous
/// backend must already be stopped (`stop_and_wait`) so its shutdown snapshot has been flushed.
///
/// See [`open_tt_at_with_gc`] for why forcing the GC here rather than via the env var matters.
pub fn reopen_tt_with_gc(dir: &tempfile::TempDir) -> Arc<TurboTasks<TurboTasksBackend>> {
    open_tt_at_with_gc(dir.path(), 2, Some(true), None)
}

/// [`reopen_tt_with_gc`] with the GC root TTL pinned, so a cross-session test can age a root out
/// inside the test rather than days later. Set at construction because the TTL is resolved once
/// when the backend is built.
pub fn reopen_tt_with_gc_ttl(
    dir: &tempfile::TempDir,
    gc_root_ttl: Duration,
) -> Arc<TurboTasks<TurboTasksBackend>> {
    open_tt_at_with_gc(dir.path(), 2, Some(true), Some(gc_root_ttl))
}

/// Reopens a backend on an existing persistence directory with GC on and the id-reuse window
/// pinned, for tests that assert on ids surviving a restart.
pub fn reopen_tt_with_id_reuse(
    dir: &tempfile::TempDir,
    id_reuse_delay_cycles: u32,
) -> Arc<TurboTasks<TurboTasksBackend>> {
    TurboTasks::new(TurboTasksBackend::new(
        BackendOptions {
            num_workers: Some(2),
            small_preallocation: true,
            storage_mode: Some(turbo_tasks_backend::StorageMode::ReadWriteOnShutdown),
            eviction_mode: EvictionMode::Full,
            gc: Some(true),
            id_reuse_delay_cycles: Some(id_reuse_delay_cycles),
            ..Default::default()
        },
        turbo_tasks_backend::turbo_backing_storage(
            dir.path(),
            &GitVersionInfo {
                describe: "test-unversioned",
                dirty: false,
            },
            BackingStorageOptions {
                is_short_session: true,
                skip_compaction: true,
                ..Default::default()
            },
        )
        .unwrap()
        .0,
    ))
}

/// A fresh persistent backend in its own temp directory, with `num_workers` workers.
pub fn create_tt_with_workers(
    name: &str,
    num_workers: usize,
) -> (Arc<TurboTasks<TurboTasksBackend>>, tempfile::TempDir) {
    let dir = create_persistence_dir(name);
    let tt = open_tt_at(dir.path(), num_workers);
    (tt, dir)
}

/// A fresh persistent backend in its own temp directory, with the default worker count.
pub fn create_tt(name: &str) -> (Arc<TurboTasks<TurboTasksBackend>>, tempfile::TempDir) {
    create_tt_with_workers(name, 2)
}

/// A fresh persistent backend built from caller-supplied [`BackendOptions`], for tests that need
/// to pin an option the helpers above don't expose. The caller owns every option, including `gc`
/// — the other helpers' GC-on default does not apply.
pub fn create_tt_with_options(
    name: &str,
    options: BackendOptions,
) -> (Arc<TurboTasks<TurboTasksBackend>>, tempfile::TempDir) {
    let dir = create_persistence_dir(name);
    let tt = TurboTasks::new(TurboTasksBackend::new(
        options,
        turbo_tasks_backend::turbo_backing_storage(
            dir.path(),
            &GitVersionInfo {
                describe: "test-unversioned",
                dirty: false,
            },
            BackingStorageOptions {
                is_short_session: true,
                skip_compaction: true,
                ..Default::default()
            },
        )
        .unwrap()
        .0,
    ));
    (tt, dir)
}
