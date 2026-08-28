#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

mod gc_fixture;
mod util;

use std::sync::Arc;

use anyhow::Result;
use turbo_tasks::{
    GcRoot, ResolvedVc, TaskId, Vc, prevent_gc,
    unmark_top_level_task_may_leak_eventually_consistent_state,
};

use crate::{
    gc_fixture::{Selector, create_selector},
    util::create_tt,
};

/// The `TaskId` backing a resolved `Vc` (its `TaskOutput` node).
fn task_id_of<T>(vc: Vc<T>) -> TaskId {
    Vc::into_raw(vc)
        .try_get_task_id()
        .expect("a resolved Vc should be backed by a task")
}

#[turbo_tasks::function]
fn leaf(n: u32) -> Vc<u32> {
    Vc::cell(n)
}

/// Resolves `leaf(n)` and returns the raw `TaskId` backing it, as a `u32`.
///
/// This runs inside a tracked task rather than at the top level of `run_once`, so the plain
/// (eventually consistent) `.resolve()` read below is legal and deterministic — the read is
/// ordered by the task graph instead of racing whatever the session happens to be doing.
#[turbo_tasks::function(operation, root)]
async fn leaf_task_id(n: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(*task_id_of(leaf(n).resolve().await?)))
}

/// A distinct leaf keyed by `n`, used as the child of a persisted root in the cross-session tests
/// so its collection can be observed independently of the shared `leaf`.
#[turbo_tasks::function]
fn orphan_leaf(n: u32) -> Vc<u32> {
    Vc::cell(n)
}

/// A `(operation, root)` op reading `orphan_leaf(n)`: a two-task "root -> subtree". Read at the top
/// level of a `run` it has no persistent parent, so it is a durable root and its child gets
/// `parent_count 1`.
#[turbo_tasks::function(operation, root)]
async fn root_with_child(n: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(*orphan_leaf(n).await? + 1))
}

#[turbo_tasks::function]
async fn branch_a() -> Result<Vc<u32>> {
    Ok(Vc::cell(1 + *leaf(10).await?))
}

#[turbo_tasks::function]
async fn branch_b() -> Result<Vc<u32>> {
    Ok(Vc::cell(2 + *leaf(20).await?))
}

/// A task that pins itself against GC while executing. Once pinned it must survive collection even
/// after it is disconnected.
#[turbo_tasks::function]
async fn pinned_branch() -> Result<Vc<u32>> {
    prevent_gc();
    Ok(Vc::cell(99))
}

/// Reads exactly one branch depending on the selector; flipping it re-executes and disconnects the
/// previously-read branch (and its subtree), which should drop that branch's `parent_count` to 0.
#[turbo_tasks::function(operation, root)]
async fn select(selector: ResolvedVc<Selector>) -> Result<Vc<u32>> {
    let use_b = *selector.await?.get();
    let value = if use_b {
        *branch_b().await?
    } else {
        *branch_a().await?
    };
    Ok(Vc::cell(value))
}

/// Like `select`, but reads `pinned_branch` instead of `branch_a` when the selector is false.
#[turbo_tasks::function(operation, root)]
async fn select_pinned(selector: ResolvedVc<Selector>) -> Result<Vc<u32>> {
    let use_b = *selector.await?.get();
    let value = if use_b {
        *branch_b().await?
    } else {
        *pinned_branch().await?
    };
    Ok(Vc::cell(value))
}

/// A plain leaf read at the top level of a `run_once`. The current task is `None` there, so the
/// leaf's task is created with no persistent parent — but the transient `Once` task that reads it
/// connects it as a child, anchoring it via `transient_ref_count`.
#[turbo_tasks::function]
async fn gc_root_leaf() -> Result<Vc<u32>> {
    Ok(Vc::cell(77))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_collects_disconnected_subtree() {
    let (tt, _persistence_dir) = create_tt("gc_collects_disconnected_subtree");
    let tt2 = tt.clone();

    let result = turbo_tasks::run_once(tt.clone(), async move {
        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        let output = select(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 11);

        // Flip: select drops branch_a; branch_a (parent_count 0) becomes a candidate.
        selector.set(true);
        assert_eq!(*output.read_strongly_consistent().await?, 22);

        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    let collected = tt2.backend().gc_for_testing(&tt2);
    assert_eq!(
        collected, 2,
        "branch_a and its cascaded child leaf(10) should both be collected"
    );
    assert_eq!(
        tt2.backend().gc_for_testing(&tt2),
        0,
        "a second GC pass must collect nothing"
    );

    // Flipping back must recompute branch_a fresh, since it was collected.
    let tt3 = tt.clone();
    let result = turbo_tasks::run_once(tt.clone(), async move {
        let selector_op = create_selector(true);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;
        let output = select(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 22);
        selector.set(false);
        assert_eq!(*output.read_strongly_consistent().await?, 11);
        let _ = &tt3;
        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    tt.stop_and_wait().await;
}

/// A task that pins itself via `prevent_gc()` must survive collection even after it is disconnected
/// from the live graph, because the pin makes it a GC root.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_does_not_collect_pinned_task() {
    let (tt, _persistence_dir) = create_tt("gc_does_not_collect_pinned_task");
    let tt2 = tt.clone();

    let result = turbo_tasks::run_once(tt.clone(), async move {
        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        let output = select_pinned(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 99);

        // Flip: select_pinned re-executes, reads branch_b, and disconnects pinned_branch.
        selector.set(true);
        assert_eq!(*output.read_strongly_consistent().await?, 22);

        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    // GC runs after the run has released activeness, so pinned_branch is disconnected
    // (parent_count 0) and otherwise collectible.
    let collected = tt2.backend().gc_for_testing(&tt2);
    assert_eq!(
        collected, 0,
        "a pinned task must not be collected even when disconnected"
    );

    // A snapshot + evict must not lose the (transient) pin. A pinned task is not forced fully
    // resident — its Meta/Data may be partially evicted — but the session-only
    // `transient_ref_count` is retained as residue (the map entry is kept), so the task stays
    // uncollectible and a subsequent GC still collects nothing.
    tt2.backend().snapshot_and_evict_for_testing(&tt2);
    assert_eq!(
        tt2.backend().gc_for_testing(&tt2),
        0,
        "pinned task must survive eviction and not be collected"
    );

    tt.stop_and_wait().await;
}

/// A parentless task read at the top level of a `run_once` is kept alive by a real anchor, not by
/// any persisted topology flag: the `run_once`'s transient `Once` task connects it as a child and
/// is never disposed, so that `transient_ref_count` keeps it uncollectible.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parentless_top_level_task_kept_by_transient_root() {
    let (tt, _persistence_dir) = create_tt("parentless_top_level_task_kept_by_transient_root");
    let tt2 = tt.clone();
    let tt3 = tt.clone();

    let root_id = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();

        assert_eq!(*gc_root_leaf().await?, 77);

        let root_id = task_id_of(gc_root_leaf().resolve().await?);
        // No persistent parent connected it, so parent_count is 0 — it is not a persistent-graph
        // child of anything.
        assert_eq!(
            tt3.backend().parent_count_for_testing(root_id),
            0,
            "a parentless top-level task has no persistent parent edge"
        );

        anyhow::Ok(root_id)
    })
    .await
    .unwrap();

    // Its `run_once`'s `Once` task is never disposed and keeps it anchored, so GC does not collect
    // it.
    assert_eq!(
        tt2.backend().parent_count_for_testing(root_id),
        0,
        "still parent_count 0 after the run"
    );
    let collected = tt2.backend().gc_for_testing(&tt2);
    assert_eq!(
        collected, 0,
        "kept alive by its (undisposed) transient Once root, not by a topology flag"
    );

    tt.stop_and_wait().await;
}

/// Disposing a root task (as `RootTask::Drop` / `root_task_dispose` does when JS stops listening to
/// a subscription) must release the anchor its child edges placed on the persistent tasks it read,
/// so the subscription's subgraph becomes collectible. Also checks the contract `RootTask::Drop`
/// relies on: disposal is idempotent and safe after the backend stopped.
///
/// Covers the [`GcRoot`] guard by the same path: it is the other way a parentless task gets
/// anchored (the `ProjectContainer` op held by a NAPI `ProjectInstance`), and it pins and releases
/// the same `transient_ref_count` this test asserts on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispose_root_task_releases_anchored_subgraph() {
    let (tt, _persistence_dir) = create_tt("dispose_root_task_releases_anchored_subgraph");

    // Spawn a real root task (as `subscribe` does) whose body reads a persistent leaf, connecting
    // it as a child of the transient root. The root sends the leaf's id out via a oneshot rather
    // than a `run_once` probe: a probe is its own transient `Once` task that would also connect the
    // leaf, and `Once` tasks are never disposed, so the leaf's count would never drop to 0.
    let (tx, rx) = tokio::sync::oneshot::channel();
    let tx = Arc::new(std::sync::Mutex::new(Some(tx)));
    let root_id = tt.spawn_root_task(move || {
        let tx = tx.lock().unwrap().take();
        Box::pin(async move {
            // The root body runs as a top-level task; unmark so the eventually-consistent leaf read
            // is allowed (as `subscribe`'s HMR handler does).
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let leaf_vc = leaf(88);
            let value = *leaf_vc.await?;
            if let Some(tx) = tx {
                let _ = tx.send(task_id_of(leaf_vc.resolve().await?));
            }
            anyhow::Ok(Vc::<u32>::cell(value))
        })
    });

    // The root connects the leaf as its only anchor: no persistent parent (parent_count 0), one
    // transient child edge (transient_ref_count 1).
    let leaf_id = rx.await.unwrap();
    for _ in 0..100 {
        if tt.backend().transient_ref_count_for_testing(leaf_id) > 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        tt.backend().parent_count_for_testing(leaf_id),
        0,
        "leaf read only by the root task has no persistent parent"
    );
    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        1,
        "the transient root task anchors the leaf via exactly one child edge"
    );

    // While the root task is live, the leaf is anchored and must not be collected.
    assert_eq!(
        tt.backend().gc_for_testing(&tt),
        0,
        "leaf anchored by the live root task must not be collected"
    );

    // Dispose the root task, shedding the leaf's transient_ref_count.
    tt.dispose_root_task(root_id);
    // Idempotent: disposing again (explicit JS dispose followed by a later `Drop`) must not panic
    // and must not underflow the child's count.

    tt.dispose_root_task(root_id);

    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        0,
        "disposing the root task released its transient_ref_count anchor on the leaf"
    );

    // Collect in a fresh run: a `run_once` keeps touched tasks active until it returns, so GC has
    // to run after it.

    turbo_tasks::run_once(tt.clone(), async move { anyhow::Ok(()) })
        .await
        .unwrap();

    // The other way the same `transient_ref_count` anchor is taken: a `GcRoot` guard (the
    // `ProjectContainer` op held by a NAPI `ProjectInstance`). Re-pinning the now-unanchored leaf
    // makes it uncollectible again, and dropping the guard releases it.
    let guard = GcRoot::pin(tt.clone(), leaf_id);
    assert_eq!(guard.task_id(), leaf_id);
    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        1,
        "the GcRoot guard pins the task"
    );
    assert_eq!(
        tt.backend().gc_for_testing(&tt),
        0,
        "a task pinned by a live GcRoot must not be collected"
    );
    drop(guard);
    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        0,
        "dropping the GcRoot released the pin"
    );

    assert_eq!(
        tt.backend().gc_for_testing(&tt),
        1,
        "the leaf becomes collectible once the root task that anchored it is disposed"
    );

    // Disposal must be safe after the backend has stopped, as a `RootTask` finalized during Node
    // worker teardown would be (the whole task map is dropped by `stop`).
    tt.stop_and_wait().await;
    tt.dispose_root_task(root_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unpin_after_stop_does_not_panic() {
    let (tt, _persistence_dir) = create_tt("unpin_after_stop_does_not_panic");

    // Pin a real task inside a session (as `prevent_gc` / `DetachedVc::new` would).
    let tt2 = tt.clone();
    let leaf_id = turbo_tasks::run_once(tt.clone(), async move {
        let id = TaskId::try_from(*leaf_task_id(7).read_strongly_consistent().await?)?;
        tt2.pin_task_for_gc(id);
        anyhow::Ok(id)
    })
    .await
    .unwrap();

    // Stop the backend — this drops the in-memory task map, so the pinned task is no longer
    // resident (exactly as at `next build` shutdown).
    tt.stop_and_wait().await;

    tt.unpin_task_for_gc(leaf_id);
}
