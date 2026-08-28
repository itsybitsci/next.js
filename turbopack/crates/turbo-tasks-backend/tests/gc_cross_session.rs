#![feature(arbitrary_self_types)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

//! Cross-session collection of GC roots.

mod gc_fixture;
mod util;

use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use anyhow::Result;
use turbo_tasks::{
    GcRoot, TurboTasks, Vc, unmark_top_level_task_may_leak_eventually_consistent_state,
};
use turbo_tasks_backend::TurboTasksBackend;

use crate::{
    gc_fixture::{create_constant, diamond_root_op},
    util::{create_persistence_dir, reopen_tt_with_gc, reopen_tt_with_gc_ttl},
};

/// Counts executions of [`orphan_leaf`], keyed by its argument. A collected task has to re-execute
/// when it is next requested, so a bump here is the observable signal that it was reclaimed —
/// whereas a task that merely got evicted restores from disk without executing.
static LEAF_EXECUTIONS: [AtomicU32; 3] = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];

fn leaf_executions(n: u32) -> u32 {
    LEAF_EXECUTIONS[n as usize].load(Ordering::Relaxed)
}

/// A leaf keyed by `n`, so each root gets a distinct child whose fate can be observed on its own.
#[turbo_tasks::function]
fn orphan_leaf(n: u32) -> Vc<u32> {
    LEAF_EXECUTIONS[n as usize].fetch_add(1, Ordering::Relaxed);
    Vc::cell(n)
}

/// A `(operation, root)` op reading `orphan_leaf(n)`: a two-task "root -> subtree". Read at the top
/// level of a session it has no persistent parent, so it is a durable GC root and its leaf gets
/// `parent_count 1`.
#[turbo_tasks::function(operation, root)]
async fn root_with_child(n: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(*orphan_leaf(n).await? + 1))
}

/// Runs GC until at least `want` tasks have been collected, or gives up after 20 passes.
///
/// Both tests need this loop rather than a fixed pass count: a root must first be demoted to
/// `FirstStale(t)` and only then age out, only one pass per session may demote
/// (`gc.rs::first_gc_pass_of_session`), and age-out needs elapsed > TTL *strictly* — so with
/// `gc.rs::now_ms` at millisecond resolution a pass sharing a millisecond with the demotion
/// collects nothing. Returns the total collected.
async fn gc_until_collected(tt: &Arc<TurboTasks<TurboTasksBackend>>, want: usize) -> usize {
    let mut collected = 0usize;
    for _ in 0..20 {
        collected += tt.backend().gc_for_testing(tt);
        if collected >= want {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    collected
}

/// Width of the diamond built for the disk-only forward-dep scrub test (see `gc_fixture`): the
/// fanout gives many chances for the racing interleaving where an `A` scrubs a not-yet-restored
/// `B`.
const DIAMOND_FANOUT: u32 = 64;

/// **A root is kept alive by being used, and reclaimed by not being used** — across process
/// restarts.
///
/// Two sibling roots are persisted in session 1; from then on only one is ever requested again. The
/// reused root and its subtree must survive every session; the abandoned one must eventually be
/// collected, taking its subtree with it. Neither outcome is reachable by the resident-scan GC
/// alone: both roots have `parent_count == 0` forever, so only the cross-session roots map
/// distinguishes them.
///
/// The TTL is forced to 0 so the age-out lands inside the test rather than days later. Demotion and
/// age-out remain two distinct steps — the pass that first notices a root is gone only starts its
/// clock — which is why each session runs two passes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reused_root_survives_sessions_that_abandon_its_sibling() {
    let dir = create_persistence_dir("reused_root_survives_sessions_that_abandon_its_sibling");

    // Session 1: build both subtrees and anchor each root with a pin, the way the embedder holds a
    // live handle to a route. The pin makes the task a durable root and gets it recorded in the
    // persisted roots map on shutdown.
    let kept_root = {
        let tt = reopen_tt_with_gc(&dir);
        let ids = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let kept = root_with_child(1);
            let dropped = root_with_child(2);
            assert_eq!(*kept.read_strongly_consistent().await?, 2);
            assert_eq!(*dropped.read_strongly_consistent().await?, 3);
            anyhow::Ok((kept.task_id(), dropped.task_id()))
        })
        .await
        .unwrap();

        // Both are anchored in session 1, so both are persisted as roots.
        let kept_pin = GcRoot::pin(tt.clone(), ids.0);
        let dropped_pin = GcRoot::pin(tt.clone(), ids.1);
        tt.backend().snapshot_and_evict_for_testing(&tt);
        drop(kept_pin);
        drop(dropped_pin);

        tt.stop_and_wait().await;
        ids.0
    };

    // Sessions 2 and 3: only root 1 is ever requested again. Session 2's first pass demotes root 2
    // (starts its clock) and a later pass ages it out; session 3 proves the collection stuck and
    // left the surviving root unharmed.
    let mut total_collected = 0usize;
    for session in 2..=3 {
        let tt = reopen_tt_with_gc_ttl(&dir, Duration::ZERO);
        let tt2 = tt.clone();
        turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            // Re-request root 1 only, and restore it so it has a resident entry to pin. Root 2 is
            // never mentioned in this session.
            assert_eq!(
                *root_with_child(1).read_strongly_consistent().await?,
                2,
                "the reused root must still compute in session {session}"
            );
            let _ = &tt2;
            anyhow::Ok(())
        })
        .await
        .unwrap();

        // Re-anchor only the reused root; root 2 has no pin this session.
        let kept_pin = GcRoot::pin(tt.clone(), kept_root);
        // The abandoned root and its leaf are the two collectible tasks here.
        total_collected += gc_until_collected(&tt, 2).await;
        drop(kept_pin);

        tt.stop_and_wait().await;
    }

    // The abandoned root and its leaf are two tasks, and nothing else in this graph is collectible
    // (root 1 is re-anchored every session, and its leaf has a parent). Requiring the pair
    // distinguishes "the sibling was reclaimed" from "GC did nothing at all".
    assert!(
        total_collected >= 2,
        "the abandoned root and its leaf should have been collected across sessions 2-3 (got \
         {total_collected})"
    );

    // Session 4: the surviving root must still be cached, and the collected one must rebuild.
    {
        let tt = reopen_tt_with_gc(&dir);
        let kept_before = leaf_executions(1);
        let dropped_before = leaf_executions(2);
        let result = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();

            // Both must still produce the right value: the survivor from cache, the collected one
            // rebuilt from scratch. A dangling edge left by the cross-session collection — or a
            // resurrected half-deleted task — would surface as a wrong value or a panic here.
            assert_eq!(*root_with_child(1).read_strongly_consistent().await?, 2);
            assert_eq!(*root_with_child(2).read_strongly_consistent().await?, 3);

            // The reused root's subtree was never collected, so its leaf is served from the
            // persisted cache without executing again.
            assert_eq!(
                leaf_executions(1),
                kept_before,
                "the reused root's subtree must be served from cache, not recomputed"
            );
            // The abandoned root's subtree was collected, so requesting it again has to re-execute.
            assert_eq!(
                leaf_executions(2),
                dropped_before + 1,
                "the abandoned root's subtree must have been collected and rebuilt"
            );
            anyhow::Ok(())
        })
        .await;
        tt.stop_and_wait().await;
        result.unwrap();
    }
}

/// Collecting an orphaned root whose subtree contains a **forward cell-dependency** to a target
/// that is **disk-only** (not restored this session) must scrub that stale reverse edge by
/// restoring the live target, rather than tripping a "target not resident, would resurrect a
/// collected task" guard.
///
/// Session 1 builds and persists the whole diamond (see the fixture above). Session 2 reopens —
/// nothing re-requests the root, so with TTL 0 it ages out — and runs GC: the root is collected and
/// its `A`/`B` children cascade concurrently, so an `A`'s `CleanupOldEdges` can open its target `B`
/// before `B` has been restored from disk. The pass completing without a panic is the assertion;
/// session 3 confirms a clean recompute (no dangling reverse edge survived).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn gc_collect_scrubs_disk_only_forward_dep_target() {
    let dir = create_persistence_dir("gc_collect_scrubs_disk_only_forward_dep_target");

    // Session 1: build the diamond and persist it. The root is anchored with a pin and snapshotted
    // so it lands in the persisted roots map — without that it is never a tracked root at all, so
    // nothing is ever collected and the cascade under test never runs.
    {
        let tt = reopen_tt_with_gc(&dir);
        let root_id = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let constant_op = create_constant();
            let constant_vc = constant_op.resolve().strongly_consistent().await?;
            // Read the root so the whole diamond is built and persisted.
            let root_op = diamond_root_op(constant_vc, DIAMOND_FANOUT);
            root_op.read_strongly_consistent().await?;
            anyhow::Ok(root_op.task_id())
        })
        .await
        .unwrap();

        let root_pin = GcRoot::pin(tt.clone(), root_id);
        // A GC pass is what admits a live root to the persisted map, so run one explicitly while
        // the root is still anchored; the snapshot alone leaves nothing for session 2 to
        // age out.
        tt.backend().gc_for_testing(&tt);
        tt.backend().snapshot_and_evict_for_testing(&tt);
        drop(root_pin);

        tt.stop_and_wait().await;
    }

    // Session 2: reopen, TTL 0, run GC. The diamond root is never re-requested, so it ages out and
    // collecting it cascades to every A/B pair.
    {
        let tt = reopen_tt_with_gc_ttl(&dir, Duration::ZERO);
        let tt2 = tt.clone();
        turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            // No panic in these passes is the real assertion.
            let collected = gc_until_collected(&tt2, DIAMOND_FANOUT as usize + 1).await;
            // The diamond is a root plus DIAMOND_FANOUT A/B pairs, so requiring more than the
            // fanout distinguishes "the subtree was reclaimed" from "GC nibbled at an edge of it".
            assert!(
                collected > DIAMOND_FANOUT as usize,
                "the orphaned diamond root subtree should be collected (got {collected})"
            );
            anyhow::Ok(())
        })
        .await
        .unwrap();
        tt.stop_and_wait().await;
    }

    // Session 3: a clean recompute must still work — no dangling reverse edge left on any target.
    {
        let tt = reopen_tt_with_gc(&dir);
        let result = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let constant_op = create_constant();
            let constant_vc = constant_op.resolve().strongly_consistent().await?;
            diamond_root_op(constant_vc, DIAMOND_FANOUT)
                .read_strongly_consistent()
                .await?;
            anyhow::Ok(())
        })
        .await;
        tt.stop_and_wait().await;
        result.unwrap();
    }
}
