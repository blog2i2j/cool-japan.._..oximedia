//! Performance and latency tests for collaborative editing components.
//!
//! Tests in this module cover:
//! - CRDT merge performance with 10+ concurrent editors using `GCounter`/`PNCounter`.
//! - Elapsed time measurement for bulk operation merges.
//! - Latency round-trip simulation via `std::sync::mpsc` channels.
//! - Concurrent editing stress tests with simulated network partitions.
//!
//! Slow tests are marked `#[ignore]` and can be run with:
//! ```text
//! cargo test -p oximedia-collab -- --ignored
//! ```

#[cfg(test)]
mod tests {
    use crate::crdt_primitives::{GCounter, GSet, PNCounter};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    // ─────────────────────────────────────────────────────────────────────────
    // CRDT merge convergence tests (fast, always run)
    // ─────────────────────────────────────────────────────────────────────────

    /// Spawn 10 threads, each incrementing a `GCounter` 100 times under their
    /// own node-id, then merge all states into one.  The merged total must equal
    /// the sum of all per-thread increments (10 × 100 = 1 000).
    #[test]
    fn test_gcounter_10_threads_convergence() {
        const THREADS: usize = 10;
        const OPS_PER_THREAD: u64 = 100;

        // Each thread builds its own local GCounter.
        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                thread::spawn(move || {
                    let node_id = format!("node-{}", i);
                    let mut counter = GCounter::new();
                    for _ in 0..OPS_PER_THREAD {
                        counter.increment(&node_id, 1);
                    }
                    counter
                })
            })
            .collect();

        // Collect all partial states.
        let partials: Vec<GCounter> = handles
            .into_iter()
            .map(|h| h.join().expect("thread should not panic"))
            .collect();

        // Merge all into a single global counter.
        let mut global = GCounter::new();
        for partial in &partials {
            global.merge(partial);
        }

        assert_eq!(
            global.value(),
            THREADS as u64 * OPS_PER_THREAD,
            "merged GCounter value must equal sum of all thread contributions"
        );
    }

    /// Spawn 10 threads using `PNCounter`: half increment, half decrement.
    /// Merged result = 5 × 100 − 5 × 50 = 500 − 250 = 250.
    #[test]
    fn test_pncounter_10_threads_convergence() {
        const THREADS: usize = 10;

        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                thread::spawn(move || {
                    let node_id = format!("node-{}", i);
                    let mut counter = PNCounter::new();
                    if i % 2 == 0 {
                        counter.increment(&node_id, 100);
                    } else {
                        counter.decrement(&node_id, 50);
                    }
                    counter
                })
            })
            .collect();

        let partials: Vec<PNCounter> = handles
            .into_iter()
            .map(|h| h.join().expect("thread should not panic"))
            .collect();

        let mut global = PNCounter::new();
        for partial in &partials {
            global.merge(partial);
        }

        // 5 threads × 100 increments − 5 threads × 50 decrements = 250.
        assert_eq!(global.value(), 250, "merged PNCounter value must be 250");
    }

    /// Merge of 1 000 operations on a `GSet` completes within a generous time bound.
    /// This is an always-run test (not `#[ignore]`); the bound is 500 ms.
    #[test]
    fn test_gset_1000_op_merge_within_budget() {
        use crate::crdt_primitives::GSet;
        let mut a = GSet::new();
        let mut b = GSet::new();
        for i in 0u32..500 {
            a.insert(format!("a-item-{}", i));
        }
        for i in 0u32..500 {
            b.insert(format!("b-item-{}", i));
        }
        let start = Instant::now();
        a.merge(&b);
        let elapsed = start.elapsed();
        // 1 000 set-union ops must complete within 500 ms on any modern CI runner.
        assert!(
            elapsed < Duration::from_millis(500),
            "GSet merge of 1000 ops took {:?}, expected < 500 ms",
            elapsed
        );
        assert_eq!(a.len(), 1000, "merged set must contain 1000 elements");
    }

    /// Latency: measure round-trip time for local sync via `mpsc` channels.
    ///
    /// We simulate a client-server pair: the client sends 1 000 messages over an
    /// in-process channel and the server echoes them back.  The median per-message
    /// latency (channel send + receive) must be below 1 ms.
    #[test]
    fn test_mpsc_sync_round_trip_latency() {
        use std::sync::mpsc;

        const MSG_COUNT: usize = 1_000;

        let (client_tx, server_rx) = mpsc::channel::<u64>();
        let (server_tx, client_rx) = mpsc::channel::<u64>();

        // Server thread: echo every received value back.
        let server = thread::spawn(move || {
            while let Ok(msg) = server_rx.recv() {
                server_tx.send(msg).expect("server send must succeed");
            }
        });

        let mut latencies = Vec::with_capacity(MSG_COUNT);
        for i in 0..MSG_COUNT as u64 {
            let t0 = Instant::now();
            client_tx.send(i).expect("client send must succeed");
            let echo = client_rx.recv().expect("client recv must succeed");
            let elapsed = t0.elapsed();
            assert_eq!(echo, i, "echo must match sent value");
            latencies.push(elapsed);
        }

        // Drop the send side so the server thread exits.
        drop(client_tx);
        server.join().expect("server thread must not panic");

        // Compute median latency.
        latencies.sort();
        let median = latencies[MSG_COUNT / 2];
        // Allow up to 1 ms per round-trip for local channel operations.
        assert!(
            median < Duration::from_millis(1),
            "median round-trip latency {:?} exceeds 1 ms budget",
            median
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Slow / large-scale tests — marked #[ignore]
    // ─────────────────────────────────────────────────────────────────────────

    /// Benchmark: merge 10 000 `GCounter` operations across 10 threads and report
    /// elapsed time.  This is marked `#[ignore]` because it is slow on weak CI.
    #[test]
    #[ignore]
    fn bench_gcounter_10k_ops_merge() {
        const THREADS: usize = 10;
        const OPS: u64 = 1_000;

        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                thread::spawn(move || {
                    let node = format!("n{}", i);
                    let mut c = GCounter::new();
                    for _ in 0..OPS {
                        c.increment(&node, 1);
                    }
                    c
                })
            })
            .collect();

        let partials: Vec<GCounter> = handles
            .into_iter()
            .map(|h| h.join().expect("no panic"))
            .collect();

        let start = Instant::now();
        let mut global = GCounter::new();
        for p in &partials {
            global.merge(p);
        }
        let elapsed = start.elapsed();

        // Informational — not a hard assertion (the test is already ignored).
        eprintln!(
            "[bench] merged {} ops across {} threads in {:?}",
            THREADS as u64 * OPS,
            THREADS,
            elapsed
        );

        assert_eq!(global.value(), THREADS as u64 * OPS);
    }

    /// Concurrent editing stress test: 16 threads each making 200 edits on a
    /// shared `GCounter` through a mutex, simulating network partitions by
    /// introducing thread yields between operations.
    #[test]
    #[ignore]
    fn stress_concurrent_editing_with_network_partitions() {
        const THREADS: usize = 16;
        const EDITS: u64 = 200;

        let shared = Arc::new(Mutex::new(GCounter::new()));

        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                let shared = Arc::clone(&shared);
                thread::spawn(move || {
                    let node = format!("editor-{}", i);
                    // Build local state (simulating offline edits / partition).
                    let mut local = GCounter::new();
                    for _ in 0..EDITS {
                        local.increment(&node, 1);
                        // Yield to simulate interleaving / network delay.
                        thread::yield_now();
                    }
                    // Re-connect: merge local state into shared.
                    let mut guard = shared.lock().expect("mutex should not be poisoned");
                    guard.merge(&local);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("no panic");
        }

        let final_value = shared.lock().expect("mutex should not be poisoned").value();

        assert_eq!(
            final_value,
            THREADS as u64 * EDITS,
            "all edits must converge after re-merging"
        );
    }

    /// Measure latency of 100 000 mpsc round-trips and verify p99 < 5 ms.
    #[test]
    #[ignore]
    fn bench_mpsc_100k_round_trip_latency() {
        use std::sync::mpsc;

        const N: usize = 100_000;

        let (tx, rx) = mpsc::channel::<u64>();
        let (back_tx, back_rx) = mpsc::channel::<u64>();

        let server = thread::spawn(move || {
            while let Ok(v) = rx.recv() {
                back_tx.send(v).expect("send back");
            }
        });

        let mut latencies = Vec::with_capacity(N);
        for i in 0..N as u64 {
            let t0 = Instant::now();
            tx.send(i).expect("send");
            let _ = back_rx.recv().expect("recv");
            latencies.push(t0.elapsed());
        }
        drop(tx);
        server.join().expect("server");

        latencies.sort();
        let p99 = latencies[(N as f64 * 0.99) as usize];
        eprintln!("[bench] p99 round-trip latency: {:?}", p99);
        assert!(
            p99 < Duration::from_millis(5),
            "p99 {:?} exceeds 5 ms budget",
            p99
        );
    }

    /// GSet: grow-only set CRDT convergence across 10 threads.
    #[test]
    fn test_gset_10_threads_convergence() {
        const THREADS: usize = 10;
        const ITEMS_PER_THREAD: usize = 50;

        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                thread::spawn(move || {
                    let mut set = GSet::new();
                    for j in 0..ITEMS_PER_THREAD {
                        set.insert(format!("thread-{}-item-{}", i, j));
                    }
                    set
                })
            })
            .collect();

        let partials: Vec<GSet<String>> = handles
            .into_iter()
            .map(|h| h.join().expect("no panic"))
            .collect();

        let mut global = GSet::new();
        for p in &partials {
            global.merge(p);
        }

        assert_eq!(
            global.len(),
            THREADS * ITEMS_PER_THREAD,
            "all {} unique items must be present after merge",
            THREADS * ITEMS_PER_THREAD
        );
    }

    /// Verify that CRDT merge is idempotent: merging the same state twice
    /// produces the same result as merging once.
    #[test]
    fn test_gcounter_merge_idempotent() {
        let mut a = GCounter::new();
        a.increment(&"node-A".to_string(), 10);

        let mut b = GCounter::new();
        b.increment(&"node-B".to_string(), 20);

        let mut merged_once = a.clone();
        merged_once.merge(&b);

        let mut merged_twice = a.clone();
        merged_twice.merge(&b);
        merged_twice.merge(&b); // merge again

        assert_eq!(
            merged_once.value(),
            merged_twice.value(),
            "CRDT merge must be idempotent"
        );
    }

    /// Verify that CRDT merge is commutative: A.merge(B) == B.merge(A).
    #[test]
    fn test_gcounter_merge_commutative() {
        let mut a = GCounter::new();
        a.increment(&"node-A".to_string(), 7);

        let mut b = GCounter::new();
        b.increment(&"node-B".to_string(), 13);

        let mut a_then_b = a.clone();
        a_then_b.merge(&b);

        let mut b_then_a = b.clone();
        b_then_a.merge(&a);

        assert_eq!(
            a_then_b.value(),
            b_then_a.value(),
            "CRDT merge must be commutative"
        );
    }
}
