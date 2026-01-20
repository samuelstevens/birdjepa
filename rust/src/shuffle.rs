//! Concurrent shuffle buffer for the pipelined loader.
//!
//! Thread-safe buffer with configurable minimum fill level for entropy.
//! Workers push, main thread pops random samples.

use rand::prelude::*;
use rand::rngs::StdRng;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Thread-safe shuffle buffer with blocking push/pop and minimum fill level.
pub struct ConcurrentShuffleBuffer<T> {
    inner: Mutex<ShuffleInner<T>>,
    capacity: usize,
    min_size: usize,
    not_full: Condvar,
    ready: Condvar,
}

struct ShuffleInner<T> {
    buffer: Vec<T>,
    rng: StdRng,
    closed: bool,
}

impl<T> ConcurrentShuffleBuffer<T> {
    /// Create a new shuffle buffer.
    ///
    /// - `capacity`: Maximum buffer size. Push blocks when full.
    /// - `min_size`: Minimum samples before pop returns. Use 0 for no minimum.
    /// - `seed`: RNG seed for deterministic shuffling.
    pub fn new(capacity: usize, min_size: usize, seed: u64) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(min_size <= capacity, "min_size must be <= capacity");
        Self {
            inner: Mutex::new(ShuffleInner {
                buffer: Vec::with_capacity(capacity),
                rng: StdRng::seed_from_u64(seed),
                closed: false,
            }),
            capacity,
            min_size,
            not_full: Condvar::new(),
            ready: Condvar::new(),
        }
    }

    /// Push a sample. Blocks if buffer is at capacity.
    /// Returns false if buffer is closed (shutdown).
    pub fn push(&self, sample: T) -> bool {
        let mut guard = self.inner.lock().unwrap();
        loop {
            if guard.closed {
                return false;
            }
            if guard.buffer.len() < self.capacity {
                guard.buffer.push(sample);
                // Notify if we have enough samples (or min_size is 0)
                if self.min_size == 0 || guard.buffer.len() >= self.min_size {
                    self.ready.notify_one();
                }
                return true;
            }
            let (new_guard, _) = self
                .not_full
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap();
            guard = new_guard;
        }
    }

    /// Try to pop a random sample with timeout.
    /// Returns Some(sample) if available, None if timed out or closed+empty.
    /// Use is_closed() to distinguish timeout vs closed.
    pub fn try_pop(&self, timeout: Duration) -> Option<T> {
        let mut guard = self.inner.lock().unwrap();

        // First try without waiting
        if let Some(sample) = self.try_pop_inner(&mut guard) {
            return Some(sample);
        }

        if guard.closed {
            return None;
        }

        // Wait once with timeout
        let (mut new_guard, _) = self.ready.wait_timeout(guard, timeout).unwrap();

        // Try again after wait
        self.try_pop_inner(&mut new_guard)
    }

    /// Internal helper: try to pop if conditions are met.
    fn try_pop_inner(&self, guard: &mut std::sync::MutexGuard<ShuffleInner<T>>) -> Option<T> {
        let len = guard.buffer.len();

        // Can pop if: min_size is 0 and we have data, OR we have >= min_size
        let can_pop = len > 0 && (self.min_size == 0 || len >= self.min_size);
        if can_pop {
            let idx = guard.rng.gen_range(0..len);
            let sample = guard.buffer.swap_remove(idx);
            self.not_full.notify_one();
            return Some(sample);
        }

        // Shutdown: drain remaining regardless of min_size
        if guard.closed && len > 0 {
            let idx = guard.rng.gen_range(0..len);
            let sample = guard.buffer.swap_remove(idx);
            self.not_full.notify_one();
            return Some(sample);
        }

        None
    }

    /// Check if buffer is closed.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }

    /// Close the buffer. Wakes all waiters, causes push() to return false.
    pub fn close(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.closed = true;
        self.not_full.notify_all();
        self.ready.notify_all();
    }

    /// Get current number of samples in the buffer.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().buffer.len()
    }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_push_pop() {
        let buffer = ConcurrentShuffleBuffer::new(10, 0, 42);

        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(buffer.push(3));

        let mut popped = vec![];
        while let Some(v) = buffer.try_pop(Duration::from_millis(10)) {
            popped.push(v);
        }

        popped.sort();
        assert_eq!(popped, vec![1, 2, 3]);
    }

    #[test]
    fn test_min_size() {
        let buffer = ConcurrentShuffleBuffer::new(10, 5, 42);

        // Push 3 samples (below min_size)
        for i in 0..3 {
            buffer.push(i);
        }

        // Should timeout (not enough samples)
        assert!(buffer.try_pop(Duration::from_millis(10)).is_none());

        // Push 2 more (now at min_size)
        buffer.push(3);
        buffer.push(4);

        // Should succeed
        assert!(buffer.try_pop(Duration::from_millis(10)).is_some());
    }

    #[test]
    fn test_close() {
        let buffer = Arc::new(ConcurrentShuffleBuffer::new(10, 5, 42));

        // Push some samples
        for i in 0..3 {
            buffer.push(i);
        }

        // Close buffer
        buffer.close();

        // Should drain remaining (ignoring min_size)
        let mut count = 0;
        while buffer.try_pop(Duration::from_millis(10)).is_some() {
            count += 1;
        }
        assert_eq!(count, 3);

        // Push should return false
        assert!(!buffer.push(99));
    }

    #[test]
    fn test_concurrent() {
        // Use min_size=0 since we just test concurrent push/pop, not min_size behavior
        let buffer = Arc::new(ConcurrentShuffleBuffer::new(100, 0, 42));
        let buffer2 = Arc::clone(&buffer);

        // Producer thread
        let producer = thread::spawn(move || {
            for i in 0..50 {
                buffer2.push(i);
            }
        });

        // Consumer: wait for samples
        thread::sleep(Duration::from_millis(50));
        let mut popped = vec![];
        while popped.len() < 50 {
            if let Some(v) = buffer.try_pop(Duration::from_millis(100)) {
                popped.push(v);
            }
        }

        producer.join().unwrap();

        popped.sort();
        assert_eq!(popped, (0..50).collect::<Vec<_>>());
    }
}
