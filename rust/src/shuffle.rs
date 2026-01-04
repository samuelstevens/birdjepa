//! Shuffle buffer for streaming data with bounded memory.
//!
//! Implements reservoir-style shuffling: fill buffer, then randomly replace
//! elements as new samples arrive.

use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::VecDeque;

/// A shuffle buffer that maintains bounded memory while providing randomness.
///
/// Algorithm:
/// 1. Fill buffer to capacity
/// 2. For each new sample, randomly swap it with an existing sample and yield the evicted one
/// 3. When input exhausted, shuffle remaining buffer and drain
pub struct ShuffleBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    rng: StdRng,
    pending_output: VecDeque<T>,
}

impl<T> ShuffleBuffer<T> {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            rng: StdRng::seed_from_u64(seed),
            pending_output: VecDeque::new(),
        }
    }

    /// Reset the buffer for a new epoch with a new seed.
    pub fn reset(&mut self, seed: u64) {
        self.buffer.clear();
        self.rng = StdRng::seed_from_u64(seed);
        self.pending_output.clear();
    }

    /// Push a sample into the buffer.
    /// Returns a sample if one was evicted (buffer was full).
    pub fn push(&mut self, sample: T) -> Option<T> {
        if self.buffer.len() < self.capacity {
            self.buffer.push(sample);
            None
        } else {
            // Buffer full, randomly swap
            let idx = self.rng.gen_range(0..self.capacity);
            let evicted = std::mem::replace(&mut self.buffer[idx], sample);
            Some(evicted)
        }
    }

    /// Signal that input is exhausted. Shuffle remaining buffer.
    pub fn finish(&mut self) {
        self.buffer.shuffle(&mut self.rng);
        self.pending_output.extend(self.buffer.drain(..));
    }

    /// Get the next output sample (from pending_output after finish).
    pub fn pop(&mut self) -> Option<T> {
        self.pending_output.pop_front()
    }

    /// Check if there are pending outputs.
    pub fn has_pending(&self) -> bool {
        !self.pending_output.is_empty()
    }

    /// Current number of samples in buffer.
    pub fn len(&self) -> usize {
        self.buffer.len() + self.pending_output.len()
    }

    /// Is the buffer empty?
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.pending_output.is_empty()
    }
}

/// A shuffled iterator that wraps an input iterator with a shuffle buffer.
pub struct ShuffledIter<I, T>
where
    I: Iterator<Item = T>,
{
    input: I,
    buffer: ShuffleBuffer<T>,
    input_exhausted: bool,
}

impl<I, T> ShuffledIter<I, T>
where
    I: Iterator<Item = T>,
{
    pub fn new(input: I, buffer_size: usize, seed: u64) -> Self {
        Self {
            input,
            buffer: ShuffleBuffer::new(buffer_size, seed),
            input_exhausted: false,
        }
    }
}

impl<I, T> Iterator for ShuffledIter<I, T>
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // First, try to get from pending output
        if let Some(sample) = self.buffer.pop() {
            return Some(sample);
        }

        // If input exhausted and buffer drained, we're done
        if self.input_exhausted {
            return None;
        }

        // Try to push from input until we get an eviction
        loop {
            match self.input.next() {
                Some(sample) => {
                    if let Some(evicted) = self.buffer.push(sample) {
                        return Some(evicted);
                    }
                    // Keep filling buffer
                }
                None => {
                    // Input exhausted, drain buffer
                    self.input_exhausted = true;
                    self.buffer.finish();
                    return self.buffer.pop();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_buffer_basic() {
        let input: Vec<i32> = (0..100).collect();
        let shuffled: Vec<i32> = ShuffledIter::new(input.into_iter(), 10, 42).collect();

        assert_eq!(shuffled.len(), 100);
        // Should contain all elements
        let mut sorted = shuffled.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        // Should be shuffled (very unlikely to be in order)
        assert_ne!(shuffled, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_shuffle_deterministic() {
        let input1: Vec<i32> = (0..100).collect();
        let input2: Vec<i32> = (0..100).collect();

        let shuffled1: Vec<i32> = ShuffledIter::new(input1.into_iter(), 10, 42).collect();
        let shuffled2: Vec<i32> = ShuffledIter::new(input2.into_iter(), 10, 42).collect();

        assert_eq!(shuffled1, shuffled2);
    }
}
