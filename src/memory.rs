use ndarray::Array1;
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// A memory entry representing a high-coherence state.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryEntry {
    /// The input embedding that triggered the memory (Key).
    pub key_vector: Array1<f32>,
    /// The natural frequency distribution at that moment (Value).
    pub freq_state: Array1<f32>,
    /// How strongly this memory was felt (Coherence).
    pub importance: f32,
    /// Step when this was stored.
    pub step: usize,
}

/// Hippocampal Memory System: Stores and recalls "Aha!" moments.
pub struct MemorySystem {
    pub storage: VecDeque<MemoryEntry>,
    pub capacity: usize,
    pub recall_threshold: f32,
}

impl MemorySystem {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: VecDeque::with_capacity(capacity),
            capacity,
            recall_threshold: 0.65, // Lowered from 0.70 to allow more recall
        }
    }

    /// Stores a state if it represents a high-coherence moment.
    pub fn store(&mut self, input: &Array1<f32>, freqs: &Array1<f32>, coherence: f32, step: usize) -> bool {
        // Only store if coherence is significantly high
        if coherence > 0.6 {
            // Check if we already have something very similar
            for entry in &self.storage {
                if self.cosine_similarity(&entry.key_vector, input) > 0.95 {
                    return false; // Already known, skip redundant store
                }
            }

            if self.storage.len() >= self.capacity {
                self.storage.pop_front(); // Forget oldest if full
            }

            self.storage.push_back(MemoryEntry {
                key_vector: input.clone(),
                freq_state: freqs.clone(),
                importance: coherence,
                step,
            });
            return true;
        }
        false
    }

    /// Attempts to find a past state that matches the current input.
    pub fn recall(&self, current_input: &Array1<f32>) -> Option<Array1<f32>> {
        let mut best_match: Option<&MemoryEntry> = None;
        let mut max_sim = self.recall_threshold;

        for entry in &self.storage {
            let sim = self.cosine_similarity(&entry.key_vector, current_input);
            if sim > max_sim {
                max_sim = sim;
                best_match = Some(entry);
            }
        }

        best_match.map(|m| m.freq_state.clone())
    }

    fn cosine_similarity(&self, v1: &Array1<f32>, v2: &Array1<f32>) -> f32 {
        let dot = v1.dot(v2);
        let n1 = v1.dot(v1).sqrt();
        let n2 = v2.dot(v2).sqrt();
        if n1 * n2 < 1e-9 { return 0.0; }
        dot / (n1 * n2)
    }
}
