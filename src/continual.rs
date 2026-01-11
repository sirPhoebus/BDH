//! Continual Learning for Harmonic BDH.
//!
//! Implements:
//! - **Experience Replay Buffer**: Stores past experiences with priority sampling
//! - **Adaptive Forgetting**: Decays unused rho states based on access frequency
//! - **Consolidation**: Protects important patterns from catastrophic forgetting

use crate::harmonic::{ComplexState, HarmonicBdh};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for continual learning.
#[derive(Clone, Debug)]
pub struct ContinualConfig {
    /// Maximum experiences in replay buffer
    pub buffer_capacity: usize,
    /// Fraction of batch from replay (0.0 = no replay, 1.0 = all replay)
    pub replay_ratio: f32,
    /// Priority exponent (higher = more focus on surprising experiences)
    pub priority_exponent: f32,
    /// Base forgetting rate for unused rho states
    pub forgetting_rate: f32,
    /// Minimum activation threshold before forgetting kicks in
    pub forgetting_threshold: f32,
    /// Consolidation strength (protects frequently-replayed patterns)
    pub consolidation_strength: f32,
    /// How often to run consolidation (every N training steps)
    pub consolidation_interval: usize,
    /// Importance decay rate (exponential moving average)
    pub importance_decay: f32,
}

impl Default for ContinualConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 10000,
            replay_ratio: 0.3,
            priority_exponent: 0.6,
            forgetting_rate: 0.001,
            forgetting_threshold: 0.01,
            consolidation_strength: 0.1,
            consolidation_interval: 100,
            importance_decay: 0.99,
        }
    }
}

/// A stored experience for replay.
#[derive(Clone)]
pub struct Experience {
    /// Input that produced this experience
    pub input: Array1<f32>,
    /// Output from forward pass
    pub output: Array1<f32>,
    /// Snapshot of rho states at time of experience
    pub rho_snapshot: Vec<Array2<f32>>,
    /// Layer energies at time of experience
    pub energies: Vec<f32>,
    /// Surprise score (reconstruction error)
    pub surprise: f32,
    /// Timestamp (for recency weighting)
    pub timestamp: usize,
}

/// Priority-based experience replay buffer.
pub struct ExperienceReplay {
    buffer: VecDeque<Experience>,
    capacity: usize,
    priority_exponent: f32,
    current_time: usize,
}

impl ExperienceReplay {
    pub fn new(capacity: usize, priority_exponent: f32) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            priority_exponent,
            current_time: 0,
        }
    }

    /// Add an experience to the buffer.
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            // Remove lowest priority experience
            let min_idx = self.find_min_priority_idx();
            self.buffer.remove(min_idx);
        }
        self.buffer.push_back(experience);
        self.current_time += 1;
    }

    /// Sample experiences with priority weighting.
    pub fn sample(&self, count: usize) -> Vec<&Experience> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let priorities: Vec<f32> = self.buffer.iter().map(|e| self.compute_priority(e)).collect();
        let total_priority: f32 = priorities.iter().sum();

        if total_priority < 1e-8 {
            // Uniform sampling if all priorities are zero
            return self.buffer.iter().take(count).collect();
        }

        let mut sampled = Vec::with_capacity(count);
        let mut sampled_indices = std::collections::HashSet::new();

        for _ in 0..count.min(self.buffer.len()) {
            let threshold = rng.gen::<f32>() * total_priority;
            let mut cumulative = 0.0f32;

            for (idx, &priority) in priorities.iter().enumerate() {
                cumulative += priority;
                if cumulative >= threshold && !sampled_indices.contains(&idx) {
                    sampled_indices.insert(idx);
                    sampled.push(&self.buffer[idx]);
                    break;
                }
            }
        }

        // Fill remaining with random if needed
        while sampled.len() < count.min(self.buffer.len()) {
            let idx = rng.gen_range(0..self.buffer.len());
            if !sampled_indices.contains(&idx) {
                sampled_indices.insert(idx);
                sampled.push(&self.buffer[idx]);
            }
        }

        sampled
    }

    /// Compute priority for an experience.
    fn compute_priority(&self, exp: &Experience) -> f32 {
        // Priority = surprise^α × recency_weight
        let recency = 1.0 / (1.0 + (self.current_time - exp.timestamp) as f32 * 0.01);
        let surprise_priority = (exp.surprise + 0.01).powf(self.priority_exponent);
        surprise_priority * recency
    }

    /// Find index of minimum priority experience.
    fn find_min_priority_idx(&self) -> usize {
        let mut min_idx = 0;
        let mut min_priority = f32::MAX;

        for (idx, exp) in self.buffer.iter().enumerate() {
            let priority = self.compute_priority(exp);
            if priority < min_priority {
                min_priority = priority;
                min_idx = idx;
            }
        }

        min_idx
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.current_time = 0;
    }
}

/// Tracks importance of each rho element for adaptive forgetting.
pub struct ImportanceTracker {
    /// Access frequency for each (layer, mode, neuron)
    access_counts: Vec<Array2<f32>>,
    /// Importance scores (exponential moving average of activation)
    importance: Vec<Array2<f32>>,
    /// Consolidation mask (protected from forgetting)
    consolidation_mask: Vec<Array2<f32>>,
    /// Decay rate for importance scores
    decay: f32,
    /// Total updates
    total_updates: usize,
}

impl ImportanceTracker {
    pub fn new(num_layers: usize, d: usize, n: usize, decay: f32) -> Self {
        Self {
            access_counts: vec![Array2::zeros((d, n)); num_layers],
            importance: vec![Array2::zeros((d, n)); num_layers],
            consolidation_mask: vec![Array2::zeros((d, n)); num_layers],
            decay,
            total_updates: 0,
        }
    }

    /// Update importance based on current rho activations.
    pub fn update(&mut self, rho: &[ComplexState]) {
        for (layer, rho_layer) in rho.iter().enumerate() {
            for ((mode, neuron), val) in rho_layer.indexed_iter() {
                let activation = val.norm();

                // Update access count
                if activation > 0.01 {
                    self.access_counts[layer][[mode, neuron]] += 1.0;
                }

                // Exponential moving average of importance
                self.importance[layer][[mode, neuron]] = self.decay
                    * self.importance[layer][[mode, neuron]]
                    + (1.0 - self.decay) * activation;
            }
        }
        self.total_updates += 1;
    }

    /// Mark patterns as consolidated (protected from forgetting).
    pub fn consolidate(&mut self, rho: &[ComplexState], strength: f32) {
        for (layer, rho_layer) in rho.iter().enumerate() {
            for ((mode, neuron), val) in rho_layer.indexed_iter() {
                let activation = val.norm();
                // Increase consolidation for active patterns
                self.consolidation_mask[layer][[mode, neuron]] = (self.consolidation_mask[layer]
                    [[mode, neuron]]
                    + strength * activation)
                    .min(1.0);
            }
        }
    }

    /// Decay consolidation over time.
    pub fn decay_consolidation(&mut self, rate: f32) {
        for mask in &mut self.consolidation_mask {
            mask.mapv_inplace(|v| (v - rate).max(0.0));
        }
    }

    /// Get forgetting mask (1.0 = keep, 0.0 = forget).
    pub fn get_forgetting_mask(
        &self,
        threshold: f32,
        base_rate: f32,
    ) -> Vec<Array2<f32>> {
        self.importance
            .iter()
            .zip(&self.consolidation_mask)
            .map(|(imp, cons)| {
                let d = imp.nrows();
                let n = imp.ncols();
                let mut mask = Array2::ones((d, n));

                for mode in 0..d {
                    for neuron in 0..n {
                        let importance = imp[[mode, neuron]];
                        let consolidation = cons[[mode, neuron]];

                        // Keep if importance > threshold or consolidated
                        if importance < threshold && consolidation < 0.5 {
                            // Gradual forgetting based on how far below threshold
                            let forget_strength =
                                base_rate * (1.0 - importance / threshold).max(0.0);
                            mask[[mode, neuron]] = 1.0 - forget_strength;
                        }
                    }
                }

                mask
            })
            .collect()
    }

    /// Get statistics for logging.
    pub fn get_stats(&self) -> (f32, f32, f32) {
        let mut total_importance = 0.0f32;
        let mut total_consolidated = 0.0f32;
        let mut count = 0usize;

        for (imp, cons) in self.importance.iter().zip(&self.consolidation_mask) {
            total_importance += imp.sum();
            total_consolidated += cons.sum();
            count += imp.len();
        }

        let avg_importance = total_importance / count as f32;
        let avg_consolidated = total_consolidated / count as f32;
        let total_access: f32 = self.access_counts.iter().map(|a| a.sum()).sum();

        (avg_importance, avg_consolidated, total_access)
    }
}

/// Adaptive forgetting mechanism for HarmonicBdh.
pub struct AdaptiveForgetting {
    tracker: ImportanceTracker,
    config: ContinualConfig,
    steps_since_consolidation: usize,
}

impl AdaptiveForgetting {
    pub fn new(num_layers: usize, d: usize, n: usize, config: ContinualConfig) -> Self {
        Self {
            tracker: ImportanceTracker::new(num_layers, d, n, config.importance_decay),
            config,
            steps_since_consolidation: 0,
        }
    }

    /// Update importance tracking after a forward pass.
    pub fn update(&mut self, model: &HarmonicBdh) {
        // Get rho states from model
        let rho: Vec<ComplexState> = (0..model.num_layers())
            .map(|l| model.get_rho(l).clone())
            .collect();
        self.tracker.update(&rho);
        self.steps_since_consolidation += 1;
    }

    /// Apply adaptive forgetting to model's rho states.
    pub fn apply_forgetting(&self, model: &mut HarmonicBdh) {
        let masks = self.tracker.get_forgetting_mask(
            self.config.forgetting_threshold,
            self.config.forgetting_rate,
        );

        for (layer, mask) in masks.iter().enumerate() {
            model.apply_rho_mask(layer, mask);
        }
    }

    /// Consolidate current patterns (call during replay).
    pub fn consolidate(&mut self, model: &HarmonicBdh) {
        let rho: Vec<ComplexState> = (0..model.num_layers())
            .map(|l| model.get_rho(l).clone())
            .collect();
        self.tracker.consolidate(&rho, self.config.consolidation_strength);
        self.steps_since_consolidation = 0;
    }

    /// Check if consolidation should run.
    pub fn should_consolidate(&self) -> bool {
        self.steps_since_consolidation >= self.config.consolidation_interval
    }

    /// Decay consolidation masks.
    pub fn decay_consolidation(&mut self) {
        self.tracker.decay_consolidation(0.001);
    }

    /// Get stats for logging.
    pub fn get_stats(&self) -> (f32, f32, f32) {
        self.tracker.get_stats()
    }
}

/// Compute surprise (reconstruction error) for an experience.
pub fn compute_surprise(input: &Array1<f32>, output: &Array1<f32>) -> f32 {
    let diff = input - output;
    let mse = diff.mapv(|d| d * d).sum() / input.len() as f32;
    mse.sqrt()
}

/// Compute diversity score between two outputs.
pub fn compute_diversity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let norm_a = a.dot(a).sqrt().max(0.001);
    let norm_b = b.dot(b).sqrt().max(0.001);
    let cosine = a.dot(b) / (norm_a * norm_b);
    1.0 - cosine.abs()
}

/// Create an experience from a forward pass.
pub fn create_experience(
    input: &Array1<f32>,
    output: &Array1<f32>,
    model: &HarmonicBdh,
    energies: &[f32],
    timestamp: usize,
) -> Experience {
    // Snapshot rho as real amplitudes
    let rho_snapshot: Vec<Array2<f32>> = (0..model.num_layers())
        .map(|l| model.get_rho(l).mapv(|c| c.norm()))
        .collect();

    let surprise = compute_surprise(input, output);

    Experience {
        input: input.clone(),
        output: output.clone(),
        rho_snapshot,
        energies: energies.to_vec(),
        surprise,
        timestamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    #[test]
    fn test_experience_replay() {
        let mut buffer = ExperienceReplay::new(100, 0.6);

        // Add experiences with varying surprise
        for i in 0..50 {
            let exp = Experience {
                input: Array1::from_elem(32, i as f32 * 0.1),
                output: Array1::from_elem(32, i as f32 * 0.05),
                rho_snapshot: vec![Array2::zeros((8, 32))],
                energies: vec![0.1],
                surprise: (i as f32 * 0.02).sin().abs(),
                timestamp: i,
            };
            buffer.push(exp);
        }

        assert_eq!(buffer.len(), 50);

        // Sample should prioritize high-surprise, recent experiences
        let samples = buffer.sample(10);
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_importance_tracker() {
        let mut tracker = ImportanceTracker::new(2, 8, 32, 0.99);

        // Simulate some rho activations
        let rho = vec![
            Array2::from_elem((8, 32), Complex32::new(0.5, 0.0)),
            Array2::from_elem((8, 32), Complex32::new(0.1, 0.0)),
        ];

        for _ in 0..10 {
            tracker.update(&rho);
        }

        let (avg_imp, avg_cons, total_access) = tracker.get_stats();
        assert!(avg_imp > 0.0);
        assert!(total_access > 0.0);
    }

    #[test]
    fn test_forgetting_mask() {
        let mut tracker = ImportanceTracker::new(1, 4, 8, 0.99);

        // Create rho with some active and some inactive neurons
        let mut rho_layer = Array2::from_elem((4, 8), Complex32::new(0.0, 0.0));
        for mode in 0..4 {
            for neuron in 0..4 {
                // Only first half active
                rho_layer[[mode, neuron]] = Complex32::new(0.5, 0.0);
            }
        }

        for _ in 0..20 {
            tracker.update(&[rho_layer.clone()]);
        }

        let masks = tracker.get_forgetting_mask(0.1, 0.1);
        assert_eq!(masks.len(), 1);

        // Active neurons should have mask closer to 1.0
        // Inactive neurons should have mask closer to 0.9 (with forgetting)
        let mask = &masks[0];
        assert!(mask[[0, 0]] > mask[[0, 7]]);
    }

    #[test]
    fn test_consolidation() {
        let mut tracker = ImportanceTracker::new(1, 4, 8, 0.99);

        let rho = vec![Array2::from_elem((4, 8), Complex32::new(0.3, 0.0))];

        // Consolidate
        tracker.consolidate(&rho, 0.5);

        let (_, avg_cons, _) = tracker.get_stats();
        assert!(avg_cons > 0.0, "Consolidation should increase mask values");
    }

    #[test]
    fn test_compute_surprise() {
        let input = Array1::from_elem(10, 1.0);
        let output = Array1::from_elem(10, 0.5);

        let surprise = compute_surprise(&input, &output);
        assert!(surprise > 0.0);
        assert!(surprise < 1.0);
    }

    #[test]
    fn test_compute_diversity() {
        let a = Array1::from_iter((0..10).map(|i| if i < 5 { 1.0 } else { 0.0 }));
        let b = Array1::from_iter((0..10).map(|i| if i >= 5 { 1.0 } else { 0.0 }));
        let c = a.clone();

        let div_ab = compute_diversity(&a, &b);
        let div_ac = compute_diversity(&a, &c);

        assert!(div_ab > div_ac, "Orthogonal vectors should have higher diversity");
    }
}
