//! Training infrastructure for Harmonic BDH.
//!
//! Provides:
//! - Simple SGD optimizer
//! - Training objectives (pattern diversity, reconstruction, prediction)
//! - Concept coherence loss for narrative-like transitions
//! - Training loop with logging

use crate::harmonic::HarmonicBdh;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f32::consts::PI;

/// Training configuration.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub log_every: usize,
    /// Objective weights
    pub diversity_weight: f32,
    pub reconstruction_weight: f32,
    pub energy_reg_weight: f32,
    /// Concept coherence weight (encourages smooth concept transitions)
    pub concept_coherence_weight: f32,
    /// Sparsity weight (encourages selective activation)
    pub sparsity_weight: f32,
    /// Temporal consistency weight (penalizes large embedding jumps)
    pub temporal_consistency_weight: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            epochs: 100,
            batch_size: 32,
            seq_len: 16,
            log_every: 10,
            diversity_weight: 1.0,
            reconstruction_weight: 0.5,
            energy_reg_weight: 0.1,
            concept_coherence_weight: 2.0,  // High for narrative coherence
            sparsity_weight: 0.2,
            temporal_consistency_weight: 1.5,  // Penalize large jumps
        }
    }
}

/// Simple SGD optimizer with momentum.
pub struct SGDOptimizer {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: Vec<Array2<f32>>,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32, param_shapes: &[(usize, usize)]) -> Self {
        let velocity = param_shapes.iter()
            .map(|&(rows, cols)| Array2::zeros((rows, cols)))
            .collect();
        
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity,
        }
    }
    
    /// Apply gradient update to parameters.
    pub fn step(&mut self, params: &mut [Array2<f32>], grads: &[Array2<f32>]) {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // v = momentum * v - lr * (grad + weight_decay * param)
            let reg_grad = grad + &(param.mapv(|p| p * self.weight_decay));
            self.velocity[i] = self.velocity[i].mapv(|v| v * self.momentum) 
                - &reg_grad.mapv(|g| g * self.learning_rate);
            
            // param += v
            *param = &*param + &self.velocity[i];
        }
    }
}

/// Training metrics.
#[derive(Clone, Debug, Default)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub total_loss: f32,
    pub diversity_loss: f32,
    pub reconstruction_loss: f32,
    pub energy_loss: f32,
    pub concept_coherence_loss: f32,
    pub sparsity_loss: f32,
    pub avg_energy: f32,
    pub avg_sparsity: f32,
    pub attractor_stability: f32,
}

/// Unsupervised training objective: maximize pattern diversity.
/// This encourages the model to use different attractor states for different inputs.
pub fn diversity_loss(outputs: &[Array1<f32>]) -> f32 {
    if outputs.len() < 2 {
        return 0.0;
    }
    
    let n = outputs[0].len();
    let mut total_similarity = 0.0f32;
    let mut count = 0;
    
    // Compute pairwise cosine similarity
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            let dot = outputs[i].dot(&outputs[j]);
            let norm_i = outputs[i].dot(&outputs[i]).sqrt().max(0.001);
            let norm_j = outputs[j].dot(&outputs[j]).sqrt().max(0.001);
            let similarity = dot / (norm_i * norm_j);
            total_similarity += similarity;
            count += 1;
        }
    }
    
    // We want to minimize similarity (maximize diversity)
    // Return average similarity as loss
    if count > 0 {
        total_similarity / count as f32
    } else {
        0.0
    }
}

/// Reconstruction loss: how well can we reconstruct the input from the output?
pub fn reconstruction_loss(inputs: &[Array1<f32>], outputs: &[Array1<f32>]) -> f32 {
    if inputs.len() != outputs.len() || inputs.is_empty() {
        return 0.0;
    }
    
    let mut total_mse = 0.0f32;
    
    for (input, output) in inputs.iter().zip(outputs.iter()) {
        let diff = input - output;
        total_mse += diff.mapv(|d| d * d).sum();
    }
    
    total_mse / inputs.len() as f32
}

/// Energy regularization: penalize very high or very low energy states.
pub fn energy_regularization(energies: &[f32], target_energy: f32) -> f32 {
    let total_energy: f32 = energies.iter().sum();
    let deviation = (total_energy - target_energy).abs();
    deviation * deviation
}

/// Concept coherence loss: encourage smooth, narrative-like concept transitions.
/// Penalizes abrupt changes in concept space between consecutive outputs.
pub fn concept_coherence_loss(outputs: &[Array1<f32>], model: &HarmonicBdh) -> f32 {
    if outputs.len() < 2 {
        return 0.0;
    }
    
    let mut total_incoherence = 0.0f32;
    let mut count = 0;
    
    // Get concept projections for each output
    let concepts: Vec<Vec<(&str, f32)>> = outputs.iter()
        .map(|o| model.get_top_concepts(o, 3))
        .collect();
    
    for i in 1..concepts.len() {
        let prev = &concepts[i - 1];
        let curr = &concepts[i];
        
        // Measure concept transition smoothness
        // Penalize if top concepts change completely between steps
        let mut shared_score = 0.0f32;
        for (name, score) in prev.iter() {
            for (name2, score2) in curr.iter() {
                if name == name2 {
                    // Reward overlapping concepts, weighted by strength
                    shared_score += score * score2;
                }
            }
        }
        
        // Incoherence = 1 - shared_score (higher when concepts change abruptly)
        total_incoherence += 1.0 - shared_score.min(1.0);
        count += 1;
    }
    
    if count > 0 {
        total_incoherence / count as f32
    } else {
        0.0
    }
}

/// Sparsity loss: encourage selective activation (not too many neurons active).
pub fn sparsity_loss(outputs: &[Array1<f32>], target_sparsity: f32) -> f32 {
    if outputs.is_empty() {
        return 0.0;
    }
    
    let mut total_deviation = 0.0f32;
    
    for output in outputs {
        let active_ratio = output.iter()
            .filter(|&&v| v.abs() > 0.01)
            .count() as f32 / output.len() as f32;
        
        // Penalize deviation from target sparsity
        let deviation = (active_ratio - (1.0 - target_sparsity)).abs();
        total_deviation += deviation * deviation;
    }
    
    total_deviation / outputs.len() as f32
}

/// Temporal consistency loss: penalize large jumps in embedding space.
/// Encourages smooth, narrative-like transitions between states.
pub fn temporal_consistency_loss(outputs: &[Array1<f32>]) -> f32 {
    if outputs.len() < 2 {
        return 0.0;
    }
    
    let mut total_jump = 0.0f32;
    let mut count = 0;
    
    for i in 1..outputs.len() {
        let prev = &outputs[i - 1];
        let curr = &outputs[i];
        
        // Compute L2 distance between consecutive outputs
        let diff = curr - prev;
        let distance = diff.mapv(|d| d * d).sum().sqrt();
        
        // Normalize by vector magnitude to get relative change
        let prev_norm = prev.mapv(|v| v * v).sum().sqrt().max(0.001);
        let relative_jump = distance / prev_norm;
        
        // Penalize large relative jumps (threshold around 0.5)
        if relative_jump > 0.3 {
            total_jump += (relative_jump - 0.3).powi(2);
        }
        count += 1;
    }
    
    if count > 0 {
        total_jump / count as f32
    } else {
        0.0
    }
}

/// Compute gradients via finite differences (simple but works for small models).
pub fn compute_numerical_gradient(
    model: &mut HarmonicBdh,
    inputs: &[Array1<f32>],
    loss_fn: impl Fn(&mut HarmonicBdh, &[Array1<f32>]) -> f32,
    epsilon: f32,
) -> f32 {
    // This is a placeholder - real gradient computation would require
    // differentiable operations. For now, we use perturbation-based updates.
    let base_loss = loss_fn(model, inputs);
    base_loss
}

/// Trainer for Harmonic BDH.
pub struct Trainer {
    pub config: TrainingConfig,
    pub metrics_history: Vec<TrainingMetrics>,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            metrics_history: Vec::new(),
        }
    }
    
    /// Train the model on a set of sequences (unsupervised).
    pub fn train_unsupervised(
        &mut self,
        model: &mut HarmonicBdh,
        sequences: &[Array2<f32>],  // Each: (seq_len, n)
    ) {
        let mut rng = thread_rng();
        
        println!("Starting unsupervised training...");
        println!("  Epochs: {}", self.config.epochs);
        println!("  Sequences: {}", sequences.len());
        println!("  Batch size: {}", self.config.batch_size);
        
        for epoch in 0..self.config.epochs {
            // Shuffle sequences
            let mut indices: Vec<usize> = (0..sequences.len()).collect();
            indices.shuffle(&mut rng);
            
            let mut epoch_metrics = TrainingMetrics {
                epoch,
                ..Default::default()
            };
            
            let mut batch_count = 0;
            
            for batch_start in (0..sequences.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(sequences.len());
                let batch_indices = &indices[batch_start..batch_end];
                
                // Process each sequence in batch
                let mut batch_outputs = Vec::new();
                let mut batch_energies = Vec::new();
                let mut batch_inputs = Vec::new();
                
                for &idx in batch_indices {
                    let sequence = &sequences[idx];
                    
                    // Forward pass through sequence
                    let mut seq_outputs = Vec::new();
                    for row in sequence.axis_iter(Axis(0)) {
                        let input = row.to_owned();
                        batch_inputs.push(input.clone());
                        
                        let (output, energies, _) = model.forward(&input);
                        seq_outputs.push(output);
                        batch_energies.extend(energies);
                    }
                    batch_outputs.extend(seq_outputs);
                }
                
                // Compute losses
                let div_loss = diversity_loss(&batch_outputs);
                let recon_loss = reconstruction_loss(&batch_inputs, &batch_outputs);
                let energy_loss_val = energy_regularization(&batch_energies, 0.3);
                let coherence_loss = concept_coherence_loss(&batch_outputs, model);
                let sparse_loss = sparsity_loss(&batch_outputs, 0.7);  // Target 70% sparsity
                let temporal_loss = temporal_consistency_loss(&batch_outputs);
                
                let total_loss = self.config.diversity_weight * div_loss
                    + self.config.reconstruction_weight * recon_loss
                    + self.config.energy_reg_weight * energy_loss_val
                    + self.config.concept_coherence_weight * coherence_loss
                    + self.config.sparsity_weight * sparse_loss
                    + self.config.temporal_consistency_weight * temporal_loss;
                
                // Accumulate metrics
                epoch_metrics.diversity_loss += div_loss;
                epoch_metrics.reconstruction_loss += recon_loss;
                epoch_metrics.energy_loss += energy_loss_val;
                epoch_metrics.concept_coherence_loss += coherence_loss;
                epoch_metrics.sparsity_loss += sparse_loss;
                epoch_metrics.total_loss += total_loss;
                
                // Compute sparsity (for display)
                let sparsity: f32 = batch_outputs.iter()
                    .map(|o| o.iter().filter(|&&v| v.abs() < 0.01).count() as f32 / o.len() as f32)
                    .sum::<f32>() / batch_outputs.len() as f32;
                epoch_metrics.avg_sparsity += sparsity;
                
                // Average energy
                let avg_e = batch_energies.iter().sum::<f32>() / batch_energies.len().max(1) as f32;
                epoch_metrics.avg_energy += avg_e;
                
                batch_count += 1;
                
                // Perturbation-based adaptation (simple evolutionary update)
                // Adjust biological config based on loss
                self.adapt_config(model, total_loss);
            }
            
            // Average metrics
            if batch_count > 0 {
                epoch_metrics.diversity_loss /= batch_count as f32;
                epoch_metrics.reconstruction_loss /= batch_count as f32;
                epoch_metrics.energy_loss /= batch_count as f32;
                epoch_metrics.concept_coherence_loss /= batch_count as f32;
                epoch_metrics.sparsity_loss /= batch_count as f32;
                epoch_metrics.total_loss /= batch_count as f32;
                epoch_metrics.avg_sparsity /= batch_count as f32;
                epoch_metrics.avg_energy /= batch_count as f32;
            }
            
            // Log progress
            if epoch % self.config.log_every == 0 || epoch == self.config.epochs - 1 {
                println!(
                    "Epoch {:4} │ Loss: {:.4} │ Div: {:.3} │ Recon: {:.3} │ Coher: {:.3} │ Spars: {:.1}%",
                    epoch,
                    epoch_metrics.total_loss,
                    epoch_metrics.diversity_loss,
                    epoch_metrics.reconstruction_loss,
                    epoch_metrics.concept_coherence_loss,
                    epoch_metrics.avg_sparsity * 100.0
                );
            }
            
            self.metrics_history.push(epoch_metrics);
        }
        
        println!("Training complete!");
    }
    
    /// Adapt model config based on loss (simple heuristic).
    fn adapt_config(&self, model: &mut HarmonicBdh, loss: f32) {
        // Gently adjust noise based on diversity loss
        // We want to maintain spontaneous activity, so don't reduce noise too aggressively
        let current_noise = model.config.noise_amplitude;
        let min_noise = 0.04;  // Keep minimum noise for spontaneous activity
        
        if loss > 0.6 {
            // High loss: increase exploration
            model.config.noise_amplitude = (current_noise * 1.02).min(0.15);
        } else if loss < 0.2 && current_noise > min_noise {
            // Only reduce if we're above minimum threshold
            model.config.noise_amplitude = (current_noise * 0.995).max(min_noise);
        }
    }
    
    /// Generate synthetic training data for testing.
    pub fn generate_synthetic_data(n: usize, num_sequences: usize, seq_len: usize) -> Vec<Array2<f32>> {
        let mut rng = thread_rng();
        let mut sequences = Vec::with_capacity(num_sequences);
        
        for _ in 0..num_sequences {
            // Generate sequences with different "pattern types"
            let pattern_type = rand::random::<u8>() % 4;
            let mut seq = Array2::zeros((seq_len, n));
            
            for t in 0..seq_len {
                let phase = t as f32 / seq_len as f32 * 2.0 * PI;
                
                for neuron in 0..n {
                    let pos = neuron as f32 / n as f32;
                    
                    let value = match pattern_type {
                        0 => (phase + pos * 2.0 * PI).sin(),  // Wave
                        1 => if pos > 0.3 && pos < 0.7 { 1.0 } else { 0.0 },  // Band
                        2 => (-((pos - 0.5).abs() * 10.0)).exp(),  // Gaussian
                        _ => (phase * (pattern_type as f32 + 1.0) + pos * 4.0 * PI).sin(),  // Harmonic
                    };
                    
                    seq[[t, neuron]] = value.max(0.0);  // Positive only
                }
            }
            
            sequences.push(seq);
        }
        
        sequences
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BiologicalConfig;
    
    #[test]
    fn test_diversity_loss() {
        let same = vec![
            Array1::from_elem(10, 1.0),
            Array1::from_elem(10, 1.0),
        ];
        let different = vec![
            Array1::from_iter((0..10).map(|i| if i < 5 { 1.0 } else { 0.0 })),
            Array1::from_iter((0..10).map(|i| if i >= 5 { 1.0 } else { 0.0 })),
        ];
        
        let loss_same = diversity_loss(&same);
        let loss_diff = diversity_loss(&different);
        
        assert!(loss_same > loss_diff, "Same patterns should have higher similarity loss");
    }
    
    #[test]
    fn test_synthetic_data() {
        let data = Trainer::generate_synthetic_data(32, 10, 8);
        
        assert_eq!(data.len(), 10);
        assert_eq!(data[0].shape(), &[8, 32]);
    }
    
    #[test]
    fn test_training_loop() {
        let config = TrainingConfig {
            epochs: 5,
            batch_size: 2,
            log_every: 2,
            ..Default::default()
        };
        
        let bio_config = BiologicalConfig {
            noise_amplitude: 0.01,
            self_excitation: 0.01,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, bio_config);
        let data = Trainer::generate_synthetic_data(32, 8, 4);
        
        let mut trainer = Trainer::new(config);
        trainer.train_unsupervised(&mut model, &data);
        
        assert_eq!(trainer.metrics_history.len(), 5);
    }
}
