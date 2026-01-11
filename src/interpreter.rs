use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// The "Inner Voice" of the brain.
/// Interprets neural states into concepts by identifying synchronized Ensembles.
pub struct Interpreter {
    pub history_buffer: Vec<String>,
    pub capacity: usize,
    pub phase_epsilon: f32, // Threshold for phase-locking (binding)
    pub top_k: usize,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            history_buffer: Vec::new(),
            capacity: 5,
            phase_epsilon: 0.15, // Approx 8-10 degrees
            top_k: 8,
        }
    }

    /// Decodes cortical output by identifying phase-locked ensembles.
    /// 
    /// # Arguments
    /// * `cortical_out`: Activation levels [Neurons]
    /// * `phases`: Oscillator phases [Neurons]
    /// * `usage`: Fatigue levels per neuron
    /// * `weights`: Projection matrix [VocabSize, Neurons]
    /// * `embedder`: To map averaged vectors back to words
    pub fn interpret_ensemble(
        &self, 
        cortical_out: &[f32], 
        phases: &[f32], 
        usage: &[f32],
        weights: &Array2<f32>,
        embedder: &crate::data::Embedder
    ) -> Vec<(String, f32)> {
        let n = cortical_out.len();
        let mut ensembles: Vec<Vec<usize>> = Vec::new();
        let mut assigned = vec![false; n];

        // 1. PRE-CALCULATE PHASORS FOR FAST SIMILARITY
        // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
        let cos_sin: Vec<(f32, f32)> = phases.iter().map(|&p| (p.cos(), p.sin())).collect();
        
        let mut active_indices: Vec<usize> = (0..n)
            .filter(|&i| cortical_out[i] >= 0.1)
            .collect();
        
        // Sort by activity descending to pick "seeds" for ensembles
        active_indices.sort_by(|&a, &b| cortical_out[b].partial_cmp(&cortical_out[a]).unwrap());

        for &i in &active_indices {
            if assigned[i] { continue; }
            
            let mut current_ensemble = vec![i];
            assigned[i] = true;
            let (ci, si) = cos_sin[i];
            let threshold = 1.0 - self.phase_epsilon;

            for &j in &active_indices {
                if assigned[j] { continue; }

                // Vectorized-ready phase similarity
                let (cj, sj) = cos_sin[j];
                let phase_diff_cos = ci * cj + si * sj;
                
                if phase_diff_cos > threshold {
                    current_ensemble.push(j);
                    assigned[j] = true;
                }
            }
            ensembles.push(current_ensemble);
        }

        // 2. DECODE ENSEMBLES IN PARALLEL
        let vocab_size = weights.nrows();
        let mut results: Vec<(String, f32)> = ensembles.into_par_iter()
            .filter_map(|ensemble| {
                if ensemble.is_empty() { return None; }

                let mut ensemble_vector = Array1::zeros(vocab_size);
                let mut total_activity = 0.0;

                for &neuron_idx in &ensemble {
                    let activity = cortical_out[neuron_idx];
                    let novelty_weight = 1.0 / (1.0 + usage[neuron_idx]);
                    let weight = activity * novelty_weight;
                    
                    // Vectorized update using ndarray
                    ensemble_vector += &(&weights.column(neuron_idx) * weight);
                    total_activity += weight;
                }

                if total_activity > 0.0 {
                    // Find normalized direction to prevent "origin collapse" in high dimensions
                    let norm = ensemble_vector.dot(&ensemble_vector).sqrt().max(1e-9);
                    let normalized_vector = ensemble_vector / norm;

                    let mut best_word_id = 0;
                    let mut max_sim = -1.0;

                    for v in 0..vocab_size {
                        let sim = normalized_vector[v];
                        if sim > max_sim {
                            max_sim = sim;
                            best_word_id = v;
                        }
                    }

                    let best_word = embedder.decode(&[best_word_id as u32]);

                    if best_word.len() > 3 && !best_word.starts_with("[") {
                        Some((best_word, total_activity))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_k);
        results
    }

    /// Legend says the "Inner Voice" speaks in narratives.
    pub fn generate_narrative(&mut self, concepts: &[(String, f32)], confidence: f32) -> String {
        if concepts.is_empty() {
             return "I feel empty.".to_string();
        }

        let (dominant, _) = &concepts[0];
        let narrative = if confidence < 0.3 {
            format!("I sense a faint glimmer of {}.", dominant)
        } else {
            format!("I am absorbed by the concept of {}.", dominant)
        };

        self.add_history(narrative.clone());
        narrative
    }

    fn add_history(&mut self, text: String) {
        self.history_buffer.push(text);
        if self.history_buffer.len() > self.capacity {
            self.history_buffer.remove(0);
        }
    }

    pub fn reflect(&self, narrative: &str, embedder: &crate::data::Embedder) -> Array1<f32> {
        let mut feedback = Array1::zeros(embedder.n);
        let mut count = 0.0;
        let words: Vec<&str> = narrative.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| s.len() > 3)
            .collect();

        for word in words {
            let tokens = embedder.tokenize(word);
            if !tokens.is_empty() && tokens[0] > 1 {
                let emb = embedder.embed_token(tokens[0]);
                feedback = feedback + &emb;
                count += 1.0;
            }
        }
        if count > 0.0 { feedback /= count; }
        
        let norm = (feedback.dot(&feedback)).sqrt();
        if norm > 1e-8 { feedback /= norm; }
        feedback * 0.3
    }
}
