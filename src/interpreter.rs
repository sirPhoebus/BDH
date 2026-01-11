use ndarray::{Array1, Array2};

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

        // 1. GROUP NEURONS INTO ENSEMBLES
        for i in 0..n {
            if assigned[i] || cortical_out[i] < 0.1 { continue; }
            
            let mut current_ensemble = vec![i];
            assigned[i] = true;

            for j in (i + 1)..n {
                if assigned[j] || cortical_out[j] < 0.1 { continue; }

                // Check for Phase-Locking (Binding)
                let phase_diff_cos = (phases[i] - phases[j]).cos();
                if phase_diff_cos > (1.0 - self.phase_epsilon) {
                    current_ensemble.push(j);
                    assigned[j] = true;
                }
            }
            ensembles.push(current_ensemble);
        }

        // 2. DECODE ENSEMBLES
        let mut results = Vec::new();
        let vocab_size = weights.nrows();

        for ensemble in ensembles {
            // Aggregate the meaning of the ensemble
            let mut ensemble_vector: Array1<f32> = Array1::zeros(vocab_size);
            let mut total_activity = 0.0;

            for &neuron_idx in &ensemble {
                let activity = cortical_out[neuron_idx];
                // Weight by novelty (inverse usage)
                let novelty_weight = 1.0 / (1.0 + usage[neuron_idx]);
                let weight = activity * novelty_weight;
                
                // The contribution of this neuron to all vocabulary directions
                for v in 0..vocab_size {
                    ensemble_vector[v] += weights[[v, neuron_idx]] * weight;
                }
                total_activity += weight;
            }

            if total_activity > 0.0 {
                let mut best_word_id = 0;
                let mut max_sim = -1.0;
                
                for v in 0..vocab_size {
                    let sim = ensemble_vector[v] / total_activity;
                    if sim > max_sim {
                        max_sim = sim;
                        best_word_id = v;
                    }
                }

                let best_word = embedder.decode(&[best_word_id as u32]);

                if best_word.len() > 3 && !best_word.starts_with("[") {
                    results.push((best_word, total_activity));
                }
            }
        }

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
