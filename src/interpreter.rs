use ndarray::Array1;
use crate::harmonic::HarmonicBdh;

/// The "Inner Voice" of the brain.
/// Interprets neural states into narrative and feeds them back.
pub struct Interpreter {
    /// History of generated thoughts (short-term buffer).
    pub history_buffer: Vec<String>,
    /// Capacity of the buffer.
    pub capacity: usize,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            history_buffer: Vec::new(),
            capacity: 5,
        }
    }

    /// Convert a set of weighted concepts into a natural language narrative.
    /// 
    /// # Arguments
    /// * `concepts`: List of (Concept Name, Activation Strength)
    /// * `confidence`: Cognitive confidence score (0.0 to 1.0)
    pub fn interpret(&mut self, concepts: &[(&str, f32)], confidence: f32) -> String {
        if concepts.is_empty() {
            return "I feel empty.".to_string();
        }

        let (dominant, strength) = concepts[0];
        
        // Modulate template based on confidence
        let narrative = if confidence < 0.3 {
            format!("I am confused, but I might be sensing {}.", dominant)
        } else if confidence < 0.6 {
            format!("I suspect there is {}.", dominant)
        } else {
            // High confidence regular path
            if strength > 0.8 {
                format!("I am clearly overwhelmed by {}.", dominant)
            } else if strength > 0.5 {
                format!("I feel a strong sense of {}.", dominant)
            } else {
                 format!("I notice {}.", dominant)
            }
        };

        // Add secondary context
        let narrative = if concepts.len() > 1 {
            let (sec, _) = concepts[1];
             format!("{} Also, {}.", narrative, sec)
        } else {
            narrative
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

    /// "Hear" the narrative: Map words back to concept vectors.
    /// This closes the loop.
    pub fn get_feedback_vector(&self, narrative: &str, brain: &HarmonicBdh) -> Array1<f32> {
        let mut feedback = Array1::zeros(brain.n);
        
        // Very simple keyword matching for "Hearing"
        // If the narrative contains a concept name, add its vector.
        // In reality, this would use the same Embedding encoder as input.
        
        // We need access to brain's concept space. 
        // Assuming HarmonicBdh has a way to lookup concepts.
        // For now, we'll implement a helper on Brain or just iterate matching names.
        
        // Since we can't iterate brain.concepts directly if private, 
        // we'll rely on a new method `get_concept_vector` in HarmonicBdh.
        
        let words: Vec<&str> = narrative.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .collect();

        for word in words {
            // Try to find a concept matching this word (case-insensitive approximation)
            if let Some(vec) = brain.get_concept_vector(word) {
                feedback += &vec;
            }
        }
        
        // Normalize
        let norm = feedback.dot(&feedback).sqrt();
        if norm > 0.0 {
            feedback /= norm;
        }
        
        feedback * 0.5 // Feedback strength
    }
}
