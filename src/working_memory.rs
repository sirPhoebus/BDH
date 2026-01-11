use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::collections::VecDeque;

/// Working Memory System (Short-Term Memory)
/// Uses a fixed-size buffer (slots) to hold recent high-salience concepts.
/// These concepts form a "context vector" that biases cortical processing.
pub struct WorkingMemory<B: Backend> {
    pub capacity: usize,
    pub slots: VecDeque<(String, Tensor<B, 1>)>, // (Concept Name, Vector)
    pub bias_strength: f32,
}

impl<B: Backend> WorkingMemory<B> {
    pub fn new(capacity: usize, bias_strength: f32) -> Self {
        Self {
            capacity,
            slots: VecDeque::with_capacity(capacity),
            bias_strength,
        }
    }
    
    /// Update memory with a new concept event.
    /// Only adds if unique from recent items to prevent looping.
    pub fn update(&mut self, concept: String, vector: Tensor<B, 1>) {
        if concept.len() < 3 || concept == "Void" || concept == "<unk>" {
            return;
        }
        
        // Prevent duplicates
        for (existing, _) in &self.slots {
            if existing == &concept {
                return;
            }
        }
        
        if self.slots.len() >= self.capacity {
            self.slots.pop_front(); // Remove oldest
        }
        
        self.slots.push_back((concept, vector));
    }
    
    /// Calculate the "Context Bias" vector.
    /// Sum of all slot vectors, weighted by recency? Or flat?
    /// For now: Flat sum, scaled by bias_strength.
    pub fn get_context_bias(&self, dim: usize, device: &B::Device) -> Tensor<B, 1> {
        if self.slots.is_empty() {
             return Tensor::zeros([dim], device);
        }
        
        let mut sum_vec = Tensor::zeros([dim], device);
        
        // Decaying weights for slots? (Oldest = Weakest)
        // Let's keep it simple: ALL slots are active context.
        for (_, vec) in &self.slots {
            sum_vec = sum_vec.add(vec.clone());
        }
        
        // Normalize?
        // Let's just scale it.
        sum_vec.mul_scalar(self.bias_strength)
    }
    
    pub fn debug_string(&self) -> String {
        let names: Vec<String> = self.slots.iter().map(|(n, _)| n.clone()).collect();
        format!("[{}]", names.join(", "))
    }
}
