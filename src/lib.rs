use ndarray::{prelude::*, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rand::{thread_rng, Rng};



pub mod lsh;
pub mod harmonic;
pub mod regions;
pub mod body;
pub mod drives;
pub mod interpreter;
pub mod harmonic_burn;
pub mod data;
pub mod training;
pub mod continual;

pub use lsh::{LshEmbedder, Vocabulary};
pub use harmonic::{
    HarmonicBdh, BiologicalConfig, DaydreamStep, ThoughtState,
    Concept, generate_brainwave, ReflectionStep,
};
pub use data::{Embedder, TrainingBatch, load_texts_from_dir, chunk_text};
pub use training::{Trainer, TrainingConfig, TrainingMetrics};
pub use continual::{
    ContinualConfig, ExperienceReplay, AdaptiveForgetting,
    Experience, ImportanceTracker, compute_surprise, compute_diversity,
    create_experience,
};

/// Helper: Layer normalization that handles zero-vectors safely.
fn layer_norm(x: &Array1<f32>) -> Array1<f32> {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.var(0.0); // Population variance
    let std = (var + 1e-6).sqrt();
    (x - mean) / std
}

fn relu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.max(0.0))
}

pub struct BdhGpu {
    // Parameters shared across L layers
    e: Array2<f32>,      // d x n (Encoder/Projection)
    d_x: Array2<f32>,    // n x d (Decoder for state x)
    d_y: Array2<f32>,    // n x d (Decoder for state y)
    b_x: Array1<f32>,    // n-dim bias for x sparsity
    b_y: Array1<f32>,    // n-dim bias for y sparsity
    
    // U is a diagonal decay matrix (represented as a vector for efficiency)
    u_decay: Array1<f32>, 
    
    pub n: usize,        // Neuron dimension
    pub d: usize,        // Low-rank dimension
}

impl BdhGpu {
    pub fn new(n: usize, d: usize) -> Self {
        let mut rng = thread_rng();
        let std_dev = 1.0 / (d as f32).sqrt();
        let dist = Normal::new(0.0, std_dev).unwrap();

        // Initialize weights
        let e = Array2::random_using((d, n), dist, &mut rng);
        let d_x = Array2::random_using((n, d), dist, &mut rng);
        let d_y = Array2::random_using((n, d), dist, &mut rng);
        
        // Negative biases encourage sparsity in ReLU (tuned for ~5% activation)
        let b_x = Array1::from_elem(n, -0.1); 
        let b_y = Array1::from_elem(n, -0.05);

        // Initialize decay factors between 0.9 and 0.999
        let u_decay = Array1::linspace(0.9, 0.999, d);

        Self { e, d_x, d_y, b_x, b_y, u_decay, n, d }
    }

    /// Forward pass through the sequence and layers.
    /// inputs: (seq_len, n)
    pub fn forward(&self, inputs: &Array2<f32>, num_layers: usize) -> Vec<(Array1<f32>, Array1<f32>)> {
        let seq_len = inputs.nrows();
        let mut outputs = Vec::with_capacity(seq_len);

        // State rho per layer: d x n matrices
        let mut rho_layers = vec![Array2::zeros((self.d, self.n)); num_layers];
        
        // Initial y_tl-1 is usually a zero vector for the first layer
        let mut y_prev_layer = Array1::zeros(self.n);

        for t in 0..seq_len {
            let mut x_curr = inputs.row(t).to_owned();
            let mut y_curr = Array1::zeros(self.n);

            for l in 0..num_layers {
                // 1. Update x using the "ReLU-LowRank" block (Eq. 8, path 1)
                // x_{t,l} = x_{t,l-1} + ReLU(D_x * LN(E * y_{t,l-1}) + b_x)
                // For first layer, use x_curr as the "previous y" to bootstrap
                let y_input = if l == 0 && y_prev_layer.sum() == 0.0 {
                    &x_curr
                } else {
                    &y_prev_layer
                };
                
                let e_y = self.e.dot(y_input);
                let ln_e_y = layer_norm(&e_y);
                let delta_x = relu(&(self.d_x.dot(&ln_e_y) + &self.b_x));
                x_curr = &x_curr + &delta_x;

                // 2. Linear Attention / Hebbian State Update
                // rho = (rho + ln_e_y * x^T) * U
                let rho = &rho_layers[l];
                
                // Outer product: (d x 1) * (1 * n)
                let update = ln_e_y.clone().insert_axis(Axis(1)).dot(&x_curr.clone().insert_axis(Axis(0)));
                
                // Element-wise application of decay vector along rows
                let mut new_rho = rho.to_owned();
                for i in 0..self.d {
                    let decay = self.u_decay[i];
                    for j in 0..self.n {
                        new_rho[[i, j]] = new_rho[[i, j]] * decay + update[[i, j]];
                    }
                }
                rho_layers[l] = new_rho.clone();

                // 3. Compute y (Eq. 8, path 2)
                // y_{t,l} = ReLU(D_y * LN(rho * x_{t,l}) + b_y) * x_{t,l}
                let rho_x = new_rho.dot(&x_curr);
                let ln_rho_x = layer_norm(&rho_x);
                let gated_y = relu(&(self.d_y.dot(&ln_rho_x) + &self.b_y));
                y_curr = &gated_y * &x_curr; // Element-wise mul for biological sparsity

                y_prev_layer = y_curr.clone();
            }
            outputs.push((x_curr, y_curr));
        }

        outputs
    }

}

pub struct ChronosBdh {
    pub n: usize,
    pub d: usize,
    pub global_time: f32,
    pub heartbeat_freq: f32,
    pub heartbeat_strength: f32, // New: Strength of heartbeat pulse
    pub rho_decay: f32,         // New: Forgetfulness factor
    pub energy_history: f32,    // New: Track energy for bias
    pub learning_rate: f32,     // New: Oja's rule learning rate
    pub last_recall_triggered: bool, // Track if recall happened this step
    
    // Associative Chaining
    pub last_output: Option<Array1<f32>>,
    pub associative_momentum: f32, // Strength of previous thought's influence


    pub rho_phase: Vec<Array2<f32>>,

    pub coupling_strength: Array2<f32>,
}


impl ChronosBdh {
    pub fn new(n: usize, d: usize, heartbeat_freq: f32, num_layers: usize) -> Self {
        let mut rng = thread_rng();
        let std_dev = 1.0 / (d as f32).sqrt();
        let dist = Normal::new(0.0, std_dev).unwrap();

        // Initialize coupling strength (projection matrix)
        let coupling_strength = Array2::random_using((d, n), dist, &mut rng);
        
        // Initialize rho phases with small noise to bootstrap activity
        // Avoiding straight zeros allows the system to have some initial resonance
        let mut rho_phase = Vec::new();
        for _ in 0..num_layers {
            rho_phase.push(Array2::random_using((d, n), Normal::new(0.0, 0.1).unwrap(), &mut rng));
        }


        Self { 
            n, 
            d, 
            global_time: 0.0, 
            heartbeat_freq, // Recommended: 0.2 - 0.8 Hz
            heartbeat_strength: 0.7, // Default: 0.7 (Range 0.6 - 0.85)
            rho_decay: 1.0,          // No decay, rely on Clamping
            energy_history: 0.0,
            learning_rate: 0.1,      // Faster learning (0.1)
            last_recall_triggered: false,
            
            last_output: None,
            associative_momentum: 0.05, // Default momentum
            
            rho_phase, 

            coupling_strength 
        }
    }

    pub fn step_time(&mut self, delta: f32) {
        self.global_time += delta;
    }

    pub fn forward_with_time(&mut self, input: &Array1<f32>, layer: usize) -> Array1<f32> {
        let mut rng = thread_rng();

        // 1. Calculate the Pulsed Heartbeat Factor
        // Range: ~0.0 to heartbeat_strength
        let heartbeat = (self.global_time * self.heartbeat_freq).sin().abs() * self.heartbeat_strength;

        // 2. Probabilistic Recall Trigger (Chaos/Random Cue)
        // Base chance: 0.5% - 3% per step
        // If energy is low (< 0.25), boost the chance significantly to prevent getting stuck
        let base_recall_prob = 0.02f64; // 2%
        let low_energy_bias = if self.energy_history < 0.25 { 0.05f64 } else { 0.0f64 };
        let _recall_prob = base_recall_prob + low_energy_bias;

        let mut current_input = input.clone();
        
        
        // 2. Heartbeat-Modulated Noise (Systole/Diastole)
        // Peak (Systole) -> High Noise (Exploration)
        // Trough (Diastole) -> Low Noise (Settling)
        // heartbeat is abs(sin), so 0..1. 
        // We want noise to scale with this.
        let noise_scale = 0.2 * heartbeat; // Max noise 0.2 at peak
        
        let noise_prob = 0.5; // Always potential for noise if systole is high
        
        if rng.gen_bool(noise_prob) && noise_scale > 0.05 {
             let noise = Array1::random_using(self.n, Normal::new(0.0, noise_scale as f64).unwrap(), &mut rng).mapv(|x| x as f32);
             current_input = &current_input + &noise;
             self.last_recall_triggered = true; // Treating modulated noise as "recall/exploration"
        } else {

             self.last_recall_triggered = false;
        }
        
        // 3. Associative Chaining (Momentum)
        // Add "ghost" of previous output
        if let Some(ref prev) = self.last_output {
             current_input = &current_input + &(prev * self.associative_momentum);
        }

        // 4. Temporal Gating
        let temporal_input = &current_input * heartbeat;


        // 4. Update Rho with Time-Weighted Hebbian Learning
        
        // Apply Decay first (Forgetting)
        // Stronger decay prevents eternal lock-in
        self.rho_phase[layer] *= self.rho_decay;

        // Calculate latent activation (d x 1)
        let latent = self.coupling_strength.dot(&temporal_input);
        
        // Calculate Energy for this step (L2 norm of latent)
        let current_energy = latent.dot(&latent).sqrt();
        self.energy_history = 0.9 * self.energy_history + 0.1 * current_energy;

        // Dynamic Self-Feedback Scaling
        // If energy is high (Active Planning), reduce feedback/update strength lightly
        // to check "overconfidence" or "attractor lock"
        let feedback_scale = if self.energy_history > 0.5 { 0.9 } else { 1.0 };

        // Hebbian Update with Clamping
        // 1. Calculate Hebbian Term
        let hebbian = latent.clone().insert_axis(Axis(1)).dot(&temporal_input.clone().insert_axis(Axis(0)));
        
        // 2. Apply Update + Decay
        // rho = (rho * decay) + (eta * hebbian * time_encoding)
        // Scaled by feedback_scale
        
        let time_encoding = (self.global_time * 0.5).cos();
        let eta = self.learning_rate;
        let decay = self.rho_decay;

        // Apply update in-place
        let rho = &mut self.rho_phase[layer];
        // Efficient ndarray operation?
        // rho = rho * decay + ...
        // Using explicit loop to combine steps and clamp
        
        for i in 0..self.d {
            for j in 0..self.n {
                let h = hebbian[[i, j]];
                let w = rho[[i, j]];
                
                let mut new_w = w * decay + (eta * h * time_encoding * feedback_scale);
                
                // 3. Stabilization: Dynamic Clamping based on Heartbeat
                // Systole (High Heartbeat): Relaxed Clamp (3.0) - Exploration
                // Diastole (Low Heartbeat): Tight Clamp (1.0) - Compression/Settling
                // heartbeat variable roughly 0.0 to 0.7. Normalize to 0..1 for phase?
                // self.global_time * freq -> sin. 
                // Let's use the calc from earlier: (self.global_time * self.heartbeat_freq).sin().abs();
                let phase_strength = (self.global_time * self.heartbeat_freq).sin().abs();
                let clamp_limit = 1.0 + (2.0 * phase_strength);
                
                new_w = new_w.clamp(-clamp_limit, clamp_limit);
                
                rho[[i, j]] = new_w;


            }
        }
        
        // Final Output Calculation for Momentum
        let final_activation = self.rho_phase[layer].sum_axis(Axis(0));
        self.last_output = Some(final_activation.clone());

        // Project back to neuron space
        final_activation
    }

}