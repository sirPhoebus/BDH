use ndarray::{prelude::*, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rand::thread_rng;

pub mod lsh;
pub mod harmonic;

pub use lsh::{LshEmbedder, Vocabulary};
pub use harmonic::{HarmonicBdh, generate_brainwave};

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