
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rayon::prelude::*;

/// Harmonic BDH model: GPU-accelerated implementation using Burn Tensors.
/// 
/// State is stored as a 4D Tensor: [Layers, Modes, Neurons, 2 (Real/Imag)]
/// Coupling is a 2D Tensor: [Modes, Neurons]
pub struct HarmonicBdhBurn<B: Backend> {
    pub n: usize,
    pub d: usize,
    pub num_layers: usize,
    
    // Tensors
    /// Complex state rho: [Layers, Modes, Neurons, 2]
    /// Last dimension: 0 = Real, 1 = Imag
    pub rho: Tensor<B, 4>,
    
    /// Coupling phases: [Modes, Neurons] (Pre-computed cosines)
    pub coupling: Tensor<B, 2>,
    
    /// Natural frequencies: [Neurons]
    pub natural_freq: Tensor<B, 1>,
    
    /// Layer frequencies: [Layers]
    pub layer_freq: Tensor<B, 1>,
    
    /// Damping per layer: [Layers]
    pub damping: Tensor<B, 1>,
    
    // Config
    base_dt: f32,
    pub noise_scale: f32,
    pub input_gain: f32,
    pub self_excitation: f32,
    
    /// Homeostatic tracking: [Neurons] (fatigue/usage moving average)
    pub usage: Tensor<B, 1>,
    pub fatigue_rate: f32,
    pub inhibition_factor: f32,
    pub coupling_strength: f32,
}

impl<B: Backend> HarmonicBdhBurn<B> {
    pub fn new(n: usize, d: usize, num_layers: usize, device: &B::Device) -> Self {
        
        // Initialize State (Zero)
        // Shape: [Layers, Modes, Neurons, 2]
        let rho = Tensor::<B, 4>::zeros([num_layers, d, n, 2], device);
        
        // Initialize Frequencies (Reduced for numerical stability)
        // 0.1 to 1.0 (Was 6.28)
        let freq_data: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32 / n as f32) * 0.9).collect();
        let natural_freq = Tensor::<B, 1>::from_floats(freq_data.as_slice(), device);

        // Initialize Layer Frequencies (Reduced)
        let layer_freq_data: Vec<f32> = (0..num_layers)
            .map(|l| 1.0 + (l as f32 * 0.5))
            .collect();
        let layer_freq = Tensor::<B, 1>::from_floats(layer_freq_data.as_slice(), device);

        // Initialize Coupling (Cosines)
        // Shape: [Modes, Neurons]
        // Calculating on CPU first then moving to Tensor is easier for initialization
        let mut coupling_data = Vec::with_capacity(d * n);
        for mode in 0..d {
            for neuron in 0..n {
                let harmonic = (mode + 1) as f32;
                let phase = (neuron as f32 / n as f32) * 6.28 * harmonic;
                coupling_data.push(phase.cos() / (d as f32).sqrt());
            }
        }
        // Tensor from 1D data, reshape to [d, n]
        // Note: from_floats_1d creates a 1D tensor. Then reshape to 2D.
        let coupling = Tensor::<B, 1>::from_floats(coupling_data.as_slice(), device)
            .reshape([d, n]);

        // Damping (constant 0.95)
        let damping = Tensor::<B, 1>::full([num_layers], 0.9, device);
        
        let usage = Tensor::<B, 1>::zeros([n], device);

        Self {
            n,
            d,
            num_layers,
            rho,
            coupling,
            natural_freq,
            layer_freq,
            damping,
            base_dt: 0.01,
            noise_scale: 0.0,
            input_gain: 0.1,
            self_excitation: 0.02,
            usage,
            fatigue_rate: 0.005,
            inhibition_factor: 0.2,
            coupling_strength: 0.01, // Subtle coupling for ensemble formation
        }
    }

    /// Set the noise scale for stochastic injection.
    pub fn set_noise(&mut self, scale: f32) {
        self.noise_scale = scale;
    }

    /// Set input gain (plasticity/sensitivity).
    pub fn set_input_gain(&mut self, gain: f32) {
        self.input_gain = gain;
    }

    /// Modulate brain parameters based on chemical state.
    /// DA (Dopamine) -> Search/Resonance (Self-Excitation)
    /// NE (Norepinephrine) -> Gain/Noise (Sensitivity)
    /// ACh (Acetylcholine) -> Input Gain (Attention to external)
    pub fn modulate_neurochem(&mut self, da: f32, ne: f32, ach: f32) {
        
        // ACh: Cholinergic modulation controls "Signal to Noise".
        // STABILITY FIX: Clamp input gain to prevent overshoot
        self.set_input_gain(ach.clamp(0.1, 0.5)); // Was: ach.max(0.1) - now capped
        
        // NE: Noradrenergic modulation - AGGRESSIVE Noise to break attractors
        // If NE is high (frustration), noise goes up to 0.5
        self.noise_scale = 0.001 + (ne * 0.5);
        
        // DA: REDUCED self-excitation to prevent runaway
        self.self_excitation = da * 0.02; // Was: da * 0.1
        
        // INCREASED damping for stability (faster energy dissipation)
        if ach < 0.2 {
             self.set_damping(0.92); // Sleep: was 0.98
        } else {
             self.set_damping(0.85); // Wake: was 0.95 (MORE damping = more stable)
        }
    }

    /// Step the dynamics forward using Euler integration on the GPU.
    /// 
    /// # Arguments
    /// * `input`: Optional input tensor [Layers, Modes, Neurons] to inject current.
    pub fn step(&mut self, input: Option<Tensor<B, 3>>) {
        // 1. Calculate derivatives
        // dRho/dt = -iw * Rho - D * Rho + Input
        
        // Extract Real and Imag parts for calculation
        // Slice syntax: [.., 0] -> Real, [.., 1] -> Imag
        // Burn 0.14 slice returns same rank. squeeze<3>(3) reduces rank 4->3.
        let real: Tensor<B, 3> = self.rho.clone().slice([0..self.num_layers, 0..self.d, 0..self.n, 0..1]).squeeze(3);
        let imag: Tensor<B, 3> = self.rho.clone().slice([0..self.num_layers, 0..self.d, 0..self.n, 1..2]).squeeze(3);
        
        // Frequencies broadcast: [Neurons] -> [Layers, Modes, Neurons]
        // Effectively natural_freq * layer_freq
        // Reshape layer_freq to [Layers, 1, 1]
        let lf = self.layer_freq.clone().reshape([self.num_layers, 1, 1]);
        // Reshape natural_freq to [1, 1, Neurons]
        let nf = self.natural_freq.clone().reshape([1, 1, self.n]);
        
        // Total Freq Omega = LF * NF
        let omega = lf.mul(nf); // [Layers, 1, Neurons] -> broadcasts to [Layers, Modes, Neurons]
        
        // Rotation:
        // dReal = Omega * Imag
        // dImag = -Omega * Real
        let d_real = omega.clone().mul(imag.clone());
        let d_imag = omega.mul(real.clone()).neg();
        
        // Damping:
        // dReal -= Damping * Real
        // dImag -= Damping * Imag
        let d = self.damping.clone().reshape([self.num_layers, 1, 1]); // [Layers, 1, 1]
        
        let mut d_real = d_real.sub(real.clone().mul(d.clone()));
        let mut d_imag = d_imag.sub(imag.clone().mul(d));

        // PHASE COHERENCE: Mean-Field Coupling (Kuramoto-style)
        // Extract first layer/mode for global bias
        if self.coupling_strength > 0.0 {
            let mean_real = real.clone().mean_dim(2); // [Layers, Modes, 1]
            let mean_imag = imag.clone().mean_dim(2);
            
            // Influence pulls oscillators toward the ensemble average
            let coupling_bias_real = mean_real.sub(real.clone()).mul_scalar(self.coupling_strength);
            let coupling_bias_imag = mean_imag.sub(imag.clone()).mul_scalar(self.coupling_strength);
            
            d_real = d_real.add(coupling_bias_real);
            d_imag = d_imag.add(coupling_bias_imag);
        }

        // Van der Pol Self-Excitation
        // dX += mu * X * (1 - r^2)
        if self.self_excitation > 0.0 {
            // Calculate r^2 = Real^2 + Imag^2
            // Use clone() because powf_scalar consumes self
            let r2 = real.clone().powf_scalar(2.0).add(imag.clone().powf_scalar(2.0));
            // Term: (1 - r^2)
            let term = r2.mul_scalar(-1.0).add_scalar(1.0); // 1 - r^2
            
            // Real part contribution
            let vdp_real = real.clone().mul(term.clone()).mul_scalar(self.self_excitation);
            d_real = d_real.add(vdp_real);
            
            // Imag part contribution? Usually VdP is on the variable itself. 
            // In complex oscillator, applying to both keeps phase but modulates amplitude.
            let vdp_imag = imag.clone().mul(term).mul_scalar(self.self_excitation);
            d_imag = d_imag.add(vdp_imag);
        }
        
        // Inject Input if present
        // Input assumed to affect Real part (driving current)
        if let Some(inp) = input {
            // inp is [Layers, Modes, Neurons] ? Or just [Neurons] broadcasted?
            // Let's assume [Layers, Modes, Neurons] for full control.
            // Apply Input Gain (Plasticity/Attention)
            let gained_inp = inp.mul_scalar(self.input_gain);
            d_real = d_real.add(gained_inp);
        }

        // Noise Injection (Langevin-like dynamics)
        if self.noise_scale > 0.0 {
            // Generate noise: [Layers, Modes, Neurons]
            // We apply it to derivative (stochastic differential equation)
            // Scale by sqrt(dt) for proper Brownian scaling, but simple scale works for now.
            let noise_dist = burn::tensor::Distribution::Normal(0.0, self.noise_scale as f64);
            let noise = Tensor::<B, 3>::random(d_real.shape(), noise_dist, &self.rho.device());
            d_real = d_real.add(noise);
        }

        
        // Update State (Euler)
        let dt = self.base_dt;
        let new_real = real.add(d_real.mul_scalar(dt));
        let new_imag = imag.add(d_imag.mul_scalar(dt));

        // LATERAL INHIBITION: Competition among neurons
        // We only apply this to the new_real part which affects future dynamics
        // and interpretation.
        // threshold = mean + alpha * std
        let flat_real = new_real.clone().slice([0..1, 0..1, 0..self.n]).reshape([self.n]);
        let mean = flat_real.clone().mean();
        let var = flat_real.clone().var(0);
        let std = var.clamp_min(0.0).sqrt();
        let threshold = mean.add(std.mul_scalar(self.inhibition_factor));
        
        // Soft inhibition: keep full value if above threshold, else scale down
        // In Burn, we use greater() instead of greater_than()
        let mask_bi = flat_real.clone().greater_equal(threshold.clone().reshape([1])).float();
        let sup_mask = mask_bi.clone().add(mask_bi.neg().add_scalar(1.0).mul_scalar(0.1)); 
        
        // HOMEOSTASIS: Update Usage (Fatigue)
        // usage = (1-rate)*usage + rate*activation
        let rate = self.fatigue_rate;
        self.usage = self.usage.clone().mul_scalar(1.0 - rate)
            .add(flat_real.clone().abs().mul_scalar(rate));

        // Apply Inhibition to the entire 3D real/imag tensors (broadcasted)
        let mask_3d = sup_mask.reshape([1, 1, self.n]).expand([self.num_layers, self.d, self.n]);
        let final_real = new_real.mul(mask_3d.clone());
        let final_imag = new_imag.mul(mask_3d);
        
        // Stack back into 4D tensor
        let shape_4d = [self.num_layers, self.d, self.n, 1];
        let new_real_4d = final_real.reshape(shape_4d);
        let new_imag_4d = final_imag.reshape(shape_4d);
        
        self.rho = Tensor::cat(vec![new_real_4d, new_imag_4d], 3);
    }

    /// Set global damping for all layers.
    pub fn set_damping(&mut self, val: f32) {
         let device = self.damping.device();
         self.damping = Tensor::<B, 1>::full([self.num_layers], val, &device);
    }

    /// Multiply layer frequencies by a scalar factor (e.g. for Sleep/Wake).
    pub fn modulate_frequencies(&mut self, factor: f32) {
         self.layer_freq = self.layer_freq.clone().mul_scalar(factor);
    }
    
    /// HEBBIAN: Modifies natural frequencies based on input correlation + homeostasis.
    /// Includes Anti-Hebbian drift and usage-based saturation.
    pub fn hebbian_learn(&mut self, input: &[f32], learning_rate: f32) {
        if input.len() != self.n || learning_rate <= 0.0 { return; }
        
        let device = self.rho.device();
        let input_tensor = Tensor::<B, 1>::from_floats(input, &device);
        
        // Get cortical output as tensor: [Neurons]
        let output = self.rho.clone()
            .slice([0..1, 0..1, 0..self.n, 0..1])
            .reshape([self.n]);
        
        let mean_act = output.clone().mean();
        let target_sparsity = 0.05; // Ideal % of active neurons

        // 1. Saturation: High usage = low learning (prevents narrow attractors)
        // saturation = (usage_data[i] / target_sparsity).min(5.0);
        let saturation = self.usage.clone().div_scalar(target_sparsity).clamp(0.0, 5.0);
        // effective_lr = learning_rate / (1.0 + saturation);
        let effective_lr = saturation.add_scalar(1.0).recip().mul_scalar(learning_rate);

        // 2. Correlation (Hebbian)
        let correlation = input_tensor.mul(output.clone());
        
        // 3. Anti-Hebbian: Subtract global average to decorrelate
        // novelty_signal = correlation - (mean_act * output[i]);
        let novelty_signal = correlation.sub(output.mul(mean_act.unsqueeze()));
        
        // 4. Update with Forgetting Leak
        let delta_f = effective_lr.mul(novelty_signal);
        self.natural_freq = self.natural_freq.clone().add(delta_f).mul_scalar(0.99999).clamp(0.01, 10.0);
    }
    
    pub fn get_usage_vec(&self) -> Vec<f32> {
        self.usage.to_data().to_vec::<f32>().unwrap()
    }

    /// Synaptic Scaling: Weakens over-active "noise" neurons 
    /// and strengthens stable "semantic" neurons.
    pub fn consolidate_synapses(&mut self) {
        let n = self.n;
        let device = self.natural_freq.device();
        
        // Calculate global average usage
        let avg_tensor = self.usage.clone().mean();
        let global_usage_avg = avg_tensor.clone().into_data().iter::<f32>().next().unwrap_or(0.0);
        
        // Prepare thresholds as tensors for comparison
        let threshold_down = Tensor::<B, 1>::full([n], global_usage_avg * 1.5, &device);
        let threshold_up_high = Tensor::<B, 1>::full([n], global_usage_avg, &device);
        let threshold_up_low = Tensor::<B, 1>::full([n], 0.01, &device);
        
        // 1. Identification Masks
        let over_active_mask = self.usage.clone().greater_equal(threshold_down).float();
        
        // Stable but quiet: 0.01 < usage < avg
        // Using multiplication as logical AND for float masks
        let quiet_mask = self.usage.clone().greater_equal(threshold_up_low).float()
            .mul(self.usage.clone().lower(threshold_up_high).float());
            
        // 2. Build scaling tensor: 1.0 - (over_active * 0.05) + (quiet * 0.005)
        let scale = Tensor::<B, 1>::full([n], 1.0, &device)
            .sub(over_active_mask.clone().mul_scalar(0.05))
            .add(quiet_mask.mul_scalar(0.005));
            
        // 3. Apply to natural_freq
        self.natural_freq = self.natural_freq.clone().mul(scale).clamp(0.01, 10.0);
        
        // 4. USAGE RESET: Refresh capacity of consolidated neurons
        // We partially reset the fatigue for over-active neurons that were scaled
        let usage_scale = Tensor::<B, 1>::full([n], 1.0, &device)
            .sub(over_active_mask.mul_scalar(0.1)); // 10% reset
        self.usage = self.usage.clone().mul(usage_scale);
    }
    
    /// Get the current state (Real part of first layer) for interpretation.
    /// Returns [Neurons] Vector.
    pub fn get_cortical_output(&self) -> Vec<f32> {
        self.rho.clone()
            .slice([0..1, 0..1, 0..self.n, 0..1])
            .reshape([self.n])
            .to_data().to_vec::<f32>().unwrap()
    }

    /// Get the phases of the oscillators (Fundamental Layer, Fundamental Mode).
    /// Returns Vec<f32> of size N.
    pub fn get_phases(&self) -> Vec<f32> {
        let real = self.rho.clone().slice([0..1, 0..1, 0..self.n, 0..1]).reshape([self.n]).to_data().to_vec::<f32>().unwrap();
        let imag = self.rho.clone().slice([0..1, 0..1, 0..self.n, 1..2]).reshape([self.n]).to_data().to_vec::<f32>().unwrap();
        
        real.into_par_iter()
            .zip(imag)
            .map(|(r, i)| i.atan2(r))
            .collect()
    }

    /// Get normalized cortical output as an ndarray for similarity matching.
    pub fn get_cortical_output_norm(&self) -> Result<ndarray::Array1<f32>, String> {
        let vec = self.get_cortical_output();
        let arr = ndarray::Array1::from(vec);
        let norm = (arr.dot(&arr)).sqrt();
        if norm < 1e-8 {
            Ok(arr)
        } else {
            Ok(arr / norm)
        }
    }
    
    /// Calculate current energy (Magnitude squared of Rho).
    /// Returns [Layers] tensor of total energy.
    pub fn get_energy(&self) -> Tensor<B, 1> {
        let real = self.rho.clone().slice([0..self.num_layers, 0..self.d, 0..self.n, 0..1]).squeeze::<3>(3);
        let imag = self.rho.clone().slice([0..self.num_layers, 0..self.d, 0..self.n, 1..2]).squeeze::<3>(3);
        
        let r2 = real.powf_scalar(2.0);
        let i2 = imag.powf_scalar(2.0);
        let mag2 = r2.add(i2); // [Layers, Modes, Neurons]
        
        mag2.sum_dim(2).sum_dim(1).squeeze::<2>(2).squeeze::<1>(1)
    }

    /// Get energies of individual neurons (Fundamental Layer, Fundamental Mode).
    /// Returns Vec<f32> of size N.
    pub fn get_neuron_energies(&self) -> Vec<f32> {
        // Take Layer 0, Mode 0
        let real = self.rho.clone().slice([0..1, 0..1, 0..self.n, 0..1]).squeeze::<3>(3).squeeze::<2>(1).squeeze::<1>(0);
        let imag = self.rho.clone().slice([0..1, 0..1, 0..self.n, 1..2]).squeeze::<3>(3).squeeze::<2>(1).squeeze::<1>(0);
        
        let mag2 = real.powf_scalar(2.0).add(imag.powf_scalar(2.0));
        mag2.to_data().to_vec::<f32>().unwrap()
    }

    /// Save state to disk (Binary Dump).
    pub fn save_state(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        
        // Save dimensions header
        file.write_all(&self.n.to_le_bytes())?;
        file.write_all(&self.d.to_le_bytes())?;
        file.write_all(&self.num_layers.to_le_bytes())?;
        
        // Save Tensor Data (Rho)
        let data = self.rho.to_data().to_vec::<f32>().unwrap();
        // Write raw floats
        for float in data {
            file.write_all(&float.to_le_bytes())?;
        }
        
        Ok(())
    }
    
    /// Load state from disk.
    pub fn load_state(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        
        // Read dimensions check
        let mut buf = [0u8; 8]; // usize is 8 byte on 64bit
        file.read_exact(&mut buf)?;
        let n: usize = usize::from_le_bytes(buf);
        file.read_exact(&mut buf)?;
        let d: usize = usize::from_le_bytes(buf);
        file.read_exact(&mut buf)?;
        let num_layers: usize = usize::from_le_bytes(buf);
        
        if n != self.n || d != self.d || num_layers != self.num_layers {
             return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Dimensions mismatch"));
        }
        
        // Read floats
        let total_floats = self.num_layers * self.d * self.n * 2;
        let mut float_buf = [0u8; 4];
        let mut data_vec = Vec::with_capacity(total_floats);
        
        for _ in 0..total_floats {
            file.read_exact(&mut float_buf)?;
            data_vec.push(f32::from_le_bytes(float_buf));
        }
        
        let device = self.rho.device();
        // 1. Create flat 1D tensor first
        let flat_tensor = Tensor::<B, 1>::from_floats(data_vec.as_slice(), &device);
        
        // 2. Reshape to 4D
        let loaded_tensor = flat_tensor.reshape([self.num_layers, self.d, self.n, 2]);
            
        self.rho = loaded_tensor;
        
        Ok(())
    }
}
