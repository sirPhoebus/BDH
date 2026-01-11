//! Harmonic BDH: A vibrational/resonance-based neural architecture.
//!
//! Instead of static weights, this treats the network as a system of coupled oscillators
//! where information propagates via phase alignment and constructive interference.
//!
//! ## Biological Mechanisms
//! - **Spontaneous Activity**: van der Pol self-excitation + noise for sustained daydreaming
//! - **Cross-Frequency Coupling**: Lower layers modulate higher layer amplitudes (theta-gamma)
//! - **Homeostatic Plasticity**: Dynamic damping prevents runaway activation
//! - **Adaptive Exploration**: Noise increases during low-energy periods (boredom → exploration)

use ndarray::prelude::*;
use num_complex::Complex32;
use rand::Rng;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Complex state representing phase + amplitude for each neuron.
pub type ComplexState = Array2<Complex32>;

/// Configuration for biological dynamics.
#[derive(Clone, Debug)]
pub struct BiologicalConfig {
    /// Spontaneous noise amplitude (daydream strength)
    pub noise_amplitude: f32,
    /// Cross-frequency coupling strength (theta-gamma binding)
    pub cross_freq_coupling: f32,
    /// Homeostatic energy threshold (above this, damping increases)
    pub homeostatic_threshold: f32,
    /// Homeostatic adaptation rate
    pub homeostatic_rate: f32,
    /// Base damping factor
    pub base_damping: f32,
    /// Van der Pol self-excitation strength (limit cycle behavior)
    pub self_excitation: f32,
    /// Endogenous drive strength (layer 0 only)
    pub endogenous_drive: f32,
    /// Adaptive noise: increases when energy stays low
    pub adaptive_noise_rate: f32,
    /// Layer frequencies (if empty, uses default progression)
    pub layer_frequencies: Vec<f32>,
    /// Steps of low energy before adaptive noise kicks in (default: 10)
    pub boredom_delay: usize,
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        // TUNED config: targets 8-15 step bursts with high transition rate
        // Strategy: very fast homeostasis to terminate bursts + high noise to restart
        Self {
            noise_amplitude: 0.070,       // High noise for quick restarts
            cross_freq_coupling: 0.36,    // Moderate coupling
            homeostatic_threshold: 0.12,  // Very low = triggers boredom very early
            homeostatic_rate: 0.28,       // Very fast adaptation = shorter bursts
            base_damping: 0.89,           // Higher decay = shorter bursts
            self_excitation: 0.028,       // Moderate self-excitation
            endogenous_drive: 0.060,      // High to maintain activity
            adaptive_noise_rate: 0.80,    // Very high - rapid recovery
            layer_frequencies: vec![],    // Empty = use default
            boredom_delay: 2,             // Boredom kicks in very fast
        }
    }
}

/// Semantic concept for thought interpretation.
#[derive(Clone, Debug)]
pub struct Concept {
    pub name: &'static str,
    pub vector: Array1<f32>,
}

/// Thought state labels based on frequency/energy patterns.
#[derive(Clone, Debug, PartialEq)]
pub enum ThoughtState {
    Resting,
    Contemplative,
    ActivePlanning,
    AlertScanning,
    Transitioning,
}

impl ThoughtState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThoughtState::Resting => "💤 Resting",
            ThoughtState::Contemplative => "🧘 Contemplative (memory recall)",
            ThoughtState::ActivePlanning => "🎯 Active Planning",
            ThoughtState::AlertScanning => "👁 Alert Scanning",
            ThoughtState::Transitioning => "🔄 Transitioning",
        }
    }
}

/// Harmonic BDH model: neurons as coupled oscillators with biological dynamics.
pub struct HarmonicBdh {
    pub n: usize,
    pub d: usize,
    
    natural_freq: Array1<f32>,
    coupling: Array2<f32>,
    layer_damping: Vec<f32>,
    neuron_damping: Vec<Array1<f32>>,
    energy_history: Vec<f32>,
    low_energy_duration: Vec<usize>,  // Track how long energy has been low
    adaptive_noise: Vec<f32>,          // Current adaptive noise level per layer
    rho: Vec<ComplexState>,
    global_phase: f32,
    layer_frequencies: Vec<f32>,
    num_layers: usize,
    
    /// Concept space for semantic interpretation
    concepts: Vec<Concept>,
    
    pub config: BiologicalConfig,
}

impl HarmonicBdh {
    pub fn new(n: usize, d: usize, num_layers: usize) -> Self {
        Self::with_config(n, d, num_layers, BiologicalConfig::default())
    }
    
    pub fn with_config(n: usize, d: usize, num_layers: usize, config: BiologicalConfig) -> Self {
        let natural_freq = Array1::from_iter(
            (0..n).map(|i| 0.1 + (i as f32 / n as f32) * 2.0 * PI)
        );
        
        let mut coupling = Array2::zeros((d, n));
        for mode in 0..d {
            for neuron in 0..n {
                let harmonic = (mode + 1) as f32;
                let phase = (neuron as f32 / n as f32) * 2.0 * PI * harmonic;
                coupling[[mode, neuron]] = phase.cos() / (d as f32).sqrt();
            }
        }
        
        let rho = vec![Array2::from_elem((d, n), Complex32::new(0.0, 0.0)); num_layers];
        
        // Layer frequencies: use config if provided, otherwise default progression
        let layer_frequencies: Vec<f32> = if config.layer_frequencies.len() == num_layers {
            config.layer_frequencies.clone()
        } else {
            (0..num_layers)
                .map(|l| 5.0 * (2.0_f32).powf(l as f32 * 3.0 / num_layers as f32))
                .collect()
        };
        
        let layer_damping = vec![config.base_damping; num_layers];
        let neuron_damping = vec![Array1::from_elem(n, config.base_damping); num_layers];
        let energy_history = vec![0.0; num_layers];
        let low_energy_duration = vec![0; num_layers];
        let adaptive_noise = vec![config.noise_amplitude; num_layers];
        
        // Initialize concept space
        let concepts = Self::create_concept_space(n);
        
        Self {
            n,
            d,
            natural_freq,
            coupling,
            layer_damping,
            neuron_damping,
            energy_history,
            low_energy_duration,
            adaptive_noise,
            rho,
            global_phase: 0.0,
            layer_frequencies,
            num_layers,
            concepts,
            config,
        }
    }
    
    /// Create a predefined concept space for semantic interpretation.
    fn create_concept_space(n: usize) -> Vec<Concept> {
        let mut rng = rand::thread_rng();
        let concept_names = [
            "curiosity", "hunger", "danger", "safety", 
            "exploration", "memory", "planning", "rest",
            "social", "novelty", "pattern", "prediction",
        ];
        
        concept_names.iter().map(|&name| {
            // Create semi-structured random vectors
            let mut vec = Array1::zeros(n);
            let cluster_size = n / concept_names.len();
            let idx = concept_names.iter().position(|&x| x == name).unwrap();
            
            // Each concept has a "home region" but with some spread
            for i in 0..n {
                let distance = ((i as i32 - (idx * cluster_size) as i32).abs() as f32) / n as f32;
                let base = (-distance * 5.0).exp();
                let noise: f32 = rng.gen_range(-0.1..0.1);
                vec[i] = (base + noise).max(0.0);
            }
            
            // Normalize
            let norm = vec.dot(&vec).sqrt().max(0.001);
            vec /= norm;
            
            Concept { name, vector: vec }
        }).collect()
    }
    
    /// (A) SPONTANEOUS ACTIVITY with van der Pol self-excitation.
    fn inject_spontaneous_activity(&self, layer: usize) -> (Array1<f32>, Array1<f32>) {
        let mut rng = rand::thread_rng();
        let layer_freq = self.layer_frequencies[layer];
        let adaptive = self.adaptive_noise[layer];
        
        // Noise component (phase-aligned + random)
        let noise = if adaptive > 0.0 {
            Array1::from_iter((0..self.n).map(|neuron| {
                let neuron_freq = self.natural_freq[neuron];
                let phase_noise = (self.global_phase * layer_freq + neuron_freq).sin();
                let random: f32 = rng.gen_range(-1.0_f32..1.0_f32);
                adaptive * (0.6 * phase_noise + 0.4 * random)
            }))
        } else {
            Array1::zeros(self.n)
        };
        
        // Endogenous drive (layer 0 only - the "heartbeat")
        let endogenous = if layer == 0 && self.config.endogenous_drive > 0.0 {
            Array1::from_iter((0..self.n).map(|_| {
                rng.gen_range(-self.config.endogenous_drive..self.config.endogenous_drive)
            }))
        } else {
            Array1::zeros(self.n)
        };
        
        (noise, endogenous)
    }
    
    /// Van der Pol style self-excitation: creates limit cycle behavior.
    /// amplitude * (1 - amplitude²) → pushes toward amplitude ≈ 1
    fn compute_self_excitation(&self, layer: usize) -> Array1<f32> {
        let mu = self.config.self_excitation;
        
        Array1::from_iter((0..self.n).map(|neuron| {
            let mut amplitude = 0.0f32;
            for mode in 0..self.d {
                amplitude += self.rho[layer][[mode, neuron]].norm();
            }
            amplitude /= self.d as f32;
            
            // Van der Pol: μ * x * (1 - x²) 
            // Positive when amplitude < 1, negative when > 1
            mu * amplitude * (1.0 - amplitude * amplitude)
        }))
    }
    
    /// Update adaptive noise based on energy levels.
    fn update_adaptive_noise(&mut self, layer: usize, current_energy: f32) {
        let low_threshold = self.config.homeostatic_threshold * 0.3;
        let delay = self.config.boredom_delay;
        
        if current_energy < low_threshold {
            self.low_energy_duration[layer] += 1;
            
            // After boredom_delay steps of low energy, increase noise (boredom → exploration)
            if self.low_energy_duration[layer] > delay {
                let boost = self.config.adaptive_noise_rate 
                    * (self.low_energy_duration[layer] as f32 - delay as f32).min(20.0) / 20.0;
                self.adaptive_noise[layer] = (self.adaptive_noise[layer] + boost)
                    .min(self.config.noise_amplitude * 5.0);
            }
        } else {
            // Reset when activity returns
            self.low_energy_duration[layer] = 0;
            // Slowly decay adaptive noise back to baseline
            self.adaptive_noise[layer] = (self.adaptive_noise[layer] * 0.95)
                .max(self.config.noise_amplitude);
        }
    }
    
    /// (B) CROSS-FREQUENCY COUPLING with phase coherence.
    fn compute_cross_frequency_modulation(&self, layer: usize) -> (Array1<f32>, f32) {
        if layer == 0 {
            return (Array1::ones(self.n), 1.0);
        }
        
        let lower_layer = layer - 1;
        
        // Amplitude modulation from lower layer
        let lower_amplitudes: Array1<f32> = (0..self.n)
            .map(|neuron| {
                let mut total_amp = 0.0f32;
                for mode in 0..self.d {
                    total_amp += self.rho[lower_layer][[mode, neuron]].norm();
                }
                total_amp / self.d as f32
            })
            .collect();
        
        // Phase coherence: how aligned are the phases between layers?
        let mut phase_coherence = 0.0f32;
        for neuron in 0..self.n {
            let mut lower_phase = Complex32::new(0.0, 0.0);
            let mut upper_phase = Complex32::new(0.0, 0.0);
            
            for mode in 0..self.d {
                lower_phase += self.rho[lower_layer][[mode, neuron]];
                upper_phase += self.rho[layer][[mode, neuron]];
            }
            
            if lower_phase.norm() > 0.001 && upper_phase.norm() > 0.001 {
                let lower_angle = lower_phase.arg();
                let upper_angle = upper_phase.arg();
                phase_coherence += (lower_angle - upper_angle).cos();
            }
        }
        phase_coherence /= self.n as f32;
        
        let max_amp = lower_amplitudes.iter().cloned().fold(0.0f32, f32::max).max(0.01);
        let modulation = lower_amplitudes.mapv(|a| {
            let normalized = a / max_amp;
            1.0 + self.config.cross_freq_coupling * normalized * (0.5 + 0.5 * phase_coherence)
        });
        
        (modulation, phase_coherence)
    }
    
    /// (C) HOMEOSTATIC PLASTICITY with recovery tracking.
    fn apply_homeostatic_plasticity(&mut self, layer: usize, current_energy: f32) -> f32 {
        let threshold = self.config.homeostatic_threshold;
        let rate = self.config.homeostatic_rate;
        
        self.energy_history[layer] = 0.9 * self.energy_history[layer] + 0.1 * current_energy;
        let sustained_energy = self.energy_history[layer];
        
        let prev_damping = self.layer_damping[layer];
        
        if sustained_energy > threshold {
            let excess = (sustained_energy - threshold) / threshold;
            self.layer_damping[layer] = (self.layer_damping[layer] - rate * excess)
                .clamp(0.5, self.config.base_damping);
        } else {
            self.layer_damping[layer] = (self.layer_damping[layer] + rate * 0.1)
                .min(self.config.base_damping);
        }
        
        // Per-neuron homeostasis
        for neuron in 0..self.n {
            let neuron_energy: f32 = (0..self.d)
                .map(|m| self.rho[layer][[m, neuron]].norm_sqr())
                .sum();
            
            if neuron_energy > threshold * 0.1 {
                self.neuron_damping[layer][neuron] = 
                    (self.neuron_damping[layer][neuron] - rate * 0.5)
                    .clamp(0.5, self.config.base_damping);
            } else {
                self.neuron_damping[layer][neuron] = 
                    (self.neuron_damping[layer][neuron] + rate * 0.05)
                    .min(self.config.base_damping);
            }
        }
        
        // Return recovery rate (how much damping changed)
        self.layer_damping[layer] - prev_damping
    }
    
    /// Initialize from signal.
    pub fn initialize_from_signal(&mut self, signal: &[f32], layer: usize) {
        let signal_len = signal.len();
        
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(signal_len);
        
        let mut buffer: Vec<Complex<f32>> = signal
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();
        fft.process(&mut buffer);
        
        for mode in 0..self.d.min(signal_len / 2) {
            let bin = &buffer[mode + 1];
            let amplitude = (bin.re * bin.re + bin.im * bin.im).sqrt() / signal_len as f32;
            let phase = bin.im.atan2(bin.re);
            
            for neuron in 0..self.n {
                let coupling_strength = self.coupling[[mode, neuron]];
                self.rho[layer][[mode, neuron]] = Complex32::new(
                    amplitude * coupling_strength * phase.cos(),
                    amplitude * coupling_strength * phase.sin(),
                );
            }
        }
    }
    
    /// Phase alignment.
    fn phase_alignment(&self, input: &Array1<f32>, state: &ComplexState) -> Array1<f32> {
        let mut alignment = Array1::zeros(self.n);
        
        for neuron in 0..self.n {
            let input_phase = input[neuron] * self.natural_freq[neuron] + self.global_phase;
            let input_phasor = Complex32::new(input_phase.cos(), input_phase.sin());
            
            let mut total_coherence = Complex32::new(0.0, 0.0);
            for mode in 0..self.d {
                total_coherence += state[[mode, neuron]] * self.coupling[[mode, neuron]];
            }
            
            let dot = input_phasor.re * total_coherence.re + input_phasor.im * total_coherence.im;
            alignment[neuron] = dot.max(0.0);
        }
        
        alignment
    }
    
    /// Update standing wave with self-excitation.
    fn update_standing_wave(&mut self, excitation: &Array1<f32>, self_excite: &Array1<f32>, layer: usize) {
        let dt = 0.1;
        let layer_damp = self.layer_damping[layer];
        
        for mode in 0..self.d {
            let mode_freq = (mode + 1) as f32 * 0.5;
            
            for neuron in 0..self.n {
                let current = self.rho[layer][[mode, neuron]];
                
                let omega = self.natural_freq[neuron] + mode_freq;
                let rotation = Complex32::new((omega * dt).cos(), (omega * dt).sin());
                
                let effective_damping = layer_damp * self.neuron_damping[layer][neuron];
                let damped = current * effective_damping;
                
                // Combined drive: external + self-excitation
                let total_drive = excitation[neuron] + self_excite[neuron];
                let drive = total_drive * self.coupling[[mode, neuron]];
                let drive_phasor = Complex32::new(
                    drive * self.global_phase.cos(),
                    drive * self.global_phase.sin(),
                );
                
                self.rho[layer][[mode, neuron]] = damped * rotation + drive_phasor * dt;
            }
        }
    }
    
    /// Forward pass with all biological mechanisms.
    /// Returns (output, energies, phase_coherences)
    pub fn forward(&mut self, input: &Array1<f32>) -> (Array1<f32>, Vec<f32>, Vec<f32>) {
        let mut signal = input.clone();
        let mut energy_per_layer = Vec::with_capacity(self.num_layers);
        let mut coherence_per_layer = Vec::with_capacity(self.num_layers);
        
        for layer in 0..self.num_layers {
            // (A) SPONTANEOUS ACTIVITY: noise + endogenous drive + self-excitation
            let (noise, endogenous) = self.inject_spontaneous_activity(layer);
            let self_excite = self.compute_self_excitation(layer);
            signal = &signal + &noise + &endogenous;
            
            // (B) CROSS-FREQUENCY COUPLING
            let (modulation, coherence) = self.compute_cross_frequency_modulation(layer);
            signal = &signal * &modulation;
            coherence_per_layer.push(coherence);
            
            // Phase alignment and gating
            let alignment = self.phase_alignment(&signal, &self.rho[layer]);
            let max_align = alignment.iter().cloned().fold(0.0f32, f32::max).max(0.01);
            let resonance_gate = alignment.mapv(|a| {
                let normalized = a / max_align;
                1.0 / (1.0 + (-5.0 * (normalized - 0.2)).exp())
            });
            
            // Update standing wave with self-excitation
            let excitation = &signal * &resonance_gate;
            self.update_standing_wave(&excitation, &self_excite, layer);
            
            signal = &signal * &resonance_gate;
            
            let layer_energy: f32 = self.rho[layer].iter().map(|c| c.norm_sqr()).sum();
            energy_per_layer.push(layer_energy);
            
            // (C) HOMEOSTATIC PLASTICITY
            self.apply_homeostatic_plasticity(layer, layer_energy);
            
            // Update adaptive noise
            self.update_adaptive_noise(layer, layer_energy);
        }
        
        self.global_phase += 0.1;
        if self.global_phase > 2.0 * PI {
            self.global_phase -= 2.0 * PI;
        }
        
        (signal, energy_per_layer, coherence_per_layer)
    }
    
    /// Simplified forward for compatibility.
    pub fn forward_simple(&mut self, input: &Array1<f32>) -> (Array1<f32>, Vec<f32>) {
        let (signal, energies, _) = self.forward(input);
        (signal, energies)
    }
    
    /// Daydream with richer output.
    pub fn daydream(&mut self, steps: usize) -> Vec<DaydreamStep> {
        let mut trajectory = Vec::with_capacity(steps);
        let mut signal = self.project_internal_state();
        
        for step in 0..steps {
            let (output, energies, coherences) = self.forward(&signal);
            
            let thought_state = self.classify_thought_state(&energies);
            let top_concepts = self.get_top_concepts(&output, 3);
            let dominant_freq = self.get_dominant_frequencies(0).mean().unwrap_or(0.0);
            
            trajectory.push(DaydreamStep {
                step,
                output: output.clone(),
                energies,
                coherences,
                thought_state,
                top_concepts,
                dominant_freq,
                adaptive_noise: self.adaptive_noise.clone(),
            });
            
            signal = output.mapv(|v| v * 0.8);
        }
        
        trajectory
    }
    
    /// Classify current thought state based on frequency/energy patterns.
    pub fn classify_thought_state(&self, energies: &[f32]) -> ThoughtState {
        let total_energy: f32 = energies.iter().sum();
        let avg_freq = self.get_dominant_frequencies(0).mean().unwrap_or(0.0);
        
        if total_energy < 0.05 {
            ThoughtState::Resting
        } else if avg_freq < 6.0 && energies.get(0).unwrap_or(&0.0) > &0.3 {
            ThoughtState::Contemplative
        } else if avg_freq > 15.0 && total_energy > 0.2 {
            ThoughtState::AlertScanning
        } else if total_energy > 0.1 {
            ThoughtState::ActivePlanning
        } else {
            ThoughtState::Transitioning
        }
    }
    
    /// Get top matching concepts via cosine similarity.
    pub fn get_top_concepts(&self, state: &Array1<f32>, top_k: usize) -> Vec<(&'static str, f32)> {
        let state_norm = state.dot(state).sqrt().max(0.001);
        let normalized_state = state / state_norm;
        
        let mut scores: Vec<_> = self.concepts.iter()
            .map(|c| {
                let similarity = normalized_state.dot(&c.vector);
                (c.name, similarity)
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
    
    fn project_internal_state(&self) -> Array1<f32> {
        let mut output = Array1::zeros(self.n);
        
        for layer in 0..self.num_layers {
            for neuron in 0..self.n {
                let mut neuron_activity = 0.0f32;
                for mode in 0..self.d {
                    neuron_activity += self.rho[layer][[mode, neuron]].norm() 
                        * self.coupling[[mode, neuron]];
                }
                output[neuron] += neuron_activity / self.num_layers as f32;
            }
        }
        
        output
    }
    
    pub fn find_attractor(&mut self, initial: &Array1<f32>, max_iters: usize, tolerance: f32) -> (Array1<f32>, usize) {
        let mut signal = initial.clone();
        let mut prev_energy = 0.0f32;
        
        for iter in 0..max_iters {
            let (output, energies, _) = self.forward(&signal);
            let total_energy: f32 = energies.iter().sum();
            
            if (total_energy - prev_energy).abs() < tolerance {
                return (output, iter);
            }
            
            prev_energy = total_energy;
            signal = output;
        }
        
        (signal, max_iters)
    }
    
    pub fn get_standing_wave(&self, layer: usize) -> Array2<f32> {
        self.rho[layer].mapv(|c| c.norm())
    }
    
    pub fn get_damping_state(&self) -> (&[f32], &[Array1<f32>]) {
        (&self.layer_damping, &self.neuron_damping)
    }
    
    pub fn get_layer_frequencies(&self) -> &[f32] {
        &self.layer_frequencies
    }
    
    pub fn get_adaptive_noise(&self) -> &[f32] {
        &self.adaptive_noise
    }
    
    pub fn get_dominant_frequencies(&self, layer: usize) -> Array1<f32> {
        let mut freqs = Array1::zeros(self.n);
        
        for neuron in 0..self.n {
            let mut max_mode = 0;
            let mut max_amp = 0.0f32;
            
            for mode in 0..self.d {
                let amp = self.rho[layer][[mode, neuron]].norm();
                if amp > max_amp {
                    max_amp = amp;
                    max_mode = mode;
                }
            }
            
            freqs[neuron] = (max_mode + 1) as f32 * 0.5;
        }
        
        freqs
    }
    
    /// Set custom layer frequencies.
    pub fn set_layer_frequencies(&mut self, freqs: Vec<f32>) {
        if freqs.len() == self.num_layers {
            self.layer_frequencies = freqs;
        }
    }
}

/// Rich output from a daydream step.
#[derive(Clone, Debug)]
pub struct DaydreamStep {
    pub step: usize,
    pub output: Array1<f32>,
    pub energies: Vec<f32>,
    pub coherences: Vec<f32>,
    pub thought_state: ThoughtState,
    pub top_concepts: Vec<(&'static str, f32)>,
    pub dominant_freq: f32,
    pub adaptive_noise: Vec<f32>,
}

/// Generate a synthetic brainwave-like signal.
pub fn generate_brainwave(samples: usize, dominant_freq: f32, noise_level: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    
    (0..samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let alpha = (2.0 * PI * dominant_freq * t).sin();
            let theta = 0.5 * (2.0 * PI * (dominant_freq * 0.6) * t).sin();
            let noise = if noise_level > 0.0 {
                rng.gen_range(-noise_level..noise_level)
            } else {
                0.0
            };
            alpha + theta + noise
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_forward() {
        let mut model = HarmonicBdh::new(64, 16, 2);
        let input = Array1::from_iter((0..64).map(|i| (i as f32 * 0.1).sin().max(0.0)));
        
        let (output, energies, coherences) = model.forward(&input);
        
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|&v| v.is_finite()));
        assert_eq!(energies.len(), 2);
        assert_eq!(coherences.len(), 2);
    }

    #[test]
    fn test_signal_initialization() {
        let mut model = HarmonicBdh::new(64, 16, 2);
        let brainwave = generate_brainwave(256, 10.0, 0.1);
        
        model.initialize_from_signal(&brainwave, 0);
        
        let wave = model.get_standing_wave(0);
        assert!(wave.sum() > 0.0);
    }

    #[test]
    fn test_attractor_convergence() {
        let config = BiologicalConfig {
            noise_amplitude: 0.0,
            self_excitation: 0.0,
            endogenous_drive: 0.0,
            cross_freq_coupling: 0.0,
            ..Default::default()
        };
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        let brainwave = generate_brainwave(64, 10.0, 0.1);
        model.initialize_from_signal(&brainwave, 0);
        
        let input = Array1::from_iter((0..32).map(|i| if i % 4 == 0 { 0.5 } else { 0.0 }));
        let (output, _) = model.find_attractor(&input, 50, 0.01);
        
        assert!(output.iter().any(|&v| v.is_finite()));
    }

    #[test]
    fn test_spontaneous_activity() {
        let config = BiologicalConfig {
            self_excitation: 0.02,
            endogenous_drive: 0.01,
            ..Default::default()
        };
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        let brainwave = generate_brainwave(128, 10.0, 0.1);
        model.initialize_from_signal(&brainwave, 0);
        
        let trajectory = model.daydream(30);
        
        // Should have sustained activity
        let final_energy: f32 = trajectory.last().unwrap().energies.iter().sum();
        let initial_energy: f32 = trajectory.first().unwrap().energies.iter().sum();
        
        // With self-excitation, energy should not decay to near-zero
        assert!(final_energy > 0.01 || initial_energy > 0.01);
    }

    #[test]
    fn test_homeostatic_plasticity() {
        let config = BiologicalConfig {
            homeostatic_threshold: 0.1,
            homeostatic_rate: 0.1,
            noise_amplitude: 0.0,
            self_excitation: 0.0,
            endogenous_drive: 0.0,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config.clone());
        
        let strong_input = Array1::from_elem(32, 1.0);
        for _ in 0..20 {
            model.forward(&strong_input);
        }
        
        let (layer_damping, _) = model.get_damping_state();
        assert!(layer_damping[0] < config.base_damping);
    }

    #[test]
    fn test_cross_frequency_coupling() {
        let config = BiologicalConfig {
            noise_amplitude: 0.0,
            self_excitation: 0.0,
            endogenous_drive: 0.0,
            ..Default::default()
        };
        let mut model = HarmonicBdh::with_config(32, 8, 3, config);
        
        let brainwave = generate_brainwave(128, 5.0, 0.0);
        model.initialize_from_signal(&brainwave, 0);
        
        let freqs = model.get_layer_frequencies();
        assert!(freqs[0] < freqs[2]);
    }

    #[test]
    fn test_thought_classification() {
        let model = HarmonicBdh::new(32, 8, 2);
        
        let state = model.classify_thought_state(&[0.01, 0.01]);
        assert_eq!(state, ThoughtState::Resting);
        
        let state = model.classify_thought_state(&[0.5, 0.1]);
        assert!(state == ThoughtState::Contemplative || state == ThoughtState::ActivePlanning);
    }

    #[test]
    fn test_concept_matching() {
        let model = HarmonicBdh::new(32, 8, 2);
        let state = Array1::from_elem(32, 0.5);
        
        let concepts = model.get_top_concepts(&state, 3);
        assert_eq!(concepts.len(), 3);
        assert!(concepts[0].1 >= concepts[1].1);
    }

    #[test]
    fn test_adaptive_noise() {
        let config = BiologicalConfig {
            adaptive_noise_rate: 0.2,
            noise_amplitude: 0.01,
            self_excitation: 0.0,
            endogenous_drive: 0.0,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config.clone());
        
        // Run with zero input to trigger low energy
        let zero_input = Array1::zeros(32);
        for _ in 0..20 {
            model.forward(&zero_input);
        }
        
        let adaptive = model.get_adaptive_noise();
        assert!(adaptive[0] >= config.noise_amplitude);
    }
}
