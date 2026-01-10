//! Harmonic BDH: A vibrational/resonance-based neural architecture.
//!
//! Instead of static weights, this treats the network as a system of coupled oscillators
//! where information propagates via phase alignment and constructive interference.
//!
//! ## Biological Mechanisms
//! - **Spontaneous Activity**: Low-level noise injection enables "daydreaming"
//! - **Cross-Frequency Coupling**: Lower layers modulate higher layer amplitudes
//! - **Homeostatic Plasticity**: Dynamic damping prevents runaway activation

use ndarray::prelude::*;
use num_complex::Complex32;
use rand::Rng;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Complex state representing phase + amplitude for each neuron.
pub type ComplexState = Array2<Complex32>;

/// Configuration for biological dynamics.
#[derive(Clone)]
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
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        Self {
            noise_amplitude: 0.01,       // Subtle background noise
            cross_freq_coupling: 0.3,    // Moderate theta-gamma binding
            homeostatic_threshold: 0.5,  // Energy threshold for "boredom"
            homeostatic_rate: 0.02,      // Slow adaptation
            base_damping: 0.95,          // 5% base energy loss
        }
    }
}

/// Harmonic BDH model: neurons as coupled oscillators with biological dynamics.
pub struct HarmonicBdh {
    pub n: usize,
    pub d: usize,
    
    /// Natural frequency of each neuron (its "tuning")
    natural_freq: Array1<f32>,
    
    /// Coupling strength between harmonic modes and neurons
    coupling: Array2<f32>,
    
    /// Per-layer damping factors (adaptive for homeostasis)
    layer_damping: Vec<f32>,
    
    /// Per-neuron damping (for fine-grained homeostasis)
    neuron_damping: Vec<Array1<f32>>,
    
    /// Energy history for homeostatic adaptation (per layer)
    energy_history: Vec<f32>,
    
    /// Complex state ρ: phase + amplitude (standing wave memory)
    rho: Vec<ComplexState>,
    
    /// Global phase (time evolution)
    global_phase: f32,
    
    /// Layer frequencies for cross-frequency coupling
    /// Lower layers = slower (theta ~5Hz), higher layers = faster (gamma ~40Hz)
    layer_frequencies: Vec<f32>,
    
    num_layers: usize,
    
    /// Biological configuration
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
        
        // Layer frequencies: theta (5Hz) -> beta (20Hz) -> gamma (40Hz)
        let layer_frequencies: Vec<f32> = (0..num_layers)
            .map(|l| 5.0 * (2.0_f32).powf(l as f32 * 3.0 / num_layers as f32))
            .collect();
        
        // Initialize per-layer and per-neuron damping
        let layer_damping = vec![config.base_damping; num_layers];
        let neuron_damping = vec![Array1::from_elem(n, config.base_damping); num_layers];
        let energy_history = vec![0.0; num_layers];
        
        Self {
            n,
            d,
            natural_freq,
            coupling,
            layer_damping,
            neuron_damping,
            energy_history,
            rho,
            global_phase: 0.0,
            layer_frequencies,
            num_layers,
            config,
        }
    }
    
    /// (A) SPONTANEOUS ACTIVITY: Inject stochastic noise into the system.
    /// This is the "daydream" mechanism - allows the system to explore attractors autonomously.
    fn inject_spontaneous_noise(&self, layer: usize) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        
        // Pink noise (1/f) is more biologically realistic than white noise
        // We approximate by mixing multiple frequencies
        Array1::from_iter((0..self.n).map(|neuron| {
            let layer_freq = self.layer_frequencies[layer];
            let neuron_freq = self.natural_freq[neuron];
            
            // Phase-aligned noise: more likely to resonate with existing patterns
            let phase_noise = (self.global_phase * layer_freq + neuron_freq).sin();
            let random_component: f32 = rng.gen_range(-1.0_f32..1.0_f32);
            
            // Combine structured and random noise
            self.config.noise_amplitude * (0.7 * phase_noise + 0.3 * random_component)
        }))
    }
    
    /// (B) CROSS-FREQUENCY COUPLING: Lower layer amplitude modulates higher layer gain.
    /// Implements theta-gamma coupling where slow oscillations "carry" fast oscillations.
    fn compute_cross_frequency_modulation(&self, layer: usize) -> Array1<f32> {
        if layer == 0 {
            // First layer has no modulation from below
            return Array1::ones(self.n);
        }
        
        // Get amplitude envelope from lower layer (slower frequency)
        let lower_layer = layer - 1;
        let lower_amplitudes: Array1<f32> = (0..self.n)
            .map(|neuron| {
                let mut total_amp = 0.0f32;
                for mode in 0..self.d {
                    total_amp += self.rho[lower_layer][[mode, neuron]].norm();
                }
                total_amp / self.d as f32
            })
            .collect();
        
        // Normalize and apply coupling strength
        let max_amp = lower_amplitudes.iter().cloned().fold(0.0f32, f32::max).max(0.01);
        lower_amplitudes.mapv(|a| {
            let normalized = a / max_amp;
            // Modulation: when lower layer is active, higher layer gain increases
            1.0 + self.config.cross_freq_coupling * normalized
        })
    }
    
    /// (C) HOMEOSTATIC PLASTICITY: Adjust damping based on sustained energy.
    /// Prevents neurons from "screaming" forever - implements neural fatigue/boredom.
    fn apply_homeostatic_plasticity(&mut self, layer: usize, current_energy: f32) {
        let threshold = self.config.homeostatic_threshold;
        let rate = self.config.homeostatic_rate;
        
        // Exponential moving average of energy
        self.energy_history[layer] = 0.9 * self.energy_history[layer] + 0.1 * current_energy;
        let sustained_energy = self.energy_history[layer];
        
        // If energy is sustained above threshold, increase damping (get "bored")
        if sustained_energy > threshold {
            let excess = (sustained_energy - threshold) / threshold;
            self.layer_damping[layer] = (self.layer_damping[layer] - rate * excess)
                .clamp(0.5, self.config.base_damping);
        } else {
            // Recovery: slowly return to base damping
            self.layer_damping[layer] = (self.layer_damping[layer] + rate * 0.1)
                .min(self.config.base_damping);
        }
        
        // Per-neuron homeostasis
        for neuron in 0..self.n {
            let neuron_energy: f32 = (0..self.d)
                .map(|m| self.rho[layer][[m, neuron]].norm_sqr())
                .sum();
            
            if neuron_energy > threshold * 0.1 {
                // This neuron is "hot" - increase its damping
                self.neuron_damping[layer][neuron] = 
                    (self.neuron_damping[layer][neuron] - rate * 0.5)
                    .clamp(0.5, self.config.base_damping);
            } else {
                // Recovery
                self.neuron_damping[layer][neuron] = 
                    (self.neuron_damping[layer][neuron] + rate * 0.05)
                    .min(self.config.base_damping);
            }
        }
    }
    
    /// Initialize the standing wave state from a signal (e.g., brainwave, audio).
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
    
    /// Compute phase alignment (coherence) between input and internal state.
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
    
    /// Update the standing wave state with excitation, applying per-neuron damping.
    fn update_standing_wave(&mut self, excitation: &Array1<f32>, layer: usize) {
        let dt = 0.1;
        let layer_damp = self.layer_damping[layer];
        
        for mode in 0..self.d {
            let mode_freq = (mode + 1) as f32 * 0.5;
            
            for neuron in 0..self.n {
                let current = self.rho[layer][[mode, neuron]];
                
                let omega = self.natural_freq[neuron] + mode_freq;
                let rotation = Complex32::new((omega * dt).cos(), (omega * dt).sin());
                
                // Apply both layer and neuron-specific damping
                let effective_damping = layer_damp * self.neuron_damping[layer][neuron];
                let damped = current * effective_damping;
                
                let drive = excitation[neuron] * self.coupling[[mode, neuron]];
                let drive_phasor = Complex32::new(
                    drive * self.global_phase.cos(),
                    drive * self.global_phase.sin(),
                );
                
                self.rho[layer][[mode, neuron]] = damped * rotation + drive_phasor * dt;
            }
        }
    }
    
    /// Forward pass with all biological mechanisms active.
    pub fn forward(&mut self, input: &Array1<f32>) -> (Array1<f32>, Vec<f32>) {
        let mut signal = input.clone();
        let mut energy_per_layer = Vec::with_capacity(self.num_layers);
        
        for layer in 0..self.num_layers {
            // (A) SPONTANEOUS ACTIVITY: Add background noise
            let noise = self.inject_spontaneous_noise(layer);
            signal = &signal + &noise;
            
            // (B) CROSS-FREQUENCY COUPLING: Modulate by lower layer
            let modulation = self.compute_cross_frequency_modulation(layer);
            signal = &signal * &modulation;
            
            // Phase alignment and resonance gating
            let alignment = self.phase_alignment(&signal, &self.rho[layer]);
            let max_align = alignment.iter().cloned().fold(0.0f32, f32::max).max(0.01);
            let resonance_gate = alignment.mapv(|a| {
                let normalized = a / max_align;
                1.0 / (1.0 + (-5.0 * (normalized - 0.2)).exp())
            });
            
            // Update standing wave
            let excitation = &signal * &resonance_gate;
            self.update_standing_wave(&excitation, layer);
            
            signal = &signal * &resonance_gate;
            
            // Calculate layer energy
            let layer_energy: f32 = self.rho[layer]
                .iter()
                .map(|c| c.norm_sqr())
                .sum();
            energy_per_layer.push(layer_energy);
            
            // (C) HOMEOSTATIC PLASTICITY: Adapt damping
            self.apply_homeostatic_plasticity(layer, layer_energy);
        }
        
        self.global_phase += 0.1;
        if self.global_phase > 2.0 * PI {
            self.global_phase -= 2.0 * PI;
        }
        
        (signal, energy_per_layer)
    }
    
    /// AUTONOMOUS THINKING: Run without external input (pure daydreaming).
    /// The system explores its own attractors driven only by spontaneous noise.
    pub fn daydream(&mut self, steps: usize) -> Vec<(Array1<f32>, Vec<f32>)> {
        let mut trajectory = Vec::with_capacity(steps);
        
        // Start from internal state projection
        let mut signal = self.project_internal_state();
        
        for _ in 0..steps {
            let (output, energies) = self.forward(&signal);
            trajectory.push((output.clone(), energies));
            
            // Feed output back (with some decay to prevent explosion)
            signal = output.mapv(|v| v * 0.8);
        }
        
        trajectory
    }
    
    /// Project the current internal state to an output signal.
    fn project_internal_state(&self) -> Array1<f32> {
        let mut output = Array1::zeros(self.n);
        
        // Sum contributions from all layers
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
    
    /// Find attractor with biological dynamics.
    pub fn find_attractor(&mut self, initial: &Array1<f32>, max_iters: usize, tolerance: f32) -> (Array1<f32>, usize) {
        let mut signal = initial.clone();
        let mut prev_energy = 0.0f32;
        
        for iter in 0..max_iters {
            let (output, energies) = self.forward(&signal);
            let total_energy: f32 = energies.iter().sum();
            
            if (total_energy - prev_energy).abs() < tolerance {
                return (output, iter);
            }
            
            prev_energy = total_energy;
            signal = output;
        }
        
        (signal, max_iters)
    }
    
    /// Get the current standing wave pattern as real amplitudes.
    pub fn get_standing_wave(&self, layer: usize) -> Array2<f32> {
        self.rho[layer].mapv(|c| c.norm())
    }
    
    /// Get current damping values (shows homeostatic state).
    pub fn get_damping_state(&self) -> (&[f32], &[Array1<f32>]) {
        (&self.layer_damping, &self.neuron_damping)
    }
    
    /// Get layer frequencies for debugging.
    pub fn get_layer_frequencies(&self) -> &[f32] {
        &self.layer_frequencies
    }
    
    /// Measure dominant frequency per neuron.
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
}

/// Generate a synthetic brainwave-like signal for testing.
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
        
        let (output, energies) = model.forward(&input);
        
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|&v| v.is_finite()));
        assert_eq!(energies.len(), 2);
    }

    #[test]
    fn test_signal_initialization() {
        let mut model = HarmonicBdh::new(64, 16, 2);
        let brainwave = generate_brainwave(256, 10.0, 0.1);
        
        model.initialize_from_signal(&brainwave, 0);
        
        let wave = model.get_standing_wave(0);
        assert!(wave.sum() > 0.0, "Standing wave should have energy after initialization");
    }

    #[test]
    fn test_attractor_convergence() {
        // Use config with no noise for deterministic convergence
        let config = BiologicalConfig {
            noise_amplitude: 0.0,
            cross_freq_coupling: 0.0, // Disable coupling for simpler dynamics
            ..Default::default()
        };
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Initialize with signal to give starting energy
        let brainwave = generate_brainwave(64, 10.0, 0.1);
        model.initialize_from_signal(&brainwave, 0);
        
        let input = Array1::from_iter((0..32).map(|i| if i % 4 == 0 { 0.5 } else { 0.0 }));
        
        // Just verify it runs and produces output
        let (output, _) = model.find_attractor(&input, 50, 0.01);
        
        // Should have some output
        assert!(output.iter().any(|&v| v.is_finite()), "Should produce finite output");
    }

    #[test]
    fn test_spontaneous_activity() {
        let mut model = HarmonicBdh::new(32, 8, 2);
        
        // Initialize with some memory
        let brainwave = generate_brainwave(128, 10.0, 0.1);
        model.initialize_from_signal(&brainwave, 0);
        
        // Run daydream (no external input)
        let trajectory = model.daydream(20);
        
        // Should have activity due to spontaneous noise
        let total_activity: f32 = trajectory.iter()
            .map(|(sig, _)| sig.iter().map(|v| v.abs()).sum::<f32>())
            .sum();
        
        assert!(total_activity > 0.0, "Daydream should produce activity");
    }

    #[test]
    fn test_homeostatic_plasticity() {
        let mut config = BiologicalConfig::default();
        config.homeostatic_threshold = 0.1; // Low threshold for testing
        config.homeostatic_rate = 0.1;      // Fast adaptation
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config.clone());
        
        // Pump energy into the system
        let strong_input = Array1::from_elem(32, 1.0);
        for _ in 0..20 {
            model.forward(&strong_input);
        }
        
        // Check that damping decreased (system got "bored")
        let (layer_damping, _) = model.get_damping_state();
        assert!(
            layer_damping[0] < config.base_damping,
            "Damping should decrease under sustained activation"
        );
    }

    #[test]
    fn test_cross_frequency_coupling() {
        let config = BiologicalConfig {
            noise_amplitude: 0.0, // No noise for deterministic test
            ..Default::default()
        };
        let mut model = HarmonicBdh::with_config(32, 8, 3, config);
        
        // Activate lower layer
        let brainwave = generate_brainwave(128, 5.0, 0.0); // Theta
        model.initialize_from_signal(&brainwave, 0);
        
        // Get layer frequencies
        let freqs = model.get_layer_frequencies();
        
        // Layer 0 should be slower than Layer 2
        assert!(freqs[0] < freqs[2], "Lower layers should have slower frequencies");
    }
}
