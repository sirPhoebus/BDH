//! Harmonic BDH: A vibrational/resonance-based neural architecture.
//!
//! Instead of static weights, this treats the network as a system of coupled oscillators
//! where information propagates via phase alignment and constructive interference.

use ndarray::prelude::*;
use num_complex::Complex32;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Complex state representing phase + amplitude for each neuron.
pub type ComplexState = Array2<Complex32>;

/// Harmonic BDH model: neurons as coupled oscillators.
pub struct HarmonicBdh {
    pub n: usize,  // Number of neurons
    pub d: usize,  // Low-rank dimension (harmonic modes)
    
    /// Natural frequency of each neuron (its "tuning")
    natural_freq: Array1<f32>,
    
    /// Coupling strength between harmonic modes and neurons
    coupling: Array2<f32>,
    
    /// Damping factor (how fast vibrations decay)
    damping: f32,
    
    /// Complex state ρ: phase + amplitude (standing wave memory)
    /// Shape: (num_layers, d, n) conceptually, stored as Vec of (d, n)
    rho: Vec<ComplexState>,
    
    /// Global phase (time evolution)
    global_phase: f32,
    
    num_layers: usize,
}

impl HarmonicBdh {
    pub fn new(n: usize, d: usize, num_layers: usize) -> Self {
        // Natural frequencies: spread across harmonic series
        // Each neuron "resonates" at a different frequency
        let natural_freq = Array1::from_iter(
            (0..n).map(|i| 0.1 + (i as f32 / n as f32) * 2.0 * PI)
        );
        
        // Coupling matrix: how strongly each mode affects each neuron
        // Initialized as harmonic basis functions
        let mut coupling = Array2::zeros((d, n));
        for mode in 0..d {
            for neuron in 0..n {
                // Each mode is a different harmonic frequency
                let harmonic = (mode + 1) as f32;
                let phase = (neuron as f32 / n as f32) * 2.0 * PI * harmonic;
                coupling[[mode, neuron]] = phase.cos() / (d as f32).sqrt();
            }
        }
        
        // Initialize rho as complex zeros (no initial vibration)
        let rho = vec![Array2::from_elem((d, n), Complex32::new(0.0, 0.0)); num_layers];
        
        Self {
            n,
            d,
            natural_freq,
            coupling,
            damping: 0.95,  // 5% energy loss per step
            rho,
            global_phase: 0.0,
            num_layers,
        }
    }
    
    /// Initialize the standing wave state from a signal (e.g., brainwave, audio).
    /// This is the "Brain Snapshot" - imprinting a pattern into the resonant memory.
    pub fn initialize_from_signal(&mut self, signal: &[f32], layer: usize) {
        let signal_len = signal.len();
        
        // Perform FFT to extract frequency components
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(signal_len);
        
        let mut buffer: Vec<Complex<f32>> = signal
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();
        fft.process(&mut buffer);
        
        // Map FFT bins to our harmonic modes and neurons
        for mode in 0..self.d.min(signal_len / 2) {
            let bin = &buffer[mode + 1]; // Skip DC component
            let amplitude = (bin.re * bin.re + bin.im * bin.im).sqrt() / signal_len as f32;
            let phase = bin.im.atan2(bin.re);
            
            // Distribute this harmonic mode across neurons based on coupling
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
    /// Returns a value in [0, 1] where 1 = perfect resonance.
    fn phase_alignment(&self, input: &Array1<f32>, state: &ComplexState) -> Array1<f32> {
        let mut alignment = Array1::zeros(self.n);
        
        for neuron in 0..self.n {
            // Convert input to phase based on neuron's natural frequency
            let input_phase = input[neuron] * self.natural_freq[neuron] + self.global_phase;
            let input_phasor = Complex32::new(input_phase.cos(), input_phase.sin());
            
            // Sum contributions from all modes
            let mut total_coherence = Complex32::new(0.0, 0.0);
            for mode in 0..self.d {
                total_coherence += state[[mode, neuron]] * self.coupling[[mode, neuron]];
            }
            
            // Coherence = how aligned the phases are
            let dot = input_phasor.re * total_coherence.re + input_phasor.im * total_coherence.im;
            alignment[neuron] = dot.max(0.0); // Only constructive interference passes
        }
        
        alignment
    }
    
    /// Update the standing wave state with a new excitation.
    fn update_standing_wave(&mut self, excitation: &Array1<f32>, layer: usize) {
        let dt = 0.1; // Time step
        
        for mode in 0..self.d {
            let mode_freq = (mode + 1) as f32 * 0.5; // Harmonic series
            
            for neuron in 0..self.n {
                let current = self.rho[layer][[mode, neuron]];
                
                // Natural oscillation: rotate by natural frequency
                let omega = self.natural_freq[neuron] + mode_freq;
                let rotation = Complex32::new((omega * dt).cos(), (omega * dt).sin());
                
                // Damping
                let damped = current * self.damping;
                
                // Excitation: input drives the oscillator
                let drive = excitation[neuron] * self.coupling[[mode, neuron]];
                let drive_phasor = Complex32::new(
                    drive * self.global_phase.cos(),
                    drive * self.global_phase.sin(),
                );
                
                // Interference: standing wave = damped rotation + new excitation
                self.rho[layer][[mode, neuron]] = damped * rotation + drive_phasor * dt;
            }
        }
    }
    
    /// Forward pass through harmonic layers.
    /// Returns (resonance_output, energy_landscape)
    pub fn forward(&mut self, input: &Array1<f32>) -> (Array1<f32>, Vec<f32>) {
        let mut signal = input.clone();
        let mut energy_per_layer = Vec::with_capacity(self.num_layers);
        
        for layer in 0..self.num_layers {
            // 1. Phase Alignment: measure resonance with internal state
            let alignment = self.phase_alignment(&signal, &self.rho[layer]);
            
            // 2. Resonance Gate: only aligned (constructive interference) signals pass
            let max_align = alignment.iter().cloned().fold(0.0f32, f32::max).max(0.01);
            let resonance_gate = alignment.mapv(|a| {
                // Soft gate based on relative alignment strength
                let normalized = a / max_align;
                1.0 / (1.0 + (-5.0 * (normalized - 0.2)).exp()) // Sigmoid, more permissive
            });
            
            // 3. Update Standing Wave with the gated signal
            let excitation = &signal * &resonance_gate;
            self.update_standing_wave(&excitation, layer);
            
            // 4. Output: resonant signal
            signal = &signal * &resonance_gate;
            
            // Track energy (sum of squared amplitudes)
            let layer_energy: f32 = self.rho[layer]
                .iter()
                .map(|c| c.norm_sqr())
                .sum();
            energy_per_layer.push(layer_energy);
        }
        
        // Advance global phase
        self.global_phase += 0.1;
        if self.global_phase > 2.0 * PI {
            self.global_phase -= 2.0 * PI;
        }
        
        (signal, energy_per_layer)
    }
    
    /// Run the attractor loop: feed output back as input until convergence.
    /// This discovers the stable "limit cycle" (standing wave pattern).
    pub fn find_attractor(&mut self, initial: &Array1<f32>, max_iters: usize, tolerance: f32) -> (Array1<f32>, usize) {
        let mut signal = initial.clone();
        let mut prev_energy = 0.0f32;
        
        for iter in 0..max_iters {
            let (output, energies) = self.forward(&signal);
            let total_energy: f32 = energies.iter().sum();
            
            // Check for convergence (stable energy = limit cycle reached)
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
    
    /// Measure the dominant frequency in each neuron's oscillation.
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
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            // Alpha wave (~10 Hz) + Theta wave (~6 Hz) + noise
            let alpha = (2.0 * PI * dominant_freq * t).sin();
            let theta = 0.5 * (2.0 * PI * (dominant_freq * 0.6) * t).sin();
            let noise = rng.gen_range(-noise_level..noise_level);
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
        assert!(output.iter().all(|&v| v >= 0.0));
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
        let mut model = HarmonicBdh::new(32, 8, 2);
        let input = Array1::from_iter((0..32).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }));
        
        let (final_state, iters) = model.find_attractor(&input, 100, 0.001);
        
        println!("Converged in {} iterations", iters);
        assert!(iters < 100, "Should converge before max iterations");
        assert!(final_state.iter().any(|&v| v > 0.0), "Should have some active neurons");
    }
}
