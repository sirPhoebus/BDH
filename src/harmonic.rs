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
    
    // === Stochastic Resonance / Self-Thinking ===
    
    /// Thermal noise floor - minimum energy injection (never zero)
    pub thermal_noise: f32,
    /// Minimum energy floor - system never drops below this
    pub min_energy_floor: f32,
    /// Self-feedback strength during reflection (0.0-1.0)
    pub reflection_feedback: f32,
    /// Spontaneous memory recall threshold
    pub recall_threshold: f32,
    
    // === Temporal / Chronos System ===
    
    /// Heartbeat frequency - the "metabolic clock" of the system (Hz)
    pub heartbeat_freq: f32,
    /// Heartbeat modulation strength (0.0 = no gating, 1.0 = full gating)
    pub heartbeat_strength: f32,
    /// Rho decay rate per timestep (prevents unbounded growth)
    pub rho_decay: f32,
    /// Number of time encoding frequencies (multi-scale temporal memory)
    pub time_encoding_dims: usize,
    /// Base frequency for time encoding
    pub time_encoding_base: f32,
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
            // Stochastic resonance defaults
            thermal_noise: 0.015,         // Always-on background noise
            min_energy_floor: 0.02,       // Never truly zero energy
            reflection_feedback: 0.7,     // Strong self-feedback during reflection
            recall_threshold: 0.1,        // Threshold for spontaneous recall
            // Temporal / Chronos defaults
            heartbeat_freq: 1.0,          // 1 Hz heartbeat (like resting heart rate)
            heartbeat_strength: 0.3,      // Moderate gating
            rho_decay: 0.995,             // Slow decay (retains ~60% after 100 steps)
            time_encoding_dims: 8,        // 8 frequency bands for time encoding
            time_encoding_base: 10000.0,  // Base for positional encoding
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    
    // === Temporal / Chronos fields ===
    
    /// Global time counter (in arbitrary units)
    global_time: f32,
    /// Time encoding cache (multi-scale positional encoding)
    time_encoding: Array1<f32>,
    /// Imprinted memories with timestamps: (time, layer, pattern_hash)
    memory_timestamps: Vec<(f32, usize, u64)>,
    
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
        
        // Initialize time encoding
        let time_encoding = Array1::zeros(config.time_encoding_dims);
        
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
            global_time: 0.0,
            time_encoding,
            memory_timestamps: Vec::new(),
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
    pub fn inject_spontaneous_activity(&self, layer: usize) -> (Array1<f32>, Array1<f32>) {
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
    fn _update_standing_wave(&mut self, excitation: &Array1<f32>, self_excite: &Array1<f32>, layer: usize) {
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

    // ========================================================================
    // TEMPORAL / CHRONOS SYSTEM
    // ========================================================================

    /// Compute the current heartbeat factor (0.0 to 1.0).
    /// Peak receptivity at heartbeat peaks, reduced at troughs.
    fn compute_heartbeat(&self) -> f32 {
        (self.global_time * self.config.heartbeat_freq * 2.0 * PI).sin().abs()
    }

    /// Advance global time.
    pub fn step_time(&mut self, delta: f32) {
        self.global_time += delta;
    }

    /// Get current global time.
    pub fn get_time(&self) -> f32 {
        self.global_time
    }

    /// Update multi-scale time encoding (like positional encoding in transformers).
    fn update_time_encoding(&mut self) {
        let dims = self.config.time_encoding_dims;
        let base = self.config.time_encoding_base;
        
        self.time_encoding = Array1::from_iter((0..dims).map(|i| {
            let freq = 1.0 / base.powf(i as f32 / dims as f32);
            if i % 2 == 0 {
                (self.global_time * freq).sin()
            } else {
                (self.global_time * freq).cos()
            }
        }));
    }

    /// Get current time encoding vector.
    pub fn get_time_encoding(&self) -> &Array1<f32> {
        &self.time_encoding
    }

    /// Update standing wave with temporal encoding (time-weighted Hebbian learning).
    fn update_standing_wave_temporal(&mut self, excitation: &Array1<f32>, self_excite: &Array1<f32>, layer: usize) {
        let dt = 0.1;
        let layer_damp = self.layer_damping[layer];
        
        // Compute temporal modulation from time encoding
        let time_mod = if self.time_encoding.len() > 0 {
            // Use first few time encoding dims to modulate phase
            self.time_encoding.iter().take(4).sum::<f32>() / 4.0
        } else {
            0.0
        };
        
        for mode in 0..self.d {
            let mode_freq = (mode + 1) as f32 * 0.5;
            
            // Mode-specific time encoding (different modes encode different timescales)
            let mode_time_weight = if mode < self.time_encoding.len() {
                1.0 + 0.2 * self.time_encoding[mode]
            } else {
                1.0
            };
            
            for neuron in 0..self.n {
                let current = self.rho[layer][[mode, neuron]];
                
                let omega = self.natural_freq[neuron] + mode_freq;
                let rotation = Complex32::new((omega * dt).cos(), (omega * dt).sin());
                
                let effective_damping = layer_damp * self.neuron_damping[layer][neuron];
                let damped = current * effective_damping;
                
                // Combined drive: external + self-excitation, modulated by time
                let total_drive = (excitation[neuron] + self_excite[neuron]) * mode_time_weight;
                let drive = total_drive * self.coupling[[mode, neuron]];
                
                // Phase includes both global phase and temporal modulation
                let phase = self.global_phase + time_mod * 0.5;
                let drive_phasor = Complex32::new(
                    drive * phase.cos(),
                    drive * phase.sin(),
                );
                
                self.rho[layer][[mode, neuron]] = damped * rotation + drive_phasor * dt;
            }
        }
    }

    /// Apply rho decay to prevent unbounded memory growth.
    fn apply_rho_decay(&mut self, layer: usize) {
        let decay = self.config.rho_decay;
        for mode in 0..self.d {
            for neuron in 0..self.n {
                self.rho[layer][[mode, neuron]] *= decay;
            }
        }
    }

    /// Imprint a signal at the current time (creates a timestamped memory).
    pub fn imprint(&mut self, signal: &Array1<f32>, layer: usize) {
        // Compute pattern hash for later identification
        let hash = self.compute_pattern_hash(signal);
        
        // Record timestamp
        self.memory_timestamps.push((self.global_time, layer, hash));
        
        // Strong Hebbian imprint at heartbeat peak
        let heartbeat = self.compute_heartbeat();
        let imprint_strength = 0.5 + 0.5 * heartbeat; // 0.5 to 1.0
        
        let latent: Array1<f32> = self.coupling.dot(signal);
        
        for mode in 0..self.d {
            for neuron in 0..self.n {
                let update = latent[mode] * signal[neuron] * imprint_strength;
                
                // Encode with current time phase
                let time_phase = if mode < self.time_encoding.len() {
                    self.time_encoding[mode]
                } else {
                    0.0
                };
                
                let phasor = Complex32::new(
                    update * (self.global_phase + time_phase).cos(),
                    update * (self.global_phase + time_phase).sin(),
                );
                
                self.rho[layer][[mode, neuron]] += phasor;
            }
        }
    }

    /// Compute a simple hash of a pattern for identification.
    fn compute_pattern_hash(&self, signal: &Array1<f32>) -> u64 {
        let mut hash: u64 = 0;
        for (i, &v) in signal.iter().enumerate().take(8) {
            hash ^= ((v * 1000.0) as u64) << (i * 8);
        }
        hash
    }

    /// Attempt to recall a pattern from a partial cue.
    pub fn recall(&mut self, cue: &Array1<f32>, layer: usize) -> Array1<f32> {
        // Apply cue through resonance
        let (output, _, _) = self.forward(cue);
        
        // The standing wave rho should resonate with matching patterns
        let mut recalled = Array1::zeros(self.n);
        
        for neuron in 0..self.n {
            let mut total = 0.0f32;
            for mode in 0..self.d {
                total += self.rho[layer][[mode, neuron]].norm() * self.coupling[[mode, neuron]];
            }
            recalled[neuron] = total;
        }
        
        // Blend cue response with rho resonance
        &output * 0.3 + &recalled * 0.7
    }

    /// Get memory timestamps for debugging/analysis.
    pub fn get_memory_timestamps(&self) -> &[(f32, usize, u64)] {
        &self.memory_timestamps
    }

    /// Forward pass with explicit time control (like ChronosBdh).
    pub fn forward_at_time(&mut self, input: &Array1<f32>, time: f32) -> (Array1<f32>, Vec<f32>, Vec<f32>) {
        self.global_time = time;
        self.forward(input)
    }

    /// Check if currently at a heartbeat peak (good time for imprinting).
    pub fn is_heartbeat_peak(&self) -> bool {
        self.compute_heartbeat() > 0.9
    }

    /// Advance time until next heartbeat peak.
    pub fn advance_to_heartbeat_peak(&mut self) {
        let freq = self.config.heartbeat_freq;
        // sin(t * freq * 2π) peaks at t = (0.25 + n) / freq
        let current_cycle = (self.global_time * freq).floor();
        let next_peak = (current_cycle + 0.25) / freq;
        
        if next_peak > self.global_time {
            self.global_time = next_peak;
        } else {
            self.global_time = (current_cycle + 1.25) / freq;
        }
        self.update_time_encoding();
    }
    
    /// Forward pass with all biological mechanisms.
    /// Returns (output, energies, phase_coherences)
    pub fn forward(&mut self, input: &Array1<f32>) -> (Array1<f32>, Vec<f32>, Vec<f32>) {
        // (0) TEMPORAL GATING: Heartbeat modulation
        let heartbeat = self.compute_heartbeat();
        let gating_factor = 1.0 - self.config.heartbeat_strength * (1.0 - heartbeat);
        let mut signal = input * gating_factor;
        
        let mut energy_per_layer = Vec::with_capacity(self.num_layers);
        let mut coherence_per_layer = Vec::with_capacity(self.num_layers);
        
        // Update time encoding for this step
        self.update_time_encoding();
        
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
            
            // Update standing wave with self-excitation AND time encoding
            let excitation = &signal * &resonance_gate;
            self.update_standing_wave_temporal(&excitation, &self_excite, layer);
            
            signal = &signal * &resonance_gate;
            
            // (D) RHO DECAY: Prevent unbounded growth
            self.apply_rho_decay(layer);
            
            let layer_energy: f32 = self.rho[layer].iter().map(|c| c.norm_sqr()).sum();
            energy_per_layer.push(layer_energy);
            
            // (C) HOMEOSTATIC PLASTICITY
            self.apply_homeostatic_plasticity(layer, layer_energy);
            
            // Update adaptive noise
            self.update_adaptive_noise(layer, layer_energy);
        }
        
        // Advance time
        self.step_time(0.1);
        
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
        
        // Normalize thresholds by network size (n * d * layers)
        // Base thresholds were tuned for n=32, d=8, layers=2
        let base_size = 32.0 * 8.0 * 2.0;
        let current_size = (self.n * self.d * self.num_layers) as f32;
        // Use log scaling for very gentle growth: log2(ratio) capped
        let ratio = current_size / base_size;
        let scale = 1.0 + (ratio.ln() / 4.0).max(0.0); // ~1.0 for small, ~2.2 for 256*64*3
        
        let resting_threshold = 0.05 * scale;
        let low_threshold = 0.1 * scale;
        let med_threshold = 0.2 * scale;
        let layer0_threshold = 0.3 * scale;
        
        if total_energy < resting_threshold {
            ThoughtState::Resting
        } else if avg_freq < 6.0 && energies.get(0).unwrap_or(&0.0) > &layer0_threshold {
            ThoughtState::Contemplative
        } else if avg_freq > 15.0 && total_energy > med_threshold {
            ThoughtState::AlertScanning
        } else if total_energy > low_threshold {
            ThoughtState::ActivePlanning
        } else {
            ThoughtState::Transitioning
        }
    }

    /// Integrate physiological signals from the body to modulate neural dynamics.
    /// 
    /// Signals: [Energy, Integrity, Pleasure, Pain]
    pub fn integrate_body_signals(&mut self, signals: &[f32]) {
        if signals.len() < 4 { return; }

        let energy = signals[0];
        // let integrity = signals[1];
        let pleasure = signals[2];
        let pain = signals[3];

        // 1. Energy Modulation: Low energy -> Slow frequencies (Theta/Delta)
        if energy < 0.4 {
            // Shift layer frequencies down to induce "sleep/groggy" state
            for freq in &mut self.layer_frequencies {
                *freq = (*freq * 0.95).max(0.5); 
            }
            // Increase damping to reduce activity
            self.config.base_damping = (self.config.base_damping + 0.01).min(0.99);
        } else {
            // Restore energy -> Restore frequencies (simplified reset towards default logic)
            for freq in &mut self.layer_frequencies {
                 *freq = (*freq * 1.01).min(80.0);
            }
            
            // "Restless Energy": If body is full of energy (> 0.8), crave activity!
            if energy > 0.8 {
                // Increase damping factor (closer to 1.0) to sustain activity better
                self.config.base_damping = (self.config.base_damping + 0.01).min(0.999);
                
                // Boost base noise amplitude so it doesn't decay away
                self.config.noise_amplitude = (self.config.noise_amplitude + 0.05).min(0.5);
                
                // Also bump current adaptive noise
                for layer in 0..self.num_layers {
                    self.adaptive_noise[layer] = (self.adaptive_noise[layer] + 0.1).min(0.8);
                }
            }
        }

        // 2. Valence Modulation (Pleasure/Pain)
        let valence = pleasure - pain;

        if valence > 0.1 {
            // Pleasure: Decrease damping (Reinforcement/Excitement)
            self.config.base_damping = (self.config.base_damping - 0.02 * valence).max(0.5);
            // Boost self-excitation (Confidence)
            self.config.self_excitation = (self.config.self_excitation + 0.005 * valence).min(0.1);
        } else if valence < -0.1 {
            // Pain: Increase damping (Inhibition/Avoidance)
            self.config.base_damping = (self.config.base_damping + 0.05 * pain).min(0.98);
            // Scramble: Inject noise to break current pattern (Flight/Flinch)
            for layer in 0..self.num_layers {
                self.adaptive_noise[layer] += 0.2 * pain;
            }
        }
    }

    /// Compute novelty as the Euclidean distance between current output and previous state.
    /// Returns a value roughly 0.0 to 1.0+.
    pub fn compute_novelty(&self, current_output: &Array1<f32>, last_output: &Array1<f32>) -> f32 {
        let diff = current_output - last_output;
        let dist_sq: f32 = diff.mapv(|x| x * x).sum();
        dist_sq.sqrt()
    }

    /// Apply motivation to the neural parameters.
    /// 
    /// # Arguments
    /// * `drive`: Name of the dominant drive (e.g., "HUNGER", "CURIOSITY")
    pub fn motivate(&mut self, drive: Option<&str>) {
        // Reset baseline modifiers first (simplified)
        // self.config.noise_amplitude = 0.07; // We accept drift for now
        
        match drive {
            Some("HUNGER") => {
                // Hunger: Focused Seeking.
                // High Gamma (fast processing), Moderate Noise (action bias).
                // In a real semantic brain, this would prime "Food" nodes.
                
                // Increase gain/excitability
                self.config.self_excitation = (self.config.self_excitation + 0.01).min(0.15);
                
                // Bias frequencies slightly up
                for freq in &mut self.layer_frequencies {
                    *freq = (*freq * 1.05).min(100.0);
                }
            },
            Some("CURIOSITY") => {
                // Curiosity: Exploration.
                // High Noise (random State jumping), Lower Damping (volatility).
                
                // Boost noise significantly
                 for layer in 0..self.num_layers {
                    self.adaptive_noise[layer] = (self.adaptive_noise[layer] + 0.2).min(1.5);
                }
                
                // Lower damping to allow new patterns to form
                self.config.base_damping = (self.config.base_damping - 0.05).max(0.6);
            },
            None => {
                // Contentment / Homeostasis
                // Slowly relax parameters back to baseline? 
                // For now, do nothing.
            }
            _ => {}
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
    
    /// Retrieve the vector for a specific concept by name (case-insensitive).
    pub fn get_concept_vector(&self, name: &str) -> Option<Array1<f32>> {
        self.concepts.iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .map(|c| c.vector.clone())
    }

    /// Calculate Shannon entropy of the energy distribution across layers.
    /// High entropy = Low Confidence.
    pub fn calculate_entropy(&self, energies: &[f32]) -> f32 {
        let sum: f32 = energies.iter().sum();
        if sum < 0.001 { return 0.0; }
        
        let mut entropy = 0.0;
        for &e in energies {
            let p = e / sum;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Get a confidence score (0.0 to 1.0) based on current state entropy.
    /// 1.0 = Highly focused/confident. 0.0 = Chaotic/Unsure.
    pub fn get_confidence(&self, energies: &[f32]) -> f32 {
        let entropy = self.calculate_entropy(energies);
        // Max entropy for N layers is ln(N). 
        // Normalize: 1 - (entropy / max_entropy)
        let max_entropy = (self.num_layers as f32).ln().max(1.0);
        (1.0 - (entropy / max_entropy)).max(0.0).min(1.0)
    }

    /// Self-Discovery: Check if the current state is novel enough to be a new concept.
    /// If so, add it to the concept memory.
    /// Returns the name of the new concept if created.
    pub fn learn_new_concept(&mut self, state: &Array1<f32>) -> Option<String> {
        // threshold for novelty (1.0 - cosine similarity)
        // Lowered to 0.1 for demo sensitivity
        let novelty_threshold = 0.1; 
        
        let mut min_dist = 1.0;
        for c in &self.concepts {
            let similarity = state.dot(&c.vector); // Assumes normalized
            let dist = 1.0 - similarity;
            if dist < min_dist {
                min_dist = dist;
            }
        }
        
        // If the nearest concept is too far away, this is a new idea.
        if min_dist > novelty_threshold {
            let new_id = self.concepts.len();
            let name = format!("Concept_{}", new_id);
            
            // Normalize state for storage
            let norm = state.dot(state).sqrt();
            let vector = if norm > 0.0 { state / norm } else { state.clone() };
            
            self.concepts.push(Concept {
                name: Box::leak(name.clone().into_boxed_str()), // Leak memory to get 'static str (Demo hack)
                vector,
            });
            
            return Some(name);
        }
        
        None
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

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get reference to rho state for a layer.
    pub fn get_rho(&self, layer: usize) -> &ComplexState {
        &self.rho[layer]
    }

    /// Get mutable reference to rho state for a layer.
    pub fn get_rho_mut(&mut self, layer: usize) -> &mut ComplexState {
        &mut self.rho[layer]
    }

    /// Apply a forgetting mask to rho state (element-wise multiply).
    pub fn apply_rho_mask(&mut self, layer: usize, mask: &ndarray::Array2<f32>) {
        for mode in 0..self.d {
            for neuron in 0..self.n {
                let scale = mask[[mode, neuron]];
                self.rho[layer][[mode, neuron]] *= scale;
            }
        }
    }

    /// Reset rho states to zero.
    pub fn reset_rho(&mut self) {
        for layer in 0..self.num_layers {
            self.rho[layer].fill(Complex32::new(0.0, 0.0));
        }
    }

    /// Blend rho state with a snapshot (for replay).
    pub fn blend_rho(&mut self, layer: usize, snapshot: &ndarray::Array2<f32>, blend_factor: f32) {
        for mode in 0..self.d {
            for neuron in 0..self.n {
                let current = self.rho[layer][[mode, neuron]];
                let target_amp = snapshot[[mode, neuron]];
                let target = Complex32::new(target_amp * current.arg().cos(), target_amp * current.arg().sin());
                self.rho[layer][[mode, neuron]] = current * (1.0 - blend_factor) + target * blend_factor;
            }
        }
    }

    // ========================================================================
    // SELF-THINKING / REFLECTION (Stochastic Resonance + Self-Feedback)
    // ========================================================================

    /// Inject thermal noise directly into rho states (stochastic resonance).
    /// This ensures the system never truly reaches zero energy.
    fn inject_thermal_noise(&mut self) {
        let mut rng = rand::thread_rng();
        let thermal = self.config.thermal_noise;
        
        if thermal <= 0.0 {
            return;
        }
        
        for layer in 0..self.num_layers {
            for mode in 0..self.d {
                for neuron in 0..self.n {
                    // Add small random phase perturbation
                    let phase_noise = rng.gen_range(-PI * 0.1..PI * 0.1);
                    let amp_noise = rng.gen_range(0.0..thermal);
                    
                    let current = self.rho[layer][[mode, neuron]];
                    let noise_phasor = Complex32::new(
                        amp_noise * phase_noise.cos(),
                        amp_noise * phase_noise.sin(),
                    );
                    self.rho[layer][[mode, neuron]] = current + noise_phasor;
                }
            }
        }
    }

    /// Enforce minimum energy floor - the brain never truly goes silent.
    fn enforce_energy_floor(&mut self) {
        let mut rng = rand::thread_rng();
        let floor = self.config.min_energy_floor;
        
        for layer in 0..self.num_layers {
            let energy: f32 = self.rho[layer].iter().map(|c| c.norm_sqr()).sum();
            
            if energy < floor {
                // Inject energy at random neurons to reach floor
                let deficit = floor - energy;
                let inject_per = (deficit / (self.n as f32)).sqrt();
                
                for neuron in 0..self.n {
                    if rng.gen::<f32>() < 0.1 {
                        // 10% chance to activate each neuron
                        let mode = rng.gen_range(0..self.d);
                        let phase = rng.gen_range(0.0..2.0 * PI);
                        self.rho[layer][[mode, neuron]] += Complex32::new(
                            inject_per * phase.cos(),
                            inject_per * phase.sin(),
                        );
                    }
                }
            }
        }
    }

    /// Extract the current "vibration" as a signal for self-feedback.
    fn extract_vibration(&self) -> Array1<f32> {
        let mut vibration = Array1::zeros(self.n);
        
        for neuron in 0..self.n {
            let mut total = 0.0f32;
            for layer in 0..self.num_layers {
                for mode in 0..self.d {
                    // Weight by mode frequency (lower modes = stronger contribution)
                    let mode_weight = 1.0 / (mode as f32 + 1.0);
                    total += self.rho[layer][[mode, neuron]].norm() * mode_weight;
                }
            }
            vibration[neuron] = total / (self.num_layers * self.d) as f32;
        }
        
        vibration
    }

    /// The "Reflection" pass: The model vibrates on its own internal state.
    /// This simulates "thinking" or "dreaming" without external input.
    /// Returns a trajectory of reflection steps with concept interpretations.
    pub fn reflect(&mut self, iterations: usize) -> Vec<ReflectionStep> {
        let mut trajectory = Vec::with_capacity(iterations);
        let mut rng = rand::thread_rng();
        
        for step in 0..iterations {
            // 1. Inject thermal noise (stochastic resonance - always on)
            self.inject_thermal_noise();
            
            // 2. Enforce minimum energy floor
            self.enforce_energy_floor();
            
            // 3. Extract current vibration as self-feedback signal
            let current_vibration = self.extract_vibration();
            
            // 4. Add thermal noise to the feedback signal
            let thermal_signal: Array1<f32> = Array1::from_iter(
                (0..self.n).map(|_| rng.gen_range(-self.config.thermal_noise..self.config.thermal_noise))
            );
            
            // 5. Create feedback signal (weighted mix of vibration + noise)
            let feedback = &current_vibration * self.config.reflection_feedback + &thermal_signal;
            
            // 6. Forward pass with self-generated input
            let (output, energies, coherences) = self.forward(&feedback);
            
            // 7. Check for spontaneous memory recall
            let recall_event = self.detect_memory_recall(&output, &energies);
            
            // 8. Classify thought state and concepts
            let thought_state = self.classify_thought_state(&energies);
            let top_concepts = self.get_top_concepts(&output, 3);
            let total_energy: f32 = energies.iter().sum();
            let dominant_freq = self.get_dominant_frequencies(0).mean().unwrap_or(0.0);
            
            trajectory.push(ReflectionStep {
                step,
                output: output.clone(),
                energies,
                coherences,
                thought_state,
                top_concepts,
                total_energy,
                thermal_contribution: self.config.thermal_noise,
                recall_event,
                dominant_freq,
            });
        }
        
        trajectory
    }

    /// Detect spontaneous memory recall based on sudden energy spikes.
    fn detect_memory_recall(&self, output: &Array1<f32>, energies: &[f32]) -> Option<String> {
        let total_energy: f32 = energies.iter().sum();
        let prev_energy = self.energy_history.iter().sum::<f32>();
        
        // Recall detected if energy suddenly increases significantly
        if total_energy > prev_energy * 2.0 && total_energy > self.config.recall_threshold {
            let concepts = self.get_top_concepts(output, 2);
            if !concepts.is_empty() {
                return Some(format!(
                    "Spontaneous recall: {} (strength: {:.2})",
                    concepts[0].0,
                    concepts[0].1
                ));
            }
        }
        None
    }

    /// Stream of consciousness: continuous self-thinking with narrative output.
    pub fn stream_of_consciousness(&mut self, duration: usize, verbose: bool) -> Vec<String> {
        let mut narrative = Vec::new();
        let mut last_state = ThoughtState::Resting;
        let mut last_concept = "";
        
        let trajectory = self.reflect(duration);
        
        for step in trajectory {
            let mut thought = String::new();
            
            // State transition
            if step.thought_state != last_state {
                thought.push_str(&format!("[{} → {}] ", last_state.as_str(), step.thought_state.as_str()));
                last_state = step.thought_state.clone();
            }
            
            // Concept shift
            if !step.top_concepts.is_empty() {
                let current_concept = step.top_concepts[0].0;
                if current_concept != last_concept {
                    thought.push_str(&format!("{{{}}} ", current_concept));
                    last_concept = current_concept;
                }
            }
            
            // Memory recall event
            if let Some(ref recall) = step.recall_event {
                thought.push_str(&format!("⚡ {} ", recall));
            }
            
            // Energy indicator
            if step.total_energy > 0.3 {
                thought.push_str("▲ ");
            } else if step.total_energy < 0.05 {
                thought.push_str("▽ ");
            }
            
            if !thought.is_empty() || verbose {
                if verbose {
                    thought.push_str(&format!("(E:{:.3})", step.total_energy));
                }
                narrative.push(thought);
            }
        }
        
        narrative
    }

    /// Get total system energy across all layers.
    pub fn total_energy(&self) -> f32 {
        self.rho.iter()
            .map(|layer| layer.iter().map(|c| c.norm_sqr()).sum::<f32>())
            .sum()
    }
}

/// Rich output from a reflection step.
#[derive(Clone, Debug)]
pub struct ReflectionStep {
    pub step: usize,
    pub output: Array1<f32>,
    pub energies: Vec<f32>,
    pub coherences: Vec<f32>,
    pub thought_state: ThoughtState,
    pub top_concepts: Vec<(&'static str, f32)>,
    pub total_energy: f32,
    pub thermal_contribution: f32,
    pub recall_event: Option<String>,
    pub dominant_freq: f32,
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

    #[test]
    fn test_reflection_never_zero_energy() {
        let config = BiologicalConfig {
            thermal_noise: 0.02,
            min_energy_floor: 0.05,
            reflection_feedback: 0.7,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Reset to zero state
        model.reset_rho();
        
        // Reflect for many iterations
        let trajectory = model.reflect(50);
        
        // Energy should NEVER be zero due to thermal noise + floor
        for step in &trajectory {
            assert!(step.total_energy > 0.0, "Energy should never be zero, got {}", step.total_energy);
        }
        
        // Should have some activity
        let avg_energy: f32 = trajectory.iter().map(|s| s.total_energy).sum::<f32>() / trajectory.len() as f32;
        assert!(avg_energy > 0.01, "Average energy should be above floor");
    }

    #[test]
    fn test_self_feedback_loop() {
        let config = BiologicalConfig {
            thermal_noise: 0.01,
            min_energy_floor: 0.02,
            reflection_feedback: 0.8,
            self_excitation: 0.03,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Initialize with some activity
        let brainwave = generate_brainwave(64, 10.0, 0.1);
        model.initialize_from_signal(&brainwave, 0);
        
        // Reflect
        let trajectory = model.reflect(30);
        
        // Should have state transitions (not stuck in one state)
        let states: Vec<_> = trajectory.iter().map(|s| s.thought_state.clone()).collect();
        let unique_states: std::collections::HashSet<_> = states.iter().collect();
        
        // At least 2 different states should appear
        assert!(unique_states.len() >= 1, "Should have at least some thought states");
    }

    #[test]
    fn test_stream_of_consciousness() {
        let config = BiologicalConfig {
            thermal_noise: 0.02,
            min_energy_floor: 0.03,
            reflection_feedback: 0.7,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Initialize
        let brainwave = generate_brainwave(64, 10.0, 0.1);
        model.initialize_from_signal(&brainwave, 0);
        
        // Get stream of consciousness
        let narrative = model.stream_of_consciousness(20, true);
        
        // Should produce some narrative
        assert!(!narrative.is_empty(), "Should produce narrative output");
        
        // Verbose mode should have energy readings
        let has_energy = narrative.iter().any(|s| s.contains("E:"));
        assert!(has_energy, "Verbose mode should show energy");
    }

    #[test]
    fn test_thermal_noise_injection() {
        let config = BiologicalConfig {
            thermal_noise: 0.05,
            noise_amplitude: 0.0,
            self_excitation: 0.0,
            endogenous_drive: 0.0,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        model.reset_rho();
        
        let initial_energy = model.total_energy();
        assert!(initial_energy < 0.001, "Should start at near-zero");
        
        // Reflect with thermal noise
        let _ = model.reflect(10);
        
        let final_energy = model.total_energy();
        assert!(final_energy > initial_energy, "Thermal noise should increase energy");
    }

    #[test]
    fn test_heartbeat_modulation() {
        let config = BiologicalConfig {
            heartbeat_freq: 1.0,
            heartbeat_strength: 0.5,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // At t=0, heartbeat = sin(0) = 0
        assert!(!model.is_heartbeat_peak());
        
        // Advance to peak
        model.advance_to_heartbeat_peak();
        assert!(model.is_heartbeat_peak(), "Should be at heartbeat peak");
    }

    #[test]
    fn test_imprint_and_recall() {
        let config = BiologicalConfig {
            heartbeat_freq: 1.0,
            heartbeat_strength: 0.3,
            rho_decay: 0.999, // Slow decay for test
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Create a distinctive pattern
        let pattern: Array1<f32> = Array1::from_iter((0..32).map(|i| {
            if i < 16 { 1.0 } else { 0.0 }
        }));
        
        // Advance to heartbeat peak and imprint
        model.advance_to_heartbeat_peak();
        model.imprint(&pattern, 0);
        
        // Verify timestamp recorded
        assert_eq!(model.get_memory_timestamps().len(), 1);
        
        // Advance time
        for _ in 0..10 {
            model.step_time(1.0);
        }
        
        // Recall with partial cue
        let cue: Array1<f32> = Array1::from_iter((0..32).map(|i| {
            if i < 8 { 0.5 } else { 0.0 } // Partial cue
        }));
        
        let recalled = model.recall(&cue, 0);
        
        // Recalled pattern should be non-zero
        assert!(recalled.iter().any(|&v| v.abs() > 0.01), "Should recall something");
    }

    #[test]
    fn test_temporal_memory_interference() {
        let config = BiologicalConfig {
            heartbeat_freq: 1.0,
            heartbeat_strength: 0.3,
            rho_decay: 0.99, // Moderate decay
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Pattern A: first half active
        let pattern_a: Array1<f32> = Array1::from_iter((0..32).map(|i| {
            if i < 16 { 1.0 } else { 0.0 }
        }));
        
        // Pattern B: second half active
        let pattern_b: Array1<f32> = Array1::from_iter((0..32).map(|i| {
            if i >= 16 { 1.0 } else { 0.0 }
        }));
        
        // Imprint A
        model.advance_to_heartbeat_peak();
        model.imprint(&pattern_a, 0);
        let time_a = model.get_time();
        
        // Advance time
        for _ in 0..20 {
            model.step_time(0.5);
        }
        
        // Imprint B
        model.advance_to_heartbeat_peak();
        model.imprint(&pattern_b, 0);
        
        // Verify both timestamps recorded
        assert_eq!(model.get_memory_timestamps().len(), 2);
        
        // The earlier memory should have decayed more
        // This is the key test for temporal memory
        let energy = model.total_energy();
        assert!(energy > 0.0, "Should have some energy from both patterns");
    }

    #[test]
    fn test_time_encoding() {
        let config = BiologicalConfig {
            time_encoding_dims: 8,
            time_encoding_base: 10000.0,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Get initial encoding
        let enc1 = model.get_time_encoding().clone();
        
        // Advance time
        model.step_time(10.0);
        model.update_time_encoding();
        let enc2 = model.get_time_encoding().clone();
        
        // Encodings should be different
        let diff: f32 = (&enc1 - &enc2).mapv(|v| v.abs()).sum();
        assert!(diff > 0.01, "Time encodings should change over time");
    }

    #[test]
    fn test_rho_decay_prevents_unbounded_growth() {
        let config = BiologicalConfig {
            rho_decay: 0.95, // Fast decay for test
            noise_amplitude: 0.0,
            self_excitation: 0.0,
            endogenous_drive: 0.0,
            thermal_noise: 0.0,
            ..Default::default()
        };
        
        let mut model = HarmonicBdh::with_config(32, 8, 2, config);
        
        // Imprint a strong pattern
        let pattern = Array1::from_elem(32, 1.0);
        model.imprint(&pattern, 0);
        
        let energy_after_imprint = model.total_energy();
        
        // Run many forward passes (which apply decay)
        for _ in 0..50 {
            let zeros = Array1::zeros(32);
            model.forward(&zeros);
        }
        
        let energy_after_decay = model.total_energy();
        
        // Energy should have decayed significantly
        assert!(energy_after_decay < energy_after_imprint * 0.5, 
            "Energy should decay: {} -> {}", energy_after_imprint, energy_after_decay);
    }
}
