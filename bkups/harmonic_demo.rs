//! Demo: Harmonic BDH with Biological Dynamics
//!
//! Run with: cargo run --bin harmonic_demo -- --help

use bdh_model::harmonic::{HarmonicBdh, BiologicalConfig, ThoughtState, generate_brainwave};
use clap::Parser;
use ndarray::Array1;

#[derive(Parser, Debug)]
#[command(name = "harmonic_demo")]
#[command(about = "Harmonic BDH: Vibrational Neural Architecture Demo")]
struct Args {
    /// Number of layers
    #[arg(short, long, default_value_t = 3)]
    layers: usize,

    /// Layer frequencies (comma-separated, e.g., "5,10,40")
    #[arg(short, long)]
    freqs: Option<String>,

    /// Cross-frequency coupling strength
    #[arg(short, long, default_value_t = 0.4)]
    coupling: f32,

    /// Van der Pol self-excitation strength
    #[arg(short, long, default_value_t = 0.02)]
    self_excite: f32,

    /// Noise amplitude
    #[arg(short, long, default_value_t = 0.03)]
    noise: f32,

    /// Number of neurons
    #[arg(long, default_value_t = 64)]
    neurons: usize,

    /// Daydream steps
    #[arg(long, default_value_t = 50)]
    steps: usize,
}

fn main() {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       HARMONIC BDH: Biological Neural Dynamics v2.0              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Parse layer frequencies if provided
    let layer_freqs: Vec<f32> = args.freqs
        .as_ref()
        .map(|s| s.split(',').filter_map(|f| f.trim().parse().ok()).collect())
        .unwrap_or_default();

    let config = BiologicalConfig {
        noise_amplitude: args.noise,
        cross_freq_coupling: args.coupling,
        self_excitation: args.self_excite,
        endogenous_drive: 0.01,
        adaptive_noise_rate: 0.15,
        homeostatic_threshold: 0.3,
        homeostatic_rate: 0.05,
        base_damping: 0.95,
        layer_frequencies: layer_freqs.clone(),
        boredom_delay: 10,
        ..Default::default()
    };

    let n = args.neurons;
    let d = 16;
    let layers = if layer_freqs.is_empty() { args.layers } else { layer_freqs.len() };

    let mut model = HarmonicBdh::with_config(n, d, layers, config.clone());

    // Apply custom frequencies if provided
    if !layer_freqs.is_empty() {
        model.set_layer_frequencies(layer_freqs.clone());
    }

    // Print configuration
    println!("┌─────────────────── Configuration ───────────────────┐");
    println!("│ Neurons: {:4}    Modes: {:2}    Layers: {}            │", n, d, layers);
    println!("│ Self-excitation: {:.3}   Coupling: {:.2}   Noise: {:.3} │", 
             args.self_excite, args.coupling, args.noise);
    println!("└─────────────────────────────────────────────────────┘\n");

    let freqs = model.get_layer_frequencies();
    println!("Layer Frequencies:");
    for (i, f) in freqs.iter().enumerate() {
        let band = match *f {
            x if x < 4.0 => "Delta",
            x if x < 8.0 => "Theta",
            x if x < 13.0 => "Alpha",
            x if x < 30.0 => "Beta",
            _ => "Gamma",
        };
        println!("  L{}: {:5.1} Hz  [{}]", i, f, band);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PART 1: BRAIN SNAPSHOT                                             ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let brainwave = generate_brainwave(256, 10.0, 0.05);
    model.initialize_from_signal(&brainwave, 0);
    
    let initial_energy: f32 = model.get_standing_wave(0).iter().map(|v| v * v).sum();
    println!("  Imprinted 10 Hz alpha wave → Initial energy: {:.4}\n", initial_energy);

    // ═══════════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PART 2: SPONTANEOUS ACTIVITY (Daydreaming)                         ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Legend: █ = 0.2 energy units, ░ = <0.2");
    println!("          Coherence shows phase alignment between layers\n");
    
    let trajectory = model.daydream(args.steps);
    
    println!("  Step │ Energy     │ Coherence  │ Noise  │ State");
    println!("  ─────┼────────────┼────────────┼────────┼─────────────────────────");
    
    let mut transitions = 0;
    let mut last_state = ThoughtState::Resting;
    let mut transition_steps = Vec::new();
    
    for step in &trajectory {
        // Create energy bar
        let energy_bar: String = step.energies.iter()
            .map(|&e| if e > 0.2 { "█" } else { "░" })
            .collect::<Vec<_>>()
            .join("");
        
        // Create coherence bar
        let coh_bar: String = step.coherences.iter()
            .map(|&c| if c > 0.5 { "●" } else if c > 0.0 { "○" } else { "·" })
            .collect::<Vec<_>>()
            .join("");
        
        // Track transitions
        if step.thought_state != last_state {
            transitions += 1;
            transition_steps.push(step.step);
            last_state = step.thought_state.clone();
        }
        
        let avg_noise: f32 = step.adaptive_noise.iter().sum::<f32>() / step.adaptive_noise.len() as f32;
        
        if step.step % 5 == 0 || step.step == trajectory.len() - 1 {
            println!(
                "  {:4} │ {:10} │ {:10} │ {:.4} │ {}",
                step.step, energy_bar, coh_bar, avg_noise, step.thought_state.as_str()
            );
        }
    }
    
    println!("\n  → {} state transitions detected at steps: {:?}\n", transitions, transition_steps);

    // ═══════════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PART 3: CROSS-FREQUENCY COUPLING                                   ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut model_cfc = HarmonicBdh::with_config(n, d, layers, config.clone());
    let theta_wave = generate_brainwave(256, 5.0, 0.0);
    model_cfc.initialize_from_signal(&theta_wave, 0);
    
    let gamma_input = Array1::from_iter((0..n).map(|i| {
        let t = i as f32 / n as f32;
        ((t * 40.0 * std::f32::consts::PI).sin() + 1.0) / 2.0
    }));
    
    // Without theta
    let mut model_no_theta = HarmonicBdh::with_config(n, d, layers, BiologicalConfig {
        noise_amplitude: 0.0,
        self_excitation: 0.0,
        endogenous_drive: 0.0,
        ..config.clone()
    });
    let (_, e1, c1) = model_no_theta.forward(&gamma_input);
    
    // With theta
    let (_, e2, c2) = model_cfc.forward(&gamma_input);
    
    println!("  Without theta modulation:");
    println!("    Layer energies: {:?}", e1.iter().map(|e| format!("{:.4}", e)).collect::<Vec<_>>());
    println!("    Phase coherence: {:?}", c1.iter().map(|c| format!("{:.3}", c)).collect::<Vec<_>>());
    
    println!("  With theta in Layer 0:");
    println!("    Layer energies: {:?}", e2.iter().map(|e| format!("{:.4}", e)).collect::<Vec<_>>());
    println!("    Phase coherence: {:?}", c2.iter().map(|c| format!("{:.3}", c)).collect::<Vec<_>>());
    
    let boost = e2.last().unwrap_or(&0.0) / e1.last().unwrap_or(&0.001).max(0.001);
    let coh_boost = c2.get(1).unwrap_or(&0.0) / c1.get(1).unwrap_or(&0.001).max(0.001);
    println!("\n  → Energy boost: {:.1}x, Coherence boost: {:.1}x\n", boost, coh_boost);

    // ═══════════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PART 4: HOMEOSTATIC PLASTICITY (Neural Fatigue + Recovery)         ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut model_homeo = HarmonicBdh::with_config(n, d, layers, BiologicalConfig {
        homeostatic_threshold: 0.1,
        homeostatic_rate: 0.1,
        noise_amplitude: 0.0,
        self_excitation: 0.0,
        endogenous_drive: 0.0,
        ..config.clone()
    });
    
    let strong_input = Array1::from_elem(n, 0.8);
    
    println!("  Phase 1: Sustained activation (fatigue)");
    println!("  Step │ L0 Damp │ L1 Damp │ L2 Damp │ Energy");
    println!("  ─────┼─────────┼─────────┼─────────┼────────");
    
    for step in 0..20 {
        let (_, energies, _) = model_homeo.forward(&strong_input);
        let (layer_damping, _) = model_homeo.get_damping_state();
        let total_e: f32 = energies.iter().sum();
        
        if step % 4 == 0 || step == 19 {
            let d0 = layer_damping.get(0).unwrap_or(&0.0);
            let d1 = layer_damping.get(1).unwrap_or(&0.0);
            let d2 = layer_damping.get(2).unwrap_or(&0.0);
            println!("  {:4} │  {:.3}  │  {:.3}  │  {:.3}  │ {:.4}", step, d0, d1, d2, total_e);
        }
    }
    
    let (fatigued_damping, _) = model_homeo.get_damping_state();
    let fatigued = fatigued_damping[0];
    
    println!("\n  Phase 2: Recovery (rest period)");
    let zero_input = Array1::zeros(n);
    let mut recovery_step = 0;
    
    for step in 0..30 {
        model_homeo.forward(&zero_input);
        let (layer_damping, _) = model_homeo.get_damping_state();
        
        if layer_damping[0] >= config.base_damping * 0.95 && recovery_step == 0 {
            recovery_step = step;
        }
    }
    
    let (recovered_damping, _) = model_homeo.get_damping_state();
    println!("  Damping: {:.3} (fatigued) → {:.3} (recovered)", fatigued, recovered_damping[0]);
    println!("  Recovery time: {} steps\n", if recovery_step > 0 { recovery_step } else { 30 });

    // ═══════════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PART 5: ATTRACTOR HOPPING                                          ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut model_hop = HarmonicBdh::with_config(n, d, layers, BiologicalConfig {
        noise_amplitude: 0.04,
        self_excitation: 0.025,
        endogenous_drive: 0.015,
        adaptive_noise_rate: 0.2,
        homeostatic_threshold: 0.2,
        ..config.clone()
    });
    
    let pattern_a = generate_brainwave(256, 8.0, 0.0);
    model_hop.initialize_from_signal(&pattern_a, 0);
    
    let daydream = model_hop.daydream(100);
    
    let mut last_freq = 0.0f32;
    let mut hops = 0;
    
    println!("  Tracking dominant frequency changes (>0.8 Hz = hop):");
    println!("  Step │ Freq  │ Event");
    println!("  ─────┼───────┼─────────────────────");
    
    for step in &daydream {
        let freq_change = (step.dominant_freq - last_freq).abs();
        
        if freq_change > 0.8 {
            hops += 1;
            println!("  {:4} │ {:5.2} │ ← Hop #{}", step.step, step.dominant_freq, hops);
            last_freq = step.dominant_freq;
        } else if step.step % 20 == 0 {
            println!("  {:4} │ {:5.2} │ (stable)", step.step, step.dominant_freq);
        }
    }
    
    println!("\n  → {} attractor hops in 100 steps (target: 2-6)\n", hops);

    // ═══════════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PART 6: INTERPRETED THOUGHT STREAM                                 ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut model_thoughts = HarmonicBdh::with_config(n, d, layers, config.clone());
    model_thoughts.initialize_from_signal(&brainwave, 0);
    
    let thoughts = model_thoughts.daydream(30);
    
    println!("  Step │ State                    │ Top Concepts");
    println!("  ─────┼──────────────────────────┼───────────────────────────────────");
    
    for step in &thoughts {
        if step.step % 5 == 0 || step.step == thoughts.len() - 1 {
            let concepts: String = step.top_concepts.iter()
                .map(|(name, score)| format!("{}({:.2})", name, score))
                .collect::<Vec<_>>()
                .join(", ");
            
            println!("  {:4} │ {:24} │ {}", 
                     step.step, 
                     step.thought_state.as_str(),
                     concepts);
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ (A) Spontaneous Activity: van der Pol + adaptive noise           ║");
    println!("║ (B) Cross-Freq Coupling:  Phase coherence modulates higher bands ║");
    println!("║ (C) Homeostatic Plasticity: Fatigue + recovery dynamics          ║");
    println!("║ (D) Attractor Hopping: {} hops / 100 steps (boredom → explore)    ║", hops);
    println!("║ (E) Semantic Readout: Concept space projection                   ║");
    println!("║                                                                  ║");
    println!("║ Next: real-time audio input → thought-to-speech via freq→phoneme ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
