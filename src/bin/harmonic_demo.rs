//! Demo: Harmonic BDH with Biological Dynamics
//!
//! Demonstrates:
//! - Spontaneous Activity (Daydreaming)
//! - Cross-Frequency Coupling (Theta-Gamma)
//! - Homeostatic Plasticity (Neural Fatigue)

use bdh_model::harmonic::{HarmonicBdh, BiologicalConfig, generate_brainwave};
use ndarray::Array1;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     HARMONIC BDH: Biological Neural Dynamics Demo            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let n = 64;
    let d = 16;
    let layers = 3;

    // Configure biological parameters
    let config = BiologicalConfig {
        noise_amplitude: 0.02,       // Moderate spontaneous activity
        cross_freq_coupling: 0.4,    // Strong theta-gamma binding
        homeostatic_threshold: 0.3,  // Moderate boredom threshold
        homeostatic_rate: 0.05,      // Medium adaptation speed
        base_damping: 0.95,
    };

    let mut model = HarmonicBdh::with_config(n, d, layers, config.clone());

    // Show layer frequencies
    let freqs = model.get_layer_frequencies();
    println!("Layer Frequencies (Hz):");
    for (i, f) in freqs.iter().enumerate() {
        let band = if *f < 8.0 { "Theta" } else if *f < 30.0 { "Beta" } else { "Gamma" };
        println!("  Layer {}: {:.1} Hz ({})", i, f, band);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // PART 1: BRAIN SNAPSHOT - Initialize with memory
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ PART 1: BRAIN SNAPSHOT ━━━");
    println!("Imprinting 10 Hz alpha wave into memory...\n");
    
    let brainwave = generate_brainwave(256, 10.0, 0.05);
    model.initialize_from_signal(&brainwave, 0);
    
    let initial_energy: f32 = model.get_standing_wave(0).iter().map(|v| v * v).sum();
    println!("  Initial memory energy: {:.4}\n", initial_energy);

    // ═══════════════════════════════════════════════════════════════════
    // PART 2: SPONTANEOUS ACTIVITY - The Daydream
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ PART 2: SPONTANEOUS ACTIVITY (Daydreaming) ━━━");
    println!("Running 30 steps with NO external input...\n");
    
    let trajectory = model.daydream(30);
    
    println!("  Step │ Activity │ Energy (L0→L2)        │ Pattern");
    println!("  ─────┼──────────┼───────────────────────┼─────────────────");
    
    for (step, (signal, energies)) in trajectory.iter().enumerate() {
        let activity: f32 = signal.iter().map(|v| v.abs()).sum();
        let active_neurons = signal.iter().filter(|&&v| v.abs() > 0.01).count();
        
        // Create energy bar
        let bar: String = energies.iter()
            .map(|&e| {
                let blocks = (e * 10.0).min(5.0) as usize;
                "█".repeat(blocks) + &"░".repeat(5 - blocks)
            })
            .collect::<Vec<_>>()
            .join("│");
        
        if step % 5 == 0 || step == trajectory.len() - 1 {
            println!(
                "  {:4} │ {:7.3} │ {} │ {} active",
                step, activity, bar, active_neurons
            );
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // PART 3: CROSS-FREQUENCY COUPLING - Theta carries Gamma
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ PART 3: CROSS-FREQUENCY COUPLING ━━━");
    println!("Testing theta (5Hz) modulation of gamma (40Hz)...\n");
    
    // Reset model
    let mut model_cfc = HarmonicBdh::with_config(n, d, layers, config.clone());
    
    // Initialize layer 0 with strong theta
    let theta_wave = generate_brainwave(256, 5.0, 0.0);
    model_cfc.initialize_from_signal(&theta_wave, 0);
    
    // Run with gamma-frequency input
    let gamma_input = Array1::from_iter((0..n).map(|i| {
        let t = i as f32 / n as f32;
        ((t * 40.0 * std::f32::consts::PI).sin() + 1.0) / 2.0
    }));
    
    println!("  Without theta in Layer 0:");
    let mut model_no_theta = HarmonicBdh::with_config(n, d, layers, config.clone());
    let (out1, e1) = model_no_theta.forward(&gamma_input);
    println!("    Output mean: {:.4}, Layer 2 energy: {:.4}", out1.mean().unwrap(), e1[2]);
    
    println!("  With theta in Layer 0:");
    let (out2, e2) = model_cfc.forward(&gamma_input);
    println!("    Output mean: {:.4}, Layer 2 energy: {:.4}", out2.mean().unwrap(), e2[2]);
    
    let boost = e2[2] / e1[2].max(0.0001);
    println!("\n  → Theta-gamma coupling boost: {:.1}x\n", boost);

    // ═══════════════════════════════════════════════════════════════════
    // PART 4: HOMEOSTATIC PLASTICITY - Getting Bored
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ PART 4: HOMEOSTATIC PLASTICITY (Neural Fatigue) ━━━");
    println!("Sustained activation → watching damping adapt...\n");
    
    let mut model_homeo = HarmonicBdh::with_config(n, d, layers, BiologicalConfig {
        homeostatic_threshold: 0.1,  // Low threshold to trigger quickly
        homeostatic_rate: 0.1,       // Fast adaptation for demo
        ..config.clone()
    });
    
    let strong_input = Array1::from_elem(n, 0.8);
    
    println!("  Step │ L0 Damping │ L1 Damping │ L2 Damping │ Total Energy");
    println!("  ─────┼────────────┼────────────┼────────────┼──────────────");
    
    for step in 0..25 {
        let (_, energies) = model_homeo.forward(&strong_input);
        let (layer_damping, _) = model_homeo.get_damping_state();
        let total_e: f32 = energies.iter().sum();
        
        if step % 4 == 0 || step == 24 {
            println!(
                "  {:4} │    {:.4}   │    {:.4}   │    {:.4}   │   {:.4}",
                step, layer_damping[0], layer_damping[1], layer_damping[2], total_e
            );
        }
    }
    
    let (final_damping, _) = model_homeo.get_damping_state();
    println!("\n  → Damping decreased from {:.2} to {:.2} (system got 'bored')\n", 
             config.base_damping, final_damping[0]);

    // ═══════════════════════════════════════════════════════════════════
    // PART 5: ATTRACTOR HOPPING - Boredom leads to new thoughts
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━ PART 5: ATTRACTOR HOPPING ━━━");
    println!("Homeostasis + noise → natural thought transitions...\n");
    
    let mut model_hop = HarmonicBdh::with_config(n, d, layers, BiologicalConfig {
        noise_amplitude: 0.05,        // Higher noise
        homeostatic_threshold: 0.2,
        homeostatic_rate: 0.08,
        ..config.clone()
    });
    
    // Initialize with pattern A
    let pattern_a = generate_brainwave(256, 8.0, 0.0);
    model_hop.initialize_from_signal(&pattern_a, 0);
    
    // Track dominant frequency changes
    let mut last_dominant = 0.0f32;
    let mut transitions = 0;
    
    println!("  Step │ Dominant Freq │ Event");
    println!("  ─────┼───────────────┼───────────────────────");
    
    let daydream = model_hop.daydream(50);
    for (step, (_, _)) in daydream.iter().enumerate() {
        let freqs = model_hop.get_dominant_frequencies(0);
        let current_dominant = freqs.mean().unwrap();
        
        if (current_dominant - last_dominant).abs() > 0.5 {
            transitions += 1;
            println!("  {:4} │    {:.2} Hz    │ ← Transition #{}", 
                     step, current_dominant, transitions);
            last_dominant = current_dominant;
        } else if step % 10 == 0 {
            println!("  {:4} │    {:.2} Hz    │ (stable)", step, current_dominant);
        }
    }
    
    println!("\n  → {} spontaneous thought transitions detected\n", transitions);

    // ═══════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ (A) Spontaneous Activity: System 'thinks' without input      ║");
    println!("║ (B) Cross-Freq Coupling:  Slow waves modulate fast waves     ║");
    println!("║ (C) Homeostatic Plasticity: Prevents runaway activation      ║");
    println!("║                                                              ║");
    println!("║ Together: Autonomous thought with natural transitions        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
