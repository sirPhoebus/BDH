//! Scaling Experiment: Test reflection quality at different neuron counts
//!
//! Metrics collected:
//! - Spontaneous recalls (count)
//! - % time in each thought state
//! - Concept chain length and variety
//! - Energy statistics

use bdh_model::{BiologicalConfig, HarmonicBdh, generate_brainwave};
use clap::Parser;
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(name = "scaling_experiment")]
#[command(about = "Test reflection quality at different neuron counts")]
struct Args {
    /// Number of neurons
    #[arg(short, long, default_value_t = 128)]
    neurons: usize,

    /// Number of reflection steps
    #[arg(short, long, default_value_t = 10000)]
    steps: usize,

    /// Thermal noise level
    #[arg(long, default_value_t = 0.015)]
    thermal: f32,

    /// Self-feedback strength
    #[arg(long, default_value_t = 0.70)]
    feedback: f32,

    /// Rho decay rate
    #[arg(long, default_value_t = 0.995)]
    decay: f32,

    /// Energy floor
    #[arg(long, default_value_t = 0.02)]
    floor: f32,

    /// Heartbeat frequency
    #[arg(long, default_value_t = 1.0)]
    heartbeat: f32,
}

#[derive(Debug, Default)]
struct ExperimentMetrics {
    total_steps: usize,
    spontaneous_recalls: usize,
    state_counts: HashMap<String, usize>,
    concept_transitions: usize,
    unique_concepts: std::collections::HashSet<String>,
    concept_chain_lengths: Vec<usize>,
    energy_min: f32,
    energy_max: f32,
    energy_sum: f32,
    zero_energy_steps: usize,
}

fn main() {
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              SCALING EXPERIMENT: {} NEURONS                   ║", args.neurons);
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let config = BiologicalConfig {
        thermal_noise: args.thermal,
        min_energy_floor: args.floor,
        reflection_feedback: args.feedback,
        rho_decay: args.decay,
        heartbeat_freq: args.heartbeat,
        // Keep other defaults
        self_excitation: 0.028,
        endogenous_drive: 0.060,
        noise_amplitude: 0.070,
        cross_freq_coupling: 0.36,
        homeostatic_threshold: 0.12,
        homeostatic_rate: 0.28,
        base_damping: 0.89,
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Neurons: {}", args.neurons);
    println!("  Steps: {}", args.steps);
    println!("  Thermal noise: {:.3}", args.thermal);
    println!("  Self-feedback: {:.1}%", args.feedback * 100.0);
    println!("  Rho decay: {:.4}", args.decay);
    println!("  Energy floor: {:.3}", args.floor);
    println!("  Heartbeat freq: {:.1} Hz", args.heartbeat);
    println!();

    // Scale d (latent dim) with neurons: d ≈ n/4, min 8, max 64
    let d = (args.neurons / 4).clamp(8, 64);
    let layers = 3;

    println!("  Latent dims (d): {}", d);
    println!("  Layers: {}", layers);
    println!();

    let mut model = HarmonicBdh::with_config(args.neurons, d, layers, config);

    // Initialize with brainwave
    let brainwave = generate_brainwave(256, 10.0, 0.1);
    model.initialize_from_signal(&brainwave, 0);

    println!("Running {} reflection steps...\n", args.steps);

    // Run reflection and collect metrics
    let trajectory = model.reflect(args.steps);

    let mut metrics = ExperimentMetrics {
        total_steps: args.steps,
        energy_min: f32::MAX,
        energy_max: 0.0,
        ..Default::default()
    };

    let mut last_concept = String::new();
    let mut current_chain_length = 0;

    for step in &trajectory {
        // Energy stats
        metrics.energy_sum += step.total_energy;
        if step.total_energy < metrics.energy_min {
            metrics.energy_min = step.total_energy;
        }
        if step.total_energy > metrics.energy_max {
            metrics.energy_max = step.total_energy;
        }
        if step.total_energy < 0.001 {
            metrics.zero_energy_steps += 1;
        }

        // State counts
        let state_name = step.thought_state.as_str().to_string();
        *metrics.state_counts.entry(state_name).or_insert(0) += 1;

        // Spontaneous recalls
        if step.recall_event.is_some() {
            metrics.spontaneous_recalls += 1;
        }

        // Concept tracking
        if !step.top_concepts.is_empty() {
            let concept = step.top_concepts[0].0.to_string();
            metrics.unique_concepts.insert(concept.clone());

            if concept != last_concept {
                if current_chain_length > 0 {
                    metrics.concept_chain_lengths.push(current_chain_length);
                }
                metrics.concept_transitions += 1;
                current_chain_length = 1;
                last_concept = concept;
            } else {
                current_chain_length += 1;
            }
        }
    }

    // Final chain
    if current_chain_length > 0 {
        metrics.concept_chain_lengths.push(current_chain_length);
    }

    // Print results
    println!("═══════════════════════════════════════════════════════════════");
    println!("                         RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("SPONTANEOUS RECALLS: {}", metrics.spontaneous_recalls);
    let recall_rate = metrics.spontaneous_recalls as f32 / args.steps as f32 * 1000.0;
    println!("  Rate: {:.2} per 1000 steps\n", recall_rate);

    println!("THOUGHT STATE DISTRIBUTION:");
    let mut states: Vec<_> = metrics.state_counts.iter().collect();
    states.sort_by(|a, b| b.1.cmp(a.1));
    for (state, count) in &states {
        let pct = (*count as f32 / args.steps as f32) * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  {:30} {:5.1}% {}", state, pct, bar);
    }
    println!();

    println!("CONCEPT DYNAMICS:");
    println!("  Unique concepts seen: {}", metrics.unique_concepts.len());
    println!("  Total transitions: {}", metrics.concept_transitions);
    let transition_rate = metrics.concept_transitions as f32 / args.steps as f32 * 100.0;
    println!("  Transition rate: {:.2} per 100 steps", transition_rate);

    if !metrics.concept_chain_lengths.is_empty() {
        let avg_chain: f32 = metrics.concept_chain_lengths.iter().sum::<usize>() as f32 
            / metrics.concept_chain_lengths.len() as f32;
        let max_chain = *metrics.concept_chain_lengths.iter().max().unwrap_or(&0);
        let min_chain = *metrics.concept_chain_lengths.iter().min().unwrap_or(&0);
        println!("  Chain lengths: min={}, avg={:.1}, max={}", min_chain, avg_chain, max_chain);
    }
    println!();

    println!("ENERGY STATISTICS:");
    let avg_energy = metrics.energy_sum / args.steps as f32;
    println!("  Min: {:.4}", metrics.energy_min);
    println!("  Avg: {:.4}", avg_energy);
    println!("  Max: {:.4}", metrics.energy_max);
    println!("  Zero-energy steps: {} ({:.2}%)", 
        metrics.zero_energy_steps,
        metrics.zero_energy_steps as f32 / args.steps as f32 * 100.0);
    println!();

    // Quality assessment
    println!("═══════════════════════════════════════════════════════════════");
    println!("                       ASSESSMENT");
    println!("═══════════════════════════════════════════════════════════════\n");

    let contemplative_pct = metrics.state_counts
        .get("🧘 Contemplative (memory recall)")
        .map(|&c| c as f32 / args.steps as f32 * 100.0)
        .unwrap_or(0.0);

    let resting_pct = metrics.state_counts
        .get("💤 Resting")
        .map(|&c| c as f32 / args.steps as f32 * 100.0)
        .unwrap_or(0.0);

    println!("Contemplative: {:.1}%", contemplative_pct);
    if contemplative_pct > 15.0 {
        println!("  ✓ Good contemplative activity");
    } else if contemplative_pct > 5.0 {
        println!("  ~ Moderate contemplative activity");
    } else {
        println!("  ⚠ Low contemplative activity - may be stuck in attractors");
    }

    println!("\nResting: {:.1}%", resting_pct);
    if resting_pct < 50.0 {
        println!("  ✓ Good activity level");
    } else if resting_pct < 70.0 {
        println!("  ~ Moderate activity");
    } else {
        println!("  ⚠ Too much resting - increase thermal noise?");
    }

    println!("\nConcept variety: {} concepts", metrics.unique_concepts.len());
    if metrics.unique_concepts.len() >= 10 {
        println!("  ✓ Good concept diversity");
    } else if metrics.unique_concepts.len() >= 6 {
        println!("  ~ Moderate diversity");
    } else {
        println!("  ⚠ Low diversity - may be locked to few attractors");
    }

    println!("\nRecalls: {}", metrics.spontaneous_recalls);
    if metrics.spontaneous_recalls >= 10 {
        println!("  ✓ Active memory recall");
    } else if metrics.spontaneous_recalls >= 3 {
        println!("  ~ Some recall activity");
    } else {
        println!("  ⚠ Few recalls - memory not activating spontaneously");
    }

    // Recommendations
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    RECOMMENDATIONS");
    println!("═══════════════════════════════════════════════════════════════\n");

    if resting_pct > 60.0 && contemplative_pct < 10.0 {
        println!("→ System seems locked. Try:");
        println!("  --thermal 0.022");
        println!("  --floor 0.025");
    } else if metrics.concept_transitions < 50 {
        println!("→ Low concept movement. Try:");
        println!("  --feedback 0.65");
        println!("  --decay 0.98");
    } else if metrics.spontaneous_recalls < 5 {
        println!("→ Low recall rate. Try:");
        println!("  --thermal 0.018");
    } else {
        println!("→ System looks healthy for {} neurons!", args.neurons);
        if args.neurons < 256 {
            println!("→ Ready to try: --neurons 256");
        } else if args.neurons < 384 {
            println!("→ Consider: --neurons 384 --feedback 0.65 --decay 0.97");
        }
    }
}
