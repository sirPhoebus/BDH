//! Self-Thinking / Reflection Demo
//!
//! Demonstrates the model "thinking to itself" without external input.
//! Uses stochastic resonance (thermal noise) and self-feedback loops
//! to create spontaneous internal narratives.

use bdh_model::{BiologicalConfig, HarmonicBdh, generate_brainwave};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "reflect_demo")]
#[command(about = "Demonstrate self-thinking / stream of consciousness")]
struct Args {
    /// Number of neurons
    #[arg(short, long, default_value_t = 64)]
    neurons: usize,

    /// Number of reflection iterations
    #[arg(short, long, default_value_t = 100)]
    steps: usize,

    /// Thermal noise level (stochastic resonance)
    #[arg(long, default_value_t = 0.02)]
    thermal: f32,

    /// Minimum energy floor
    #[arg(long, default_value_t = 0.03)]
    floor: f32,

    /// Self-feedback strength (0.0-1.0)
    #[arg(long, default_value_t = 0.7)]
    feedback: f32,

    /// Verbose output (show energy at each step)
    #[arg(short, long)]
    verbose: bool,

    /// Initialize with a brainwave signal
    #[arg(long)]
    init: bool,
}

fn main() {
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           SELF-THINKING / REFLECTION DEMO                  ║");
    println!("║   The model thinks to itself without external input        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let config = BiologicalConfig {
        thermal_noise: args.thermal,
        min_energy_floor: args.floor,
        reflection_feedback: args.feedback,
        self_excitation: 0.03,
        endogenous_drive: 0.04,
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Neurons: {}", args.neurons);
    println!("  Thermal noise: {:.3}", args.thermal);
    println!("  Energy floor: {:.3}", args.floor);
    println!("  Self-feedback: {:.1}%", args.feedback * 100.0);
    println!();

    let mut model = HarmonicBdh::with_config(args.neurons, 16, 3, config);

    // Optionally initialize with a brainwave pattern
    if args.init {
        println!("Initializing with alpha-wave pattern...\n");
        let brainwave = generate_brainwave(128, 10.0, 0.1);
        model.initialize_from_signal(&brainwave, 0);
    } else {
        println!("Starting from near-zero state (cold start)...\n");
    }

    println!("═══════════════════ STREAM OF CONSCIOUSNESS ═══════════════════\n");

    // Run reflection
    let trajectory = model.reflect(args.steps);

    // Display as stream of consciousness
    let mut last_state = trajectory[0].thought_state.clone();
    let mut last_concept = "";
    let mut thought_buffer = String::new();

    for step in &trajectory {
        // State transition
        if step.thought_state != last_state {
            if !thought_buffer.is_empty() {
                println!("{}", thought_buffer);
                thought_buffer.clear();
            }
            println!("\n[{} → {}]", last_state.as_str(), step.thought_state.as_str());
            last_state = step.thought_state.clone();
        }

        // Concept shift
        if !step.top_concepts.is_empty() {
            let current_concept = step.top_concepts[0].0;
            if current_concept != last_concept {
                thought_buffer.push_str(&format!(" {{{}}} ", current_concept));
                last_concept = current_concept;
            }
        }

        // Memory recall event
        if let Some(ref recall) = step.recall_event {
            println!("  ⚡ {}", recall);
        }

        // Energy indicator (verbose mode)
        if args.verbose {
            thought_buffer.push_str(&format!("[E:{:.3}] ", step.total_energy));
        }
    }

    if !thought_buffer.is_empty() {
        println!("{}", thought_buffer);
    }

    println!("\n══════════════════════════════════════════════════════════════\n");

    // Summary statistics
    let total_energy: f32 = trajectory.iter().map(|s| s.total_energy).sum();
    let avg_energy = total_energy / trajectory.len() as f32;
    let min_energy = trajectory.iter().map(|s| s.total_energy).fold(f32::MAX, f32::min);
    let max_energy = trajectory.iter().map(|s| s.total_energy).fold(0.0f32, f32::max);

    let state_counts: std::collections::HashMap<_, usize> = trajectory.iter()
        .fold(std::collections::HashMap::new(), |mut acc, s| {
            *acc.entry(s.thought_state.as_str()).or_insert(0) += 1;
            acc
        });

    let recall_count = trajectory.iter().filter(|s| s.recall_event.is_some()).count();

    println!("Summary:");
    println!("  Total steps: {}", args.steps);
    println!("  Energy - Avg: {:.4}, Min: {:.4}, Max: {:.4}", avg_energy, min_energy, max_energy);
    println!("  Spontaneous recalls: {}", recall_count);
    println!();
    println!("State distribution:");
    for (state, count) in &state_counts {
        let pct = (*count as f32 / trajectory.len() as f32) * 100.0;
        println!("  {}: {:.1}%", state, pct);
    }

    // Verify key property: energy never zero
    let zero_count = trajectory.iter().filter(|s| s.total_energy < 0.001).count();
    if zero_count == 0 {
        println!("\n✓ Energy never dropped to zero (stochastic resonance working)");
    } else {
        println!("\n⚠ Warning: {} steps had near-zero energy", zero_count);
    }
}
