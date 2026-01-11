//! Benchmark: Compare spontaneous activity variants
//!
//! Tests different configurations for self-sustaining daydream dynamics:
//! - Baseline (current model)
//! - + Weak endogenous drive
//! - + Van der Pol nonlinearity
//! - Both combined

use bdh_model::{HarmonicBdh, BiologicalConfig, ThoughtState};
use clap::Parser;
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(name = "benchmark_variants")]
#[command(about = "Benchmark spontaneous activity variants")]
struct Args {
    /// Number of daydream steps
    #[arg(short, long, default_value_t = 200)]
    steps: usize,

    /// Number of neurons
    #[arg(short, long, default_value_t = 64)]
    neurons: usize,

    /// Number of runs per variant for averaging
    #[arg(short, long, default_value_t = 3)]
    runs: usize,
}

/// Metrics from a daydream run.
#[derive(Debug, Clone, Default)]
struct DaydreamMetrics {
    total_transitions: usize,
    avg_burst_duration: f32,
    state_distribution: HashMap<String, f32>,
    avg_energy: f32,
    avg_dominant_freq: f32,
    num_bursts: usize,
    time_in_resting: f32,
}

/// Analyze a daydream trajectory for key metrics.
fn analyze_daydream(trajectory: &[bdh_model::DaydreamStep]) -> DaydreamMetrics {
    let mut metrics = DaydreamMetrics::default();
    let total_steps = trajectory.len() as f32;
    
    if trajectory.is_empty() {
        return metrics;
    }
    
    // Count state occurrences
    let mut state_counts: HashMap<String, usize> = HashMap::new();
    let mut transitions = 0;
    let mut last_state = &trajectory[0].thought_state;
    
    // Track bursts (non-Resting periods)
    let mut in_burst = false;
    let mut burst_lengths = Vec::new();
    let mut current_burst_len = 0;
    
    let mut total_energy = 0.0f32;
    let mut total_freq = 0.0f32;
    
    for step in trajectory {
        // State counting
        let state_name = step.thought_state.as_str().to_string();
        *state_counts.entry(state_name).or_insert(0) += 1;
        
        // Transition counting
        if &step.thought_state != last_state {
            transitions += 1;
            last_state = &step.thought_state;
        }
        
        // Burst tracking (non-Resting = burst)
        let is_resting = step.thought_state == ThoughtState::Resting;
        if !is_resting {
            if !in_burst {
                in_burst = true;
                current_burst_len = 0;
            }
            current_burst_len += 1;
        } else {
            if in_burst {
                burst_lengths.push(current_burst_len);
                in_burst = false;
            }
        }
        
        // Energy and frequency
        total_energy += step.energies.iter().sum::<f32>();
        total_freq += step.dominant_freq;
    }
    
    // Final burst if still in one
    if in_burst && current_burst_len > 0 {
        burst_lengths.push(current_burst_len);
    }
    
    metrics.total_transitions = transitions;
    metrics.num_bursts = burst_lengths.len();
    metrics.avg_burst_duration = if burst_lengths.is_empty() {
        0.0
    } else {
        burst_lengths.iter().sum::<usize>() as f32 / burst_lengths.len() as f32
    };
    
    // State distribution as percentages
    for (state, count) in state_counts {
        let pct = (count as f32 / total_steps) * 100.0;
        if state.contains("Resting") {
            metrics.time_in_resting = pct;
        }
        metrics.state_distribution.insert(state, pct);
    }
    
    metrics.avg_energy = total_energy / total_steps;
    metrics.avg_dominant_freq = total_freq / total_steps;
    
    metrics
}

/// Run a single variant configuration.
fn run_variant(
    name: &str,
    config: BiologicalConfig,
    neurons: usize,
    steps: usize,
    runs: usize,
) -> DaydreamMetrics {
    let mut all_metrics = Vec::new();
    
    for _ in 0..runs {
        let mut model = HarmonicBdh::with_config(neurons, 16, 3, config.clone());
        
        // Initialize with some memory
        let brainwave = bdh_model::generate_brainwave(256, 10.0, 0.05);
        model.initialize_from_signal(&brainwave, 0);
        
        let trajectory = model.daydream(steps);
        let metrics = analyze_daydream(&trajectory);
        all_metrics.push(metrics);
    }
    
    // Average metrics across runs
    let n = all_metrics.len() as f32;
    let mut avg = DaydreamMetrics::default();
    
    for m in &all_metrics {
        avg.total_transitions += m.total_transitions;
        avg.num_bursts += m.num_bursts;
        avg.avg_burst_duration += m.avg_burst_duration;
        avg.avg_energy += m.avg_energy;
        avg.avg_dominant_freq += m.avg_dominant_freq;
        avg.time_in_resting += m.time_in_resting;
    }
    
    avg.total_transitions = (avg.total_transitions as f32 / n) as usize;
    avg.num_bursts = (avg.num_bursts as f32 / n) as usize;
    avg.avg_burst_duration /= n;
    avg.avg_energy /= n;
    avg.avg_dominant_freq /= n;
    avg.time_in_resting /= n;
    
    // Merge state distributions
    let mut merged_states: HashMap<String, f32> = HashMap::new();
    for m in &all_metrics {
        for (state, pct) in &m.state_distribution {
            *merged_states.entry(state.clone()).or_insert(0.0) += pct / n;
        }
    }
    avg.state_distribution = merged_states;
    
    avg
}

fn print_metrics(name: &str, m: &DaydreamMetrics) {
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ {:^59} │", name);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Transitions: {:3}    Bursts: {:3}    Avg Burst: {:5.1} steps  │", 
             m.total_transitions, m.num_bursts, m.avg_burst_duration);
    println!("│ Avg Energy: {:6.4}   Avg Freq: {:5.2} Hz                    │", 
             m.avg_energy, m.avg_dominant_freq);
    println!("│ Time in Resting: {:5.1}%                                     │", 
             m.time_in_resting);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ State Distribution:                                         │");
    
    let mut states: Vec<_> = m.state_distribution.iter().collect();
    states.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (state, pct) in states.iter().take(5) {
        let bar_len = (*pct / 5.0).min(12.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("│   {:30} {:5.1}% {}     │", state, pct, bar);
    }
    println!("└─────────────────────────────────────────────────────────────┘");
}

fn main() {
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     SPONTANEOUS ACTIVITY VARIANT BENCHMARK                    ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Steps: {:4}   Neurons: {:3}   Runs per variant: {:2}            ║", 
             args.steps, args.neurons, args.runs);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Define variants
    let variants: Vec<(&str, BiologicalConfig)> = vec![
        // 1. BASELINE - current defaults
        ("1. BASELINE (current)", BiologicalConfig {
            noise_amplitude: 0.02,
            self_excitation: 0.015,
            endogenous_drive: 0.008,
            cross_freq_coupling: 0.3,
            adaptive_noise_rate: 0.1,
            homeostatic_threshold: 0.3,
            homeostatic_rate: 0.05,
            base_damping: 0.95,
            layer_frequencies: vec![],
        }),
        
        // 2. WEAK ENDOGENOUS DRIVE - stronger L0 injection
        ("2. + Strong Endogenous Drive", BiologicalConfig {
            noise_amplitude: 0.02,
            self_excitation: 0.015,
            endogenous_drive: 0.025,  // 3x stronger
            cross_freq_coupling: 0.3,
            adaptive_noise_rate: 0.1,
            homeostatic_threshold: 0.3,
            homeostatic_rate: 0.05,
            base_damping: 0.95,
            layer_frequencies: vec![],
        }),
        
        // 3. VAN DER POL - stronger self-excitation
        ("3. + Van der Pol (μ=0.04)", BiologicalConfig {
            noise_amplitude: 0.02,
            self_excitation: 0.04,  // Stronger limit cycle behavior
            endogenous_drive: 0.008,
            cross_freq_coupling: 0.3,
            adaptive_noise_rate: 0.1,
            homeostatic_threshold: 0.3,
            homeostatic_rate: 0.03,  // Slower homeostasis
            base_damping: 0.96,      // Slightly less decay
            layer_frequencies: vec![],
        }),
        
        // 4. BOTH - combined
        ("4. BOTH (endogenous + vdP)", BiologicalConfig {
            noise_amplitude: 0.025,
            self_excitation: 0.035,
            endogenous_drive: 0.02,
            cross_freq_coupling: 0.35,
            adaptive_noise_rate: 0.15,
            homeostatic_threshold: 0.25,
            homeostatic_rate: 0.04,
            base_damping: 0.96,
            layer_frequencies: vec![],
        }),
        
        // 5. AGGRESSIVE - push the limits
        ("5. AGGRESSIVE (high μ, high drive)", BiologicalConfig {
            noise_amplitude: 0.03,
            self_excitation: 0.06,
            endogenous_drive: 0.03,
            cross_freq_coupling: 0.4,
            adaptive_noise_rate: 0.2,
            homeostatic_threshold: 0.2,
            homeostatic_rate: 0.06,
            base_damping: 0.97,
            layer_frequencies: vec![],
        }),
        
        // 6. TUNED - aggressive but with faster homeostasis to limit burst length
        ("6. TUNED (aggr + fast homeo)", BiologicalConfig {
            noise_amplitude: 0.035,
            self_excitation: 0.05,
            endogenous_drive: 0.025,
            cross_freq_coupling: 0.4,
            adaptive_noise_rate: 0.25,
            homeostatic_threshold: 0.15,  // Lower threshold = faster "boredom"
            homeostatic_rate: 0.12,       // Fast adaptation
            base_damping: 0.94,           // More decay
            layer_frequencies: vec![],
        }),
        
        // 7. BURSTY - designed for short frequent bursts
        ("7. BURSTY (short bursts)", BiologicalConfig {
            noise_amplitude: 0.04,
            self_excitation: 0.045,
            endogenous_drive: 0.03,
            cross_freq_coupling: 0.35,
            adaptive_noise_rate: 0.3,
            homeostatic_threshold: 0.1,   // Very low threshold
            homeostatic_rate: 0.15,       // Very fast adaptation
            base_damping: 0.92,           // High decay
            layer_frequencies: vec![],
        }),
        
        // 8. OPTIMAL - balance between BURSTY and TUNED
        ("8. OPTIMAL (balanced)", BiologicalConfig {
            noise_amplitude: 0.045,
            self_excitation: 0.055,
            endogenous_drive: 0.028,
            cross_freq_coupling: 0.38,
            adaptive_noise_rate: 0.28,
            homeostatic_threshold: 0.12,
            homeostatic_rate: 0.14,
            base_damping: 0.93,
            layer_frequencies: vec![],
        }),
    ];

    let mut results = Vec::new();

    for (name, config) in &variants {
        print!("Running {}...", name);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        let metrics = run_variant(name, config.clone(), args.neurons, args.steps, args.runs);
        results.push((name, metrics));
        
        println!(" done.");
    }

    // Print all results
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                         RESULTS                               ");
    println!("═══════════════════════════════════════════════════════════════");

    for (name, metrics) in &results {
        print_metrics(name, metrics);
    }

    // Summary comparison
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│                    COMPARISON SUMMARY                       │");
    println!("├───────────────────────────┬────────┬────────┬───────┬───────┤");
    println!("│ Variant                   │ Trans. │ Bursts │ B.Dur │ %Rest │");
    println!("├───────────────────────────┼────────┼────────┼───────┼───────┤");
    
    for (name, m) in &results {
        let short_name = if name.len() > 25 { &name[..25] } else { name };
        println!("│ {:25} │ {:6} │ {:6} │ {:5.1} │ {:5.1} │", 
                 short_name, m.total_transitions, m.num_bursts, 
                 m.avg_burst_duration, m.time_in_resting);
    }
    println!("└───────────────────────────┴────────┴────────┴───────┴───────┘");

    // Find best variant
    let best = results.iter()
        .max_by(|a, b| {
            // Score: more transitions + lower resting time + reasonable burst duration
            let score_a = a.1.total_transitions as f32 * 2.0 
                + a.1.num_bursts as f32 
                - a.1.time_in_resting * 0.5
                + (a.1.avg_burst_duration.min(15.0)) * 0.5;
            let score_b = b.1.total_transitions as f32 * 2.0 
                + b.1.num_bursts as f32 
                - b.1.time_in_resting * 0.5
                + (b.1.avg_burst_duration.min(15.0)) * 0.5;
            score_a.partial_cmp(&score_b).unwrap()
        });

    if let Some((name, _)) = best {
        println!("\n★ RECOMMENDED: {} ★", name);
    }

    // Target comparison
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│                     TARGET METRICS                          │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ • 4-12 transitions per 100 steps                            │");
    println!("│ • Bursts lasting 5-15 steps                                 │");
    println!("│ • <50% time in Resting                                      │");
    println!("│ • Mix of Active Planning, Exploration, Contemplative        │");
    println!("└─────────────────────────────────────────────────────────────┘");
}
