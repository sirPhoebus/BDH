//! Demo: Harmonic BDH with resonance dynamics
//!
//! This demonstrates:
//! 1. Initializing from a "brainwave" signal
//! 2. Finding attractor states (limit cycles)
//! 3. Observing energy flow and phase alignment

use bdh_model::{HarmonicBdh, generate_brainwave};
use ndarray::Array1;

fn main() {
    println!("=== Harmonic BDH: Vibrational Neural Architecture ===\n");

    let n = 128;      // Neurons
    let d = 32;       // Harmonic modes
    let layers = 3;

    let mut model = HarmonicBdh::new(n, d, layers);

    // --- Part 1: Brain Snapshot Initialization ---
    println!("1. BRAIN SNAPSHOT: Initializing from synthetic brainwave...");
    let brainwave = generate_brainwave(512, 10.0, 0.05); // ~10 Hz alpha wave
    model.initialize_from_signal(&brainwave, 0);
    
    let initial_wave = model.get_standing_wave(0);
    let initial_energy: f32 = initial_wave.iter().map(|&v| v * v).sum();
    println!("   Initial standing wave energy: {:.4}", initial_energy);
    
    // Show dominant frequencies in first few neurons
    let freqs = model.get_dominant_frequencies(0);
    println!("   Dominant frequencies (first 8 neurons): {:?}", 
             &freqs.as_slice().unwrap()[..8]);
    println!();

    // --- Part 2: Forward Pass with Resonance ---
    println!("2. RESONANCE DYNAMICS: Processing input signal...");
    
    // Create a test input that partially matches the brainwave pattern
    let input = Array1::from_iter((0..n).map(|i| {
        let t = i as f32 / n as f32;
        ((t * 10.0 * std::f32::consts::PI).sin() + 1.0) / 2.0 // ~10 Hz, positive
    }));
    
    println!("   Input: {} neurons, mean={:.4}, max={:.4}", 
             input.len(), input.mean().unwrap(), input.iter().cloned().fold(0.0f32, f32::max));

    // Run forward pass
    let (output, energies) = model.forward(&input);
    
    println!("   Output: mean={:.4}, sparsity={:.1}%",
             output.mean().unwrap(),
             (output.iter().filter(|&&v| v < 0.01).count() as f32 / n as f32) * 100.0);
    println!("   Layer energies: {:?}", energies);
    println!();

    // --- Part 3: Attractor Discovery (Limit Cycle) ---
    println!("3. ATTRACTOR DISCOVERY: Finding stable resonance pattern...");
    
    // Start with a sparse excitation
    let seed = Array1::from_iter((0..n).map(|i| if i % 8 == 0 { 0.5 } else { 0.0 }));
    
    let (attractor_state, iterations) = model.find_attractor(&seed, 200, 0.0001);
    
    println!("   Converged in {} iterations", iterations);
    let active_neurons = attractor_state.iter().filter(|&&v| v > 0.01).count();
    println!("   Active neurons in attractor: {} / {} ({:.1}%)", 
             active_neurons, n, (active_neurons as f32 / n as f32) * 100.0);
    println!();

    // --- Part 4: Zero-Shot Resonance Test ---
    println!("4. ZERO-SHOT RESONANCE: Introducing a new frequency...");
    
    // Create a different frequency input (not seen during initialization)
    let novel_input = Array1::from_iter((0..n).map(|i| {
        let t = i as f32 / n as f32;
        ((t * 7.0 * std::f32::consts::PI).sin() + 1.0) / 2.0 // ~7 Hz (theta)
    }));
    
    let (novel_output, novel_energies) = model.forward(&novel_input);
    
    println!("   Novel (7 Hz) input response:");
    println!("   - Output mean: {:.4}", novel_output.mean().unwrap());
    println!("   - Energy: {:?}", novel_energies);
    
    // Compare with the original frequency
    let (original_output, original_energies) = model.forward(&input);
    println!("   Original (10 Hz) input response:");
    println!("   - Output mean: {:.4}", original_output.mean().unwrap());
    println!("   - Energy: {:?}", original_energies);
    println!();

    // --- Part 5: Energy Evolution ---
    println!("5. ENERGY EVOLUTION: Watching the standing wave stabilize...");
    let mut model2 = HarmonicBdh::new(64, 16, 2);
    let pulse = Array1::from_iter((0..64).map(|i| if i == 32 { 1.0 } else { 0.0 }));
    
    for step in 0..10 {
        let (_, energies) = model2.forward(&pulse);
        let total: f32 = energies.iter().sum();
        let bar = "█".repeat((total * 20.0).min(40.0) as usize);
        println!("   Step {:2}: E={:.4} {}", step, total, bar);
    }
    println!();

    println!("=== Demo Complete ===");
    println!("\nKey Insights:");
    println!("• Standing wave memory persists without recalculating attention");
    println!("• Phase alignment gates information flow (constructive interference)");
    println!("• Attractor states = stable concepts (limit cycles)");
    println!("• Novel frequencies can induce new resonance patterns (zero-shot)");
}
