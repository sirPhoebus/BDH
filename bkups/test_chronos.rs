use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use bdh_model::ChronosBdh;
use std::f32::consts::PI;

fn main() {
    let n = 50;
    let d = 20;
    let freq = 1.0;
    let num_layers = 1;

    let mut chronos = ChronosBdh::new(n, d, freq, num_layers);
    println!("Created ChronosBdh with N={}, D={}, Freq={}", n, d, freq);

    // 1. Generate Input Signal
    let mut rng = rand::thread_rng();
    let input: Array1<f32> = Array1::random_using(n, Uniform::new(0.0f32, 1.0f32), &mut rng);
    println!("Input Signal Norm: {}", input.dot(&input).sqrt());

    // 2. Advance time to a heartbeat peak
    // Heartbeat = sin(t * freq).abs()
    // Peak at t * freq = PI / 2
    let time_to_peak = PI / (2.0 * freq);
    chronos.step_time(time_to_peak);
    println!("Stepped time to {}. Heartbeat: {}", chronos.global_time, (chronos.global_time * freq).sin().abs());

    // 3. Imprint the signal
    let _ = chronos.forward_with_time(&input, 0);
    println!("Imprinted signal at t={}", chronos.global_time);

    // 4. Wait 100 "heartbeats"
    // One beat = PI (since we take abs(sin), period is PI... wait, sin(x) period 2PI, abs(sin(x)) period PI?)
    // sin(t) repeats every 2PI. abs(sin(t)) peaks at PI/2, 3PI/2.
    // Let's advance by 100 * 2PI / freq just to be sure we are far ahead.
    let wait_time = 100.0 * 2.0 * PI / freq;
    chronos.step_time(wait_time);
    println!("advanced time to {}", chronos.global_time);

    // 5. Test reconstruction (feed zeros)
    let zeros = Array1::zeros(n);
    let output = chronos.forward_with_time(&zeros, 0);
    
    // 6. Check correlation
    let dot = input.dot(&output);
    let input_norm = input.dot(&input).sqrt();
    let output_norm = output.dot(&output).sqrt();
    let cosine_sim = dot / (input_norm * output_norm);
    
    println!("Output Norm: {}", output_norm);
    println!("Reconstruction Cosine Similarity: {}", cosine_sim);

    if cosine_sim > 0.9 {
        println!("SUCCESS: Signal reconstructed.");
    } else {
        println!("FAILURE: Signal lost or distorted.");
    }
}
