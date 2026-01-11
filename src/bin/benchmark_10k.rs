use bdh_model::harmonic_burn::HarmonicBdhBurn;
use burn_wgpu::{Wgpu, WgpuDevice, AutoGraphicsApi};
use std::time::Instant;

fn main() {
    println!("Initializing 10k Neuron Benchmark (WGPU)...");
    
    // Setup WGPU backend
    // Burn 0.14 Wgpu alias: type Wgpu<F = f32, I = i32>
    type Backend = Wgpu<f32, i32>;
    let device = WgpuDevice::BestAvailable;

    let n = 10_000;
    let d = 16;
    let layers = 4;
    
    println!("Creating model with N={}, D={}, Layers={}...", n, d, layers);
    let start_init = Instant::now();
    let mut model: HarmonicBdhBurn<Backend> = HarmonicBdhBurn::new(n, d, layers, &device);
    println!("Initialization took: {:?}", start_init.elapsed());
    
    // Warmup
    println!("Warming up GPU...");
    model.step(None);
    
    println!("Starting Benchmark (100 steps)...");
    let start_bench = Instant::now();
    
    for _ in 0..100 {
        model.step(None);
    }
    
    // Force synchronization (compute energy to pull data back)
    let _energy = model.get_energy().to_data(); 
    
    let duration = start_bench.elapsed();
    println!("Benchmark Complete.");
    println!("Total Time: {:?}", duration);
    println!("Time per step: {:?}", duration / 100);
    println!("Steps per second: {:.2}", 100.0 / duration.as_secs_f32());
}
