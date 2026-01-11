use bdh_model::harmonic::{BiologicalConfig, ThoughtState};
use bdh_model::regions::{BrainRegion, RegionType};
use ndarray::Array1;
use rand::Rng;
use std::thread;
use std::time::Duration;

fn main() {
    println!("Initializing Modular Brain Architecture...");
    
    // 1. Create Regions
    let mut sensory = BrainRegion::new("Visual Cortex", RegionType::Sensory, 64, 16, 3);
    let mut associative = BrainRegion::new("Association Area", RegionType::Associative, 64, 16, 3);
    let mut executive = BrainRegion::new("Prefrontal Cortex", RegionType::Executive, 64, 16, 3);
    
    // 2. Connect Regions: Sensory -> Associative -> Executive
    // In this simple demo, we handle the routing manually in the loop, 
    // but the struct supports storing connections.
    sensory.connect_to(1); // just symbolic indices for now
    associative.connect_to(2);
    
    println!("Regions Created:");
    println!("  - {}: {:?}", sensory.name, sensory.region_type);
    println!("  - {}: {:?}", associative.name, associative.region_type);
    println!("  - {}: {:?}", executive.name, executive.region_type);
    
    println!("\nStarting Simulation (20 steps)...");
    println!("Step | Sensory Energy (Gamma) | Assoc Energy (Theta) | Exec Energy (Delta) | Action");
    println!("-----+------------------------+----------------------+---------------------+-------");
    
    let mut rng = rand::thread_rng();
    
    // Initial input
    let mut input = Array1::from_iter((0..64).map(|_| rng.gen_range(-0.5..0.5)));
    
    // Transfer buffers
    let mut sensory_out = Array1::zeros(64);
    let mut assoc_out = Array1::zeros(64);
    
    for step in 0..20 {
        // --- 1. SENSORY PASS ---
        // Occasionally inject a "visual stimulus"
        if step % 5 == 0 {
            input = Array1::from_iter((0..64).map(|_| rng.gen_range(-1.0..1.0)));
        } else {
            input = Array1::zeros(64); // Silence (noise only)
        }
        
        // Add feedback from executive to sensory (Top-down attention) - scaled down
        // input += &executive.project_output(&executive_out) * 0.2;
        
        let (s_out, s_energies) = sensory.process(&input);
        sensory_out = s_out;
        
        // --- 2. ASSOCIATIVE PASS ---
        // Input is bottom-up from sensory
        let (a_out, a_energies) = associative.process(&sensory_out);
        assoc_out = a_out;
        
        // --- 3. EXECUTIVE PASS ---
        // Input is bottom-up from associative
        let (e_out, e_energies) = executive.process(&assoc_out);
        // executive_out = e_out; // for next loop if we had feedback
        
        // --- REPORTING ---
        let s_state = sensory.bdh.classify_thought_state(&s_energies);
        let e_state = executive.bdh.classify_thought_state(&e_energies);
        
        let action = if e_state == ThoughtState::ActivePlanning { "PLANNING" } else { "..." };
        
        println!("{:4} | {:<22} | {:<20} | {:<19} | {}", 
            step, 
            format!("{:.3} ({})", s_energies.iter().sum::<f32>(), get_icon(&s_state)),
            format!("{:.3}", a_energies.iter().sum::<f32>()),
            format!("{:.3} ({})", e_energies.iter().sum::<f32>(), get_icon(&e_state)),
            action
        );
        
        // Small delay for readability
        thread::sleep(Duration::from_millis(100));
    }
}

fn get_icon(state: &ThoughtState) -> &str {
    match state {
        ThoughtState::Resting => "💤",
        ThoughtState::Contemplative => "🧘",
        ThoughtState::ActivePlanning => "🎯",
        ThoughtState::AlertScanning => "👁",
        ThoughtState::Transitioning => "🔄",
    }
}
