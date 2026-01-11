use bdh_model::harmonic::{BiologicalConfig, ThoughtState};
use bdh_model::regions::{BrainRegion, RegionType};
use bdh_model::body::BodyState;
use bdh_model::drives::Drives;
use ndarray::Array1;
use rand::Rng;
use std::thread;
use std::time::Duration;

fn main() {
    println!("Initializing Motivational Core Simulation...");
    println!("Dynamics: Hunger (Survival) dominates Curiosity (Exploration).");

    let mut brain = BrainRegion::new("Cortex", RegionType::Associative, 64, 16, 3);
    let mut body = BodyState::new();
    let mut drives = Drives::new();
    
    let mut rng = rand::thread_rng();
    let steps = 60;
    
    // Track previous output for novelty detection
    let mut last_output = Array1::zeros(64);
    let mut input = Array1::zeros(64);
    
    println!("\nStep | Energy | Change | Active Drive | Action      | Brain | Outcome");
    println!("-----+--------+--------+--------------+-------------+-------+-------------------");
    
    for step in 0..steps {
        // 1. Body Update (Metabolism)
        // Static decay per step
        body.energy = (body.energy - 0.015).max(0.0);
        
        // 2. Brain Process
        let (output, energies) = brain.process(&input);
        let state = brain.bdh.classify_thought_state(&energies);
        
        // 3. Compute Novelty & Update Drives
        let novelty = brain.bdh.compute_novelty(&output, &last_output);
        last_output = output.clone();
        
        drives.update(body.energy, novelty);
        let active_drive = drives.get_dominant_drive();
        
        // 4. Motivate Brain (Modulate parameters based on drive)
        brain.bdh.motivate(active_drive.as_deref());
        
        // 5. Interpret Action
        // If Hunger is active -> FORAGE
        // If Curiosity is active -> EXPLORE
        // Else -> IDLE/SLEEP
        let action = match active_drive.as_deref() {
            Some("HUNGER") => "FORAGE",
            Some("CURIOSITY") => "EXPLORE",
            _ => "IDLE",
        };
        
        // 6. Environment Response
        let mut outcome = "";
        let mut reward_valence = 0.0;
        
        if action == "FORAGE" {
            // Harder to find food than to explore
            if rng.gen::<f32>() > 0.8 {
                body.energy = (body.energy + 0.4).min(1.0);
                reward_valence = 0.5;
                outcome = "Found Food!";
            }
        } else if action == "EXPLORE" {
            // Exploring guarantees some novelty input next step
            outcome = "Wandering...";
            // Influence next input to be random (novel)
            input = Array1::from_iter((0..64).map(|_| rng.gen_range(-1.0..1.0)));
            
            // Force wake up brain if sleeping
            if state == ThoughtState::Resting {
                 brain.bdh.inject_spontaneous_activity(0); // Kick
            }
        } else {
            // Static input (boring)
             input = input.mapv(|v| v * 0.9);
        }
        
        // Body update from action cost/reward
        body.update(if action == "IDLE" {0.0} else {0.3}, reward_valence);

        println!("{:4} | {:<6.2} | {:<6.2} | {:<12} | {:<11} | {:<5} | {}", 
            step, 
            body.energy, 
            novelty,
            active_drive.as_deref().unwrap_or("-"),
            action,
            get_icon(&state),
            outcome
        );
        
        thread::sleep(Duration::from_millis(50));
    }
}

fn get_icon(state: &ThoughtState) -> &str {
    match state {
        ThoughtState::Resting => "💤",
        ThoughtState::Contemplative => "Co",
        ThoughtState::ActivePlanning => "🎯",
        ThoughtState::AlertScanning => "👁",
        ThoughtState::Transitioning => "🔄",
    }
}
