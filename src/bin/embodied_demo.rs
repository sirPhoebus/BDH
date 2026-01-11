use bdh_model::harmonic::{BiologicalConfig, ThoughtState};
use bdh_model::regions::{BrainRegion, RegionType};
use bdh_model::body::BodyState;
use ndarray::Array1;
use rand::Rng;
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    println!("Initializing Embodied Brain Simulation...");
    println!("Controls: The brain must 'decision make' to keep energy up and avoid pain.");
    
    // 1. Setup Brain (Single 'General' Region for simplicity, or use specific)
    let mut cortex = BrainRegion::new("Cortex", RegionType::Associative, 64, 16, 3);
    
    // 2. Setup Body
    let mut body = BodyState::new();
    
    // 3. Environment Loop
    let mut rng = rand::thread_rng();
    let steps = 50;
    
    println!("\nStep | Energy | Integ | Valence | Brain State      | Action      | Outcome");
    println!("-----+--------+-------+---------+------------------+-------------+-------------------");
    
    // Initial input
    let mut input = Array1::zeros(64);
    
    for step in 0..steps {
        // --- A. Perception ---
        // Input is noise + body signals injected as context
        let body_signals = body.get_signals(); // [Energy, Integrity, Pleasure, Pain]
        
        // Inject low energy as a specific signal pattern (e.g., neurons 0-10)
        if body.energy < 0.3 {
            for i in 0..10 { input[i] = 1.0; } // "Hunger" signal
        }
        
        // --- B. Cognition ---
        // 1. Integrate signals into brain dynamics (modulate parameters)
        cortex.bdh.integrate_body_signals(&body_signals);
        
        // 2. Process
        let (_, energies) = cortex.process(&input);
        let state = cortex.bdh.classify_thought_state(&energies);
        
        // --- C. Action Selection ---
        // Map brain state to action
        // Heuristic mapping for demo:
        // ActivePlanning -> Forage (High Risk, High Reward)
        // Resting -> Sleep (Safe, Low Energy efficient)
        // Transitioning/Scanning -> Explore (Low Risk, Chance of Reward)
        // Contemplative -> Do Nothing (Energy drain)
        
        let (action, intensity) = match state {
            ThoughtState::ActivePlanning => ("FORAGE", 0.8),
            ThoughtState::Resting => ("SLEEP", 0.1),
            ThoughtState::AlertScanning | ThoughtState::Transitioning => ("EXPLORE", 0.5),
            ThoughtState::Contemplative => ("IDLE", 0.2),
        };
        
        // --- D. Environment Response ---
        // Simulate world
        let randomness: f32 = rng.gen();
        let mut feedback_valence = 0.0;
        let mut outcome = "Nothing";
        
        match action {
            "FORAGE" => {
                if randomness > 0.7 {
                    feedback_valence = 0.8; // Found food!
                    body.energy = (body.energy + 0.3).min(1.0);
                    outcome = "FOUND FOOD! (+Energy)";
                } else if randomness < 0.2 {
                    feedback_valence = -0.5; // Hurt!
                    outcome = "THORN PRICK! (-Pain)";
                } else {
                    outcome = "Searching...";
                }
            },
            "SLEEP" => {
                body.energy = (body.energy + 0.05).min(1.0); // Slow recovery
                feedback_valence = 0.1; // Comfort
                outcome = "Resting...";
            },
            "EXPLORE" => {
                if randomness > 0.9 {
                    feedback_valence = 0.4;
                    outcome = "Interesting smell";
                }
            },
            _ => {},
        }
        
        // --- E. Update Body ---
        body.update(intensity, feedback_valence);
        
        // Decay input
        input = input.mapv(|v| v * 0.5);
        
        // Reporting
        println!("{:4} | {:<6.2} | {:<5.2} | {:<7.2} | {:<16} | {:<11} | {}", 
            step, 
            body.energy, 
            body.integrity, 
            body.pleasure_pain, 
            get_icon(&state),
            action,
            outcome
        );
        
        if body.energy <= 0.0 {
            println!("\n💀 DIED OF STARVATION 💀");
            break;
        }
        
        thread::sleep(Duration::from_millis(50));
    }
}

fn get_icon(state: &ThoughtState) -> &str {
    match state {
        ThoughtState::Resting => "💤 Rest",
        ThoughtState::Contemplative => "Co Think",
        ThoughtState::ActivePlanning => "🎯 Plan",
        ThoughtState::AlertScanning => "👁 Scan",
        ThoughtState::Transitioning => "🔄 Trans",
    }
}
