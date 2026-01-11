use bdh_model::harmonic::{BiologicalConfig, ThoughtState};
use bdh_model::regions::{BrainRegion, RegionType};
use bdh_model::interpreter::Interpreter;
use ndarray::Array1;
use std::thread;
use std::time::Duration;

fn main() {
    println!("Initializing Self-Discovery Simulation...");
    println!("The brain encounters a new, repeated pattern and crystallizes it into a concept.");

    let mut brain = BrainRegion::new("Cortex", RegionType::Associative, 64, 16, 3);
    let mut interpreter = Interpreter::new();
    
    // Create a stable "Mystery Object" pattern (e.g., a specific sensory input)
    let mystery_pattern = Array1::from_iter((0..64).map(|i| (i as f32 * 0.5).sin()));
    let normal_pattern = Array1::zeros(64);
    
    let mut input = normal_pattern;

    println!("\nStep | Input Type | Learned? | Inner Voice (Narrative)");
    println!("-----+------------+----------+--------------------------------------------------");
    
    for step in 0..25 {
        // Feed the mystery pattern from step 5-15
        let input_type = if step > 5 && step < 20 {
            input = mystery_pattern.clone();
            "MYSTERY"
        } else {
            input = input.mapv(|v| v * 0.9); // Decay
            "Void"
        };
        
        // 1. Process
        let (output, energies) = brain.process(&input);
        
        // 2. Self-Discovery Learning
        // Only learn if signal is strong (energy high)
        let total_energy: f32 = energies.iter().sum();
        let new_concept = if total_energy > 0.1 {
            brain.bdh.learn_new_concept(&output)
        } else {
            None
        };
        
        // 3. Interpret
        let top_concepts_raw = brain.bdh.get_top_concepts(&output, 2);
        let top_concepts: Vec<(&str, f32)> = top_concepts_raw.iter().map(|&(n, s)| (n, s)).collect();
        let narrative = interpreter.interpret(&top_concepts, 0.9); // Assume high confidence for demo clarity

        println!("{:4} | {:<10} | {:<8} | {}", 
            step, 
            input_type,
            new_concept.clone().unwrap_or("-".to_string()),
            narrative
        );
        
        thread::sleep(Duration::from_millis(150));
    }
}
