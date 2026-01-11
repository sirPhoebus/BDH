use bdh_model::harmonic::{BiologicalConfig, ThoughtState};
use bdh_model::regions::{BrainRegion, RegionType};
use bdh_model::interpreter::Interpreter;
use ndarray::Array1;
use rand::Rng;
use std::thread;
use std::time::Duration;

fn main() {
    println!("Initializing Metacognition Simulation...");
    println!("The brain monitors its own confidence levels. High noise reduces confidence.");

    let mut brain = BrainRegion::new("Cortex", RegionType::Associative, 64, 16, 3);
    let mut interpreter = Interpreter::new();
    let mut rng = rand::thread_rng();
    
    // Start with a clear thought
    let hunger_vec = brain.bdh.get_concept_vector("planning").unwrap();
    let mut input = hunger_vec;

    println!("\nStep | Noise | Conf  | Brain State      | Inner Voice (Narrative)");
    println!("-----+-------+-------+------------------+--------------------------------------------------");
    
    for step in 0..25 {
        // 1. Process Input
        let (output, energies) = brain.process(&input);
        let state = brain.bdh.classify_thought_state(&energies);
        
        // 2. Metacognition: Calculate Confidence
        let confidence = brain.bdh.get_confidence(&energies);
        
        // 3. Interpret with Confidence
        let top_concepts_raw = brain.bdh.get_top_concepts(&output, 2);
        let top_concepts: Vec<(&str, f32)> = top_concepts_raw.iter().map(|&(n, s)| (n, s)).collect();
        let narrative = interpreter.interpret(&top_concepts, confidence);
        
        // 4. Input Dynamics (Dynamic Noise injection)
        // Steps 0-8: Low noise (Clear thought)
        // Steps 8-16: High noise (Confusion)
        // Steps 16+: Recovery
        let noise_level = if step > 8 && step < 16 { 0.8 } else { 0.1 };
        
        let noise = Array1::from_iter((0..64).map(|_| rng.gen_range(-noise_level..noise_level)));
        
        // Feedback loop + Noise
        let feedback = interpreter.get_feedback_vector(&narrative, &brain.bdh);
        input = input.mapv(|v| v * 0.5) + feedback + noise;

        println!("{:4} | {:<5.1} | {:<5.2} | {:<16} | {}", 
            step, 
            noise_level,
            confidence,
            get_icon(&state),
            narrative
        );
        
        thread::sleep(Duration::from_millis(150));
    }
}

fn get_icon(state: &ThoughtState) -> &str {
    match state {
        ThoughtState::Resting => "💤 Resting",
        ThoughtState::Contemplative => "🧘 Contemplative",
        ThoughtState::ActivePlanning => "🎯 Planning",
        ThoughtState::AlertScanning => "👁 Scanning",
        ThoughtState::Transitioning => "🔄 Trans",
    }
}
