use bdh_model::harmonic::{BiologicalConfig, ThoughtState};
use bdh_model::regions::{BrainRegion, RegionType};
use bdh_model::interpreter::Interpreter;
use ndarray::Array1;
use std::thread;
use std::time::Duration;

fn main() {
    println!("Initializing Inner Voice Simulation...");
    println!("The brain will 'think' and then 'hear' its own thoughts.");

    let mut brain = BrainRegion::new("Cortex", RegionType::Associative, 64, 16, 3);
    let mut interpreter = Interpreter::new();
    
    // Inject initial "Hunger" thought to start the chain
    println!("INITIAL INJECTION: Hunger");
    let hunger_vec = brain.bdh.get_concept_vector("hunger").unwrap();
    let mut input = hunger_vec;

    println!("\nStep | Brain State      | Top Concepts             | Inner Voice (Narrative)");
    println!("-----+------------------+--------------------------+--------------------------------------------------");
    
    for step in 0..20 {
        // 1. Process Input
        let (output, energies) = brain.process(&input);
        let state = brain.bdh.classify_thought_state(&energies);
        
        // 2. Extract Concepts
        // Get top 2 concepts to form a sentence
        let top_concepts_raw = brain.bdh.get_top_concepts(&output, 2);
        // Convert to slice for interpreter
        let top_concepts: Vec<(&str, f32)> = top_concepts_raw.iter().map(|&(n, s)| (n, s)).collect();
        
        // 3. Interpret (Generate Narrative)
        let narrative = interpreter.interpret(&top_concepts);
        
        // 4. Feedback Loop (Hearing)
        // Convert narrative back to vector
        let feedback = interpreter.get_feedback_vector(&narrative, &brain.bdh);
        
        // Next input is the feedback (Self-talk)
        // We decay the old input and add the new "voice"
        input = input.mapv(|v| v * 0.2) + feedback;

        // Display
        // Setup strings for nice formatting
        let concept_str = format!("{}, {}", 
            top_concepts.get(0).map(|x| x.0).unwrap_or("-"),
            top_concepts.get(1).map(|x| x.0).unwrap_or("-")
        );
        
        println!("{:4} | {:<16} | {:<24} | {}", 
            step, 
            get_icon(&state),
            concept_str,
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
