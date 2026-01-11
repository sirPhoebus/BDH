use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use bdh_model::ChronosBdh;
use rand::thread_rng;

fn main() {
    let n = 256;
    let d = 64;

    let freq = 0.5; // Heartbeat freq
    let num_layers = 1;

    let mut chronos = ChronosBdh::new(n, d, freq, num_layers);
    
    // Associative Chaining & Stabilization Config
    chronos.rho_decay = 1.0;            // No decay, rely on clamping
    chronos.associative_momentum = 0.02; // Reduced momentum

    
    println!("Starting Associative Chaining Experiment (N={}, D={})", n, d);

    // Create Concept Vectors for tracking "thoughts"
    let num_concepts = 10;
    let mut concepts = Vec::new();
    let mut rng = thread_rng();
    for i in 0..num_concepts {
        let vec = Array1::random_using(n, Uniform::new(0.0f32, 1.0f32), &mut rng);
        let norm = vec.dot(&vec).sqrt();
        concepts.push((format!("Concept_{}", i), vec / norm));
    }

    let steps = 10_000;
    let mut recall_count = 0;
    let mut contemplative_steps = 0;
    let mut active_concept: Option<usize> = None;
    let mut concept_transitions = 0;
    let mut low_energy_steps = 0;
    
    // Time step size
    let dt = 0.1; 

    // Zero input for autonomous running (except internal recalls)
    let mut current_input = Array1::zeros(n);
    let feedback_strength = 0.95; // Strong feedback for continuity


    // LEARNING PHASE (Sequence Imprinting A -> B -> C)
    println!("Imprinting Sequence of {} concepts...", num_concepts);
    
    // 1. Individual Imprinting (Base representations)
    for (_, vec) in &concepts {
        for _ in 0..10 {
            chronos.step_time(dt);
            let _ = chronos.forward_with_time(vec, 0);
        }
    }
    
    // 2. Transition Imprinting (A -> B)
    // Overlap previous concept with next concept to build Hebbian links
    for i in 0..(num_concepts - 1) {
        let vec_a = &concepts[i].1;
        let vec_b = &concepts[i+1].1;
        
        println!("Linking {} -> {}", concepts[i].0, concepts[i+1].0);
        
        for _ in 0..10 {
            chronos.step_time(dt);
            // Superposition input: Strong B + Fading A
            let input = vec_b + &(vec_a * 0.5); 
            let _ = chronos.forward_with_time(&input, 0);
        }
    }

    println!("Learning Complete. Starting Free-Run...");


    println!("Step, Time, Energy, State, ActiveConcept");

    for step in 0..steps {
        chronos.step_time(dt);
        let output = chronos.forward_with_time(&current_input, 0);
        
        // Recurrence: Next input is dampened output + tanh to prevent explosion
        current_input = (&output * feedback_strength).mapv(|x| x.tanh());

        
        // Metrics
        let energy = output.dot(&output).sqrt();
        
        // 1. Recalls (Now Heartbeat-Modulated Noise)
        if chronos.last_recall_triggered {
            recall_count += 1;
        }

        // 2. Contemplative State (Low-Medium Energy)
        // Adjust thresholds based on observation
        let state = if energy < 0.1 {
            low_energy_steps += 1;
            "Resting"
        } else if energy < 2.0 { // Arbitrary broad range for "thinking"
            contemplative_steps += 1;
            "Contemplative"
        } else {
            "Active"
        };
        
        // 3. Concept Chains
        let output_norm = energy.max(1e-6);
        let normalized_output = &output / output_norm;
        
        let mut max_sim = 0.0;
        let mut best_concept = None;
        
        for (i, (_, vec)) in concepts.iter().enumerate() {
            let sim = normalized_output.dot(vec);
            if sim > max_sim {
                max_sim = sim;
                best_concept = Some(i);
            }
        }
        
        
        // Lower threshold for "recognizing" a concept
        let concept_str = if max_sim > 0.35 { 
            if active_concept != best_concept {
                concept_transitions += 1;

                active_concept = best_concept;
            }
            format!("{} ({:.2})", concepts[best_concept.unwrap()].0, max_sim)
        } else {
            format!("None ({:.2})", max_sim)
        };


        // Sparse logging
        if step % 100 == 0 {
            println!("{}, {:.1}, {:.4}, {}, {}", step, chronos.global_time, energy, state, concept_str);
        }
    }

    println!("--- Experiment Results ---");
    println!("Total Steps: {}", steps);
    println!("Recalls Triggered (Systole noise): {} ({:.2}%)", recall_count, (recall_count as f32 / steps as f32) * 100.0);
    println!("Contemplative Steps: {} ({:.2}%)", contemplative_steps, (contemplative_steps as f32 / steps as f32) * 100.0);
    println!("Concept Transitions: {}", concept_transitions);
    println!("Low Energy Steps: {}", low_energy_steps);
}
