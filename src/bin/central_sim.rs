use bdh_model::harmonic_burn::HarmonicBdhBurn;
use bdh_model::body::BodyState;
use bdh_model::drives::Drives;
use bdh_model::interpreter::Interpreter;
use burn_wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use std::time::Duration;
use std::thread;
use ndarray::Array1;

fn main() {
    println!("Initializing CENTRAL SIM (Grand Unification)...");
    
    // 1. Setup GPU Backend
    type Backend = Wgpu<f32, i32>;
    let device = WgpuDevice::BestAvailable;
    let n_neurons = 1000; // Can scale to 10k, using 1k for nice console visualization
    let d = 16;
    let layers = 4;
    
    println!("1. Cortex: Booting GPU Kernel ({} neurons)...", n_neurons);
    let mut cortex: HarmonicBdhBurn<Backend> = HarmonicBdhBurn::new(n_neurons, d, layers, &device);
    
    if std::path::Path::new("brain.state").exists() {
        println!("   found existing brain state! Loading...");
        match cortex.load_state("brain.state") {
            Ok(_) => println!("   State loaded successfully."),
            Err(e) => println!("   Failed to load state: {}", e),
        }
    } else {
        println!("   No saved state found. Starting fresh.");
    }
    
    // 2. Setup Physiology
    println!("2. Body: Initializing homeostatic systems...");
    let mut body = BodyState::new();
    
    // 3. Setup Motivation
    println!("3. Drives: Calibrating intrinsic motivators...");
    let mut drives = Drives::new();
    
    // 4. Setup Interpreter
    println!("4. Interpreter: Linking Wernicke's Area...");
    let mut interpreter = Interpreter::new();
    
    // 5. Data Pipeline
    println!("5. Data: Loading corpus from data/books...");
    let books = bdh_model::data::load_texts_from_dir("data/books").unwrap();
    if books.is_empty() {
        panic!("No books found! Run get_books first.");
    }
    
    // Train Embedder
    println!("   Training Embedder on {} books...", books.len());
    let embedder = bdh_model::data::Embedder::train_on_corpus(&books, 5000, n_neurons).unwrap();
    println!("   Embedder ready. Vocab size: {}", embedder.vocab_size);
    
    // Initialize Reader State
    let mut current_book_idx = 0;
    let mut current_tokens: Vec<u32> = embedder.tokenize(&books[0]);
    let mut token_ptr = 0;
    
    // 6. Concept Memory (Restored)
    let mut concept_memory: Vec<(String, Array1<f32>)> = Vec::new();
    
    // Semantic Grounding: Use the Embedder to find real vectors for concepts
    let safety_vec = embedder.embed_text("safety calm secure protect").row(0).to_owned();
    let danger_vec = embedder.embed_text("danger pain threat hurt").row(0).to_owned();
    
    // Normalize? Embedder output is usually not normalized.
    // Let's assume using them raw is fine for dot product scoring if consistent.
    concept_memory.push(("Safety".to_string(), safety_vec.to_vec().into()));
    concept_memory.push(("Danger".to_string(), danger_vec.to_vec().into()));
    
    println!("\nSIMULATION STARTING. Reading: Book 0\n");
    println!("Step | Energy | Valnc | Drive  | Conf | Input Word      | Prediction      | Inner Voice");
    println!("-----+--------+-------+--------+------+-----------------+-----------------+-------------------------------------");

    let mut step = 0;
    
    loop {
        step += 1;
        
        // --- A. Body Update (Metabolism) ---
        body.update(0.005, 0.0); // Constant decay, energy comes from DATA processing (simulated below)
        
        // --- B. Drive Update (Data Hunger) ---
        // Hunger increases if we run out of tokens or haven't read in a while.
        // Simplified: Hunger is inverse of Energy.
        // Construct Novelty: If the current word is "rare" (high ID), it's novel.
        let is_novel = if token_ptr < current_tokens.len() {
             current_tokens[token_ptr] > 1000 // Rare words are > 1000
        } else { false };
        let novelty = if is_novel { 0.8 } else { 0.1 };
        
        drives.update(body.energy, novelty);
        let current_drive = drives.get_dominant_drive();
        
        // --- C. Brain Modulation ---
        // Sleep/Wake cycle based on energy
        if body.energy < 0.2 {
            cortex.modulate_frequencies(0.5); 
        } else {
            cortex.modulate_frequencies(1.0);
        }
        
        // Drive Modulation
        let modulation = drives.get_modulation();
        cortex.set_damping(modulation.damping_target);
        
        // --- Body Modulation (Embodiment) ---
        // Combine Drive Noise Bias with Body Pain
        let valence = body.pleasure_pain;
        let pain_noise = if valence < -0.1 { 0.2 * valence.abs() } else { 0.0 };
        // Base noise is Drive Bias + Pain Effect
        // If Pleasure (valence > 0.1), we reduce noise further (focus)
        let pleasure_damping = if valence > 0.1 { 0.5 } else { 1.0 };
        
        let pleasure_damping = if valence > 0.1 { 0.5 } else { 1.0 };
        
        cortex.set_noise((modulation.noise_bias + pain_noise) * pleasure_damping);
        
        // Plasticity Modulation via Arousal
        // High Arousal = High Gain (Alert, taking in info)
        // Low Arousal = Low Gain (Sleepy, ignore stats)
        cortex.set_input_gain(body.arousal.max(0.1));


        // --- D. Input Processing (Reading OR Dreaming) ---
        let input_word: String;
        let mut prediction = "".to_string();
        let mut is_dreaming = false;
        
        // DREAM MODE TRIGGER:
        // Use Drives: If seeking/playing/resting, we might dream if no external input
        // For now, keep the simple energy trigger + Drive trigger
        if body.energy < 0.2 || (current_drive.is_none() && step > 100) {
             is_dreaming = true;
        }
        
        if is_dreaming {
            // --- DREAMING (Reflection Loop) ---
            // 1. Interpret current state -> Narrative
            let cortical_out = cortex.get_cortical_output();
            let output_arr = Array1::from(cortical_out);
            
            // Re-use logic for best concept (duplicated from below, maybe refactor later)
            // Ideally we interpret what we *just* saw.
            // For loop closure: We need a narrative to reflect on.
            // Let's use the PREVIOUS narrative (stored in interpreter history?) 
            // Or generate a new one now.
            
            // Let's decode nearest token for simplicity of "Input Word" display
            let (next_token, _conf) = embedder.decode_nearest(&output_arr);
            input_word = next_token.clone();
            prediction = format!("(Reflecting) {}", input_word);
            
            // 2. Reflect: Narrative -> Vector
            // We construct a narrative string "I notice X"
            let self_talk = format!("I notice {}", input_word);
            let feedback_vector = interpreter.reflect(&self_talk, &embedder);
            
            // 3. Feed it back
             let emb_vec = feedback_vector.to_vec();
             let input_tensor = Tensor::<Backend, 1>::from_floats(emb_vec.as_slice(), &device);
             let input_3d = input_tensor.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);
            
            cortex.step(Some(input_3d));
            
            // Dreaming cost
            body.energy = (body.energy - 0.001).max(0.0);
            
        } else {
            // --- READING ---
            // 1. Prediction: Before seeing the word, what did we expect?
            // Decode *previous* step's output (which is current state before new input)
            let cortical_out = cortex.get_cortical_output();
            let out_vec = Array1::from(cortical_out);
            let (predicted_token, _conf) = embedder.decode_nearest(&out_vec);
            prediction = predicted_token;

            // 2. Read actual word
            if token_ptr < current_tokens.len() {
                let token_id = current_tokens[token_ptr];
                input_word = embedder.decode(&[token_id]);
                
                // Get embedding [Neurons]
                let emb = embedder.embed_token(token_id); // Array1<f32>
                let emb_vec = emb.to_vec();
                
                let input_tensor = Tensor::<Backend, 1>::from_floats(emb_vec.as_slice(), &device);
                // Reshape [Neurons] -> [1, 1, Neurons] -> Broadcast
                let input_3d = input_tensor.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);
                
                cortex.step(Some(input_3d));
                
                body.energy = (body.energy + 0.01).min(1.0);
                token_ptr += 1;
            } else {
                cortex.step(None);
                input_word = "<END>".to_string();
                
                if drives.hunger > 0.8 {
                    current_book_idx = (current_book_idx + 1) % books.len();
                    current_tokens = embedder.tokenize(&books[current_book_idx]);
                    token_ptr = 0;
                    println!("--> Opening Book {}...", current_book_idx);
                }
            }
        }
        
        // --- E. Readout ---
        if step % 20 == 0 {
            let cortical_out = cortex.get_cortical_output();

            let energies = cortex.get_energy().to_data().to_vec::<f32>().unwrap();
            
            // Confidence (Entropy)
            let sum_e: f32 = energies.iter().sum();
            let entropy = if sum_e > 0.001 {
                energies.iter().map(|e| {
                    let p = e / sum_e;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                }).sum::<f32>()
            } else { 0.0 };
            let max_entropy = (layers as f32).ln().max(1.0);
            let confidence = (1.0 - (entropy / max_entropy)).max(0.0).min(1.0);
            
            // Interpretation
            let output_arr = Array1::from(cortical_out);
            let mut best_concept = "Void";
            let mut best_score = 0.0;
            
            // Match against concepts (initially empty or seeded)
            for (name, vec) in &concept_memory {
                let score = output_arr.dot(vec);
                if score.abs() > best_score {
                    best_score = score.abs();
                    best_concept = name;
                }
            }
            // Also match against current input word?
            // "Learning": If confidence is high but concept is unknown, make new concept.
            
            // Self-Discovery / Association
            // If we have high energy (focus) but don't recognize the pattern (low score),
            // we should associate the *Current Input Word* with this Brain State.
            // DEBUG: Print why we aren't learning
            if best_score < 0.8 && sum_e > 0.5 && input_word.len() > 2 && input_word != "<unk>" {
                 // println!("   [Learning] Score: {:.2}, SumE: {:.2}. Associating '{}'", best_score, sum_e, input_word);
                 concept_memory.push((input_word.clone(), output_arr.clone()));
                 // Update best_concept immediately for this step
                 best_concept = &concept_memory.last().unwrap().0;
                 best_score = 1.0; 
            } else if input_word.len() > 2 {
                 // println!("   [No Learn] Score: {:.2} (Concept: {}), SumE: {:.2}", best_score, best_concept, sum_e);
            }

            // --- EMBODIMENT LOOP ---
            let env_valence = match best_concept {
                "Safety" => 0.2, // Pleasure
                "Danger" => -0.5, // Pain
                _ => 0.0, // Neutral
            };
            if env_valence != 0.0 {
                body.update(0.1, env_valence);
            }
            // -----------------------

            let concepts = vec![(best_concept, best_score)];
            let narrative = interpreter.interpret(&concepts, confidence);

            println!("{:4} | {:<4.2}   | {:>+5.2} | {:<6} | {:<4.2} | {:<15} | {:<15} | {}", 
                step, 
                body.energy, 
                body.pleasure_pain,
                current_drive.unwrap_or("-".to_string()), 
                confidence,
                input_word.chars().take(15).collect::<String>(),
                prediction.chars().take(15).collect::<String>(),
                narrative
            ); 

        }
        
        
        // Auto-Save
        if step % 100 == 0 {
             let _ = cortex.save_state("brain.state");
        }

        // Read speed
        thread::sleep(Duration::from_millis(10));
    }
}
