use bdh_model::harmonic_burn::HarmonicBdhBurn;
use bdh_model::body::BodyState;
use bdh_model::drives::Drives;
use bdh_model::interpreter::Interpreter;
use burn_wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use std::time::Duration;
use std::thread;
use ndarray::Array1;
use std::collections::VecDeque;
use serde::Serialize;
use axum::{
    routing::get,
    response::IntoResponse,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    Router,
};
use tower_http::services::ServeDir;
use tokio::sync::broadcast;
use std::sync::Arc;

#[derive(Serialize, Clone)]
struct BrainStateUpdate {
    energies: Vec<f32>,
    dopamine: f32,
    norepinephrine: f32,
    acetylcholine: f32,
    input_word: String,
    prediction: String,
    inner_voice: String,        // [NEW] Narrative thought
    memory_event: Option<Vec<f32>>,
}

/// A simple episodic memory of a high-affect event.
#[derive(Clone)]
#[allow(dead_code)]
struct EpisodicMemory {
    embedding: Array1<f32>,
    concept: String,
    valence: f32, // How strongly it was felt
}

struct EpisodicBuffer {
    memories: VecDeque<EpisodicMemory>,
    capacity: usize,
}

impl EpisodicBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            memories: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn add(&mut self, embedding: Array1<f32>, concept: String, valence: f32) {
        if self.memories.len() >= self.capacity {
            self.memories.pop_front(); // Forget oldest
        }
        self.memories.push_back(EpisodicMemory { embedding, concept, valence });
    }

    fn sample(&self) -> Option<EpisodicMemory> {
        if self.memories.is_empty() { return None; }
        // Simple random sample
        let idx = rand::random::<usize>() % self.memories.len();
        self.memories.get(idx).cloned()
    }
}

#[tokio::main]
async fn main() {
    println!("Initializing CENTRAL SIM (Grand Unification)...");
    
    // 0. Setup Networking
    let (tx, _rx) = broadcast::channel::<BrainStateUpdate>(16);
    let tx_shared = Arc::new(tx);
    
    let app_tx = tx_shared.clone();
    let app = Router::new()
        .route("/ws", get(move |ws| ws_handler(ws, app_tx)))
        .fallback_service(ServeDir::new("viz"));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("   Visualizer server running at http://localhost:3000");
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

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
    
    // Train BPE Embedder
    println!("   Training BPE Embedder on {} books (Scalable)...", books.len());
    let embedder = bdh_model::data::Embedder::train_on_corpus(&books, 8000, n_neurons).unwrap(); // 8k vocab
    println!("   Embedder ready. Vocab size: {}", embedder.vocab_size);
    
    // Initialize Reader State
    let mut episodic_buffer = EpisodicBuffer::new(50);
    let mut current_book_idx = 0;
    let mut current_tokens: Vec<u32> = embedder.tokenize(&books[0]);
    let mut token_ptr = 0;
    
    // 6. Concept Memory (Restored)
    let mut concept_memory: Vec<(String, Array1<f32>)> = Vec::new();
    
    // Semantic Grounding: Enrich vocabulary of thoughts
    let concepts_to_add = vec![
        ("Safety", "safety calm secure protect"),
        ("Danger", "danger pain threat hurt"),
        ("Mystery", "mystery unknown puzzle hidden"),
        ("Action", "action move fast run jump"),
        ("Thought", "thought dream mind think logic"),
        ("Machine", "machine metal gear clock electric"),
        ("Life", "life grow organic leaf breath"),
    ];

    for (name, text) in concepts_to_add {
        let vec = embedder.embed_text(text).row(0).to_owned();
        concept_memory.push((name.to_string(), vec.to_vec().into()));
    }
    
    println!("\nSIMULATION STARTING. Reading: Book 0\n");
    println!("Step | Energy | Valnc | Drive  | Conf | Input Word      | Prediction      | Inner Voice");
    println!("-----+--------+-------+--------+------+-----------------+-----------------+-------------------------------------");

    let mut step = 0;
    
    // 7. Goal Vectors (Drive Targets)
    // We create "Attractors" for drives.
    // "Food" for Hunger. "Mystery" for Curiosity.
    let food_vec = embedder.embed_text("food eat delicious satisfy").row(0).to_owned();
    let novelty_vec = embedder.embed_text("mystery strange new discover").row(0).to_owned();
    let food_vec = Tensor::<Backend, 1>::from_floats(food_vec.to_vec().as_slice(), &device);
    let novelty_vec = Tensor::<Backend, 1>::from_floats(novelty_vec.to_vec().as_slice(), &device);

    let mut last_narrative = "".to_string();
    let mut is_sleeping = false;
    let mut current_memory_event: Option<Vec<f32>> = None;

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
        
        // --- C. Brain Modulation (Neurochemical) ---
        // 1. Update Neurotransmitters
        let da = body.chemicals.dopamine;
        let ne = body.chemicals.norepinephrine;
        let ach = body.chemicals.acetylcholine;
        
        // 2. Modulate Physics
        cortex.modulate_neurochem(da, ne, ach);
        
        // Sleep frequency modulation (Low ACh)
        if ach < 0.2 {
             cortex.modulate_frequencies(0.5);
        } else {
             cortex.modulate_frequencies(1.0);
        }
        
        // Drive Modulation (shifts targets, physics handled by chemicals now)
        let _modulation = drives.get_modulation();
        // We override damping from neurochem if drive is strong?
        // Let's let Neurochem be the final arbiter of physics.
        
        // Calculate Bias Tensor here to be available for both Reading and Dreaming (if needed)
        // 2. Drive Goal Injection (Bias)
        // Instead of just damping, we inject a "Wanting" vector.
        let mut bias_input: Option<Tensor<Backend, 1>> = None;
        
        match current_drive.as_deref() {
            Some("HUNGER") => {
                 bias_input = Some(food_vec.clone());
            },
            Some("CURIOSITY") => {
                 bias_input = Some(novelty_vec.clone());
            },
            _ => {},
        }
        
        // Prepare final bias tensor (Goal + Echo)
        // Echo: Feed back last narrative weakly
        let mut total_bias_vec = if let Some(ref b) = bias_input {
             b.clone().mul_scalar(0.2) // Weak goal
        } else {
             Tensor::<Backend, 1>::zeros([n_neurons], &device)
        };

        // Echo Loop (Working Memory)
        // Every step, we hear a faint echo of our last thought.
        if !last_narrative.is_empty() && step % 5 == 0 {
             let echo_vec = interpreter.reflect(&last_narrative, &embedder);
             let echo_tensor = Tensor::<Backend, 1>::from_floats(echo_vec.to_vec().as_slice(), &device);
             total_bias_vec = total_bias_vec.add(echo_tensor.mul_scalar(0.1)); // Very weak echo
        }

        // Expand Bias for Addition
        let bias_3d = total_bias_vec.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);


        // --- D. Agency / Action Selection (Homeostatic + Conflict) ---
        
        // 1. Perception/Appraisal
        // Check alignment with Drives (Foraging)
        let cortical_out = cortex.get_cortical_output();
        let state_vec = Array1::from(cortical_out.clone());
        let goal_match = if let Some(ref goal) = bias_input {
             // Approximate dot product with Goal Vector
             // (Assuming normalized vectors, this is cosine sim)
             let goal_vec: Vec<f32> = goal.clone().into_data().iter::<f32>().collect();
             let goal_arr = Array1::from_vec(goal_vec);
             state_vec.dot(&goal_arr)
        } else {
             0.0 
        };

        #[derive(PartialEq, Clone, Copy)]
        enum Action {
            Read,
            Skip,      
            Reflect,   
            Dream,     
        }

        // Hysteresis Logic for Sleep/Wake
        // We use the 'is_sleeping' flag to stay asleep until recharged
        if body.energy < 0.35 {
             is_sleeping = true;
        } else if body.energy > 0.85 {
             is_sleeping = false;
        }

        // 2. Policy (Homeostatic + Conflict Resolution)
        let action = if is_sleeping {
             Action::Dream
        } else if body.chemicals.norepinephrine > 0.9 {
             Action::Skip 
        } else if drives.hunger > 0.85 && goal_match < 0.2 {
             // FORAGING MODE
             Action::Skip 
        } else if drives.curiosity > 0.8 && step % 50 == 0 {
             Action::Reflect
        } else {
             Action::Read
        };
        
        let mut input_word = "".to_string();
        let mut prediction = "".to_string();
        
        match action {
             Action::Dream => {
                // --- DREAMING (Episodic Replay + Chaining) ---
                prediction = "(Internal)".to_string();
                
                // Higher chance for a memory if we are stuck
                let replay_trigger = rand::random::<f32>() < 0.3; // 30% jump chance
                
                let input_tensor = if replay_trigger {
                    if let Some(mem) = episodic_buffer.sample() {
                         input_word = format!("*{}*", mem.concept);
                         Tensor::<Backend, 1>::from_floats(mem.embedding.to_vec().as_slice(), &device)
                    } else {
                        // Fallback to Chaining
                        let output_arr = Array1::from(cortex.get_cortical_output());
                        let (next_token, _) = embedder.decode_nearest(&output_arr);
                        input_word = format!("~{}~", next_token);
                        
                        let self_talk = format!("I imagine {}", next_token);
                        let feedback_vector = interpreter.reflect(&self_talk, &embedder);
                        Tensor::<Backend, 1>::from_floats(feedback_vector.to_vec().as_slice(), &device)
                    }
                } else {
                    // Standard Associative Chaining
                    let output_arr = Array1::from(cortex.get_cortical_output());
                    let (mut next_token, _) = embedder.decode_nearest(&output_arr);
                    
                    // Anti-Loop: If word is "imagine" or too frequent, inject a random concept
                    if next_token == "imagine" || next_token == "Safety" || next_token == "t" {
                         let (rand_token, _) = embedder.get_random_token();
                         next_token = rand_token;
                    }

                    input_word = format!("~{}~", next_token);
                    let self_talk = format!("I feel {}", next_token);
                    let feedback_vector = interpreter.reflect(&self_talk, &embedder);
                    Tensor::<Backend, 1>::from_floats(feedback_vector.to_vec().as_slice(), &device)
                };

                // Apply Input
                let input_3d = input_tensor.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);
                cortex.step(Some(input_3d));
                
                // Dreaming recovers Energy significantly to break the fast-cycle flutter
                body.energy = (body.energy + 0.1).min(1.0); // FASTER RECOVERY
             },
             
             Action::Reflect => {
                 // --- REFLECTION ---
                 if !last_narrative.is_empty() {
                      input_word = "self".to_string();
                      let echo_vec = interpreter.reflect(&last_narrative, &embedder);
                      let echo_tensor = Tensor::<Backend, 1>::from_floats(echo_vec.to_vec().as_slice(), &device);
                      let input_3d = echo_tensor.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);
                      cortex.step(Some(input_3d));
                 } else {
                     cortex.step(None); 
                 }
             },
             
             Action::Skip => {
                 // --- SKIPPING (Foraging) ---
                 input_word = ">>".to_string();
                 token_ptr += 10;
                 if token_ptr >= current_tokens.len() { token_ptr = 0; } 
                 body.energy = (body.energy - 0.002).max(0.0);
                 cortex.step(None);
             },
             
             Action::Read => {
                 // --- READING ---
                // 1. Prediction
                let cortical_out = cortex.get_cortical_output();
                let out_vec = Array1::from(cortical_out);
                let (predicted_token, _confidence) = embedder.decode_nearest(&out_vec);
                prediction = predicted_token;
    
                // 2. Read actual word
                if token_ptr < current_tokens.len() {
                    let token_id = current_tokens[token_ptr];
                    input_word = embedder.decode(&[token_id]);
                    
                    let emb = embedder.embed_token(token_id);
                    let emb_vec = emb.to_vec();
                    let input_tensor = Tensor::<Backend, 1>::from_floats(emb_vec.as_slice(), &device);
                    let input_3d = input_tensor.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);
                    
                    let final_input = input_3d.add(bias_3d);
                    cortex.step(Some(final_input));
                    
                    body.energy = (body.energy - 0.001).max(0.0); // 5x LONGER ATTENTION SPAN
                    token_ptr += 1;
                } else {
                    cortex.step(None);
                    input_word = "<END>".to_string();
                    // Keep Hunger book-switching logic? Yes.
                    if drives.hunger > 0.8 {
                        current_book_idx = (current_book_idx + 1) % books.len();
                        current_tokens = embedder.tokenize(&books[current_book_idx]);
                        token_ptr = 0;
                        println!("--> Opening Book {}...", current_book_idx);
                    }
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

            // --- EMBODIMENT LOOP (Refined) ---
            // 1. Concept-based Valence (Safety/Danger)
            let mut env_valence = match best_concept {
                "Safety" => 0.2, 
                "Danger" => -0.3,
                _ => 0.0,
            };
            
            // 2. Epistemic Valence (Coherence/Confusion)
            // High Confidence = Pleasure (Flow). Low Confidence = Anxiety.
            let epistemic_reward = (confidence - 0.5) * 0.1; 
            env_valence += epistemic_reward;
            
            // 3. Goal Alignment (if Hunger, finding Food = Reward)
            // Just use simple "Novelty Reward" for now if Curiosity is high
            if drives.curiosity > 0.6 && input_word != "<unk>" && action == Action::Read {
                 env_valence += 0.3; // BOOSTED
            }

            // 4. Intrinsic Reward (Dreaming/Chaining)
            if action == Action::Dream {
                 env_valence += 0.1; // BOOSTED
                 // Bonus if we actually recalled a memory (Look for '*' markers)
                 if input_word.contains('*') {
                     env_valence += 0.4; // RECALL BONUS
                 }
            }

            if env_valence != 0.0 {
                 body.update(0.1, env_valence);
            }
            
            // --- MEMORY ENCODING ---
            if body.pleasure_pain.abs() > 0.5 || body.arousal > 0.8 {
                 let cortical_out = cortex.get_cortical_output();
                 let state_vec = Array1::from(cortical_out);
                 current_memory_event = Some(state_vec.to_vec());
                 episodic_buffer.add(state_vec, best_concept.to_string(), body.pleasure_pain);
            }
            // -----------------------

            let concepts = vec![(best_concept, best_score)];
            let narrative = interpreter.interpret(&concepts, confidence);
            last_narrative = narrative.clone();

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
        
        

            // --- BROADCAST STATE (for 3D viz) ---
            // Throttled to prevent saturating websocket at high speeds
            if step % 10 == 0 {
                let current_energies = cortex.get_neuron_energies(); 
                let update = BrainStateUpdate {
                    energies: current_energies,
                    dopamine: body.chemicals.dopamine,
                    norepinephrine: body.chemicals.norepinephrine,
                    acetylcholine: body.chemicals.acetylcholine,
                    input_word: input_word.clone(),
                    prediction: prediction.clone(),
                    inner_voice: last_narrative.clone(), // Broadcast narrative
                    memory_event: current_memory_event,
                };
                let _ = tx_shared.send(update);
                current_memory_event = None; // Clear after broadcast
            }

        // Auto-Save
        if step % 100 == 0 {
             let _ = cortex.save_state("brain.state");
        }

        // Read speed - REMOVED SLEEP FOR OVERCLOCKING
        // thread::sleep(Duration::from_millis(10));
    }
}

async fn ws_handler(ws: WebSocketUpgrade, tx: Arc<broadcast::Sender<BrainStateUpdate>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, tx))
}

async fn handle_socket(mut socket: WebSocket, tx: Arc<broadcast::Sender<BrainStateUpdate>>) {
    let mut rx = tx.subscribe();
    while let Ok(update) = rx.recv().await {
        let msg = serde_json::to_string(&update).unwrap();
        if socket.send(Message::Text(msg.into())).await.is_err() {
            break;
        }
    }
}
