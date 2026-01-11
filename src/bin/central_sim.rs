use bdh_model::harmonic_burn::HarmonicBdhBurn;
use bdh_model::data::acquisition::{download_gutenberg_book, GUTENBERG_IDS};
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

/// Learning quality metrics - tracks if the brain is actually improving
struct LearningMetrics {
    // Track which concepts the brain recognizes (diversity measure)
    concept_counts: std::collections::HashMap<String, usize>,
    total_recognitions: usize,
    
    // Semantic coherence over time (embedding similarity)
    coherence_samples: Vec<f32>,
    
    // Novel learning: how many new concepts discovered
    concepts_at_start: usize,
    concepts_now: usize,
    
    // Dream/Read ratio (healthy = balanced)
    dream_steps: usize,
    read_steps: usize,
    
    // Energy stability (learning requires sustained attention)
    energy_samples: Vec<f32>,
    
    // History for trend analysis
    coherence_history: Vec<(usize, f32)>,  // (step, avg_coherence)
    
    // Episodic memory engagement
    memory_recalls: usize,
    memory_stores: usize,
}

impl LearningMetrics {
    fn new(initial_concepts: usize) -> Self {
        Self {
            concept_counts: std::collections::HashMap::new(),
            total_recognitions: 0,
            coherence_samples: Vec::new(),
            concepts_at_start: initial_concepts,
            concepts_now: initial_concepts,
            dream_steps: 0,
            read_steps: 0,
            energy_samples: Vec::new(),
            coherence_history: Vec::new(),
            memory_recalls: 0,
            memory_stores: 0,
        }
    }
    
    fn record_concept(&mut self, concept: &str) {
        *self.concept_counts.entry(concept.to_string()).or_insert(0) += 1;
        self.total_recognitions += 1;
    }
    
    fn record_coherence(&mut self, score: f32, step: usize) {
        self.coherence_samples.push(score);
        if self.coherence_samples.len() > 100 {
            self.coherence_samples.remove(0);
        }
        
        // Record history every 100 steps
        if step > 0 && step % 100 == 0 && !self.coherence_samples.is_empty() {
            let avg = self.coherence_samples.iter().sum::<f32>() / self.coherence_samples.len() as f32;
            self.coherence_history.push((step, avg));
        }
    }
    
    fn record_energy(&mut self, energy: f32) {
        self.energy_samples.push(energy);
        if self.energy_samples.len() > 100 {
            self.energy_samples.remove(0);
        }
    }
    
    fn record_action(&mut self, is_reading: bool) {
        if is_reading { self.read_steps += 1; } else { self.dream_steps += 1; }
    }
    
    fn record_concept_learned(&mut self) {
        self.concepts_now += 1;
    }
    
    fn record_memory_store(&mut self) { self.memory_stores += 1; }
    fn record_memory_recall(&mut self) { self.memory_recalls += 1; }
    
    fn concept_diversity(&self) -> f32 {
        if self.total_recognitions == 0 { return 0.0; }
        let unique = self.concept_counts.len() as f32;
        let total = self.total_recognitions as f32;
        // Entropy-like measure: higher = more diverse, max = 1.0
        unique / total.sqrt()
    }
    
    fn avg_coherence(&self) -> f32 {
        if self.coherence_samples.is_empty() { return 0.0; }
        self.coherence_samples.iter().sum::<f32>() / self.coherence_samples.len() as f32
    }
    
    fn coherence_trend(&self) -> f32 {
        if self.coherence_history.len() < 3 { return 0.0; }
        let n = self.coherence_history.len();
        let first = self.coherence_history[0].1;
        let last = self.coherence_history[n - 1].1;
        last - first
    }
    
    fn learning_rate(&self) -> f32 {
        (self.concepts_now - self.concepts_at_start) as f32
    }
    
    fn energy_stability(&self) -> f32 {
        if self.energy_samples.len() < 2 { return 0.0; }
        let mean: f32 = self.energy_samples.iter().sum::<f32>() / self.energy_samples.len() as f32;
        let variance: f32 = self.energy_samples.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / self.energy_samples.len() as f32;
        1.0 / (1.0 + variance.sqrt()) // Higher = more stable
    }
    
    fn print_benchmark(&self, step: usize) {
        let trend = self.coherence_trend();
        let trend_symbol = if trend > 0.01 { "📈" } else if trend < -0.01 { "📉" } else { "➡️" };
        
        println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║                  LEARNING BENCHMARK @ Step {:>6}                        ║", step);
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  Concept Diversity:  {:.3} ({} unique concepts seen)              ║", 
                 self.concept_diversity(), self.concept_counts.len());
        println!("║  Semantic Coherence: {:.3} {} (trend: {:+.3})                      ║", 
                 self.avg_coherence(), trend_symbol, trend);
        println!("║  Concepts Learned:   {} (started with {})                           ║", 
                 self.concepts_now, self.concepts_at_start);
        println!("║  Energy Stability:   {:.3}                                            ║", 
                 self.energy_stability());
        println!("║  Read/Dream Ratio:   {}/{}                                         ║", 
                 self.read_steps, self.dream_steps);
        println!("║  Memory: {} stores, {} recalls                                         ║", 
                 self.memory_stores, self.memory_recalls);
        
        // Show top concepts
        let mut top: Vec<_> = self.concept_counts.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        let top3: Vec<_> = top.iter().take(3).map(|(k, v)| format!("{}:{}", k, v)).collect();
        println!("║  Top Concepts:       {:?}                                              ║", top3);
        println!("╚══════════════════════════════════════════════════════════════════════════╝\n");
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
    let current_tokens: Vec<u32> = embedder.tokenize(&books[0]);
    
    // PRE-EMBED: Process entire corpus in parallel using all CPU cores
    let window_size = 64; // Semantic chunks
    println!("   Pre-embedding corpus (parallel, window={})...", window_size);
    let mut preembedded: Vec<ndarray::Array1<f32>> = embedder.preembed_corpus_parallel(&current_tokens, window_size);
    let mut chunk_ptr = 0;
    println!("   Created {} semantic chunks from {} tokens ({:.1}x compression)", 
             preembedded.len(), current_tokens.len(), 
             current_tokens.len() as f32 / preembedded.len() as f32);
    
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
    
    // Learning quality benchmarks
    let mut metrics = LearningMetrics::new(concept_memory.len());
    let benchmark_interval = 500; // Print benchmark every N steps

    loop {
        step += 1;
        
        // --- A. Body Update (Metabolism) ---
        body.update(0.005, 0.0); // Constant decay, energy comes from DATA processing (simulated below)
        
        // --- B. Drive Update (Data Hunger) ---
        // Hunger increases if we run out of chunks or haven't read in a while.
        // Simplified: Hunger is inverse of Energy.
        // Novelty: Later chunks in a book tend to have rarer concepts
        let is_novel = chunk_ptr > preembedded.len() / 2; // Second half of book is "novel"
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
                 chunk_ptr += 5; // Skip 5 chunks (~320 tokens)
                 if chunk_ptr >= preembedded.len() { chunk_ptr = 0; } 
                 body.energy = (body.energy - 0.01).max(0.0);
                 cortex.step(None);
             },
             
             Action::Read => {
                 // --- READING (BATCH MODE: 64 tokens per physics step) ---
                // 1. Prediction
                let cortical_out = cortex.get_cortical_output();
                let out_vec = Array1::from(cortical_out);
                let (predicted_token, _confidence) = embedder.decode_nearest(&out_vec);
                prediction = predicted_token;
    
                // 2. Read semantic chunk (64 tokens at once)
                if chunk_ptr < preembedded.len() {
                    let emb = &preembedded[chunk_ptr];
                    input_word = format!("[chunk:{}]", chunk_ptr);
                    
                    let emb_vec = emb.to_vec();
                    let input_tensor = Tensor::<Backend, 1>::from_floats(emb_vec.as_slice(), &device);
                    let input_3d = input_tensor.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);
                    
                    let final_input = input_3d.add(bias_3d);
                    cortex.step(Some(final_input));
                    
                    // APPROACH 3: Hebbian weight learning - modify brain based on input
                    cortex.hebbian_learn(&emb_vec, 0.001); // Small learning rate
                    
                    // Energy cost scaled by window size (64 tokens worth)
                    body.energy = (body.energy - 0.001 * window_size as f32).max(0.0);
                    chunk_ptr += 1;
                } else {
                    cortex.step(None);
                    input_word = "<LOADING>".to_string();
                    
                    // AUTO BOOK DOWNLOAD: Fetch new book from Gutenberg when content exhausted
                    let book_idx = (current_book_idx + step) % GUTENBERG_IDS.len();
                    let book_id = GUTENBERG_IDS[book_idx];
                    println!("\n--> Finished book! Downloading new book (Gutenberg #{})...", book_id);
                    
                    match download_gutenberg_book(book_id, "data/books") {
                        Ok(path) => {
                            // Load and embed the new book
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                let new_tokens = embedder.tokenize(&content);
                                preembedded = embedder.preembed_corpus_parallel(&new_tokens, window_size);
                                chunk_ptr = 0;
                                current_book_idx = book_idx;
                                println!("--> Loaded book {} ({} chunks)", book_id, preembedded.len());
                            }
                        }
                        Err(e) => {
                            eprintln!("--> Failed to download book {}: {}", book_id, e);
                            // Fallback: loop current book
                            chunk_ptr = 0;
                        }
                    }
                }
             }
        }
        
        // --- E. Readout (every step is now meaningful - processing 64 tokens) ---
        if step % 1 == 0 {
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
            
            // Match against concepts using NORMALIZED cosine similarity
            // This makes the score scale-invariant (0.0 to 1.0)
            let output_norm = (output_arr.dot(&output_arr)).sqrt().max(1e-8);
            let output_normalized = &output_arr / output_norm;
            
            for (name, vec) in &concept_memory {
                let vec_norm = (vec.dot(vec)).sqrt().max(1e-8);
                let vec_normalized = vec / vec_norm;
                let score = output_normalized.dot(&vec_normalized);
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
            // APPROACH 2: Lowered threshold from 0.8 to 0.3 for faster learning
            if best_score < 0.3 && sum_e > 0.3 && input_word.len() > 2 && input_word != "<unk>" && !input_word.starts_with("[") {
                 concept_memory.push((input_word.clone(), output_arr.clone()));
                 metrics.record_concept_learned();
                 // Update best_concept immediately for this step
                 best_concept = &concept_memory.last().unwrap().0;
                 best_score = 1.0; 
            }
            
            // Record learning metrics
            metrics.record_concept(best_concept);
            metrics.record_coherence(best_score, step);
            metrics.record_energy(body.energy);
            metrics.record_action(action == Action::Read);

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

            // Only print benchmark at regular intervals (reduces terminal clutter)
            if step % benchmark_interval == 0 && step > 0 {
                metrics.print_benchmark(step);
                
                // FILE PROBE: Write current state to probe.json for external monitoring
                let probe_data = format!(
                    r#"{{"step":{},"coherence":{:.4},"diversity":{:.4},"concepts_learned":{},"energy_stability":{:.3},"read_steps":{},"dream_steps":{},"chunk_ptr":{},"book_chunks":{}}}"#,
                    step, 
                    metrics.avg_coherence(),
                    metrics.concept_diversity(),
                    metrics.concepts_now,
                    metrics.energy_stability(),
                    metrics.read_steps,
                    metrics.dream_steps,
                    chunk_ptr,
                    preembedded.len()
                );
                let _ = std::fs::write("probe.json", probe_data);
            }

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
