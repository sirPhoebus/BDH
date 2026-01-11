use bdh_model::harmonic_burn::HarmonicBdhBurn;
use bdh_model::data::acquisition::{download_gutenberg_book, GUTENBERG_IDS};
use bdh_model::body::BodyState;
use bdh_model::drives::Drives;
use bdh_model::interpreter::Interpreter;
use bdh_model::memory::MemorySystem;
use burn_wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

use ndarray::Array1;
use std::collections::VecDeque;
use serde::Serialize; // Added Deserialize just in case, but definitely need Chunk
use bdh_model::data::Chunk; // Import Chunk explicitly
use bdh_model::working_memory::WorkingMemory;
use std::process::Command;


use axum::{
    routing::get,
    response::IntoResponse,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    Router,
};
use tower_http::services::ServeDir;
use tokio::sync::broadcast;
use std::sync::Arc;

const BENCHMARK_INTERVAL: usize = 10; // Lowered for faster verification

/// Helper: Filter out low-quality concepts (Stopwords, suffixes, short tokens)
fn is_useful_concept(word: &str) -> bool {
    let word_lower = word.to_lowercase();
    // 1. Length Check
    if word.len() < 3 { return false; } // Revert to > 2 for quality
    
    // 2. Stopwords
    let stopwords = ["the", "and", "that", "have", "for", "with", "you", "this", "but", "his", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "got", "was", "were", "had", "bin", "did", "are", "is", "am"];
    if stopwords.contains(&word_lower.as_str()) { return false; }
    
    // 3. Suffix / Artifact Heuristics
    // Common BPE artifacts (e.g., 'ing', 'ment', 'tion' if standalone)
    let suffixes = ["ing", "ment", "tion", "ness", "ity", "ous", "ent", "est", "ist", "ful", "less", "able", "ible", "al", "ic"];
    if suffixes.contains(&word_lower.as_str()) { return false; }
    
    // 4. Special Characters
    if word.starts_with("[") || word.starts_with("<") || word.contains(|c: char| !c.is_alphanumeric()) { return false; }
    
    true
}

#[derive(Serialize, Clone)]
struct BrainStateUpdate {
    energies: Vec<f32>,
    dopamine: f32,
    norepinephrine: f32,
    acetylcholine: f32,
    input_word: String,
    prediction: String,
    inner_voice: String,
    memory_event: Option<Vec<f32>>,
    working_memory: String, // [NEW] WM Content
    mode: String,          // [NEW] Reading/Reflecting
}

// --- PERSISTENCE ---
fn save_concepts(concepts: &Vec<(String, Array1<f32>)>) {
    let plain_data: Vec<(String, Vec<f32>)> = concepts.iter()
        .map(|(k, v)| (k.clone(), v.to_vec()))
        .collect();
    let json = serde_json::to_string(&plain_data).unwrap();
    std::fs::write("concept_memory.json", json).unwrap_or_default();
}

fn load_concepts() -> Option<Vec<(String, Array1<f32>)>> {
    if let Ok(content) = std::fs::read_to_string("concept_memory.json") {
        if let Ok(plain_data) = serde_json::from_str::<Vec<(String, Vec<f32>)>>(&content) {
            let loaded: Vec<(String, Array1<f32>)> = plain_data.iter()
                .map(|(k, v)| (k.clone(), Array1::from(v.clone())))
                .collect();
            println!("   [Persistence] Loaded {} concepts from disk.", loaded.len());
            return Some(loaded);
        }
    }
    None
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
    _concepts_at_start: usize,
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
            _concepts_at_start: initial_concepts,
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
    
    fn record_concept_learned(&mut self, concept: &str, _step: usize) {
        if !self.concept_counts.contains_key(concept) {
             // Synaptic Scaling Filter: Only count tokens > 2 chars as a "Learned Concept"
             if concept.len() > 2 && !concept.starts_with("[") {
                self.concepts_now += 1;
             }
        }
        self.record_concept(concept);
    }
    
    fn record_memory_store(&mut self) { self.memory_stores += 1; }
    fn record_memory_recall(&mut self) { self.memory_recalls += 1; }
    
    fn concept_diversity(&self) -> f32 {
        if self.total_recognitions == 0 { return 0.0; }
        let unique = self.concept_counts.len() as f32;
        let total = self.total_recognitions as f32;
        // Entropy-like measure: higher = more diverse
        // User suggestion: unique / log(total) to prevent artificial decay
        unique / (total.ln().max(1.0))
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
    

    
    fn energy_stability(&self) -> f32 {
        if self.energy_samples.len() < 2 { return 0.0; }
        let mean: f32 = self.energy_samples.iter().sum::<f32>() / self.energy_samples.len() as f32;
        let variance: f32 = self.energy_samples.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / self.energy_samples.len() as f32;
        1.0 / (1.0 + variance.sqrt()) // Higher = more stable
    }
    
    fn print_benchmark(&self, _step: usize) {
        let trend = self.coherence_trend();
        let _trend_symbol = if trend > 0.01 { "📈" } else if trend < -0.01 { "📉" } else { "➡️" };
        
        /*
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
        */
    }
}

#[tokio::main]
async fn main() {
    // Reset probe.json to signal fresh start
    let _ = std::fs::write("probe.json", r#"{"step": 0}"#);

    // Auto-launch visualization (non-blocking)
    let _ = Command::new("python")
        .arg("monitor_probe.py")
        .spawn()
        .map_err(|e| eprintln!("Warning: Failed to launch monitor_probe.py: {}", e));
    
    // println!("Initializing CENTRAL SIM (Grand Unification)...");
    
    // 0. Setup Networking
    let (tx, _rx) = broadcast::channel::<BrainStateUpdate>(16);
    let tx_shared = Arc::new(tx);
    
    let app_tx = tx_shared.clone();
    let app = Router::new()
        .route("/ws", get(move |ws| ws_handler(ws, app_tx)))
        .fallback_service(ServeDir::new("viz"));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    // println!("   Visualizer server running at http://localhost:3000");
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // 1. Setup GPU Backend
    type Backend = Wgpu<f32, i32>;
    let device = WgpuDevice::BestAvailable;
    let n_neurons = 1000; // Can scale to 10k, using 1k for nice console visualization
    let d = 16;
    let layers = 4;
    
    // println!("1. Cortex: Booting GPU Kernel ({} neurons)...", n_neurons);
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
    let mut preembedded: Vec<Chunk> = embedder.preembed_corpus_parallel(&current_tokens, window_size);
    let mut chunk_ptr = 0;
    let mut last_prediction: Option<Array1<f32>> = None;

    println!("   Created {} semantic chunks from {} tokens ({:.1}x compression)", 
             preembedded.len(), current_tokens.len(), 
             current_tokens.len() as f32 / preembedded.len() as f32);
    
    // 6. Concept Memory (Restored)
    let mut concept_memory: Vec<(String, Array1<f32>)> = Vec::new();
    
    // PERSISTENCE: Try loading first
    if let Some(loaded) = load_concepts() {
        concept_memory = loaded;
    } else {
        // Fallback to defaults
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
    }
    
    // 7. Hippocampal Memory (Hippocampus)
    println!("7. Memory: Initializing vector store...");
    let mut memory = MemorySystem::new(100);
    
    // 6b. Working Memory (Short-Term Context)
    println!("   Working Memory: Allocating 3 slots...");
    let mut wm = WorkingMemory::<Backend>::new(3, 1.5); // Capacity 3, Bias Strength 1.5 (Strong driver)
    
    println!("\nSIMULATION STARTING. Reading: Book 0\n");
    /*
    println!("Step | Energy | Valnc | Drive  | Conf | Input Word      | Prediction      | Inner Voice");
    println!("-----+--------+-------+--------+------+-----------------+-----------------+-------------------------------------");
    */

    let mut step = 0;
    
    // 7. Goal Vectors (Drive Targets)
    // We create "Attractors" for drives.
    // "Food" for Hunger. "Mystery" for Curiosity.
    let food_vec = embedder.embed_text("food eat delicious satisfy").row(0).to_owned();
    let novelty_vec = embedder.embed_text("mystery strange new discover").row(0).to_owned();
    let food_vec = Tensor::<Backend, 1>::from_floats(food_vec.to_vec().as_slice(), &device);
    let novelty_vec = Tensor::<Backend, 1>::from_floats(novelty_vec.to_vec().as_slice(), &device);

    let mut last_narrative = String::new();
    let mut current_memory_event: Option<Vec<f32>> = None;
    let mut current_stimulus: Option<Array1<f32>> = None; 
    let mut current_word: Option<String> = None;
    let mut is_sleeping = false;
    
    // Learning quality benchmarks
    let mut metrics = LearningMetrics::new(concept_memory.len());

    // Reflection State
    let mut is_reflecting = false;
    let mut reflection_timer = 0;
    const READING_PHASE_LEN: usize = 200;
    const REFLECTION_PHASE_LEN: usize = 50;

    loop {
        step += 1;
        
        // --- A. Body Update (Metabolism) ---
        body.update(0.005, 0.0); // Constant decay, energy comes from DATA processing (simulated below)
        
        // --- B. Drive Update (Data Hunger) ---
        // Surprise Signal: inverse of prediction confidence
        // We use normalized output to get a stable similarity score
        let surprise = if let Ok(out_norm) = cortex.get_cortical_output_norm() {
            let (_, sim) = embedder.decode_nearest(&out_norm);
            (1.0 - sim).max(0.0).min(1.0)
        } else {
            0.5 // Default surprise if no output
        };
        
        drives.update(body.energy, surprise);
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

        // Add Working Memory Context Bias
        let wm_bias = wm.get_context_bias(n_neurons, &device);
        total_bias_vec = total_bias_vec.add(wm_bias);

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
        // Reflection Timer Logic
        if is_reflecting {
             reflection_timer += 1;
             if reflection_timer > REFLECTION_PHASE_LEN {
                 is_reflecting = false;
                 reflection_timer = 0;
                 // println!("   [State] Resume Reading (World Input Active)");
             }
        } else {
             reflection_timer += 1;
             if reflection_timer > READING_PHASE_LEN {
                 is_reflecting = true;
                 reflection_timer = 0;
                 // println!("   [State] Entering Reflection (Inner Monologue...)");
             }
        }

        let action = if is_sleeping {
             Action::Dream
        } else if is_reflecting {
             Action::Reflect // Override other drives during Reflection Phase
        } else if body.chemicals.norepinephrine > 0.9 {
             Action::Skip 
        } else if drives.hunger > 0.85 && goal_match < 0.2 {
             // FORAGING MODE
             Action::Skip 
        } else {
             Action::Read
        };
        
        let mut input_word = "".to_string();
        let mut prediction = "".to_string();
        
        // Initialize loop variables
        let mut surprise = 0.0;
        let mut learning_rate_modulation = 1.0;
        
        match action {
             Action::Dream => {
                // --- DREAMING (Episodic Replay + Chaining) ---
                prediction = "(Internal)".to_string();
                
                // Higher chance for a memory if we are stuck
                let replay_trigger = rand::random::<f32>() < 0.3; // 30% jump chance
                
                let input_tensor = if replay_trigger {
                    if let Some(mem) = episodic_buffer.sample() {
                         input_word = format!("*{}*", mem.concept);
                         current_stimulus = Some(mem.embedding.clone());
                         
                         // LOOSE ASSOCIATION: Apply past bias during dreaming
                         memory.recall_threshold = 0.55; 
                         if let Some(past_freqs) = memory.recall(&mem.embedding) {
                             let current_freqs = cortex.natural_freq.to_data().to_vec::<f32>().unwrap();
                             let current_arr = Array1::from(current_freqs);
                             let blended: Array1<f32> = (&current_arr * 0.9) + (&past_freqs * 0.1);
                             cortex.natural_freq = Tensor::<Backend, 1>::from_floats(blended.as_slice().unwrap(), &device);
                             metrics.record_memory_recall();
                         }

                         Tensor::<Backend, 1>::from_floats(mem.embedding.to_vec().as_slice(), &device)
                    } else {
                        // Fallback to Chaining
                        let output_arr = Array1::from(cortex.get_cortical_output());
                        let (next_token, _) = embedder.decode_nearest(&output_arr);
                        input_word = format!("~{}~", next_token);
                        
                        let self_talk = format!("I imagine {}", next_token);
                        let feedback_vector = interpreter.reflect(&self_talk, &embedder);
                        current_stimulus = Some(feedback_vector.clone());
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
                    current_stimulus = Some(feedback_vector.clone());
                    Tensor::<Backend, 1>::from_floats(feedback_vector.to_vec().as_slice(), &device)
                };

                // Apply Input
                let input_3d = input_tensor.reshape([1, 1, n_neurons]).expand([layers, d, n_neurons]);
                cortex.step(Some(input_3d));
                
                // Dreaming recovers Energy significantly to break the fast-cycle flutter
                body.energy = (body.energy + 0.1).min(1.0); // FASTER RECOVERY
             },
             
             Action::Reflect => {
                 // --- REFLECTION (INNER MONOLOGUE) ---
                 // Sensory Deprivation: No external input. Purely driven by WM Bias (added earlier).
                 current_stimulus = None;
                 input_word = "   [Reflecting...]".to_string();
                 
                 // 1. Process Thinking Step (Zero Input + WM Bias)
                 // We need to pass a Zero Tensor + Bias.
                 // We already added bias to total_bias_vec.
                 // So we just need to pass a Zero Input Tensor.
                 let zero_input = Tensor::<Backend, 1>::zeros([n_neurons], &device)
                         .reshape([1, 1, n_neurons])
                         .expand([layers, d, n_neurons]);
                 
                 // Add bias
                 let final_input = zero_input.add(bias_3d);
                 
                 cortex.step(Some(final_input));
                 
                 // 2. Decode Thought
                 let cortical_out = cortex.get_cortical_output();
                 let out_vec = Array1::from(cortical_out.clone());
                 let (thought_token, confidence) = embedder.decode_nearest(&out_vec);
                 
                 prediction = format!("(Thought: {})", thought_token);
                 
                 // SILENT MODE: Only print via Probe.json
                 if is_useful_concept(&thought_token) {
                      if confidence > 0.25 {
                          // println!("   [Inner Monologue] '{}' ({:.2})", thought_token, confidence);
                          // FEEDBACK: Thinking about it makes it stronger in WM
                          let thought_tensor = Tensor::<Backend, 1>::from_floats(out_vec.as_slice().unwrap(), &device);
                          wm.update(thought_token, thought_tensor);
                          
                          // PERSISTENCE: Opportunistic Save
                          if step % 200 == 0 {
                              save_concepts(&concept_memory);
                          }
                      }
                 }
             },
              
              Action::Skip => {
                  // --- SKIPPING (Foraging) ---
                  input_word = ">>".to_string();
                  chunk_ptr += 5; // Skip 5 chunks (~320 tokens)
                  if chunk_ptr >= preembedded.len() { chunk_ptr = 0; } 
                   body.energy = (body.energy - 0.01).max(0.0);
                   current_stimulus = None;
                   cortex.step(None);
              },
              
             Action::Read => {
                 // --- READING (BATCH MODE: 64 tokens per physics step) ---
                 // 1. Prediction for display
                 let cortical_out = cortex.get_cortical_output();
                 let out_vec = Array1::from(cortical_out);
                 let (predicted_token, _confidence) = embedder.decode_nearest(&out_vec);
                 prediction = predicted_token;
     
                 if chunk_ptr < preembedded.len() {
                     let chunk = &preembedded[chunk_ptr];
                     current_stimulus = Some(chunk.vector.clone());
                     current_word = Some(chunk.label.clone());
                     input_word = format!("[chunk:{}]", chunk_ptr);
                     
                     // MEMORY RECALL
                     memory.recall_threshold = 0.65;
                     if let Some(past_freqs) = memory.recall(&chunk.vector) {
                         let current_freqs = cortex.natural_freq.to_data().to_vec::<f32>().unwrap();
                         let current_arr = Array1::from(current_freqs);
                         let blended: Array1<f32> = (&current_arr * 0.9) + (&past_freqs * 0.1);
                         cortex.natural_freq = Tensor::<Backend, 1>::from_floats(blended.as_slice().unwrap(), &device);
                         metrics.record_memory_recall();
                     }
 
                     // --- PREDICTIVE CODING: ERROR SIGNAL ---
                     surprise = 0.0;
                     learning_rate_modulation = 1.0;
                     let array_emb = &chunk.vector;
                     
                     if let Some(last_pred) = &last_prediction {
                          let input_norm = (array_emb.dot(array_emb)).sqrt().max(1e-8);
                          let input_normalized = array_emb / input_norm;
                          let similarity = last_pred.dot(&input_normalized);
                          surprise = (1.0 - similarity).max(0.0).min(1.0);
                          
                          learning_rate_modulation = 1.0 + (surprise * 2.0);
                          if surprise > 0.5 {
                              body.release(0.0, surprise * 0.1, 0.0);
                          }
                     }
     
                     // Send to Cortex
                     let input_tensor: Tensor<Backend, 3> = Tensor::<Backend, 1>::from_floats(array_emb.as_slice().unwrap(), &device)
                         .reshape([1, 1, n_neurons])
                         .expand([layers, d, n_neurons]);
                         
                     let final_input = input_tensor.add(bias_3d);
                     
                     
                     cortex.step(Some(final_input));
     
                     // Hebbian Learning
                     let base_lr = 0.01;
                     // Only learn during Read action here
                     cortex.hebbian_learn(array_emb.as_slice().unwrap(), base_lr * learning_rate_modulation);
                     
                     // Update Prediction
                     let current_out = cortex.get_cortical_output_norm().unwrap_or(Array1::zeros(n_neurons));
                     last_prediction = Some(current_out);
                     
                     body.energy = (body.energy - 0.001 * window_size as f32).max(0.0);
                     chunk_ptr += 1;
                 } else {
                     cortex.step(None);
                     input_word = "<LOADING>".to_string();
                     
                     // AUTO BOOK DOWNLOAD
                     let book_idx = (current_book_idx + step) % GUTENBERG_IDS.len();
                     let book_id = GUTENBERG_IDS[book_idx];
                     
                     match download_gutenberg_book(book_id, "data/books") {
                         Ok(path) => {
                             if let Ok(content) = std::fs::read_to_string(&path) {
                                 let new_tokens = embedder.tokenize(&content);
                                 preembedded = embedder.preembed_corpus_parallel(&new_tokens, window_size);
                                 chunk_ptr = 0;
                                 current_book_idx = book_idx;
                             }
                         }
                         Err(e) => {
                             chunk_ptr = 0; // Loop if fail
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
            
            // RICH INTERPRETATION: Use Ensemble decoding with Dynamic Phase Epsilon
            // Tighten epsilon during high arousal (precision), loosen during dreaming (creativity)
            let arousal_bias = (body.arousal - 0.5) * 0.1; // -0.05 to +0.05
            let dynamic_epsilon = if action == Action::Dream { 
                0.20 + arousal_bias 
            } else { 
                0.12 + arousal_bias 
            };
            interpreter.phase_epsilon = dynamic_epsilon;

            let phases = cortex.get_phases();
            let usage = cortex.get_usage_vec();
            let ensemble_concepts = interpreter.interpret_ensemble(
                &cortical_out, 
                &phases, 
                &usage, 
                &embedder.projection,
                &embedder
            );
            
            let mut best_concept = "Void".to_string();
            let mut best_score = 0.0;
            
            if !ensemble_concepts.is_empty() {
                best_concept = ensemble_concepts[0].0.clone();
                best_score = ensemble_concepts[0].1;
                
                // Track concepts found in ensembles
                for (name, _) in &ensemble_concepts {
                    metrics.record_concept(name);
                }
            }

            // Narrative generation
            let narrative = interpreter.generate_narrative(&ensemble_concepts, confidence);
            last_narrative = narrative.clone();
            input_word = narrative; // Update input_word for display/reflection

            // --- SELF-DISCOVERY & ASSOCIATIVE LEARNING ---
            // If the ensemble decoding is weak (best_score < 0.5) but internal focus is high (sum_e > 0.3),
            // we first check our associative memory for a past match.
            if best_score < 0.5 {
                let output_arr = Array1::from(cortical_out.clone());
                let output_norm = (output_arr.dot(&output_arr)).sqrt().max(1e-8);
                let output_normalized = &output_arr / output_norm;

                for (name, vec) in &concept_memory {
                    let vec_norm = (vec.dot(vec)).sqrt().max(1e-8);
                    let vec_normalized = vec / vec_norm;
                    let score = output_normalized.dot(&vec_normalized);
                    if score.abs() > best_score {
                        best_score = score.abs();
                        best_concept = name.to_string();
                    }
                }
                
                // Debug why learning stops
                // Debug why learning stops
                if step % 200 == 0 {
                    // SILENT MODE
                    // println!("   [Debug] Step {}: Best Match '{}' ({:.4})", step, best_concept, best_score);
                    // println!("   [WM] Context: {}", wm.debug_string());
                }

                // If still unrecognized, associate the current *Input Stimulus* word as a NEW concept
                if let Some(word_str) = &current_word {
                    // QUALITY CONTROL: Filter using our helper (Stopwords, Suffixes, Length)
                    if best_score < 0.5 && sum_e > 0.01 && is_useful_concept(word_str) && word_str != "<unk>" {
                         // SILENT MODE
                         // println!("   [!] LEARNED NEW CONCEPT: '{}' (E={:.4}, Score={:.4})", word_str, sum_e, best_score);
                         concept_memory.push((word_str.clone(), output_arr));
                         metrics.record_concept_learned(word_str.as_str(), step);
                         best_concept = word_str.clone();
                         best_score = 1.0; 
                    }
                }
            }
            // ----------------------------------------------

            // ----------------------------------------------
            
            // 7. Update Working Memory
            if best_score > 0.4 && best_concept != "Void" && best_concept != "<unk>" {
                let thought_tensor = Tensor::<Backend, 1>::from_floats(cortical_out.as_slice(), &device);
                wm.update(best_concept.clone(), thought_tensor);
            }

            // MEMORY STORAGE: "Aha!" moment storage
            if best_score > 0.45 && best_concept.len() > 3 && !best_concept.starts_with("[") {
                if let Some(stim) = &current_stimulus { 
                    let current_freqs = cortex.natural_freq.to_data().to_vec::<f32>().unwrap();
                    if memory.store(stim, &Array1::from(current_freqs), best_score, step) {
                        metrics.record_memory_store();
                        if step % 10 == 0 {
                            // println!("   [Memory] Stored high-coherence state: '{}' ({:.2})", best_concept, best_score);
                        }
                    }
                }
            }
            
            // Record learning metrics
            metrics.record_concept(&best_concept);
            metrics.record_coherence(best_score, step);
            metrics.record_energy(body.energy);
            metrics.record_action(action == Action::Read);

            // --- EMBODIMENT LOOP (Refined) ---
            // 1. Concept-based Valence (Safety/Danger)
            let mut env_valence = match best_concept.as_str() {
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

            // --- HOMEOSTATIC REGULATION ---
            // NUCLEUS COERULEUS: Boredom / Frustration
            // If diversity is low, we are stuck in a loop. INCREASE NOISE.
            let diversity = metrics.concept_diversity();
            // AGGRESSIVE: Raise threshold to 0.5 to force noise if stuck in attractors
            if diversity < 0.5 && step > 500 {
                 let frustration = (0.5 - diversity) * 5.0; // 0.0 to 1.0
                 // Boost NE significantly to break attractors
                 body.chemicals.norepinephrine += frustration * 0.1;
                 // reduce dopamine to stop reinforcement of current state
                 body.chemicals.dopamine *= 0.95;
            }


            if env_valence != 0.0 {
                 body.update(0.1, env_valence);
            }
            
            // --- MEMORY ENCODING ---
            if body.pleasure_pain.abs() > 0.5 || body.arousal > 0.8 {
                 let cortical_out = cortex.get_cortical_output();
                 let state_vec = Array1::from(cortical_out);
                 current_memory_event = Some(state_vec.to_vec());
                 episodic_buffer.add(state_vec, best_concept.clone(), body.pleasure_pain);
            }
            // -----------------------


            // Only print benchmark at regular intervals (reduces terminal clutter)
            if step % BENCHMARK_INTERVAL == 0 && step > 0 {
                metrics.print_benchmark(step);
                
                // FILE PROBE: Write current state to probe.json for external monitoring
                let _ = std::fs::write("probe.json", format!(
                r#"{{
                    "step": {},
                    "coherence": {:.4},
                    "diversity": {:.4},
                    "concepts_learned": {},
                    "energy_stability": {:.4},
                    "read_steps": {},
                    "dream_steps": {},
                    "mem_stores": {},
                    "mem_recalls": {},
                    "chunk_ptr": {},
                    "book_chunks": {},
                    "surprise": {:.4},
                    "cortical_energy": {:.4}
                }}"#,
                step,
                cortex.get_phases().iter().map(|p| p.sin()).sum::<f32>().abs() / n_neurons as f32, // Simplified coherence
                metrics.concept_diversity(),
                metrics.concepts_now,
                body.energy,
                metrics.read_steps,
                metrics.dream_steps,
                metrics.memory_stores,
                metrics.memory_recalls,
                chunk_ptr,
                preembedded.len(),
                surprise,
                sum_e
            ));
            
                // PERSISTENCE (Guaranteed Save every 500 steps)
                if step % 500 == 0 {
                    save_concepts(&concept_memory);
                }
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
                    inner_voice: prediction.clone(), // Use prediction as inner voice too
                    memory_event: current_memory_event,
                    
                    // NEW FIELDS for Silent Observability
                    working_memory: wm.debug_string(),
                    mode: if is_reflecting { "Reflecting".to_string() } else { "Reading".to_string() },
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
