//! Data loading, tokenization, and embedding for BDH training.
//!
//! Provides:
//! - Corpus loading from directories of text files
//! - BPE tokenization via HuggingFace tokenizers
//! - Random projection embedding to neuron space
//! - Data acquisition helpers (Gutenberg, Wikipedia)

use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rand::thread_rng;
use std::fs;
use std::io;
use walkdir::WalkDir;

/// Load all .txt files from a directory.
pub fn load_texts_from_dir(dir_path: &str) -> io::Result<Vec<String>> {
    let mut texts = Vec::new();
    
    for entry in WalkDir::new(dir_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "txt" {
                    if let Ok(content) = fs::read_to_string(path) {
                        // Filter out very short files
                        if content.len() > 500 {
                            texts.push(content);
                        }
                    }
                }
            }
        }
    }
    
    Ok(texts)
}

/// Split a corpus into chunks suitable for training.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    
    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);
        
        if end >= words.len() {
            break;
        }
        start += chunk_size - overlap;
    }
    
    chunks
}

use tokenizers::Tokenizer;
use tokenizers::models::bpe::{BpeTrainer, BpeBuilder};
use tokenizers::pre_tokenizers::whitespace::Whitespace;

/// Embedder: tokenizes text and projects to neuron-dimensional space.
/// Uses HuggingFace Tokenizers BPE for scalable vocabulary.
pub struct Embedder {
    tokenizer: Tokenizer,
    projection: Array2<f32>,   // vocab_size x n
    pub vocab_size: usize,
    pub n: usize,
}

impl Embedder {
    /// Create a new embedder with a pre-trained tokenizer file.
    pub fn from_pretrained(tokenizer_path: &str, n: usize) -> io::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            
        let vocab_size = tokenizer.get_vocab_size(true);
        let projection = Self::init_projection(vocab_size, n);
        
        Ok(Self {
            tokenizer,
            projection,
            vocab_size,
            n,
        })
    }
    
    /// Create a new embedder and train BPE on corpus.
    pub fn train_on_corpus(texts: &[String], vocab_size: usize, n: usize) -> io::Result<Self> {
        println!("Training BPE Tokenizer on {} texts...", texts.len());
        
        let builder = BpeBuilder::new();
        let bpe = builder.build().unwrap(); 
        let mut tokenizer = Tokenizer::new(bpe); // This wraps BPE in ModelWrapper
        
        tokenizer.with_pre_tokenizer(Whitespace::default());
        
        let mut trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(vec![
                tokenizers::AddedToken::from("<pad>", true),
                tokenizers::AddedToken::from("<unk>", true),
                tokenizers::AddedToken::from("<bos>", true),
                tokenizers::AddedToken::from("<eos>", true),
            ])
            .build();
            
        // Write corpus to temp file for training
        let temp_corpus_path = "temp_corpus.txt";
        {
            use std::io::Write;
            let mut file = fs::File::create(temp_corpus_path)?;
            for text in texts {
                writeln!(file, "{}", text)?;
            }
        }
        
        // HACK: Use TokenizerImpl with explicit Wrappers to match BPE model with BpeTrainer
        use tokenizers::tokenizer::TokenizerImpl;
        use tokenizers::{NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper};
        
        // Explicitly typed TokenizerImpl using wrappers for components but BPE for model
        let mut local_tokenizer: TokenizerImpl<
            tokenizers::models::bpe::BPE,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper
        > = TokenizerImpl::new(tokenizers::models::bpe::BPE::default());
        
        local_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::Whitespace(Whitespace::default()));

        local_tokenizer.train_from_files(&mut trainer, vec![temp_corpus_path.to_string()])
             .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
             
        // Clean up
        let _ = fs::remove_file(temp_corpus_path);
        
        // Now wrap the trained model
        let tokenizer = Tokenizer::new(local_tokenizer.get_model().clone());
            
        let actual_vocab_size = tokenizer.get_vocab_size(true);
        println!("Trained BPE with {} tokens", actual_vocab_size);
        
        let projection = Self::init_projection(actual_vocab_size, n);
        
        Ok(Self {
            tokenizer,
            projection,
            vocab_size: actual_vocab_size,
            n,
        })
    }
    
    /// Initialize random projection matrix.
    fn init_projection(vocab_size: usize, n: usize) -> Array2<f32> {
        let mut rng = thread_rng();
        let std_dev = 1.0 / (n as f32).sqrt();
        let dist = Normal::new(0.0, std_dev).unwrap();
        Array2::random_using((vocab_size, n), dist, &mut rng)
    }
    
    /// Save tokenizer to file (JSON).
    pub fn save_tokenizer(&self, path: &str) -> io::Result<()> {
        self.tokenizer.save(path, true)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
    }
    
    /// Tokenize text and return token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        if let Ok(encoding) = self.tokenizer.encode(text, false) {
            encoding.get_ids().to_vec()
        } else {
            vec![]
        }
    }
    
    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_else(|_| "".to_string())
    }
    
    /// Embed a single token ID to n-dimensional space.
    pub fn embed_token(&self, token_id: u32) -> Array1<f32> {
        let id = token_id as usize;
        if id < self.vocab_size {
            self.projection.row(id).to_owned()
        } else {
            Array1::zeros(self.n)
        }
    }
    
    /// Embed text to sequence of n-dimensional vectors.
    pub fn embed_text(&self, text: &str) -> Array2<f32> {
        let tokens = self.tokenize(text);
        if tokens.is_empty() {
             return Array2::zeros((0, self.n));
        }
        let mut embeddings = Array2::zeros((tokens.len(), self.n));
        
        for (i, &token_id) in tokens.iter().enumerate() {
            let emb = self.embed_token(token_id);
            embeddings.row_mut(i).assign(&emb);
        }
        
        embeddings
    }
    
    /// Decode finding the nearest token to the given vector (Cosine Similarity).
    pub fn decode_nearest(&self, query_vec: &Array1<f32>) -> (String, f32) {
        let mut best_sim = -1.0;
        // BPE vocab is usually smaller than hash space, iterating 5000 is fast enough.
        // We iterate 0..vocab_size
        let mut best_id = 0;
        
        // Normalize query
        let q_norm = query_vec.dot(query_vec).sqrt();
        if q_norm < 1e-6 {
            return ("Void".to_string(), 0.0);
        }
        
        // This iteration should be optimized for scale, but for 5k-10k it's fine.
        for id in 0..self.vocab_size {
            if id >= self.projection.nrows() { continue; }
            let emb = self.projection.row(id);
            let dot = emb.dot(query_vec);
            let emb_norm = emb.dot(&emb).sqrt();
            let sim = dot / (q_norm * emb_norm);
            
            if sim > best_sim {
                best_sim = sim;
                best_id = id;
            }
        }
        
        let token = self.tokenizer.decode(&[best_id as u32], false).unwrap_or("<err>".to_string());
        (token, best_sim)
    }
}

/// Data acquisition: download Project Gutenberg texts.
pub mod acquisition {
    use std::fs::{self, File};
    use std::io::{self, Write};
    use std::path::Path;
    
    /// List of popular Gutenberg book IDs (expanded corpus for richer semantics).
    /// Categories: Adventure, Sci-Fi, Philosophy, Drama, Mystery, Fantasy
    pub const GUTENBERG_IDS: &[u32] = &[
        // === ADVENTURE ===
        120,    // Treasure Island
        103,    // Around the World in 80 Days
        164,    // Twenty Thousand Leagues
        18857,  // The Count of Monte Cristo
        1184,   // The Count of Monte Cristo (alt)
        2083,   // The Scarlet Pimpernel
        76,     // Huckleberry Finn
        74,     // Tom Sawyer
        
        // === SCI-FI / SPECULATIVE ===
        84,     // Frankenstein
        35,     // Time Machine
        36,     // War of the Worlds
        62,     // A Princess of Mars
        159,    // The Island of Dr. Moreau
        5200,   // Metamorphosis
        
        // === MYSTERY / HORROR ===
        1661,   // Sherlock Holmes
        2852,   // Hound of the Baskervilles
        244,    // A Study in Scarlet
        345,    // Dracula
        43,     // Jekyll and Hyde
        
        // === CLASSICS / DRAMA ===
        1342,   // Pride and Prejudice
        158,    // Emma
        105,    // Persuasion
        1260,   // Jane Eyre
        768,    // Wuthering Heights
        98,     // Tale of Two Cities
        1400,   // Great Expectations
        174,    // Picture of Dorian Gray
        11,     // Alice in Wonderland
        12,     // Through the Looking Glass
        
        // === PHILOSOPHY / IDEAS ===
        1232,   // The Prince (Machiavelli)
        1497,   // Republic (Plato)
        3600,   // Thus Spake Zarathustra
        4280,   // Meditations (Marcus Aurelius)
        5827,   // The Problems of Philosophy
        
        // === MYTHOLOGY / EPIC ===
        6130,   // The Iliad
        1727,   // The Odyssey
        22381,  // Metamorphoses (Ovid)
        
        // === ADDITIONAL CLASSICS ===
        2701,   // Moby Dick
        2600,   // War and Peace
        4300,   // Ulysses
        1952,   // The Yellow Wallpaper
        514,    // Little Women
    ];
    
    /// Download a single Gutenberg book.
    pub fn download_gutenberg_book(book_id: u32, output_dir: &str) -> io::Result<String> {
        let url = format!(
            "https://www.gutenberg.org/cache/epub/{}/pg{}.txt",
            book_id, book_id
        );
        
        fs::create_dir_all(output_dir)?;
        let output_path = format!("{}/gutenberg_{}.txt", output_dir, book_id);
        
        if Path::new(&output_path).exists() {
            println!("  Book {} already exists, skipping", book_id);
            return Ok(output_path);
        }
        
        println!("  Downloading book {}...", book_id);
        
        let response = reqwest::blocking::get(&url)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        
        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Failed to download book {}: {}", book_id, response.status())
            ));
        }
        
        let content = response.text()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        
        // Strip Gutenberg header/footer
        let clean_content = strip_gutenberg_header(&content);
        
        let mut file = File::create(&output_path)?;
        file.write_all(clean_content.as_bytes())?;
        
        Ok(output_path)
    }
    
    /// Download multiple Gutenberg books.
    pub fn download_gutenberg_corpus(book_ids: &[u32], output_dir: &str) -> io::Result<Vec<String>> {
        println!("Downloading {} Gutenberg books...", book_ids.len());
        
        let mut paths = Vec::new();
        for &id in book_ids {
            match download_gutenberg_book(id, output_dir) {
                Ok(path) => paths.push(path),
                Err(e) => eprintln!("  Warning: Failed to download book {}: {}", id, e),
            }
        }
        
        println!("Downloaded {} books successfully", paths.len());
        Ok(paths)
    }
    
    /// Strip Gutenberg header and footer from text.
    fn strip_gutenberg_header(text: &str) -> String {
        let lines: Vec<&str> = text.lines().collect();
        
        // Find start marker
        let start = lines.iter()
            .position(|line| line.contains("*** START OF"))
            .map(|i| i + 1)
            .unwrap_or(0);
        
        // Find end marker
        let end = lines.iter()
            .position(|line| line.contains("*** END OF"))
            .unwrap_or(lines.len());
        
        lines[start..end].join("\n")
    }
    
    /// Download Wikipedia article (simple version using API).
    pub fn download_wikipedia_article(title: &str, output_dir: &str) -> io::Result<String> {
        let url = format!(
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles={}",
            urlencoding::encode(title)
        );
        
        fs::create_dir_all(output_dir)?;
        let safe_title = title.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        let output_path = format!("{}/wiki_{}.txt", output_dir, safe_title);
        
        if Path::new(&output_path).exists() {
            return Ok(output_path);
        }
        
        println!("  Downloading Wikipedia: {}...", title);
        
        let response = reqwest::blocking::get(&url)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        
        let json: serde_json::Value = response.json()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        
        // Extract text from JSON response
        let pages = json["query"]["pages"].as_object()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid Wikipedia response"))?;
        
        let content = pages.values()
            .next()
            .and_then(|page| page["extract"].as_str())
            .unwrap_or("");
        
        let mut file = File::create(&output_path)?;
        file.write_all(content.as_bytes())?;
        
        Ok(output_path)
    }
}

/// Training data batch.
#[derive(Clone)]
pub struct TrainingBatch {
    /// Input embeddings (batch_size, seq_len, n)
    pub inputs: Vec<Array2<f32>>,
    /// Target token IDs for next-step prediction
    pub targets: Vec<Vec<u32>>,
}

impl TrainingBatch {
    /// Create batches from texts.
    pub fn from_texts(embedder: &Embedder, texts: &[String], seq_len: usize, batch_size: usize) -> Vec<Self> {
        let mut all_inputs = Vec::new();
        let mut all_targets = Vec::new();
        
        for text in texts {
            let tokens = embedder.tokenize(text);
            if tokens.len() < seq_len + 1 {
                continue;
            }
            
            // Create input/target pairs
            for start in 0..(tokens.len() - seq_len) {
                let input_tokens = &tokens[start..start + seq_len];
                let target_tokens = &tokens[start + 1..start + seq_len + 1];
                
                // Embed inputs
                let mut input_emb = Array2::zeros((seq_len, embedder.n));
                for (i, &token_id) in input_tokens.iter().enumerate() {
                    input_emb.row_mut(i).assign(&embedder.embed_token(token_id));
                }
                
                all_inputs.push(input_emb);
                all_targets.push(target_tokens.to_vec());
            }
        }
        
        // Create batches
        let mut batches = Vec::new();
        for chunk in all_inputs.chunks(batch_size).zip(all_targets.chunks(batch_size)) {
            batches.push(TrainingBatch {
                inputs: chunk.0.to_vec(),
                targets: chunk.1.to_vec(),
            });
        }
        
        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_text() {
        let text = "The quick brown fox jumps over the lazy dog";
        let chunks = chunk_text(text, 4, 1);
        
        assert!(!chunks.is_empty());
        assert!(chunks[0].split_whitespace().count() <= 4);
    }
    
    #[test]
    fn test_embedder_train() {
        let texts = vec![
            "Hello world, this is a test.".to_string(),
            "Another sentence for tokenization.".to_string(),
            "The quick brown fox jumps over the lazy dog.".to_string(),
        ];
        
        let embedder = Embedder::train_on_corpus(&texts, 100, 32).unwrap();
        
        assert!(embedder.vocab_size > 0);
        assert_eq!(embedder.n, 32);
        
        let tokens = embedder.tokenize("Hello world");
        assert!(!tokens.is_empty());
        
        let embeddings = embedder.embed_text("Hello world");
        assert_eq!(embeddings.ncols(), 32);
    }
}
