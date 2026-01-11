//! Training binary for Harmonic BDH.
//!
//! Usage:
//!   cargo run --bin train -- --help
//!   cargo run --bin train -- --download   # Download Gutenberg corpus
//!   cargo run --bin train -- --data-dir ./data/gutenberg --epochs 50

use bdh_model::{
    HarmonicBdh, BiologicalConfig,
    Trainer, TrainingConfig,
    Embedder, load_texts_from_dir, chunk_text,
};
use bdh_model::data::acquisition;
use clap::Parser;
use ndarray::Array2;

#[derive(Parser, Debug)]
#[command(name = "train")]
#[command(about = "Train Harmonic BDH on text corpus")]
struct Args {
    /// Download Gutenberg corpus first
    #[arg(long)]
    download: bool,

    /// Directory containing training data
    #[arg(long, default_value = "./data/gutenberg")]
    data_dir: String,

    /// Output directory for models/tokenizers
    #[arg(long, default_value = "./output")]
    output_dir: String,

    /// Number of neurons
    #[arg(short, long, default_value_t = 128)]
    neurons: usize,

    /// Number of harmonic modes
    #[arg(short, long, default_value_t = 32)]
    modes: usize,

    /// Number of layers
    #[arg(short, long, default_value_t = 3)]
    layers: usize,

    /// Vocabulary size for tokenizer
    #[arg(long, default_value_t = 4000)]
    vocab_size: usize,

    /// Sequence length for training
    #[arg(long, default_value_t = 16)]
    seq_len: usize,

    /// Training epochs
    #[arg(short, long, default_value_t = 50)]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value_t = 16)]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.01)]
    lr: f32,

    /// Use synthetic data (for testing)
    #[arg(long)]
    synthetic: bool,

    /// Number of synthetic sequences
    #[arg(long, default_value_t = 100)]
    num_synthetic: usize,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            HARMONIC BDH TRAINING                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Download corpus if requested
    if args.download {
        println!("Downloading Gutenberg corpus...\n");
        acquisition::download_gutenberg_corpus(
            acquisition::GUTENBERG_IDS,
            &args.data_dir,
        )?;
        println!();
    }

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Initialize model
    let bio_config = BiologicalConfig {
        noise_amplitude: 0.02,
        self_excitation: 0.015,
        endogenous_drive: 0.01,
        cross_freq_coupling: 0.3,
        ..Default::default()
    };

    let mut model = HarmonicBdh::with_config(
        args.neurons,
        args.modes,
        args.layers,
        bio_config,
    );

    println!("Model configuration:");
    println!("  Neurons: {}", args.neurons);
    println!("  Modes: {}", args.modes);
    println!("  Layers: {}", args.layers);
    println!();

    // Prepare training data
    let sequences: Vec<Array2<f32>> = if args.synthetic {
        println!("Using synthetic training data ({} sequences)...\n", args.num_synthetic);
        Trainer::generate_synthetic_data(args.neurons, args.num_synthetic, args.seq_len)
    } else {
        println!("Loading corpus from {}...", args.data_dir);
        
        // Load texts
        let texts = load_texts_from_dir(&args.data_dir)?;
        if texts.is_empty() {
            println!("No text files found! Use --download to fetch Gutenberg corpus.");
            println!("Or use --synthetic for testing with generated data.\n");
            return Ok(());
        }
        
        println!("  Loaded {} text files", texts.len());
        
        // Chunk texts
        let chunks: Vec<String> = texts.iter()
            .flat_map(|t| chunk_text(t, 200, 50))
            .collect();
        println!("  Created {} text chunks", chunks.len());
        
        // Train embedder
        let embedder = Embedder::train_on_corpus(&chunks, args.vocab_size, args.neurons)?;
        
        // Save tokenizer
        let tokenizer_path = format!("{}/tokenizer.json", args.output_dir);
        embedder.save_tokenizer(&tokenizer_path)?;
        println!("  Saved tokenizer to {}", tokenizer_path);
        
        // Embed texts to sequences
        println!("  Embedding {} chunks to neuron space...", chunks.len().min(500));
        let seqs: Vec<Array2<f32>> = chunks.iter()
            .take(500)  // Limit for memory
            .map(|chunk| embedder.embed_text_sparse(chunk, 0.1))
            .filter(|seq| seq.nrows() >= args.seq_len)
            .map(|seq| {
                // Truncate to seq_len
                let rows = seq.nrows().min(args.seq_len);
                seq.slice(ndarray::s![..rows, ..]).to_owned()
            })
            .collect();
        
        println!("  Prepared {} training sequences\n", seqs.len());
        seqs
    };

    if sequences.is_empty() {
        println!("No training data available!");
        return Ok(());
    }

    // Configure trainer
    let train_config = TrainingConfig {
        epochs: args.epochs,
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        learning_rate: args.lr,
        log_every: args.epochs / 10,
        ..Default::default()
    };

    println!("Training configuration:");
    println!("  Epochs: {}", train_config.epochs);
    println!("  Batch size: {}", train_config.batch_size);
    println!("  Learning rate: {}", train_config.learning_rate);
    println!();

    // Train
    println!("━━━ TRAINING ━━━\n");
    let mut trainer = Trainer::new(train_config);
    trainer.train_unsupervised(&mut model, &sequences);

    // Summary
    println!("\n━━━ TRAINING SUMMARY ━━━");
    if let Some(first) = trainer.metrics_history.first() {
        if let Some(last) = trainer.metrics_history.last() {
            println!("  Initial loss: {:.4}", first.total_loss);
            println!("  Final loss: {:.4}", last.total_loss);
            println!("  Improvement: {:.1}%", 
                     (1.0 - last.total_loss / first.total_loss.max(0.001)) * 100.0);
        }
    }

    // Test the trained model
    println!("\n━━━ MODEL TEST ━━━");
    let daydream = model.daydream(10);
    println!("Daydream after training:");
    for step in &daydream {
        if step.step % 3 == 0 {
            let concepts: String = step.top_concepts.iter()
                .take(2)
                .map(|(name, score)| format!("{}({:.2})", name, score))
                .collect::<Vec<_>>()
                .join(", ");
            println!("  Step {:2}: {} │ {}", step.step, step.thought_state.as_str(), concepts);
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TRAINING COMPLETE                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
