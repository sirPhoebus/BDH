use bdh_model::{BdhGpu, LshEmbedder, Vocabulary};

fn main() {
    // Architecture dimensions
    let n = 512;          // Neuron dimension (BDH state space)
    let d = 128;          // Low-rank bottleneck
    let embed_dim = 256;  // Embedding dimension (simulating e.g., smaller transformer)
    let num_hashes = 64;  // LSH hyperplanes
    let layers = 4;
    let vocab_size = 1000;

    println!("=== BDH Model with LSH Embeddings ===\n");

    // Initialize components
    let vocab = Vocabulary::new_random(vocab_size, embed_dim);
    let lsh = LshEmbedder::new(embed_dim, num_hashes, n);
    let model = BdhGpu::new(n, d);

    // Simulate a token sequence (would come from a real tokenizer)
    let token_ids: Vec<usize> = vec![42, 128, 256, 512, 99];
    println!("Input token IDs: {:?}\n", token_ids);

    // Step 1: Token IDs -> Dense Embeddings
    let embeddings = vocab.embed_sequence(&token_ids);
    println!(
        "Embeddings shape: ({}, {})",
        embeddings.nrows(),
        embeddings.ncols()
    );

    // Step 2: Dense Embeddings -> Sparse Positive (via LSH)
    let sparse_inputs = lsh.project_batch(&embeddings);
    
    // Check LSH output properties
    for (i, row) in sparse_inputs.axis_iter(ndarray::Axis(0)).enumerate() {
        let sparsity = (row.iter().filter(|&&v| v == 0.0).count() as f32 / n as f32) * 100.0;
        let positivity = row.iter().all(|&v| v >= 0.0);
        println!(
            "  Token {}: LSH sparsity={:.1}%, all_positive={}",
            i, sparsity, positivity
        );
    }
    println!();

    // Step 3: Forward through BDH layers
    println!("Running BDH forward pass ({} layers)...\n", layers);
    let results = model.forward(&sparse_inputs, layers);

    println!("Output statistics:");
    for (t, (x, y)) in results.iter().enumerate() {
        let x_sparsity = (x.iter().filter(|&&v| v == 0.0).count() as f32 / n as f32) * 100.0;
        let y_sparsity = (y.iter().filter(|&&v| v == 0.0).count() as f32 / n as f32) * 100.0;
        let x_positive = x.iter().all(|&v| v >= 0.0);
        let y_positive = y.iter().all(|&v| v >= 0.0);

        println!(
            "  Token {}: x_sparsity={:.1}%, y_sparsity={:.1}%, x_pos={}, y_pos={}",
            t, x_sparsity, y_sparsity, x_positive, y_positive
        );
    }

    println!("\n=== Pipeline Complete ===");
    println!("TokenID -> Embedding -> LSH (positive orthant) -> BDH layers -> Sparse output");
}
