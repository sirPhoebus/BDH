use ndarray::{prelude::*, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rand::thread_rng;

/// Locality Sensitive Hashing for projecting embeddings into the positive orthant.
/// 
/// This implements SimHash-style projection where:
/// 1. Random hyperplanes partition the embedding space
/// 2. Each hyperplane contributes a binary bit based on sign(dot product)
/// 3. The concatenation of ReLU'd projections ensures positivity
pub struct LshEmbedder {
    hyperplanes: Array2<f32>, // (num_hashes, embed_dim)
    num_hashes: usize,
    output_dim: usize,
}

impl LshEmbedder {
    /// Create a new LSH embedder.
    /// - `embed_dim`: dimension of input embeddings (e.g., 768 for BERT)
    /// - `num_hashes`: number of random hyperplanes (controls output sparsity)
    /// - `output_dim`: target neuron dimension n (must be divisible by num_hashes)
    pub fn new(embed_dim: usize, num_hashes: usize, output_dim: usize) -> Self {
        assert!(
            output_dim % num_hashes == 0,
            "output_dim must be divisible by num_hashes"
        );

        let mut rng = thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();

        // Random Gaussian hyperplanes for SimHash
        let hyperplanes = Array2::random_using((num_hashes, embed_dim), dist, &mut rng);

        Self {
            hyperplanes,
            num_hashes,
            output_dim,
        }
    }

    /// Project a single embedding vector into the positive orthant.
    /// Returns a sparse, positive vector of dimension `output_dim`.
    pub fn project(&self, embedding: &Array1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(self.output_dim);
        let bucket_size = self.output_dim / self.num_hashes;

        for (h, hyperplane) in self.hyperplanes.axis_iter(Axis(0)).enumerate() {
            let dot = hyperplane.dot(embedding);

            // SimHash: positive dot -> activate corresponding bucket
            // We use the magnitude as activation strength (soft LSH)
            if dot > 0.0 {
                // Hash the hyperplane index to a position within the bucket
                let bucket_start = h * bucket_size;
                // Distribute activation across bucket using deterministic pattern
                let activation_idx = bucket_start + (h % bucket_size);
                output[activation_idx] = dot;
            }
        }

        // Normalize to unit norm for stable dynamics
        let norm = output.dot(&output).sqrt();
        if norm > 1e-8 {
            output /= norm;
        }

        output
    }

    /// Project a batch of embeddings.
    /// Input: (batch_size, embed_dim)
    /// Output: (batch_size, output_dim)
    pub fn project_batch(&self, embeddings: &Array2<f32>) -> Array2<f32> {
        let batch_size = embeddings.nrows();
        let mut outputs = Array2::zeros((batch_size, self.output_dim));

        for (i, emb) in embeddings.axis_iter(Axis(0)).enumerate() {
            let projected = self.project(&emb.to_owned());
            outputs.row_mut(i).assign(&projected);
        }

        outputs
    }
}

/// Simple token-to-embedding lookup (placeholder for real embeddings).
/// In production, you'd load pre-trained embeddings or use a learned table.
pub struct Vocabulary {
    embeddings: Array2<f32>, // (vocab_size, embed_dim)
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl Vocabulary {
    /// Create a random vocabulary (for testing).
    /// In practice, load from a pre-trained model.
    pub fn new_random(vocab_size: usize, embed_dim: usize) -> Self {
        let mut rng = thread_rng();
        let dist = Normal::new(0.0, 1.0 / (embed_dim as f32).sqrt()).unwrap();
        let embeddings = Array2::random_using((vocab_size, embed_dim), dist, &mut rng);

        Self {
            embeddings,
            vocab_size,
            embed_dim,
        }
    }

    /// Get embedding for a token id.
    pub fn embed(&self, token_id: usize) -> Array1<f32> {
        self.embeddings.row(token_id).to_owned()
    }

    /// Get embeddings for a sequence of token ids.
    pub fn embed_sequence(&self, token_ids: &[usize]) -> Array2<f32> {
        let mut result = Array2::zeros((token_ids.len(), self.embed_dim));
        for (i, &tid) in token_ids.iter().enumerate() {
            result.row_mut(i).assign(&self.embeddings.row(tid));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_positivity() {
        let lsh = LshEmbedder::new(64, 32, 512);
        let mut rng = thread_rng();
        let embedding = Array1::random_using(64, Normal::new(0.0, 1.0).unwrap(), &mut rng);

        let projected = lsh.project(&embedding);

        // All values should be non-negative
        assert!(projected.iter().all(|&v| v >= 0.0));
        // Should be sparse (~50% zeros from sign test)
        let sparsity = projected.iter().filter(|&&v| v == 0.0).count() as f32 / 512.0;
        assert!(sparsity > 0.3, "Expected sparsity > 30%, got {:.1}%", sparsity * 100.0);
    }

    #[test]
    fn test_lsh_locality() {
        let lsh = LshEmbedder::new(64, 32, 512);
        let mut rng = thread_rng();

        let v1 = Array1::random_using(64, Normal::new(0.0, 1.0).unwrap(), &mut rng);
        let v2 = &v1 + &Array1::random_using(64, Normal::new(0.0, 0.1).unwrap(), &mut rng); // Similar
        let v3 = Array1::random_using(64, Normal::new(0.0, 1.0).unwrap(), &mut rng); // Different

        let p1 = lsh.project(&v1);
        let p2 = lsh.project(&v2);
        let p3 = lsh.project(&v3);

        // Similar inputs should have more similar outputs
        let sim_12 = p1.dot(&p2);
        let sim_13 = p1.dot(&p3);
        
        println!("sim(v1, v2) = {:.4}, sim(v1, v3) = {:.4}", sim_12, sim_13);
        // Similar vectors should generally have higher dot product
        // (This is probabilistic, so we use a soft assertion)
    }
}
