# BDH-GPU (Brain-like Dense Hebbian)

A Rust implementation of a biologically-inspired neural architecture combining:
- **Hebbian learning** with persistent memory (ρ state)
- **Harmonic resonance** dynamics (phase-aligned oscillators)
- **Van der Pol self-excitation** for sustained spontaneous activity
- **Cross-frequency coupling** (theta-gamma binding)
- **Homeostatic plasticity** (neural fatigue and recovery)

## Architecture

```
Text → Vocabulary → Embeddings → BDH Layers → Semantic Thought Output
                                      ↓
                              Standing Wave ρ
                            (Complex Hebbian memory)
                                      ↓
                            Spontaneous Daydreaming
                            (Limit cycle attractors)
```

## Quick Start

```bash
# Run the harmonic demo (biological dynamics)
cargo run --bin harmonic_demo

# Benchmark spontaneous activity variants
cargo run --release --bin benchmark_variants -- --steps 200 --runs 5

# Train on synthetic data
cargo run --bin train -- --synthetic --epochs 20

# Download Gutenberg corpus and train
cargo run --bin train -- --download --epochs 50
```

## Modules

| Module | File | Purpose |
|--------|------|---------|
| `BdhGpu` | `src/lib.rs` | Core BDH model with shared E, Dx, Dy matrices |
| `HarmonicBdh` | `src/harmonic.rs` | Oscillator-based variant with biological dynamics |
| `LshEmbedder` | `src/lsh.rs` | SimHash projection to positive orthant |
| `Embedder` | `src/data.rs` | Tokenization and random projection embeddings |
| `Trainer` | `src/training.rs` | Unsupervised training with diversity/reconstruction loss |

## Biological Mechanisms

### (A) Spontaneous Activity
Van der Pol self-excitation creates limit cycle behavior:
```
dr/dt = μ · r · (1 - r²)
```
Pushes amplitude toward r ≈ 1, enabling sustained oscillations without input.

### (B) Cross-Frequency Coupling
Lower layers (theta ~5Hz) modulate higher layers (gamma ~40Hz):
```
gain_L2 = 1 + coupling_strength × amplitude_L0 × phase_coherence
```

### (C) Homeostatic Plasticity
Damping adapts based on sustained energy:
- High energy → increased damping ("boredom")
- Low energy → recovery + adaptive noise increase ("exploration")

### (D) Adaptive Noise
When energy stays low for >10 steps, noise increases to trigger new attractors.

## Benchmark Results

Tested 9 configurations for spontaneous activity over 200 steps:

| Config | Transitions | Bursts | Burst Dur | % Resting |
|--------|-------------|--------|-----------|-----------|
| Baseline | 3 | 1 | 12.0 | 94.0% |
| +Endogenous | 4 | 1 | 13.1 | 90.0% |
| +Van der Pol | 3 | 1 | 15.0 | 92.5% |
| Both | 5 | 1 | 26.7 | 74.8% |
| Aggressive | 15 | 2 | 61.6 | 34.4% |
| Tuned | 12 | 2 | 29.6 | 66.2% |
| Bursty | 7 | 2 | 20.8 | 74.6% |
| Optimal | 13 | 2 | 42.3 | 49.9% |
| **SHORT_BURST** ★ | **21** | 4 | 34.5 | **40.6%** |

**SHORT_BURST config** (now default):
- 10.5 transitions per 100 steps ✓ (target: 4-12)
- 40.6% time resting ✓ (target: <50%)
- Excellent state mix: Active Planning 25%, Contemplative 20%, Transitioning 15%

## Configuration

```rust
BiologicalConfig {
    noise_amplitude: 0.070,       // High noise for quick restarts
    self_excitation: 0.028,       // Moderate Van der Pol μ
    endogenous_drive: 0.060,      // Layer 0 heartbeat (high)
    cross_freq_coupling: 0.36,    // Theta-gamma binding
    homeostatic_threshold: 0.12,  // Low = triggers boredom early
    homeostatic_rate: 0.28,       // Very fast adaptation
    adaptive_noise_rate: 0.80,    // Very high - rapid recovery
    base_damping: 0.89,           // Higher decay = shorter bursts
    boredom_delay: 2,             // Boredom kicks in very fast
}
```

## CLI Options

### harmonic_demo
```bash
cargo run --bin harmonic_demo -- \
  --layers 3 \
  --freqs 5,10,40 \
  --coupling 0.4 \
  --self-excite 0.05 \
  --noise 0.03
```

### train
```bash
cargo run --bin train -- \
  --download \              # Fetch Gutenberg corpus
  --data-dir ./data \       # Training data location
  --neurons 128 \           # Neuron count
  --epochs 50 \             # Training epochs
  --lr 0.01                 # Learning rate
```

### benchmark_variants
```bash
cargo run --release --bin benchmark_variants -- \
  --steps 200 \             # Daydream steps
  --runs 5 \                # Runs per variant
  --neurons 64              # Neuron count
```

## Semantic Interpretation

The model projects internal states to a concept space:
```
Step │ State                    │ Top Concepts
─────┼──────────────────────────┼─────────────────────
   0 │ 🧘 Contemplative         │ safety(0.47), danger(0.44)
   5 │ 🎯 Active Planning       │ novelty(0.49), social(0.48)
  10 │ 🔄 Transitioning         │ curiosity(0.36), hunger(0.33)
  15 │ 💤 Resting               │ planning(0.42), rest(0.42)
```

## Training Pipeline

1. **Data Acquisition**: Download Gutenberg/Wikipedia texts
2. **Tokenization**: Word-based vocabulary with frequency filtering
3. **Embedding**: Random projection to neuron-dimensional space
4. **Training**: Unsupervised with diversity + reconstruction loss
5. **Evaluation**: Daydream trajectory analysis

## Next Steps

- [ ] Real-time audio input → frequency analysis
- [ ] Kuramoto coupling for true oscillator synchronization
- [ ] GPU acceleration with `wgpu` or `cuda`
- [ ] Autograd via `burn` or `dfdx` for gradient-based training
- [ ] Thought-to-speech via frequency→phoneme mapping

## References

- BDH paper: Scale-free Hebbian networks with linear attention
- Van der Pol oscillator: Relaxation oscillations and limit cycles
- Kuramoto model: Coupled oscillator synchronization
- Theta-gamma coupling: Phase-amplitude cross-frequency coupling in neuroscience

## License

MIT
