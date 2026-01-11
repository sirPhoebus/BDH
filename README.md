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

# Self-thinking / reflection demo (stream of consciousness)
cargo run --bin reflect_demo -- --steps 100 --verbose

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
| `ExperienceReplay` | `src/continual.rs` | Priority-based replay buffer for continual learning |
| `AdaptiveForgetting` | `src/continual.rs` | Importance-based rho decay to prevent catastrophic forgetting |

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

### (E) Continual Learning
Experience replay + adaptive forgetting prevents catastrophic forgetting:
```
New Input → Forward → Store in Replay Buffer (prioritized by surprise)
                    ↓
          Periodic Replay (mix 30% old + 70% new)
                    ↓
          Update rho + Apply Forgetting Mask (decay unused states)
```
- **Replay Buffer**: Prioritizes surprising/diverse experiences
- **Importance Tracking**: Monitors which rho states are frequently activated
- **Consolidation**: Protects replayed patterns from forgetting

### (F) Temporal / Chronos System (Dual-Clock)
Global heartbeat + multi-scale time encoding for temporal memory:
```
                    ┌─────────────────────────────────────┐
                    │         HEARTBEAT CLOCK             │
                    │    sin(t × freq) → Gating Factor    │
                    │    Peak = high plasticity           │
                    │    Trough = filtering noise         │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
    Input ──► Heartbeat Gate ──► Forward ──► Rho Update with Time Encoding
                                                    │
                                    ┌───────────────┴───────────────┐
                                    │     MULTI-SCALE ENCODING       │
                                    │  PE(t,i) = sin(t / 10000^(i/d))│
                                    │  Different modes = timescales  │
                                    └────────────────────────────────┘
```
- **Heartbeat Modulation**: `sin(t × freq).abs()` gates input receptivity
- **Time-Weighted Learning**: Rho updates encode "when" via phase modulation
- **Rho Decay**: `rho *= 0.995` per step prevents unbounded growth
- **Imprint/Recall**: Explicit memory operations with timestamps

### (G) Self-Thinking / Stochastic Resonance
The model can "think to itself" without external input:
```
┌─────────────────────────────────────────────────────────────┐
│                    REFLECTION LOOP                           │
│                                                              │
│   Thermal Noise ──┬──> rho states ──> Extract Vibration     │
│         ↑         │         ↑              │                 │
│         │         │         │              ▼                 │
│   Energy Floor ───┘    Self-Feedback <── Forward Pass        │
│                              ↑              │                 │
│                              └──────────────┘                │
│                                                              │
│   Output: Stream of Consciousness (concepts, states, recalls)│
└─────────────────────────────────────────────────────────────┘
```
- **Thermal Noise**: Always-on background noise prevents zero-energy states
- **Energy Floor**: System never truly goes silent (like biological alpha/theta rhythms)
- **Self-Feedback**: Output becomes next input (internal narrative)
- **Spontaneous Recall**: Energy spikes trigger "memory" surfacing

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

## Continual Learning Configuration

```rust
ContinualConfig {
    buffer_capacity: 10000,       // Max experiences stored
    replay_ratio: 0.3,            // 30% of batch from replay
    priority_exponent: 0.6,       // Higher = more focus on surprising experiences
    forgetting_rate: 0.001,       // Base decay for unused rho states
    forgetting_threshold: 0.01,   // Importance below this triggers forgetting
    consolidation_strength: 0.1,  // Protection for replayed patterns
    consolidation_interval: 100,  // Steps between consolidation
    importance_decay: 0.99,       // EMA decay for importance scores
}
```

Usage:
```rust
let trainer = Trainer::with_continual_learning(
    TrainingConfig::default(),
    ContinualConfig::default(),
);
trainer.train_unsupervised(&mut model, &data);
```

## Self-Thinking Configuration

```rust
BiologicalConfig {
    thermal_noise: 0.015,         // Always-on background noise
    min_energy_floor: 0.02,       // Never truly zero energy
    reflection_feedback: 0.7,     // Self-feedback strength (0.0-1.0)
    recall_threshold: 0.1,        // Threshold for spontaneous recall
    ..Default::default()
}
```

Usage:
```rust
let mut model = HarmonicBdh::with_config(64, 16, 3, config);

// Reflect (think without input)
let trajectory = model.reflect(100);

// Or get stream of consciousness narrative
let narrative = model.stream_of_consciousness(50, false);
for thought in &narrative {
    println!("{}", thought);
}
```

## Temporal Memory Configuration

```rust
BiologicalConfig {
    heartbeat_freq: 1.0,          // 1 Hz metabolic clock
    heartbeat_strength: 0.3,      // Gating strength (0.0-1.0)
    rho_decay: 0.995,             // Memory decay per step
    time_encoding_dims: 8,        // Multi-scale encoding bands
    time_encoding_base: 10000.0,  // Positional encoding base
    ..Default::default()
}
```

Usage:
```rust
let mut model = HarmonicBdh::with_config(64, 16, 3, config);

// Imprint at heartbeat peak
model.advance_to_heartbeat_peak();
model.imprint(&signal, 0);

// Later: recall from partial cue
model.step_time(100.0);
let recalled = model.recall(&partial_cue, 0);
```

## Next Steps

- [x] Experience replay buffer for continual learning
- [x] Adaptive forgetting with importance tracking
- [x] Self-thinking via stochastic resonance
- [x] Stream of consciousness narrative generation
- [x] Temporal memory with heartbeat modulation (Chronos)
- [x] Multi-scale time encoding
- [x] Rho decay for bounded memory
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
