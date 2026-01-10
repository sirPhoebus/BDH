# BDH-GPU (Brain-like Dense Hebbian)

A Rust implementation of a biologically-inspired neural architecture combining:
- **Hebbian learning** with persistent memory (ρ state)
- **Harmonic resonance** dynamics (phase-aligned oscillators)
- **LSH embeddings** for positive-orthant projections

## Architecture

```
TokenID → Embedding → LSH (positive orthant) → BDH Layers → Sparse Output
                                                    ↓
                                            Standing Wave ρ
                                          (Hebbian memory)
```

### Two Modes

| Mode | File | Description |
|------|------|-------------|
| **Standard BDH** | `src/lib.rs` | Low-rank Hebbian updates, ReLU sparsity |
| **Harmonic BDH** | `src/harmonic.rs` | Complex ρ (phase+amplitude), resonance gating |

## Quick Start

```bash
# Standard BDH with LSH embeddings
cargo run

# Harmonic/vibrational demo
cargo run --bin harmonic_demo

# Run tests
cargo test
```

## Key Concepts

### Standard BDH
- **ρ update**: `ρ = ρ * U + outer(E·y, x)` — Hebbian accumulation with decay
- **Sparsity**: ReLU + negative biases → ~5% neuron activation
- **Memory**: O(n) in neurons, O(1) in sequence length (unbounded context)

### Harmonic BDH
- **Complex ρ**: Phase encodes timing, amplitude encodes strength
- **Phase alignment**: Constructive interference gates information flow
- **Attractor states**: Limit cycles = stable concepts
- **Brain snapshot**: FFT of signal initializes standing wave patterns

## Modules

| Module | Purpose |
|--------|---------|
| `BdhGpu` | Core BDH model with shared E, Dx, Dy matrices |
| `HarmonicBdh` | Oscillator-based variant with phase dynamics |
| `LshEmbedder` | SimHash projection to positive orthant |
| `Vocabulary` | Token → embedding lookup (placeholder) |

## Example Output

```
=== Harmonic BDH: Vibrational Neural Architecture ===

1. BRAIN SNAPSHOT: Initializing from synthetic brainwave...
   Initial standing wave energy: 0.6206

2. RESONANCE DYNAMICS: Processing input signal...
   Output: mean=0.0140, sparsity=50.8%

3. ATTRACTOR DISCOVERY: Finding stable resonance pattern...
   Converged in 64 iterations

4. ZERO-SHOT RESONANCE: Introducing a new frequency...
   10 Hz response: 3x stronger than 7 Hz (model tuned to 10 Hz)
```

## Next Steps

- [ ] Kuramoto coupling for true oscillator synchronization
- [ ] Real WAV/EEG loading via `hound` crate
- [ ] Autograd via `burn` or `dfdx` for training Dx, Dy, E
- [ ] GPU acceleration with `wgpu` or `cuda`

## References

- BDH paper: Scale-free Hebbian networks with linear attention
- Kuramoto model: Coupled oscillator synchronization
- Cymatics: Standing wave pattern formation
