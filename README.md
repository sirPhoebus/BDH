# BDH-GPU: Embodied Cognitive Architecture

A Rust implementation of a biologically-inspired neural architecture that merges **Hebbian learning**, **Harmonic resonance**, and **Embodied Cognition**.

## Grand Unification

The system is now an **Embodied Cognitive Architecture**, consisting of three coupled feedback loops:

1.  **The Brain (Cortex)**:
    *   **GPU-Accelerated**: Using `burn` + `wgpu` for massive parallel dynamics.
    *   **Harmonic**: Neurons oscillate and bind via phase coherence (Theta-Gamma coupling).
    *   **Plastic**: Weights evolve via Hebbian learning modulated by Arousal.

2.  **The Body (Physiology)**:
    *   **Homeostasis**: Tracks Energy, Integrity, and Arousal.
    *   **Valence**: Pain/Pleasure signals drive the brain's "Noise" level (Thrashing vs. Settling).
    *   **Arousal**: Stress/Excitement modulates Plasticity (Input Gain).

3.  **The Mind (Interpreter)**:
    *   **Reflection Loop**: The brain's output is decoded into language ("I notice X").
    *   **Self-Talk**: This narrative is re-embedded into a vector and fed back as input ("Dreaming").
    *   **Drives**: Explicit Hunger and Curiosity states shift the brain between "Seeking", "Playing", and "Resting" modes.

## Architecture

```
        Environment / Corpus
                │
                ▼
        [Semantic Encoder] <───┐
                │              │ (Reflection Loop)
                ▼              │
        ┌────────────────┐     │
        │  CORTEX (GPU)  │ ────┘
        │   Harmonic ρ   │
        └───────┬────────┘
                │ (Modulation: Noise, Damping, Gain)
                ▲
        ┌───────┴────────┐
        │     DRIVES     │ (Hunger, Curiosity)
        └───────┬────────┘
                ▲
        ┌───────┴────────┐
        │      BODY      │ (Energy, Valence, Arousal)
        └────────────────┘
```

## Quick Start

### Grand Unification Simulation
Run the complete system with Body, Brain, and Interpreter loop:
```bash
cargo run --bin central_sim
```
*Watch the console for the interplay of Energy (metabolism), Valence (emotional reaction to concepts), and Mode switching (Reading vs. Dreaming).*

### Other Demos
```bash
# Old CPU-based harmonic demo
cargo run --bin harmonic_demo

# Train embeddings
cargo run --bin train -- --download --epochs 20
```

## Modules

| Module | File | Purpose |
|--------|------|---------|
| `HarmonicBdhBurn` | `harmonic_burn.rs` | **GPU Core**. Tensor-based implementation of harmonic dynamics. |
| `BodyState` | `body.rs` | Physiological simulation (Energy, Integrity, Arousal). |
| `Drives` | `drives.rs` | Intrinsic motivation (Hunger, Curiosity) -> Brain Parameters. |
| `Interpreter` | `interpreter.rs` | Language-to-Vector reflection loop ("Inner Voice"). |
| `Embedder` | `data.rs` | Semantic grounding (Word <-> Vector). |

## Biological Mechanisms

### (A) Embodiment Loop
Concepts are not just Math; they are Feelings.
- **Safety**: Triggers Positive Valence -> Reduces Neural Noise -> Stabilizes State.
- **Danger**: Triggers Negative Valence (Pain) -> Increases Neural Noise -> Destabilizes State (Flight/Fight).

### (B) Arousal-Modulated Plasticity
- **High Arousal** (Stress/Excitement): Increases Input Gain. The brain "pay attentions" and learns instantly.
- **Low Arousal** (Sleep/Boredom): Decreases Input Gain. The brain ignores external stimuli and consolidates.

### (C) Reflection (Dreaming)
When biological drives (Energy) are low, the system disconnects from external text and enters a **Default Mode Network** state:
1.  Decode current neural state -> "I notice Safety"
2.  Extract keyword -> "Safety"
3.  Embed keyword -> Vector input
4.  Feed back into brain -> Associative drift

## Roadmap
- [x] GPU Acceleration (`burn`)
- [x] Embodiment (Body State)
- [x] Explicit Drives
- [x] Interpreter / Reflection Loop
- [x] Semantic Grounding
- [ ] Metacognition (Confidence monitoring)
- [ ] Real-time Audio Input
- [ ] Multi-Modal binding

## License
MIT
