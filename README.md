# BDH-GPU: Embodied Cognitive Architecture

A Rust implementation of a biologically-inspired neural architecture that merges **Hebbian learning**, **Harmonic resonance**, and **Embodied Cognition**.

This project simulates a brain that doesn't just process data, but **feels**, **needs**, and **dreams**.

## 🧠 Grand Unification (Central Sim)

The core executable `central_sim` integrates three originally separate feedback loops into a single cohesive entity:

1.  **The Brain (Cortex)**:
    *   **GPU-Accelerated**: Using `burn` + `wgpu` for massive parallel dynamics.
    *   **Harmonic**: Neurons oscillate and bind via phase coherence (Theta-Gamma coupling).
    *   **Plastic**: Weights evolve via Hebbian learning, modulated dynamically by Neurotransmitters.

2.  **The Body (Physiology)**:
    *   **Homeostasis**: Tracks Energy, Integrity, and Arousal.
    *   **Neurochemistry**: 
        *   **Dopamine (DA)**: Driven by reward/prediction error.
        *   **Norepinephrine (NE)**: Driven by stress/arousal.
        *   **Acetylcholine (ACh)**: Modulate attention and memory encoding.
    *   **Drives**: Explicit **Hunger** (for data) and **Curiosity** (for novelty) shift the agent's behavior.

3.  **The Mind (Cognition)**:
    *   **Episodic Memory**: Hippocampal replay of high-valence events.
    *   **Associative Chaining**: "Dreaming" state where thoughts drift based on semantic similarity.
    *   **Interpreter**: Decodes neural states into English narrative ("I feel...").

## 📊 Real-Time Telemetry

The system exposes its internal state in real-time via `probe.json`. A Python companion tool visualizes this data live.

### Included Visualization (`monitor_probe.py`)
Plots the following metrics in real-time:
- **Coherence**: How synchronized the brain states are.
- **Diversity**: The richness of concepts currently being activated.
- **Energy Stability**: Metabolic health of the simulation.
- **Concepts Learned**: Accumulation of novel patterns.

### 3D Visualization (Web)
The simulation also hosts a local web server for 3D monitoring.
- **URL**: `http://localhost:3000`
- **Features**: Real-time 3D view of neuronal firing, region highlighting, and neurotransmitter levels.

## 🚀 Quick Start

### 1. Prerequisites
- **Rust**: Latest stable.
- **Python**: 3.8+ with `matplotlib` for the visualizer.
  ```bash
  pip install matplotlib
  ```

### 2. Run the Simulation + Monitor

**Terminal 1 (Visualization):**
Start this first to catch the early data.
```bash
python monitor_probe.py
```

**Terminal 2 (The Brain):**
```bash
cargo run --release --bin central_sim
```

*Note: The simulation runs in "Silent Mode" in the terminal to optimize performance. Watch the graphs window for activity.*

## 🏗️ Architecture

```
        Environment / Corpus
                │
                ▼
        [Semantic Encoder] <───┐
                │              │ (Dream/Reflection Loop)
                ▼              │
        ┌────────────────┐     │
        │  CORTEX (GPU)  │ ────┘
        │   Harmonic ρ   │
        └───────┬────────┘
                │ (Modulation: DA, NE, ACh)
                ▲
        ┌───────┴────────┐
        │   NEUROCHEM    │ (Drives: Hunger, Curiosity)
        └───────┬────────┘
                ▲
        ┌───────┴────────┐
        │      BODY      │ (Energy, Valence, Arousal)
        └────────────────┘
```

## 🧩 Modules

| Module | File | Purpose |
|--------|------|---------|
| `central_sim` | `src/bin/central_sim.rs` | **Main Entrypoint**. The "Grand Loop" integrating all systems. |
| `monitor_probe.py` | `monitor_probe.py` | **Telemetry**. Real-time graphing tool. |
| `HarmonicBdhBurn` | `src/harmonic_burn.rs` | **GPU Core**. Tensor-based physics of the neural substrate. |
| `BodyState` | `src/body.rs` | Physiological sumulation. |
| `Drives` | `src/drives.rs` | Intrinsic motivation engine. |
| `MemorySystem` | `src/memory.rs` | Vector database for Hippocampal/Episodic storage. |
| `Continual` | `src/continual.rs` | Adaptive forgetting and experience replay logic. |

## 🔮 Roadmap

- [x] GPU Acceleration (`burn` + `wgpu`)
- [x] Bio-Chemical Modulation (DA/NE/ACh)
- [x] Embodiment & Homeostasis
- [x] Episodic Memory & Replay
- [x] Real-time Telemetry Probing
- [ ] Metacognition (Confidence-aware planning)
- [ ] Multi-Modal Input (Audio/Visual binding)
- [ ] Sleep-Wake Cycles (Circadian Rhythm)

## License
MIT
