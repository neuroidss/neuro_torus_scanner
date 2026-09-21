# 🧠 NeuroCanvas × TBP.Monty & VLA-JEPA

### Real-Time 3D Embodied Active Inference, CUDA Optic-Flow Raycasting, Egocentric Rotating Manifolds, and Biologically Plausible Local Plasticity

[![DOI:10.1016/j.neuron.2018.09.023](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2018.09.023-purple.svg)](https://doi.org/10.1016/j.neuron.2018.09.023)
[![DOI:10.1016/j.neuron.2013.03.007](https://img.shields.io/badge/DOI-10.1016%2Fj.neuron.2013.03.007-blue.svg)](https://doi.org/10.1016/j.neuron.2013.03.007)
[![DOI:10.1088/1741-2552/aacfe4](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Faacfe4-orange.svg)](https://doi.org/10.1088/1741-2552/aacfe4)
[![DOI:10.3389/fncir.2018.00053](https://img.shields.io/badge/DOI-10.3389%2Ffncir.2018.00053-green.svg)](https://doi.org/10.3389/fncir.2018.00053)
[![arXiv:2506.09985](https://img.shields.io/badge/arXiv-2506.09985-red.svg)](https://arxiv.org/abs/2506.09985)
[![arXiv:2602.10098](https://img.shields.io/badge/arXiv-2602.10098-yellow.svg)](https://arxiv.org/abs/2602.10098)

---

## 📑 Table of Contents
1. [Paradigm Shift: Embodied Active Inference vs. Mechanical BCI](#1-paradigm-shift-embodied-active-inference-vs-mechanical-bci)
2. [Prefrontal Neurodynamics & The Cortical Engine](#2-prefrontal-neurodynamics--the-cortical-engine)
   - 2.1 [FCz / Supplementary Motor Area (SMA) & Body-Frame Kinematics](#21-fcz--supplementary-motor-area-sma--body-frame-kinematics)
   - 2.2 [32-Slot Theta-Gamma PAC Comet Tail (Lisman & Jensen, 2013)](#22-32-slot-theta-gamma-pac-comet-tail-lisman--jensen-2013)
   - 2.3 [Working Memory 2.0 & Infragranular Beta Gating (Miller et al., 2018)](#23-working-memory-20--infragranular-beta-gating-miller-et-al-2018)
   - 2.4 [120-Edge Volume-Conduction-Free $ci\text{PLV}$ (Bruña et al., 2018)](#24-120-edge-volume-conduction-free-ciplv-bruña-et-al-2018)
   - 2.5 [Strict Elimination of Backpropagation: Hebbian Oja + Widrow-Hoff Delta Rule](#25-strict-elimination-of-backpropagation-hebbian-oja--widrow-hoff-delta-rule)
3. [VLA-JEPA World Modeling & The Ecological Aperture Problem](#3-vla-jepa-world-modeling--the-ecological-aperture-problem)
   - 3.1 [Aperture Collapse in Vision Transformers (Gibson Optic Flow Theory)](#31-aperture-collapse-in-vision-transformers-gibson-optic-flow-theory)
   - 3.2 [High-Throughput Parallel CUDA Tensor Raycaster ($>120\text{ FPS}$)](#32-high-throughput-parallel-cuda-tensor-raycaster-120text-fps)
   - 3.3 [Action Conditioning in Latent Feature Space ($2048\text{D}$)](#33-action-conditioning-in-latent-feature-space-2048textd)
4. [Egocentric Rotating Navigation Engine](#4-egocentric-rotating-navigation-engine)
   - 4.1 [Body-Centric Coordinate Invariance (Forward/Strafe/Yaw)](#41-body-centric-coordinate-invariance-forwardstrafeyaw)
   - 4.2 [Rotating Map Transformation Matrix & Compass Needle](#42-rotating-map-transformation-matrix--compass-needle)
   - 4.3 [Recursive Backtracking Maze Topology](#43-recursive-backtracking-maze-topology)
5. [Live Diagnostics & Dual-Sensor Telemetry](#5-live-diagnostics--dual-sensor-telemetry)
6. [Quickstart & Controls Reference](#6-quickstart--controls-reference)
7. [Comprehensive Scientific Bibliography & DOIs](#7-comprehensive-scientific-bibliography--dois)

---

## 1. Paradigm Shift: Embodied Active Inference vs. Mechanical BCI

Traditional Brain-Computer Interfaces (BCIs) treat neural activity as an external steering joystick, attempting to classify discrete categories (e.g., *Forward, Backward, Left, Right*). This introduces severe quantization noise, eliminates continuous motor persistence, and violates basic cortical physiology.

**NeuroCanvas × TBP.Monty & VLA-JEPA** unifies the **Thousand Brains Theory (Hawkins et al., 2017, 2025/2026)** with **Joint-Embedding Predictive Architectures (Assran et al., 2025; Sun et al., 2026)**:
* **The Cortex Does Not Classify Buttons:** Medial prefrontal cortex (**FCz**, Supplementary Motor Area / pre-SMA / dACC) maintains continuous, structured trajectories across abstract reference frames.
* **Separation of World Prediction and Motor Execution:** 
  1. The **Frontal Executive Heterarchy** (L4/L2-3 HTM columns) monitors cortical phase-amplitude coupling and predicts the next target state in a high-dimensional latent space ($z_{target} \in \mathbb{R}^{2048}$).
  2. The **Motor Readout (M1 / Action Layer)** resolves inverse kinematics through local synaptic error minimization, translating latent intentions into continuous body-frame propulsion $(v_{fwd}, v_{str})$.
* **Zero Artificial Backpropagation:** Synaptic weights are updated strictly through local biophysical plasticity (Hebbian competitive learning, Oja's rule, and the Widrow-Hoff Delta rule). Global gradient backpropagation (Adam, SGD) is strictly forbidden.

```
                  EMBODIED BRAIN-BODY ACTIVE INFERENCE ARCHITECTURE
                  
       ┌─────────────────────────────────────────────────────────────────┐
       │             HIGH-DENSITY MESOSCOPIC EEG (16-CH @ FCz)           │
       │    120-Edge Directed ciPLV • 32 Theta-Gamma PAC Slices          │
       └───────────────────────────────┬─────────────────────────────────┘
                                       │ Raw Phase-Amplitude Tensor
                                       ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │            CORTICAL HTM WORKING MEMORY (L4 / L2-3)              │
       │   • 4,096 Columns with Topological Receptive Fields             │
       │   • kWTA Sparsity: k=80 active (1.95% density)                  │
       │   • Calcium Trace Integration • 16 Emergent Attractor Concepts  │
       └───────────────┬─────────────────────────────────┬───────────────┘
                       │                                 │
         Synaptic      │                                 │ Predicted Concept
         Plasticity    │                                 │ Weights w ∈ R^16
         (Oja's Rule)  ▼                                 ▼
       ┌───────────────────────────────┐ ┌───────────────────────────────┐
       │   HEBBIAN ALIGNMENT MATRIX    │ │   PREDICTED LATENT EMBEDDING  │
       │      W_concept ∈ R^(16×2048)  │ │      z_pred ∈ R^2048          │
       └───────────────┬───────────────┘ └───────────────┬───────────────┘
                       │                                 │
                       │ Latent Feedback                 │ Cosine Comparison
                       ▼                                 ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │                   V-JEPA 2 EMBEDDED WORLD MODEL                 │
       │   Real-Time 3D Optic Flow (256×256) ──► Latent State z ∈ R^2048 │
       └───────────────────────────────┬─────────────────────────────────┘
                                       │ Fused State (z_real + w_HTM)
                                       ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │            MOTOR ACTION READOUT (WIDROW-HOFF DELTA)             │
       │   Biologically Plausible Error Correction: dW = η · e ⊗ s       │
       │   Output: Continuous Body Propulsion (v_fwd, v_strafe, turn)    │
       └───────────────────────────────┬─────────────────────────────────┘
                                       │ 125 FPS Kinematic Commands
                                       ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │                EGOCENTRIC ROTATING MAZE RUNNER                  │
       │   • Avatar Centered Facing UP • World Rotates Clockwise on Left │
       │   • 32-Slot PAC Comet Trail • True North Needle                 │
       └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Prefrontal Neurodynamics & The Cortical Engine

### 2.1 FCz / Supplementary Motor Area (SMA) & Body-Frame Kinematics
Electrode site **FCz** is positioned directly over the medial wall of the frontal cortex:
* **pre-SMA & SMA Proper:** Responsible for internally generated motor sequences, temporal chunking, and action representation in ego-centric/body coordinates rather than Cartesian world space.
* **Dorsal Anterior Cingulate Cortex (dACC):** Operates as a prediction error and conflict resolution hub (Alexander & Brown, 2011; Womelsdorf et al., 2010), evaluating whether intended trajectories match incoming sensory states.
* **Body-Frame Invariance:** Forward movement is defined along the avatar’s current viewing vector ($\vec{u}_{fwd}$), while lateral shifts represent true strafing ($\vec{u}_{str}$), eliminating the unnatural mental rotation required by stationary world-fixed frames.

### 2.2 32-Slot Theta-Gamma PAC Comet Tail (Lisman & Jensen, 2013)
Under the Lisman-Jensen phase-amplitude coupling (PAC) model, working memory sequences are time-division multiplexed into high-gamma subcycles ($30\text{--}100\text{ Hz}$) nested within an endogenous theta carrier ($4\text{--}8\text{ Hz}$):
* Each theta cycle is discretized into **32 temporal phase slices**.
* **Slot 0 Subtraction (Downbeat Anchor):** Slices $1\dots31$ are phase-referenced against the initial anchor (Slot 0), yielding a zero-drift relative trajectory $\text{traj}_{32} \in \mathbb{R}^{32 \times 2}$.
* **Spectral Comet Trail:** Rendered in real time on the 2D navigator as a dynamic gradient comet tail projecting from the avatar's center. It reveals the forward predictive lookahead sweep of the cortex before physical movement is executed.
* **Dynamic Vector Amplitude:** Arrow lengths are never clamped to fixed pixel lengths; they scale dynamically with **thrust**, **temporal persistence** ($\mu = \tanh(2 \|\vec{F}\|)$), and **HTM predictive confidence**.

### 2.3 Working Memory 2.0 & Infragranular Beta Gating (Miller et al., 2018)
In accordance with **Working Memory 2.0**:
* **Superficial Layers (L2/3):** Gamma bursts ($30\text{--}100\text{ Hz}$) encoding active representations and sensory transitions.
* **Deep Layers (L5/6):** Infragranular beta rhythms ($15\text{--}30\text{ Hz}$) exerting top-down inhibitory control over superficial gamma.
* **Dynamic Gating:** Deep-layer beta desynchronization opens the gate, allowing high-gamma activation and motor intent transmission. Re-emerging beta terminates the action chunk and clears the buffer.

### 2.4 120-Edge Volume-Conduction-Free $ci\text{PLV}$ (Bruña et al., 2018)
Scalp EEG is corrupted by zero-lag volume conduction. To isolate genuine physiological phase locking across the 16-channel array, NeuroCanvas computes the **Corrected Imaginary Phase-Locking Value ($ci\text{PLV}$)** across all 120 electrode pairs:

$$ci\text{PLV}_{j, k} = \frac{\frac{1}{T} \Im \left\lbrace \sum_{t=1}^T \dot{x}_j(t) \cdot \dot{x}_k^*(t) \right\rbrace}{\sqrt{1 - \left( \frac{1}{T} \Re \left\lbrace \sum_{t=1}^T \dot{x}_j(t) \cdot \dot{x}_k^*(t) \right\rbrace \right)^2}}$$

Any non-cerebral common-mode artifact (e.g., muscle tension, blink, contact polarization) has zero imaginary phase shift and vanishes ($\sin(0) = 0$).

### 2.5 Strict Elimination of Backpropagation: Hebbian Oja + Widrow-Hoff Delta Rule
The neocortex does not calculate global loss derivatives or employ gradient descent. In `MontyVLA`:
1. **Hebbian Alignment (Oja's Rule):** When a cortical concept $k$ is selected by competitive Winner-Take-All inhibition, its associative projection to the visual latent space updates locally:
   $$\mathbf{W}_{\text{concept}}[k] \leftarrow \mathbf{W}_{\text{concept}}[k] + \eta_{\text{align}} \left( \vec{z}_{\text{JEPA}} - \mathbf{W}_{\text{concept}}[k] \right)$$
   followed by unit $L_2$ normalization to enforce synaptic stability.
2. **Motor Output Synapses (Widrow-Hoff Delta Rule):** Action readout is performed by a single synaptic matrix $\mathbf{W}_{\text{motor}}$ mapping the joint state $\vec{s}_{\text{VLA}} = [\vec{z}_{\text{JEPA}}, \vec{w}_{\text{HTM}}]$ to muscle drives:
   $$\vec{a}_{\text{pred}} = \mathbf{W}_{\text{motor}} \cdot \vec{s}_{\text{VLA}}, \quad \vec{e} = \vec{a}_{\text{target}} - \vec{a}_{\text{pred}}$$
   $$\mathbf{W}_{\text{motor}} \leftarrow \mathbf{W}_{\text{motor}} + \eta_{\text{motor}} (\vec{e} \otimes \vec{s}_{\text{VLA}})$$
   This ensures strict biological plausibility: updates are computed entirely from pre-synaptic activation and post-synaptic error without Adam, backpropagation, or computation graphs.

---

## 3. VLA-JEPA World Modeling & The Ecological Aperture Problem

### 3.1 Aperture Collapse in Vision Transformers (Gibson Optic Flow Theory)
Under the ecological optics framework (Gibson, 1950, 1979; Marr & Ullman, 1981):
* **The Failure of Flat Shading:** Self-supervised Vision Transformers (ViT) patchify images into local $16 \times 16$ windows. If walls are rendered with uniform solid colors, $\nabla I = 0$ within all interior patches. No spatial luminance gradient exists across frames, causing the aperture problem: the ViT cannot distinguish forward locomotion from standing still.
* **Procedural Texture Synthesis:** `GPURaycaster` injects high-frequency procedural texture gradients:
  - **Wall Block Mortar:** Calculated from ray hit coordinates $u = (x_{\text{hit}} + y_{\text{hit}}) \pmod 1$ and normalized vertical height $v$.
  - **Perspective Floor Striping:** Raymarched as a function of screen vertical coordinate $y$, generating exponential spatial compression toward the horizon.
  - **Optic Flow Induction:** Forward movement generates rapid radial flow across the floor and lateral expansion across wall patches, allowing V-JEPA's tubelet attention ($2 \times 16 \times 16$) to immediately register non-zero latent velocity $\|\Delta z_t\| > 0.40$.

### 3.2 High-Throughput Parallel CUDA Tensor Raycaster ($>120\text{ FPS}$)
Custom CPU raycasting loops (`for ray in range(256): while not hit:`) bottleneck Python interpreters, dropping frame rates to 10–20 FPS. 

`GPURaycaster` executes entirely within PyTorch GPU tensors:
1. **Parallel Raymarching:** Evaluates 256 rays $\times$ 180 depth steps ($46,080$ spatial points) simultaneously via broadcasting.
2. **Zero CPU Roundtrips:** Ray-wall intersections, distance calculations, fish-eye correction, wall height clamping, and texture generation occur in video memory.
3. **Execution Time:** $<0.25\text{ ms}$ on NVIDIA RTX architectures, ensuring a locked $120\text{ FPS}$ pipeline.

### 3.3 Action Conditioning in Latent Feature Space ($2048\text{D}$)
Following the VLA-JEPA framework (Sun et al., 2026):
* Visual frames from the 3D sensor are passed to `VLA_JEPA_Wrapper`, producing a $2048\text{D}$ self-supervised world state representation $\vec{z}_t \in \mathbb{R}^{2048}$.
* Rather than predicting raw pixels, the model matches latent predictions. The real-time metric $\|\Delta z_t\| = \|\vec{z}_t - \vec{z}_{t-1}\|_2$ quantifies the rate of environmental state change, providing an objective metric of sensory flow.

---

## 4. Egocentric Rotating Navigation Engine

### 4.1 Body-Centric Coordinate Invariance (Forward/Strafe/Yaw)
Physical movement is mapped directly to the avatar's internal orientation $\theta_{\text{avatar}}$:
* **Longitudinal Drive ($v_{\text{fwd}}$):** Actuated by pushing forward on the stick/BCI ($-ly$).
* **Lateral Drive ($v_{\text{str}}$):** Actuated by pushing sideways ($lx$).
* **Angular Yaw ($\dot{\theta}$):** Actuated by prefrontal sagitta curvature ($-rx \cdot 3.2$) or keyboard turning keys (`Q` / `E`).

The resulting velocity in world coordinates is:

$$\vec{v}_{\text{world}} = v_{\text{fwd}} \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix} + v_{\text{str}} \begin{bmatrix} \sin\theta \\ -\cos\theta \end{bmatrix}$$

```
                           AVATAR BODY REFERENCE FRAME
                                      ▲
                                      │ +v_fwd (Forward / UP on Screen)
                                      │
           -v_strafe (Strafe Left) ◄──┼──► +v_strafe (Strafe Right)
                                      │
                                      ▼
                                      -v_fwd (Backward / DOWN on Screen)
```

### 4.2 Rotating Map Transformation Matrix & Compass Needle
* **Egocentric Centering:** The avatar is permanently locked at the center of the viewport $(C_x, C_y)$, facing straight **UP** ($(0, -1)$ in screen coordinates).
* **World Rotation:** Turning left ($\dot{\theta} > 0$) rotates the visual world clockwise (to the right) by $-\theta$.
* **Coordinate Mapping:** Any world coordinate $(x, y)$ is transformed to screen pixel space $(X_{\text{screen}}, Y_{\text{screen}})$ via:

$$\begin{bmatrix} X_{\text{screen}} \\ Y_{\text{screen}} \end{bmatrix} = \begin{bmatrix} C_x \\ C_y \end{bmatrix} + \text{CELL\_SIZE} \cdot \begin{bmatrix} \sin\theta & -\cos\theta \\ -\cos\theta & -\sin\theta \end{bmatrix} \begin{bmatrix} x - x_{\text{avatar}} \\ y - y_{\text{avatar}} \end{bmatrix}$$

* **Compass Rose:** A red needle with a marker **[N]** continuously tracks the orientation of True North $(0, 1)$ relative to the avatar's current heading.

### 4.3 Recursive Backtracking Maze Topology
* **Perfect Grid Carving:** Built on an odd grid ($11 \times 11$). Steps jump by two cells while explicitly breaking down the wall between: `grid[y + dy][x + dx] = 0`.
* **Zero Isolated Wall Islands:** Guarantees 100% topological connectivity from start $(1, 1)$ to exit $(dim - 2, dim - 2)$.
* **Allocentric Wavefront BFS:** Breadth-First Search precalculates the exact distance gradient field $D(x, y)$ to the exit across all traversable corridors.

---

## 5. Live Diagnostics & Dual-Sensor Telemetry

```
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│     EGOCENTRIC ROTATING MAP (2D)      │ │        V-JEPA 3D VISION SENSOR        │
│                                       │ │                                       │
│  • Avatar locked at center facing UP  │ │  • Parallel CUDA Raycaster (256×256) │
│  • World rotates around avatar        │ │  • Quake-style textured optic flow    │
│  • 32-Slot PAC Comet Trail (Lisman)   │ │  • High-frequency brick & floor grid  │
│  • Dynamic Human Intention Vector     │ │                                       │
│  • Dynamic Monty Action Head Vector   │ │                                       │
│  • Red True North Compass Needle      │ │                                       │
└───────────────────────────────────────┘ └───────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME NEUROPHYSIOLOGICAL TELEMETRY                        │
│  • V-JEPA 2 State: Ego-Velocity ||dz|| (Flow) • World Prediction Similarity     │
│  • HTM Working Memory: Top Active Concepts (L2/3) • Autonomy Score [0–100%]     │
│  • Kinematics: Body Forward / Strafe • Sagitta (rx) • Temporal Bias (ry)        │
│  • Spectral Connectivity: 120-Edge Directed ciPLV Network (65–100 Hz Ripples)   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

* **Ego-Velocity $\|\Delta z_t\|$:** Shows real-time latent displacement. Values above $0.200$ confirm that V-JEPA is actively perceiving 3D optic flow.
* **World Prediction Sim ($\cos(z_{\text{pred}}, z_{\text{real}})$):** Quantifies how accurately the HTM working memory anticipates the incoming visual state.
* **Autonomy Score ($\Omega_{\text{LTM}}$):** Measures agreement between Monty's Action Head output and human teacher demonstrations over a rolling 150-step queue.

---

## 6. Quickstart & Controls Reference

### 1. Launch VLA-JEPA Server (Optional, Port 6001)
```bash
python jepa_server.py --port 6001 --device cuda
```
*(If the server is offline, the script gracefully switches to lightweight standalone mode without crashing).*

### 2. Launch NeuroCanvas Maze Engine
```bash
python neuro_monty_maze_learning.py
```

### Controls:
| Input | Action | Functional Description |
| :--- | :--- | :--- |
| **BCI Joystick / FCz** | **Body Movement** | Longitudinal forward/back ($-ly$) and lateral strafe ($lx$). |
| **BCI Sagitta ($rx$)** | **Yaw Steering** | Trajectory curvature turns the avatar heading and rotates the world. |
| `W` / `UP` | **Forward** | Thrust along current avatar heading. |
| `S` / `DOWN` | **Backward** | Reverse along current avatar heading. |
| `A` | **Strafe Left** | Lateral slide perpendicular to heading. |
| `D` | **Strafe Right** | Lateral slide perpendicular to heading. |
| `Q` / `LEFT` | **Turn Left** | Rotates avatar left (world rotates clockwise/right). |
| `E` / `RIGHT` | **Turn Right** | Rotates avatar right (world rotates counter-clockwise/left). |
| `SPACE` | **Autopilot Toggle** | Transfers control from Human Teacher to Autonomous Monty. |
| `R` | **Reset Maze** | Generates a new random recursive backtracking maze. |

---

## 7. Comprehensive Scientific Bibliography & DOIs

1. **Lisman, J. E., & Jensen, O. (2013).** The theta-gamma neural code. *Neuron*, 77(6), 1002–1016. [DOI: 10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)
2. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).** Working Memory 2.0. *Neuron*, 100(2), 463–475. [DOI: 10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)
3. **Bruña, R., Maestú, F., & Pereda, E. (2018).** Phase Locking Value revisited: teaching new tricks to an old dog. *Journal of Neural Engineering*, 15(5), 056011. [DOI: 10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)
4. **Nolte, G., et al. (2004).** Identifying true brain interaction from EEG data using the imaginary part of coherency. *Clinical Neurophysiology*, 115(10), 2292–2307. [DOI: 10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)
5. **Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018).** Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits*, 12, 53. [DOI: 10.3389/fncir.2018.00053](https://doi.org/10.3389/fncir.2018.00053)
6. **Hawkins, J., Leadholm, N., & Clay, V. (2025/2026).** The Thousand Brains Theory 2.0: An Extension for the Long-Range Connections of the Neocortical Heterarchy. *arXiv preprint*, [arXiv:2507.05888](https://arxiv.org/abs/2507.05888).
7. **Hawkins, J., Ahmad, S., & Cui, Y. (2017).** A theory of how columns in the neocortex enable learning the structure of the world. *Frontiers in Neural Circuits*, 11, 81. [DOI: 10.3389/fncir.2017.00081](https://doi.org/10.3389/fncir.2017.00081)
8. **Assran, M., et al. (2025).** V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. *arXiv preprint*, [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).
9. **Sun, J., et al. (2026).** VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. *arXiv preprint*, [arXiv:2602.10098](https://arxiv.org/abs/2602.10098).
10. **Gibson, J. J. (1950).** *The Perception of the Visual World.* Houghton Mifflin.
11. **Gibson, J. J. (1979).** *The Ecological Approach to Visual Perception.* Houghton Mifflin. [ISBN: 9780898599596]
12. **Marr, D., & Ullman, S. (1981).** Directional selectivity and its use in early visual processing. *Proceedings of the Royal Society of London. Series B. Biological Sciences*, 211(1183), 151–180. [DOI: 10.1098/rspb.1981.0001](https://doi.org/10.1098/rspb.1981.0001)
13. **Dickey, C. W., et al. (2022).** Widespread ripples synchronize human cortical activity during sleep, waking, and memory recall. *PNAS*, 119(28), e2107797119. [DOI: 10.1073/pnas.2107797119](https://doi.org/10.1073/pnas.2107797119)
14. **Chen, J., Zhang, C., Hu, P., Min, B., & Wang, L. (2024).** Flexible control of sequence working memory in the macaque frontal cortex. *Neuron*, 112(20), 3502–3514. [DOI: 10.1016/j.neuron.2024.07.024](https://doi.org/10.1016/j.neuron.2024.07.024)
15. **Fan, Y., Wang, M., Ding, N., & Luo, H. (2024).** Two-dimensional neural geometry underpins hierarchical organization of sequence in human working memory. *Nature Human Behaviour*, 8, 2150–2163. [DOI: 10.1038/s41562-024-02047-8](https://doi.org/10.1038/s41562-024-02047-8)
16. **Friston, K. (2010).** The free-energy principle: a unified brain theory?. *Nature Reviews Neuroscience*, 11(2), 127–138. [DOI: 10.1038/nrn2787](https://doi.org/10.1038/nrn2787)
17. **Oja, E. (1982).** Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267–273. [DOI: 10.1007/BF00275687](https://doi.org/10.1007/BF00275687)
18. **Widrow, B., & Hoff, M. E. (1960).** Adaptive switching circuits. *IRE WESCON Convention Record*, 4, 96–104.
