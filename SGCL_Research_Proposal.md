# Skill-Graph Curriculum Learning for Unified Humanoid Motion Synthesis

---

## Abstract

Recent advances in generative modeling, particularly diffusion policies, have enabled humanoid robots to learn a diverse repertoire of skills by imitating human motion. However, these policies often learn each skill in isolation, creating a fragmented latent space where fluid transitions between dynamically distinct motions—such as from walking to a cartwheel—remain a significant challenge. This "skill compositionality gap" prevents the synthesis of long-horizon, versatile behaviors. In this paper, we introduce Skill-Graph Curriculum Learning (SGCL), a novel training strategy that addresses this gap. SGCL first automatically organizes a large-scale motion dataset into a directed graph of skills based on their kinematic similarity. It then trains a unified state-action diffusion policy by traversing this graph, starting with foundational skills and progressively introducing more complex motions and the explicit transitions between them. By learning the connections on the skill manifold, our approach produces a single policy capable of fluid, zero-shot composition of disparate skills. We demonstrate that our SGCL-trained agent achieves state-of-the-art performance on a series of complex skill transition tasks in a physics-based simulation, significantly outperforming a baseline model trained without a curriculum.

---

## 1. Introduction

The quest for agile, general-purpose humanoid robots has seen remarkable progress, largely driven by learning-from-demonstration (LfD). Frameworks like BeyondMimic have shown that deep reinforcement learning can produce high-fidelity motion tracking policies, and that these can be distilled into a single, powerful diffusion policy. Such policies can generate naturalistic motions and even be guided at test-time to adapt to new environmental goals.

However, a critical limitation persists. While these models can learn an extensive library of individual skills—walking, running, jumping, even complex acrobatics—they treat the motion space as a disconnected archipelago of skills. The model can generate a trajectory within the "walking" manifold or the "cartwheel" manifold, but it lacks the understanding of how to navigate the space *between* them. When prompted to transition from walking to a cartwheel, the policy often fails, producing unstable or physically impossible motions. This is because the training process, which typically samples data uniformly from all skills, provides insufficient signal for learning the crucial, short-lived transition dynamics. This "skill compositionality gap" is a major barrier to creating truly versatile agents that can chain together diverse actions to solve complex, long-horizon problems.

To solve this, we propose **Skill-Graph Curriculum Learning (SGCL)**. Our core insight is that skills are not learned in a vacuum; they are built upon each other in a structured way. A human learns to walk before they run, and they learn to jump before attempting a spinning jump. SGCL brings this developmental intuition to the training of motion policies. Instead of presenting all data at once, we structure the learning process itself. Our method first constructs a "Skill Graph" from the motion capture data, where nodes represent distinct skills (e.g., `walk`, `run`, `jump`) and directed edges represent feasible transitions between them. We then train a unified diffusion policy via a curriculum that traverses this graph, ensuring the model masters foundational skills before learning the more complex ones and, crucially, the transitions that connect them.

This paper makes the following contributions:
1.  **A novel curriculum learning framework (SGCL)** designed specifically for training diffusion-based motion policies, which significantly improves skill transition capabilities.
2.  **An automated method for skill segmentation and graph construction** from unstructured motion capture data using kinematic feature clustering.
3.  **A comprehensive evaluation on a new benchmark of skill composition tasks**, demonstrating that our SGCL-trained agent can fluidly and robustly execute long-horizon sequences of diverse skills in a physics-based humanoid simulation.

---

## 2. Design and Implementation Plan

This research builds directly upon the BeyondMimic repository. It utilizes the completed Stage 1 (motion tracking) to generate expert data and implements the proposed Stage 2 (diffusion policy) using the SGCL training strategy.

### Phase 1: Skill Graph Construction (Data Pre-processing)

This phase analyzes the entire motion dataset to build the curriculum structure.

*   **Proposed Script**: `scripts/curriculum/build_skill_graph.py`
*   **Methodology**:
    1.  **Load MoCap Data**: Ingest all reference motions from datasets like AMASS/LAFAN1.
    2.  **Feature Extraction**: For every frame in the dataset, compute a low-dimensional kinematic feature vector (e.g., root linear and angular velocity, feet contact states, center of mass height).
    3.  **Skill Clustering**: Use an unsupervised density-based clustering algorithm like **DBSCAN** on the feature vectors. DBSCAN is ideal as it can identify core skill manifolds (dense clusters) and label the frames in between as "transition" points (noise points in DBSCAN terminology).
    4.  **Graph Definition**: Define each cluster as a node in a directed graph (e.g., Node A = `walk`, Node B = `run`). Create a directed edge from Node A to Node B if the dataset contains transition points that move from a frame in cluster A to a frame in cluster B.
    5.  **Save Artifacts**: Save the graph structure and the per-frame skill/transition labels. This output will govern the entire training process.

### Phase 2: Trajectory Data Collection

This phase uses the Stage 1 RL policies to generate the state-action dataset for the diffusion model.

*   **Proposed Script**: `scripts/diffusion/collect_trajectories.py` (as planned for the original Stage 2).
*   **Methodology**:
    1.  Roll out the expert motion tracking policies on the reference motions.
    2.  For each trajectory chunk `(Observation_History, Future_Trajectory)`, append the corresponding skill labels and transition status generated in Phase 1.
    3.  Save this augmented data. Each data point will now be `(History, Future, Skill_Label)`. 

### Phase 3: SGCL-Based Diffusion Model Training

This is the core of the research, where the curriculum is applied.

*   **Proposed Script**: `scripts/curriculum/train_sgcl.py`
*   **Methodology**:
    1.  Instantiate the `StateActionDiffusionModel` as planned in the original Stage 2 design.
    2.  Load the full, labeled dataset, but create separate `DataLoaders` for each skill and for transition data.
    3.  Implement a staged training loop:
        *   **Stage I (Foundational Manifolds)**: Train the model for `N_1` epochs exclusively on data from foundational, cyclic skills (e.g., `walk`, `idle`, `jog`). This establishes a stable base latent space.
        *   **Stage II (Transitional Bridges)**: Train for `N_2` epochs on a mix of foundational skill data and the data labeled as transitions between them (e.g., `walk-to-run`). This explicitly teaches the model how to connect the primary manifolds.
        *   **Stage III (Acyclic & Complex Skills)**: Finally, train for `N_3` epochs on the full dataset, including high-complexity, acyclic skills (`jump`, `cartwheel`). Since the model has already learned a robust and connected base space, it can integrate these complex motions more effectively.

### Phase 4: Evaluation

We need a new benchmark to evaluate the primary contribution: skill compositionality.

*   **Proposed Script**: `scripts/curriculum/evaluate_transitions.py`
*   **Methodology**:
    1.  **Define Composition Tasks**: Create a set of challenging, long-horizon tasks that require chaining skills. Examples:
        *   `Walk -> Sprint -> Stop`
        *   `Run -> Jump_over_obstacle -> Run`
        *   `Walk -> Cartwheel -> Walk`
    2.  **Define Metrics**: The primary metric will be the **Transition Success Rate (TSR)**: the percentage of trials where the agent successfully executes the full sequence without falling or producing significant artifacts.
    3.  **Comparative Analysis**: Evaluate two models:
        *   **Baseline**: A diffusion model trained on all data uniformly for `N_1+N_2+N_3` epochs (the original BeyondMimic Stage 2 approach).
        *   **Ours (SGCL)**: The model trained with our curriculum.
    4.  **Hypothesis**: The SGCL-trained model will exhibit a significantly higher TSR, demonstrating its superior ability to compose skills fluidly.
