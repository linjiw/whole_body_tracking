# Teacher-Guided Learning on a Skill-Graph Manifold for Humanoid Motion Synthesis

---

## Abstract

Generative diffusion policies have enabled humanoid robots to learn a diverse repertoire of skills by imitating human motion. However, these policies often learn each skill in isolation, creating a fragmented latent space where fluid transitions between dynamically distinct motions—such as from walking to a cartwheel—remain a significant challenge. This "skill compositionality gap" prevents the synthesis of long-horizon, versatile behaviors. In this paper, we introduce a novel training framework that solves this problem by intelligently structuring the learning process. Our framework first models the entire motion dataset as a **Skill Graph**, identifying distinct skills and the feasible transitions that connect them. It then employs a **Teacher-Student** dynamic, where a lightweight "teacher" algorithm guides a "student" (a Transformer-based diffusion policy) through this graph. The teacher assesses the student's performance in real-time and dynamically adapts the curriculum, focusing on skills and transitions at the very boundary of the student's competence. By combining the explicit structure of the Skill Graph with adaptive, teacher-guided pacing, our method produces a single, unified policy that demonstrates unprecedented fluidity in composing disparate skills, effectively closing the compositionality gap.

---

## 1. Introduction

The quest for agile, general-purpose humanoid robots has seen remarkable progress, largely driven by learning-from-demonstration (LfD). Frameworks like BeyondMimic have shown that deep reinforcement learning can produce high-fidelity motion tracking policies, and that these can be distilled into a single, powerful diffusion policy. Such policies can generate naturalistic motions and even be guided at test-time to adapt to new environmental goals.

However, a critical limitation persists. While these models can learn an extensive library of individual skills—walking, running, jumping, even complex acrobatics—they treat the motion space as a disconnected archipelago of skills. The model can generate a trajectory within the "walking" manifold or the "cartwheel" manifold, but it lacks the understanding of how to navigate the space *between* them. This is because the training process, which typically samples data uniformly, provides insufficient signal for learning the rare and difficult dynamics of skill transitions. This "skill compositionality gap" is a major barrier to creating truly versatile agents.

A static, pre-defined curriculum is a step in the right direction, but it is a blunt instrument. It does not adapt to the model's unique learning trajectory. To build these bridges effectively, the training process must be both **structured** and **adaptive**. It must emulate a real teacher who uses a textbook (the structure) but adjusts the lesson plan (the adaptation) based on the student's progress.

To this end, we propose a unified framework that integrates two key innovations:
1.  **Skill Graph Construction**: We first perform a comprehensive analysis of the motion dataset to build a **Skill Graph**. This graph serves as our structured "textbook," explicitly mapping out foundational skills, advanced skills, and the valid transition pathways between them.
2.  **Teacher-Guided Pacing**: We then introduce a **Teacher** algorithm that intelligently guides our **Student** (a Transformer-based diffusion policy) through this graph. The Teacher monitors the Student's performance and dynamically adjusts the curriculum, focusing on specific skills and transitions that are challenging but still learnable. It decides *what* to learn from the graph and *when* to learn it.

By combining the explicit structure of the Skill Graph with the adaptive feedback of a teacher-student loop, our framework ensures that the model's training is always focused on the most productive areas of the problem space. This allows the policy to systematically master not only a diverse set of skills but, most importantly, the challenging transitional dynamics that connect them.

---

## 2. Design and Implementation Plan

This research builds directly upon the BeyondMimic repository. It utilizes the completed Stage 1 (motion tracking) to generate expert data and implements the proposed Stage 2 (diffusion policy) using our novel teacher-guided training strategy.

### Phase 1: Skill Graph Construction

*(This foundational phase provides the structure for the curriculum.)*

*   **Objective**: To automatically cluster motion data into discrete skills (`walk`, `run`, `jump`) and identify the transition points between them, creating a directed graph.
*   **Proposed Script**: `scripts/curriculum/build_skill_graph.py`
*   **Methodology**: Use a density-based clustering algorithm (e.g., DBSCAN) on kinematic features from the MoCap dataset. The resulting clusters become nodes (skills) and the points between clusters define the edges (transitions).

### Phase 2: The Student (The Generative Model)

*   **Model**: A **Transformer-based State-Action Diffusion Model**. The self-attention mechanism is critical for modeling the long-range temporal dependencies in complex, multi-skill motion sequences.
*   **Implementation-Friendliness**: This uses the same planned architecture as the original BeyondMimic Stage 2, leveraging the repository's existing `torch` foundation.

### Phase 3: The Teacher-Guided Training Loop

*(This is the core of our contribution, merging the Skill Graph with the Teacher algorithm.)*

*   **Proposed Script**: `scripts/curriculum/train_teacher_guided.py`
*   **Components**:
    1.  **The Student**: The `StateActionDiffusionModel` instance.
    2.  **The Teacher**: A new `TeacherPacingAlgorithm` class.
*   **Methodology**: The training process is an adaptive loop governed by the Teacher:
    1.  **Initialization**: The Teacher identifies the root nodes of the Skill Graph (e.g., `idle`, `walk`) and sets them as the initial **"active curriculum"**. The data sampling probability `P` is distributed only among these active skills.
    2.  **Training Stage**: For a number of epochs, the Student trains on data sampled exclusively from the active curriculum. The Teacher continuously adjusts the sampling probabilities `P` *within* this active set, focusing more on skills where the Student shows high learning loss.
    3.  **Mastery Assessment**: At the end of each epoch, the Teacher checks if the skills in the active curriculum have been "mastered." Mastery is defined as the validation loss for a skill remaining below a set threshold for several consecutive epochs.
    4.  **Curriculum Expansion**: Once a skill is mastered, the Teacher **consults the Skill Graph** and adds the mastered skill's immediate neighbors (and the corresponding transition data) to the "active curriculum" for the next stage of training. For example, after `walk` is mastered, the Teacher adds `run` and the `walk-to-run-transition` to the active set.
    5.  **Repeat**: The process repeats, with the Teacher intelligently expanding the curriculum along the graph's pathways, until the entire Skill Graph has been mastered. This ensures a logical progression from simple to complex and guarantees that the crucial transition dynamics are learned.

### Phase 4: Evaluation

*   **Objective**: To prove that our method enables superior skill compositionality.
*   **Metric**: The primary metric will be the **Transition Success Rate (TSR)**, measuring the model's ability to successfully execute a commanded sequence of disparate skills (e.g., `Walk -> Jump -> Walk`) without failure.
*   **Comparative Analysis**: We will compare our **TGCL-trained model** against a **baseline model** trained with a uniform data distribution (the original BeyondMimic approach). The hypothesis is that our model will achieve a dramatically higher TSR, demonstrating the effectiveness of our structured, adaptive learning framework.
