Our key articles are dataset tech-reports:
- [A Video Dataset for Classroom Group Engagement Recognition](https://www.nature.com/articles/s41597-025-04987-w)
- [DIPSER: A Dataset for In-Person Student Engagement Recognition in the Wild](https://arxiv.org/abs/2502.20209)

---

The following was AI-generated, consider dropping. Review is not priority yet.

---

# Literature Review: Classroom Group Engagement Recognition

## 1. Project Overview

**Goal:** Develop a system that recognizes group engagement levels in classroom video recordings using deep learning. The system classifies 10-second video segments into three engagement levels (Low, Medium, High) and provides temporal analysis of engagement patterns.

**Key distinction:** Unlike individual engagement detection from webcam (e.g., DAiSEE), this project focuses on group-level engagement in authentic classroom environments with multiple students visible in the frame.

---

## 2. What is Group Engagement?

Group engagement extends individual engagement to the collective level. It encompasses:

- **Behavioral dimension:** Active participation in discussions, task collaboration, hand-raising, note-taking
- **Cognitive dimension:** Active knowledge seeking, critical thinking, question asking
- **Emotional dimension:** Interest in content, positive emotional investment, enthusiasm

In classroom settings, group engagement is observable through:
- Synchronized behaviors (group note-taking, collective attention)
- Interaction patterns (discussion frequency, peer explanation)
- Postural cues (leaning forward, orientation toward teacher/materials)

The ICAP framework (Interactive, Constructive, Active, Passive) provides a theoretical foundation for understanding engagement modes and their relationship to learning outcomes.

---

## 3. Benchmark Datasets

### 3.1 OUC-CGE (Primary Dataset)

**Paper:** *"A Video Dataset for Classroom Group Engagement Recognition"* (Lu et al., Nature Scientific Data, 2025)

OUC-CGE is the first benchmark dedicated to group engagement analysis in authentic classroom settings.

**Dataset characteristics:**
- 12 hours 50 minutes of video
- 7,705 segments (10-second clips)
- Three engagement levels: Low (0), Medium (1), High (2)
- Multiple camera angles: front, side, back views
- Two classroom layouts: round-table and chess-board
- 17 undergraduate participants across 21 sessions
- Resolution: 1280×720, 30fps

**Engagement definitions:**
- **Low:** Majority disengaged—eyes barely open, yawning, gazes away, slumped posture
- **Medium:** Moderate involvement—leaning forward, neutral expression, one-hand head support
- **High:** Fully absorbed—intent focus, active note-taking, positive emotions, discussion participation

**Baseline results:**

| Model | Accuracy |
|-------|----------|
| C2D | 90.0% |
| I3D | 94.6% |
| X3D | 93.7% |
| SLOW-NLN | 97.0% |
| SlowFast | 97.5% |
| SLOW | 97.8% |

**Why SLOW works best:** The dataset's engagement patterns are dominated by low-frequency behaviors (sustained postures, slow movements). SLOW's 4fps sampling captures these effectively while filtering high-frequency noise.

**Dataset access:** Publicly available at OSF (https://osf.io/brd2c/)

### 3.2 DIPSER (Cross-Dataset Validation)

**Paper:** *"DIPSER: A Dataset for In-Person Student Engagement Recognition in the Wild"* (Marquez-Carpintero et al., 2025)

In-person classroom dataset designed to assess student attention in face-to-face settings.

**Dataset characteristics:**
- 51.3 hours of video, 54 subjects (Caucasian)
- Individual cameras per student (640×480) + context cameras (1280×720, ~9fps)
- Smartwatch sensor data: heart rate, accelerometer, gyroscope
- 5-level attention labels (1-5 scale) + 9 emotion categories
- Multiple evaluators: 4 experts + self-report per student
- 9 different educational scenarios (lectures, tests, presentations, group work)

**Pre-extracted features included:**
- MediaPipe Face Mesh (facial landmarks)
- MediaPipe BlazePose (body skeleton)
- MediaPipe Hand Landmarks
- Head pose estimation
- Gaze estimation

**Relevance for this project:**
1. **Cross-dataset validation:** Different university, demographics (Caucasian vs. Asian), setup
2. **Multimodal data:** Pre-extracted pose and landmarks enable multimodal experiments
3. **True generalization test:** If model works on both OUC-CGE and DIPSER, it's more robust

**Label mapping consideration:** DIPSER has individual attention (5-level), OUC-CGE has group engagement (3-level). For cross-dataset evaluation, aggregate individual labels to group level.

### 3.3 DAiSEE (Reference Dataset)

**Paper:** *"DAiSEE: Towards User Engagement Recognition in the Wild"* (Gupta et al., 2016)

Individual engagement in online learning contexts.
- 9,068 clips from 112 subjects
- Four states: Engagement, Boredom, Confusion, Frustration
- Four levels per state (0-3)
- Webcam recordings during e-learning

**Relevance:** Different domain (online vs. in-person). Provides historical context for engagement recognition research.

### 3.4 Other Relevant Datasets

**EngageWild:** 78 subjects, 5-minute videos, four engagement levels, varied settings

**EngageNet:** Combines behavioral and cognitive engagement, web-based platform data

---

## 4. Video Understanding Architectures

Group engagement recognition is fundamentally a video classification task requiring spatio-temporal modeling.

### 4.1 2D CNN Approaches (Spatial Only)

**C2D (Convolutional 2D)**
- Applies 2D convolutions to stacked frames
- Treats temporal dimension as channel dimension
- Fast but weak temporal modeling
- Baseline approach, ~90% on OUC-CGE

### 4.2 3D CNN Approaches (Spatio-Temporal)

**I3D (Inflated 3D ConvNet)**
- *Paper:* "Quo Vadis, Action Recognition?" (Carreira & Zisserman, 2017)
- Inflates 2D ImageNet-pretrained filters to 3D
- Joint spatio-temporal feature extraction
- 3×3×3 convolution kernels
- ~94.6% on OUC-CGE

**C3D**
- Early 3D CNN architecture
- 3D convolutions throughout
- Computationally expensive

**X3D (Expand 3D)**
- *Paper:* "X3D: Expanding Architectures for Efficient Video Recognition" (Feichtenhofer, 2020)
- Progressive expansion along multiple axes (depth, width, temporal)
- Efficient video understanding
- ~93.7% on OUC-CGE

### 4.3 Two-Stream and Multi-Pathway Approaches

**SlowFast Networks**
- *Paper:* "SlowFast Networks for Video Recognition" (Feichtenhofer et al., 2019)
- Dual-pathway design:
  - **Slow pathway:** Low temporal resolution (4fps), heavy 3D convolutions, captures spatial semantics
  - **Fast pathway:** High temporal resolution (32fps), lightweight, captures rapid motion
- Lateral connections fuse pathways
- ~97.5% on OUC-CGE

**SLOW (Single Pathway)**
- Slow pathway only from SlowFast
- 4fps sampling rate
- Best performance on OUC-CGE (~97.8%)
- Works well because engagement behaviors are low-frequency

**SLOW-NLN (with Non-Local Networks)**
- Adds non-local (self-attention) modules to SLOW
- Captures long-range dependencies
- ~97.0% on OUC-CGE

### 4.4 Transformer-Based Approaches

**Vision Transformer (ViT)**
- Self-attention on image patches
- Requires large-scale pretraining

**Video Transformers (TimeSformer, ViViT)**
- Extend ViT to video with temporal attention
- Computationally expensive
- May be overkill for this task given CNN performance

### 4.5 Architecture Selection Guidance

For OUC-CGE specifically:
- **Start with:** SLOW (best reported results, simpler than SlowFast)
- **Compare:** I3D (good balance), SlowFast (if fast motion matters)
- **Baseline:** C2D (sanity check)
- **Advanced:** Add attention mechanisms, experiment with backbones

---

## 5. Key Concepts

### 5.1 Temporal Modeling in Video

**Frame rate and sampling:**
- Raw video: 30fps
- SLOW pathway: 4fps (every 8th frame)
- Fast pathway: 32fps (higher temporal resolution)

**Clip duration:**
- OUC-CGE uses 10-second clips
- At 4fps: 40 frames per clip
- At 30fps: 300 frames per clip

**Why low frame rate works for engagement:**
- Engagement states evolve slowly (seconds, not milliseconds)
- Body posture, attention direction change gradually
- High frame rate captures noise (fidgeting, blinks)

### 5.2 Backbone Networks

All OUC-CGE baselines use **ResNet-50** as backbone:
- Well-understood, pretrained weights available
- Good balance of depth and efficiency
- Can be replaced with EfficientNet, ConvNeXt for experiments

### 5.3 Multi-View Considerations

OUC-CGE includes front, side, and back camera views:
- Different views capture different engagement cues
- Front: facial expressions, eye contact
- Side: posture, peer interaction
- Back: overall class attention direction

Model should generalize across views or use view-specific processing.

---

## 6. Multimodal Approaches: Pose and Facial Landmarks

### 6.1 Motivation for Multimodal Features

While video-based models (SLOW, I3D) achieve high accuracy on OUC-CGE (~98%), they have limitations:

1. **Limited interpretability:** Difficult to explain *why* a prediction was made
2. **Appearance dependency:** May overfit to specific classroom appearance
3. **Computational cost:** 3D CNNs require significant GPU resources
4. **Generalization uncertainty:** May not transfer to different demographics/setups

Pose and facial landmarks offer complementary advantages:
- **Interpretable:** "Head-down posture" or "low eye openness" are explainable
- **Domain-invariant:** Skeleton representation abstracts away appearance
- **Efficient:** Landmark models are much lighter than video models
- **Cross-dataset potential:** May generalize better to DIPSER

### 6.2 Pose Estimation for Engagement

**MediaPipe BlazePose:**
- Real-time body pose estimation (33 landmarks)
- Works on single RGB frames
- Provides landmarks for: head, shoulders, arms, hands, torso

**Engagement cues from pose:**
- Head orientation (looking down vs. forward)
- Posture (upright vs. slouched)
- Hand activity (writing, raising hand)
- Body orientation (facing teacher vs. peer)

**Model approaches:**
- **GCN (Graph Convolutional Network):** Treat skeleton as graph, model spatial and temporal relations
- **Sequence models:** Flatten landmarks → LSTM/GRU/Transformer

### 6.3 Facial Landmark Features

**MediaPipe Face Mesh:**
- 468 facial landmarks
- Real-time on CPU

**Derived features:**
- **Head pose:** Yaw, pitch, roll angles from landmark geometry
- **Eye Aspect Ratio (EAR):** Detects drowsiness/closed eyes
- **Mouth Aspect Ratio:** Open mouth may indicate boredom/yawning
- **Gaze direction:** Estimated from eye landmarks (approximate)

**Temporal patterns:**
- Frequency of head movements
- Duration of downward gaze
- Eye closure patterns

### 6.4 Multimodal Fusion Strategies

**Early fusion:**
- Concatenate pose + landmark features
- Single temporal model processes combined features
- Simple, but assumes modalities have same temporal dynamics

**Late fusion:**
- Separate models for each modality
- Combine predictions (average, learned weighting)
- Allows different architectures per modality

**Attention-based fusion:**
- Learn to weight modalities dynamically
- Can adapt based on input (e.g., if face is occluded, rely on pose)

### 6.5 Hypothesis for Generalization

Training on OUC-CGE with video model achieves 98% accuracy but may not transfer to DIPSER because:
- Different demographics (appearance mismatch)
- Different camera angles and resolution
- Different classroom environments

Pose/landmark models may generalize better because:
- Skeleton representation is invariant to appearance
- Geometric features (head angle) transfer across populations
- Less overfitting to dataset-specific visual patterns

**Experiment:** Compare video-only vs. pose-only vs. multimodal generalization from OUC-CGE to DIPSER.

---

## 7. Key Research Papers

### Dataset and Benchmark Papers

1. **OUC-CGE Paper** (Lu et al., 2025)
   - Primary reference for dataset and baselines
   - Defines group engagement annotation protocol
   - Establishes benchmark results
   - *Read thoroughly before starting implementation*

2. **DIPSER Paper** (Marquez-Carpintero et al., 2025)
   - In-person classroom dataset with multimodal data
   - 54 subjects, 51.3 hours, individual attention labels
   - Pre-extracted pose and landmarks included
   - *Cross-dataset validation reference*

3. **DAiSEE Paper** (Gupta et al., 2016)
   - Individual engagement benchmark (online learning)
   - Useful for historical context

### Video Understanding Architecture Papers

4. **"Quo Vadis, Action Recognition?"** (Carreira & Zisserman, 2017)
   - Introduces I3D architecture
   - Inflation of 2D CNNs to 3D
   - *Essential for understanding 3D convolutions*

5. **"SlowFast Networks for Video Recognition"** (Feichtenhofer et al., 2019)
   - Dual-pathway architecture
   - State-of-the-art action recognition
   - *Core architecture for this project*

6. **"X3D: Expanding Architectures for Efficient Video Recognition"** (Feichtenhofer, 2020)
   - Efficient video understanding
   - *Alternative architecture to explore*

7. **"Non-local Neural Networks"** (Wang et al., 2018)
   - Self-attention for video
   - *Useful for attention mechanism experiments*

### Pose and Landmark Papers

8. **"BlazePose: On-device Real-time Body Pose Tracking"** (Bazarevsky et al., 2020)
   - MediaPipe pose estimation (33 landmarks)
   - Real-time, lightweight
   - *Implementation reference*

9. **"Attention Mesh: High-fidelity Face Mesh"** (Grishchenko et al., 2020)
   - MediaPipe Face Mesh (468 landmarks)
   - *Implementation reference for facial features*

### Foundational Papers

10. **"Deep Residual Learning"** (He et al., 2016)
    - ResNet architecture (backbone for baselines)

11. **"Attention Is All You Need"** (Vaswani et al., 2017)
    - Transformer architecture
    - *Background for attention experiments*

### Engagement Recognition Papers

12. **"The Faces of Engagement"** (Whitehill et al., 2014)
    - Early engagement detection work

13. **"Attentive or Not?"** (Goldberg et al., 2021)
    - ICAP framework application
    - *Theoretical grounding for attention labels*

---

## 8. Practical Considerations

### 7.1 Why OUC-CGE Reports High Accuracy

The ~98% accuracy on OUC-CGE is notably high. Reasons:
- **Coarse granularity:** Three classes (Low/Medium/High) are visually distinct
- **Spatial dominance:** Static posture features are highly discriminative
- **Limited variation:** Fixed classroom, consistent lighting, repeated participants
- **Task simplicity:** Group-level classification averages individual noise

This doesn't mean the task is "solved"—generalization to new classrooms, lighting conditions, and participant populations remains challenging.

### 7.2 Potential Research Directions

- **Cross-view generalization:** Train on one view, test on another
- **Efficiency:** Achieve similar accuracy with smaller/faster models
- **Interpretability:** Which regions/frames drive predictions?
- **Fine-grained analysis:** Can we detect individual engagement within group?
- **Temporal segmentation:** Automatically find engagement state transitions

### 7.3 Implementation Frameworks

**PyTorchVideo:** Facebook's library for video understanding
- Pretrained SlowFast, I3D, X3D models
- Easy model loading and fine-tuning

**MMAction2:** OpenMMLab's action recognition toolbox
- Comprehensive model zoo
- Flexible configuration system

**Official OUC-CGE code:** Available on GitHub
- Baseline implementations
- Data loading utilities

---

## 9. Privacy Considerations

Classroom video analysis raises privacy concerns:

1. **Local processing:** Run inference on-premises, no cloud upload
2. **Aggregate metrics:** Report class-level engagement, not individual tracking
3. **No face storage:** Process frames for engagement, don't save facial data
4. **Pose-based option:** Pose/landmark models don't require storing facial images
5. **Consent:** Ensure proper consent for any classroom recordings
6. **Purpose limitation:** Use only for educational feedback, not surveillance

---

## 10. Summary: Project Approach

Based on this review:

1. **Primary dataset:** OUC-CGE (group engagement, real classroom, ~98% baseline)

2. **Validation dataset:** DIPSER (cross-dataset generalization test)

3. **Primary architecture:** SLOW (best baseline for video)

4. **Multimodal extension:** Pose + facial landmarks for interpretability and generalization

5. **Experiments:**
   - Reproduce baseline results on OUC-CGE
   - Compare: C2D, I3D, X3D, SlowFast, SLOW
   - Implement pose-only and landmark-only models
   - Compare multimodal fusion strategies
   - Evaluate cross-dataset generalization on DIPSER

6. **Key hypothesis:** Multimodal (pose/landmarks) may generalize better than video-only

7. **Application:** Process classroom recordings, generate engagement reports

---

## 11. Recommended Reading Order

1. **Week 1:** OUC-CGE paper + DIPSER paper (understand datasets)
2. **Week 2:** SlowFast paper + I3D paper (video architectures)
3. **Week 3:** BlazePose/MediaPipe papers (pose extraction)
4. **Week 4:** Run official baseline code, reproduce results
5. **Ongoing:** Reference other papers as needed

---

*This review provides foundation for the project. The student should read primary papers and expand understanding through implementation.*
