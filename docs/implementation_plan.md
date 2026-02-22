# Implementation Plan: Classroom Group Engagement Recognition

## Project Goal

Develop a system that recognizes group engagement levels in classroom video recordings. The system should:
- Classify 10-second video segments into three engagement levels (Low, Medium, High)
- Reproduce and analyze baseline results on OUC-CGE dataset
- Explore multimodal approaches incorporating pose and facial landmarks
- Validate generalization on DIPSER dataset
- Generate post-session reports with engagement timeline visualization

---

## Phase 1: Setup and Data

### 1.1 Environment
- Python environment with video dependencies
- Clone OUC-CGE repository, verify baseline runs
- Configure experiment tracking

### 1.2 OUC-CGE Dataset Preparation
- Download dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/BRD2C)
- Implement data pipeline compatible with video models
- Explore key video properties, analyze label distirbutions

### 1.3 DIPSER Dataset
- Upload dataset under DVC. If excessively large, then strip of redundant data.
  Preliminary discussions show that we need "context cameras"
- Design label mapping: 5-level individual → 3-level group
- Note: pre-extracted pose/landmarks available

**Deliverable:** Working data pipelines for both datasets

---

## Phase 2: Baseline Reproduction

### 2.1 OUC-CGE Baseline
- Implement/integrate SLOWFAST model
- Reproduce reported results on OUC-CGE dataset: ~97%
- Create and conduct 1-2 experiments: at least one for architecture and/or video properties (fps, network depth)
- Measure available batch locally: very likely remote GPU host required
    - if that's the case: fix `requirements.txt`, rebuild docker image, access vast.ai cloud provider

### 2.2 More Baselines
- Reproduce results with some other model: SLOW, for example
- Compare with SLOWFAST baseline

### 2.3 Cross-Dataset Baseline (DIPSER)
- Apply OUC-CGE-trained model to DIPSER context cameras
- Measure zero-shot transfer performance
- Analyze failure modes
- Formulate several hypothesis how to address failure modes 

**Deliverables:**
- Reproduced baseline results on OUC-CGE (~97-98%)
- Cross-dataset transfer results on DIPSER
- Analysis of generalization gap

---

Thoughts. How to evaluate generalization.
- measure zero-shot on DIPSER after OUC-CGE
- measure zero-shot on OUC-CGE after DIPSER
- train on both datasets, measure validation metrics

---

## Phase 3: Architecture Experiments

Key question:
- what are we trying to do?

For example for "what are the best zero-shot transferable models?"
- Add more models (fine-tune on OUC-CGE) and compare zero-shot quality on DIPSER
- measure quality on DIPSER after training
- will usage of temporal modeling or human pose (and/or other spatial feature) improve generalizability

### 3.? Highly flexible
- Add new models for comparison, conduct experiments ro improve metric: TBD which exactly metrics on which datasets
Possibilities for experiments:
- backbone
- temporal settings (framerate, clip duration, number of frames)
- creative use of attention mechanisms to extract non-local features

---

## Phase 4: Multimodal Experiments (Pose and Landmarks)

- are facial landmarks relevant with given resolution?
- setup up pipelines for facial landmarks, human pose extraction. Plan their temporal modeling
- likely precompute will be preferable

### 4.1 Comparisons
- It can be interesting to compare pose-only model with image-only and fused ones
- pose can be used as GCN or LSTM on flattened pose keypoints
- consider various fusion strategies: cross-attention(?), concatenation after temporal modeling, prediction combination
- which modality provides most generalization: provide typical validation comparison

---

## Phase 6: Finalizing

### 6.1 Experiments wrap-up

Theoretical results:
- Outline key results regarding metrics and generalization: are they worthy? are they novel?
- If there is a strong yes, what are we going to do about it?

Experiments due diligence:
- check experiments' reproducibility
- rerun key experiments with different seeds, complete all experiments

### 6.2 Model Export
- choose, optimize and export the best model
- verify metrics, measure performance

### 6.3 Inference Pipeline
- Implement end-to-end inference pipeline:
  - Video input
  - Pose/landmark extraction (if multimodal)
  - Model output
- measure performance

### 6.4 Integration
- Vibecode web or desktop application to wrap inference pipeline
- (maybe scalable on-demand servers that inference live-feed? privacy? unfamiliar web tech stack?)
- test on some videofeeds from a dataset
- plan test phase

### 6.5 Field test
- live-test and record session
- follow testing procedure created in 6.4

---

## Phase 7: Report feature

### 7.1 Report Generation
- Design report format:
  - Session summary (duration, overall engagement distribution)
  - Engagement timeline visualization
  - Key moments: periods of low/high engagement with timestamps
  - Summarization of engagement dips with external AI model (provide transcribed lecture with inserted cues as context). Cues as `<ENGAGEMENT_DIP>` and `</ENGAGEMENT_DIP>` or whatever
- Implement report generation module
- Test on recorded field test sesssion

---

## Phase 8: Documentation and presentation preparations

- code clean-up
- repository documentation
- create presentation: focus on key results, live-testing and purpose of the project, not on the technical details. Though architecture details as well as in-depth implementation knowledge should be prepared as well
