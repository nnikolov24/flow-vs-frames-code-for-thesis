# flow-vs-frames-code-for-thesis
Bachelor’s thesis code comparing motion-only (optical-flow) and appearance-only (RGB) X3D models for football (soccer) event classification.
# Flow vs. Frames — Motion-only vs Appearance-only Models for Football Event Recognition

This repository contains the code for my **Bachelor’s thesis in Cognitive Science & Artificial Intelligence** at **Tilburg University**.

**Thesis title:**  
*Flow vs. Frames: Comparing Motion-only and Appearance-only Models for Football (Soccer) Event Recognition*

The goal of this project is to compare how well **motion-only video representations** (optical flow) perform relative to **appearance-only representations** (RGB frames) when recognizing key football events.

---

## Task

3-class classification of short football clips:
- scoring  
- tackling  
- red cards  

---

## Method

- **Architecture:** X3D-S (3D CNN)
- **Appearance-only model:** RGB frames with temporal shuffling during training to suppress motion cues
- **Motion-only model:** Dense Farnebäck optical flow
- **Dataset:** Football Match Actions Videos (Kaggle, Refaat 2022) link: https://www.kaggle.com/datasets/itarek898/football-match-actions-video-dataset
- **Evaluation:** 5-fold cross-validation + held-out test set

Both models use the same backbone, training procedure, and evaluation protocol to ensure a fair comparison.

---

## Main Findings

- Both models achieve very high performance
- The motion-only model slightly outperforms the appearance-only model on the test set
- Motion cues are especially effective for tackling events
- Motion-based recognition depends strongly on temporal context
- Appearance-based recognition remains robust even with few frames

---

## Repository Structure

```text
CODING SUBMISSION/
├── PREPROCESSING/
├── DATASET CLASSES/
├── Models/
├── Training/
├── Evaluation/
├── RUN_KFOLD/
├── ablation(changing frames)/
└── Plotting and visualization/


Usage (High-level)
Preprocess videos (frame extraction and optical flow computation)

Train appearance-only or motion-only X3D-S models

Evaluate on validation and test sets

Run k-fold cross-validation and clip-length ablation studies

Dataset files are not included due to size and licensing constraints.

Author
Nikolay Nikolov
BSc Cognitive Science & Artificial Intelligence
Tilburg University
