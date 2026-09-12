# Phase 3: Explainability & Attribution Failure Analysis

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Investigation Scope:** Systematic Taxonomy and Observational Audit of Predictive Attribution Failures in the Frozen P3 Stage-2 Model  
**Date:** September 12, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Executive Summary of Failure Taxonomy

The objective of Phase 3 is not merely to measure global attribution averages, but to characterize **systematic failure modes and their empirical properties**. We audited 10 systematic failure patterns across the 40-case evaluation cohort and whole-dataset classification outputs, distinguishing strictly between **`OBSERVED`** data, **`INFERRED`** interpretations, and **`UNKNOWN`** factors.

---

## 2. Detailed Audit of the 10 Failure Patterns

### Pattern 1: Correct Prediction + Correct Anatomy (Regional Grounding)
- **Definition:** The model correctly predicts presence ($P \ge 0.5$), correctly classifies the exact anatomical location, and attribution exhibits its closest proximity to the lesion and highest arterial concentration.
- **Representative Case:** `topaneu_center1_mr_056` (Presence $P = 0.8205$, Location = `Right Supraclinoid ICA`).
- **Classification Status:**
  - **OBSERVED:** Present in $16.12\%$ of whole-cohort positive cases (4 cases in the 40-case cohort). These cases exhibit the lowest mean centroid distance ($19.46\text{ mm}$ for Layer IG) and highest mean aneurysm attribution mass fraction ($0.011533$ / $1.15\%$), with vessel containment reaching $6.21\%$.
  - **INFERRED:** True positive predictions with exact location matches reflect closer alignment with regional vascular structures than misaligned or negative predictions.
  - **UNKNOWN:** Why this high-grounding regime is restricted to a minority of cases under the current architecture.

### Pattern 2: Correct Prediction + Wrong Anatomy (Anatomical Decoupling)
- **Definition:** The model correctly predicts presence ($P \ge 0.5$), but predicts a discordant anatomical location, and attribution mass is focused on distant vascular segments.
- **Representative Case:** `topaneu_center1_mr_148` (Presence $P = 0.6523$, GT = Basilar/PICA; Model predicts Anterior Circulation).
- **Classification Status:**
  - **OBSERVED:** Observed across $55.59\%$ of positive cases in the full dataset. In the 40-case cohort, misaligned cases exhibit a mean centroid distance of $34.62\text{ mm}$ (Layer IG).
  - **INFERRED:** The presence classification head can achieve high confidence from general vascular morphology or regional asymmetry without requiring the location head to correctly identify the anatomical artery.
  - **UNKNOWN:** Whether presence queries and location queries attend to distinct spatial sub-regions within the $7 \times 7 \times 7$ bottleneck.

### Pattern 3: Wrong Location Prediction + Vessel-Aligned Attribution (Residual Regional Evidence)
- **Definition:** The model predicts presence ($P \ge 0.5$), but assigns an incorrect discrete location class that nonetheless belongs to the same parent arterial trunk, while attribution remains spatially near the true lesion.
- **Representative Case:** `topaneu_center1_mr_017` (Presence $P = 0.2302$, GT = Right ICA C7-PCom junction; attribution focuses on the carotid siphon).
- **Classification Status:**
  - **OBSERVED:** Attribution is spatially closer to the GT lesion than in misaligned cases, exhibiting a mean centroid distance of approximately **$26.59\text{ mm}$** (vs. $34.62\text{ mm}$ in misaligned cases and $55.05\text{ mm}$ in false negatives).
  - **INFERRED:** This is consistent with residual regional anatomical information being preserved in the bottleneck representation despite incorrect discrete classification by the multi-class linear head.
  - **UNKNOWN:** Whether hierarchical classification or segmentation-guided decoding would resolve the discrete location error.

### Pattern 4: High-Confidence False Positive with Vascular Containment
- **Definition:** The model predicts presence ($P \ge 0.5$) in an aneurysm-free scan, with attribution localized to normal arterial structures.
- **Representative Case:** `topaneu_center1_mr_018` (Presence $P = 0.9312$, GT presence = 0).
- **Classification Status:**
  - **OBSERVED:** False positive attribution exhibits relatively high vascular containment ($5.55\%$, higher than the $2.89\%$ observed in true negatives). Background attribution is $94.45\%$.
  - **INFERRED:** This is consistent with vascular-structure-driven false positives, potentially including tortuous or bifurcation structures (e.g. carotid siphon, basilar bifurcation), rather than diffuse non-vascular parenchymal noise.
  - **UNKNOWN:** The precise morphological features (curvature, branching angles, local vessel caliber) of normal healthy vasculature that trigger false-positive alarms across a wider population.

### Pattern 5: High Vessel Containment with Low Aneurysm Mass Fraction (Vessel-Dominated Representation)
- **Definition:** Attribution shows measurable overlap with the general arterial tree ($4\%\text{--}6\%$), but the raw attribution mass fraction within the lesion dome remains $<0.5\%$.
- **Classification Status:**
  - **OBSERVED:** In true positives, Layer IG vessel containment averages $5.08\%$, whereas the aneurysm attribution mass fraction averages $0.005341$ ($0.534\%$). Simultaneously, top-5% attribution voxels intersect $79.35\%\text{--}88.51\%$ of the lesion volume.
  - **INFERRED:** Healthy vasculature occupies $\sim 14\%$ of the intracranial volume, whereas the aneurysm dome occupies $<0.1\%$. The coarse $7 \times 7 \times 7$ bottleneck feature map captures broader parent arterial context rather than sharply delineating the focal aneurysm boundary.
  - **UNKNOWN:** The theoretical maximum attribution mass fraction achievable given the resolution constraints of a $7^3$ feature map.

### Pattern 6: Aneurysm Attribution with Discordant Parent Vessel Label
- **Definition:** Attribution overlaps the focal aneurysm region, but the classification head outputs an anatomically contradictory parent vessel label.
- **Classification Status:**
  - **OBSERVED:** Observed primarily in cases near the circle of Willis midline (e.g. Anterior Communicating Artery vs. Internal Carotid Artery bifurcations).
  - **INFERRED:** Dense arterial clustering at the skull base and circle of Willis leads to spatial overlap within the effective receptive field of individual $7^3$ bottleneck tokens ($\approx 32\text{ mm}$ isotropic).
  - **UNKNOWN:** The extent to which inter-subject anatomical variability in circle of Willis geometry contributes to this labeling ambiguity.

### Pattern 7: Low Gradient Specificity under Global Channel Pooling (Grad-CAM Attenuation)
- **Definition:** Attribution map displays low spatial contrast and diffuse spread across the brain volume.
- **Classification Status:**
  - **OBSERVED:** Standard 3D Grad-CAM exhibits poor localization (mean centroid distance of $52.25\text{ mm}$, mean aneurysm mass fraction $0.000013$) and negative deletion faithfulness ($\Delta P = -0.0551$).
  - **INFERRED:** The coarse $7 \times 7 \times 7$ classification representation and global channel pooling across spatial dimensions may contribute to reduced spatial specificity.
  - **UNKNOWN:** Whether locally weighted CAM variants (e.g. Grad-CAM++ or HiResCAM) fully compensate for spatial pooling in 3D medical volumes.

### Pattern 8: Parenchymal / Nonvascular Attribution
- **Definition:** Attribution mass resides outside the segmented vascular tree and outside the ground-truth aneurysm volume.
- **Classification Status:**
  - **OBSERVED:** Background attribution averages $95.66\%$ on the $224^3$ grid for Layer Integrated Gradients across the cohort.
  - **INFERRED:** This is consistent with trilinear upsampling from coarse $7^3$ tokens ($\approx 32\text{ mm}$ per voxel) to the $224^3$ grid, which spreads token activations into adjacent nonvascular brain parenchyma. It may also reflect surrounding contextual features used by convolutional filters.
  - **UNKNOWN:** The exact proportion of background attribution attributable to interpolation diffusion versus authentic nonvascular contextual reliance (e.g. potential shortcut learning).

### Pattern 9: Attention-Gradient Disagreement
- **Definition:** Cross-attention query weights and gradient-based attributions diverge in spatial focus.
- **Classification Status:**
  - **OBSERVED:** While Layer Integrated Gradients shows high agreement with Cross-Attention ($r = 0.9037$), standard Grad-CAM shows near-zero correlation ($r = 0.0589$), and Grad-CAM++ shows moderate correlation ($r = 0.4973$).
  - **INFERRED:** Cross-Attention weights capture query-key similarity distributions, while gradient attribution incorporates classification weight magnitudes and signs. Methods that incorporate gradient directions and path integration align more closely with attention queries than simple channel-averaged gradients.
  - **UNKNOWN:** The effect of multi-head query diversity on gradient alignment across individual cross-attention heads.

### Pattern 10: Presence-Location Confidence Decoupling
- **Definition:** Binary presence head predicts high presence probability ($P \ge 0.90$), but location logits are diffuse, uniform, or contradict the presence confidence.
- **Classification Status:**
  - **OBSERVED:** The presence classification head (`cls_head_list[0]`, 2 queries) and location head (`cls_head_list[1]`, 16 queries) operate in parallel on the bottleneck without mutual cross-head conditioning.
  - **INFERRED:** Independent classification heads allow presence detection to decouple from location localization, contributing to high presence sensitivity alongside low exact-location accuracy ($16.12\%$).
  - **UNKNOWN:** Whether joint multi-task conditioning or sequential presence-to-location gating would enforce consistent confidence across heads.

---

## 3. Candidate Architectural Limitations Identified by the Audit

The failure analysis identifies two candidate architectural limitations for future investigation:

1. **Coarse Bottleneck Representation for Classification ($7 \times 7 \times 7$):**
   - **OBSERVED:** The classification pathway operates exclusively on the compressed $7 \times 7 \times 7$ bottleneck tensor, where each token corresponds to an effective receptive field of $\approx 32\text{ mm}$ isotropic.
   - **INFERRED:** This coarse spatial resolution is a candidate limitation that may restrict the model's ability to delineate micro-aneurysms ($<5\text{ mm}$) and resolve dense adjacent vascular bifurcations.

2. **Limited Direct Access to High-Resolution Decoder Features:**
   - **OBSERVED:** The model contains dual high-resolution U-Net decoders that generate precise vascular segmentations ($86.07\%$ vessel Dice in Phase 2). However, these high-resolution decoder features and skip connections do not directly inform the classification heads during inference.
   - **INFERRED:** The separation between the high-resolution segmentation decoders and the coarse classification bottleneck is a candidate limitation that may prevent classification heads from utilizing verified vascular contours.

*Neither candidate limitation is claimed to be causally responsible for classification errors without formal ablation experiments in subsequent work.*
