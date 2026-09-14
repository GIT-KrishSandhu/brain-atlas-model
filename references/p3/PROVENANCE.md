# P3 Model Provenance & Heritage

## Overview
This document records the provenance, authorship, training configuration, and lineage of the official P3 model used as the foundational baseline in this research project.

- **Model Name:** `ResEncoderUNet_two_seg_with_cls_modality`
- **Authors:** Pengcheng Shi, et al. (Team "bravecowcow")
- **Competition:** RSNA 2025 Intracranial Aneurysm Detection Challenge (Kaggle)
- **Standing:** 2nd Place Solution
- **Reference Paper:** *Intracranial Aneurysm Classification and Segmentation via Tri-Axial ROI and Multi-Task Learning* (arXiv:2606.26706)
- **License:** Apache License 2.0 (inherited from the competition solution repository)

---

## Checkpoint Provenance
- **Original Source:** Kaggle dataset `pengchengshi/rsna-2nd-stage2-models` / `Dataset660_26classes_resize224_4661`
- **File Name:** `checkpoint_final.pth`
- **Checkpoint SHA256:** `E60B539D025A8ECCCF77D3FF1A45EF6888B5E059671CC8E24FFF572D520FA521`
- **Trainer Name:** `nnUNetTrainer_ResEncoderUNet_two_seg_with_cls_modality_CE_DC_AWDC_onlyMirror01_lr4e3_100epochs`
- **Total Parameters:** 109,359,299
- **State Dict Tensor Count:** 583
- **Trained Epochs:** 100 epochs on Stage-2 multi-task data

---

## Key Training Objectives (Pretrained Baseline)
The original P3 Stage-2 checkpoint was trained with a composite multi-task objective:
1. **Presence Binary Classification:** BCE loss with positive weighting on scalar presence logit.
2. **13-Class Location Classification:** Multi-label cross-entropy / BCE loss for 13 intracranial aneurysm locations.
3. **4-Class Modality Classification:** Classification of imaging modality.
4. **Vascular Segmentation (`seg_layers_1`):** Combined Cross-Entropy + Dice loss across 15 classes (13 vessel classes + 1 aneurysm class + 1 background).
5. **Territory Segmentation (`seg_layers_2`):** Combined Cross-Entropy + Dice loss across 14 classes (13 anatomical territories + 1 background).
