import sys, os, time, json
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

# Select 4 real cases: 2 positive, 2 negative
cases = [
    ('topaneu_center1_mr_017', 1.0, 1), # positive, MRA
    ('topaneu_center1_mr_028', 1.0, 1), # positive, MRA
    ('topaneu_center1_mr_001', 0.0, 1), # negative, MRA
    ('topaneu_center1_mr_005', 0.0, 1), # negative, MRA
]

print(f"Loading and preprocessing {len(cases)} cases for tiny overfit test...")
data_list = []
for cid, label, mod in cases:
    path = rf'D:\NLP_Project\topaneu_release\images\{cid}_0000.nii.gz'
    raw = nib.load(path).get_fdata().astype(np.float32)
    # Resample to 64^3 for fast iteration
    factors = [64.0 / s for s in raw.shape]
    res = zoom(raw, factors, order=1)
    norm = (res - 290.1886) / 1933.2790
    norm = np.clip(norm, -7905.17, 7954.08)
    t_img = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()
    t_lbl = torch.tensor([[label]], dtype=torch.float32)
    t_mod = torch.tensor([mod], dtype=torch.long)
    data_list.append((cid, t_img, t_lbl, t_mod))
    print(f"  Loaded {cid}: label={label}, shape={t_img.shape}")

# Instantiate B1 model
print("\nInstantiating ResEncoderUNet_two_seg_with_cls_modality (B1)...")
model = ResEncoderUNet_two_seg_with_cls_modality(
    in_channels=1,
    out_channels_1=15,
    out_channels_2=14,
    cls_head_num_classes_list=[1, 13],
    cls_drop_out_list=[0.0, 0.0],
    cls_query_num_list=[2, 16],
    use_cross_attention=True,
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
    n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={"eps": 1e-05, "affine": True},
    dropout_op=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
    deep_supervision=True
)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.25]))
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

model.train()
print("\nStarting overfit training loop (20 iterations)...")
losses = []
t0 = time.time()

for step in range(1, 21):
    step_loss = 0.0
    correct = 0
    total = len(data_list)
    optimizer.zero_grad()
    
    for cid, x, y, _ in data_list:
        cls_preds = model(x, only_forward_cls=True)
        pred = cls_preds[0]
        loss = criterion(pred, y)
        loss.backward()
        step_loss += loss.item()
        
        prob = torch.sigmoid(pred).item()
        pred_label = 1.0 if prob > 0.5 else 0.0
        if pred_label == y.item():
            correct += 1
            
    optimizer.step()
    avg_loss = step_loss / total
    acc = correct / total * 100.0
    losses.append(avg_loss)
    
    if step % 2 == 0 or step == 1:
        print(f"  Step {step:02d}/20 | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")

t_elapsed = time.time() - t0
print(f"\nTiny overfit test completed in {t_elapsed:.2f}s!")
print(f"Initial loss: {losses[0]:.4f} -> Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

# Evaluate final predictions
model.eval()
print("\nFinal Predictions on Tiny Subset:")
with torch.no_grad():
    for cid, x, y, _ in data_list:
        p = torch.sigmoid(model(x, only_forward_cls=True)[0]).item()
        print(f"  {cid}: True={y.item():.0f}, Predicted Prob={p:.4f} ({'CORRECT' if (p>0.5) == (y.item()==1.0) else 'WRONG'})")

print("\n=== TINY SUBSET OVERFIT TEST COMPLETE ===")
