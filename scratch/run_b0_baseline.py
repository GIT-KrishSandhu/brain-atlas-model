import sys, os, time, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score

# 1. Identify valid cases in topaneu_release
images = {os.path.basename(f).replace('_0000.nii.gz', ''): f 
          for f in glob.glob(r'D:\NLP_Project\topaneu_release\images\*.nii.gz') 
          if not os.path.basename(f).startswith('._')}

pos_cases = []
neg_cases = []

for cid, img_path in sorted(images.items()):
    jf = os.path.join(r'D:\NLP_Project\topaneu_release\location_jsons', f'{cid}.json')
    if os.path.exists(jf):
        with open(jf) as fp:
            d = json.load(fp)
        if d.get('locations') and len(d['locations']) > 0:
            pos_cases.append(cid)
        else:
            neg_cases.append(cid)

print(f"Total available cases: {len(pos_cases)} positive, {len(neg_cases)} negative.")

# Controlled dataset split:
# Train: 8 positive, 8 negative (16 cases)
# Val: 4 positive, 4 negative (8 cases)
np.random.seed(42)
train_pos = pos_cases[:8]
train_neg = neg_cases[:8]
val_pos = pos_cases[8:12]
val_neg = neg_cases[8:12]

train_cids = [(c, 1.0) for c in train_pos] + [(c, 0.0) for c in train_neg]
val_cids = [(c, 1.0) for c in val_pos] + [(c, 0.0) for c in val_neg]

np.random.shuffle(train_cids)
np.random.shuffle(val_cids)

print(f"Train cases: {len(train_cids)} (8 pos, 8 neg)")
print(f"Val cases: {len(val_cids)} (4 pos, 4 neg)")

def load_and_preprocess(cid, target_dim=64):
    path = images[cid]
    raw = nib.load(path).get_fdata().astype(np.float32)
    factors = [float(target_dim) / s for s in raw.shape]
    res = zoom(raw, factors, order=1)
    norm = (res - 290.1886) / 1933.2790
    norm = np.clip(norm, -7905.17, 7954.08)
    return torch.from_numpy(norm).unsqueeze(0).float() # (1, 64, 64, 64)

print("Pre-loading train dataset...")
train_data = []
for cid, label in train_cids:
    t_x = load_and_preprocess(cid)
    train_data.append((t_x, label, cid))

print("Pre-loading val dataset...")
val_data = []
for cid, label in val_cids:
    t_x = load_and_preprocess(cid)
    val_data.append((t_x, label, cid))

# 2. Simple 3D CNN Architecture (B0 Baseline)
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # 32^3
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # 16^3
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), # 8^3
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1) # 1x1x1
        )
        self.classifier = nn.Linear(128, 1)
        
    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

model = Simple3DCNN()
param_count = sum(p.numel() for p in model.parameters())
print(f"\nB0 Model constructed! Parameter count: {param_count:,} ({param_count/1e3:.1f} K)")

# 3. Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 10
batch_size = 4

print(f"\nStarting B0 training ({epochs} epochs, batch_size={batch_size}, lr=1e-3)...")
history = []
t0 = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    np.random.shuffle(train_data)
    epoch_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        xb = torch.stack([item[0] for item in batch])
        yb = torch.tensor([[item[1]] for item in batch], dtype=torch.float32)
        
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(batch)
        num_batches += 1
        
    train_loss = epoch_loss / len(train_data)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_preds_list = []
    val_targets_list = []
    
    with torch.no_grad():
        for t_x, label, _ in val_data:
            xb = t_x.unsqueeze(0)
            yb = torch.tensor([[label]], dtype=torch.float32)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            prob = torch.sigmoid(out).item()
            val_preds_list.append(prob)
            val_targets_list.append(label)
            
    val_loss = val_loss / len(val_data)
    history.append((epoch, train_loss, val_loss))
    
    print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

train_time = time.time() - t0
print(f"\nB0 training finished in {train_time:.2f}s!")

# 4. Final Evaluation on Validation Set
val_preds = np.array(val_preds_list)
val_targets = np.array(val_targets_list)
val_binary_preds = (val_preds >= 0.5).astype(float)

auroc = roc_auc_score(val_targets, val_preds)
auprc = average_precision_score(val_targets, val_preds)
cm = confusion_matrix(val_targets, val_binary_preds, labels=[0.0, 1.0])
tn, fp, fn, tp = cm.ravel()

acc = (tp + tn) / (tp + tn + fp + fn)
sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # recall
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

print("\n=== B0 BASELINE EVALUATION METRICS ===")
print(f"Validation Cases: {len(val_targets)} (4 pos, 4 neg)")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print(f"Sensitivity / Recall: {sens:.4f} ({sens*100:.1f}%)")
print(f"Specificity: {spec:.4f} ({spec*100:.1f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.1f}%)")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp}):")
print(f"  [[TN={tn}, FP={fp}],")
print(f"   [FN={fn}, TP={tp}]]")

# Save results json
results = {
    "experiment": "B0 - Minimal 3D CNN Baseline",
    "architecture": "Simple3DCNN (4 Conv3D + BN + ReLU + MaxPool + AdaptiveAvgPool + Linear)",
    "parameter_count": param_count,
    "input_resolution": "64x64x64",
    "batch_size": batch_size,
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "epochs": epochs,
    "train_cases": len(train_data),
    "val_cases": len(val_data),
    "initial_train_loss": history[0][1],
    "final_train_loss": history[-1][1],
    "final_val_loss": history[-1][2],
    "auroc": float(auroc),
    "auprc": float(auprc),
    "accuracy": float(acc),
    "sensitivity": float(sens),
    "specificity": float(spec),
    "precision": float(prec),
    "f1": float(f1),
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "training_time_seconds": round(train_time, 2)
}

with open(r'D:\NLP_Project\scratch\b0_results.json', 'w') as fp:
    json.dump(results, fp, indent=2)

print("\nSaved B0 results to scratch/b0_results.json!")
