import os, glob, json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cases = [
    ('Normal Case', 'topaneu_center1_mr_001', 'No aneurysm (Negative)'),
    ('Aneurysm Case 1', 'topaneu_center1_mr_017', 'Aneurysm at Location 28'),
    ('Aneurysm Case 2', 'topaneu_center1_mr_028', 'Aneurysm at Location 32'),
    ('Multi-Aneurysm Case', 'topaneu_center1_mr_024', 'Multiple Aneurysms (Locs 23, 24, 25)')
]

os.makedirs(r'D:\NLP_Project\scratch\visualizations', exist_ok=True)

for title, cid, desc in cases:
    img_path = rf'D:\NLP_Project\topaneu_release\images\{cid}_0000.nii.gz'
    vessel_path = rf'D:\NLP_Project\topaneu_release\vessel_masks\{cid}.nii.gz'
    loc_path = rf'D:\NLP_Project\topaneu_release\location_masks\{cid}.nii.gz'
    
    img = nib.load(img_path).get_fdata()
    vessel = nib.load(vessel_path).get_fdata() if os.path.exists(vessel_path) else None
    loc = nib.load(loc_path).get_fdata() if os.path.exists(loc_path) else None
    
    # Find slice with aneurysm if positive, else middle axial slice
    if loc is not None and np.any(loc > 0):
        z_indices = np.where(loc > 0)[2]
        z_slice = int(np.median(z_indices))
    elif vessel is not None and np.any(vessel > 0):
        z_indices = np.where(vessel > 0)[2]
        z_slice = int(np.median(z_indices))
    else:
        z_slice = img.shape[2] // 2
        
    slice_img = img[:, :, z_slice]
    slice_vessel = vessel[:, :, z_slice] if vessel is not None else np.zeros_like(slice_img)
    slice_loc = loc[:, :, z_slice] if loc is not None else np.zeros_like(slice_img)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"{title}: {cid} ({desc}) - Slice z={z_slice}", fontsize=14, fontweight='bold')
    
    # 1. Original scan
    axes[0].imshow(np.rot90(slice_img), cmap='gray')
    axes[0].set_title("Original MRA/CTA Scan")
    axes[0].axis('off')
    
    # 2. Vessel segmentation
    axes[1].imshow(np.rot90(slice_img), cmap='gray')
    vessel_overlay = np.ma.masked_where(slice_vessel == 0, slice_vessel)
    axes[1].imshow(np.rot90(vessel_overlay), cmap='autumn', alpha=0.6)
    axes[1].set_title("Vessel Anatomy Mask")
    axes[1].axis('off')
    
    # 3. Aneurysm annotation
    axes[2].imshow(np.rot90(slice_img), cmap='gray')
    loc_overlay = np.ma.masked_where(slice_loc == 0, slice_loc)
    axes[2].imshow(np.rot90(loc_overlay), cmap='spring', alpha=0.8)
    axes[2].set_title("Aneurysm Location Mask")
    axes[2].axis('off')
    
    # 4. Composite (Original + Vessel + Aneurysm)
    axes[3].imshow(np.rot90(slice_img), cmap='gray')
    axes[3].imshow(np.rot90(vessel_overlay), cmap='cool', alpha=0.5)
    axes[3].imshow(np.rot90(loc_overlay), cmap='hot', alpha=0.85)
    axes[3].set_title("Composite (Vessel + Aneurysm)")
    axes[3].axis('off')
    
    out_file = rf'D:\NLP_Project\scratch\visualizations\{cid}.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {out_file}")

print("All 4 visualizations generated successfully!")
