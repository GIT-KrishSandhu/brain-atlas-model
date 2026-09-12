import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import (
    ResEncoderUNet_two_seg_with_cls_modality
)

def test_memory_and_gradients():
    print("=" * 60)
    print("TESTING MEMORY AND GRADIENT HOOKING STRATEGIES")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ResEncoderUNet_two_seg_with_cls_modality(
        in_channels=1, out_channels_1=15, out_channels_2=14,
        cls_head_num_classes_list=[1, 13], cls_drop_out_list=[0.0, 0.0],
        cls_query_num_list=[2, 16], use_cross_attention=True,
        n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320],
        kernel_sizes=[[3, 3, 3]] * 6, strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6], n_conv_per_stage_decoder=[1] * 5,
        conv_bias=True, norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=False
    ).to(device)
    model.eval()
    
    ckpt_path = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['network_weights'], strict=True)
    print("Model loaded successfully.")
    
    # Strategy 1: Bottleneck Grad-CAM & Attention extraction
    print("\n--- Strategy 1: Bottleneck Grad-CAM & Cross-Attention ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    dummy_input = torch.randn(1, 1, 224, 224, 224, device=device)
    
    # Forward encoder under torch.no_grad()
    with torch.no_grad():
        conv_enc_outputs = [model.conv_encoder_blocks[0](dummy_input)]
        for i in range(1, len(model.conv_encoder_blocks)):
            conv_enc_outputs.append(model.conv_encoder_blocks[i](conv_enc_outputs[-1]))
        lres_input = conv_enc_outputs[-1] # shape (1, 320, 7, 7, 7)
        
    print(f"lres_input shape: {lres_input.shape}")
    
    # Enable gradient on lres_input
    lres_feature = lres_input.detach().clone().requires_grad_(True)
    
    # Hook attention weights on presence head
    attn_dict = {}
    def hook_attn(module, input, output):
        attn_dict['weights'] = output[1] # (B, query_num, 343)
        
    h = model.cls_head_list[0].pooling.cross_attention.register_forward_hook(hook_attn)
    
    # Forward through presence classification head
    presence_logit = model.cls_head_list[0](lres_feature)[0, 0]
    print(f"Presence logit: {presence_logit.item():.4f}")
    print(f"Attention weights shape: {attn_dict['weights'].shape}")
    
    # Backward to get bottleneck gradients for Grad-CAM
    presence_logit.backward()
    grad_bottleneck = lres_feature.grad
    print(f"Bottleneck grad shape: {grad_bottleneck.shape}")
    print(f"Bottleneck grad finite: {torch.isfinite(grad_bottleneck).all().item()}")
    
    # Compute 3D Grad-CAM
    # weights: GAP over spatial dimensions (D, H, W = 7, 7, 7)
    alpha = grad_bottleneck.mean(dim=(2, 3, 4), keepdim=True) # (1, 320, 1, 1, 1)
    gradcam_7 = torch.relu((alpha * lres_feature).sum(dim=1, keepdim=True)) # (1, 1, 7, 7, 7)
    gradcam_224 = nn.functional.interpolate(gradcam_7, size=(224, 224, 224), mode='trilinear', align_corners=False)
    print(f"Grad-CAM (224^3) shape: {gradcam_224.shape}")
    print(f"Grad-CAM min/max: {gradcam_224.min().item():.4f} / {gradcam_224.max().item():.4f}")
    
    h.remove()
    vram_strat1 = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak VRAM for Strategy 1: {vram_strat1:.3f} GB")
    
    # Strategy 2: Test input gradient under AMP (for Integrated Gradients)
    print("\n--- Strategy 2: Testing Input Gradient for Integrated Gradients with AMP ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    dummy_input_ig = torch.randn(1, 1, 224, 224, 224, device=device, requires_grad=True)
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            conv_enc_outputs = [model.conv_encoder_blocks[0](dummy_input_ig)]
            for i in range(1, len(model.conv_encoder_blocks)):
                conv_enc_outputs.append(model.conv_encoder_blocks[i](conv_enc_outputs[-1]))
            lres_input = conv_enc_outputs[-1]
            logit = model.cls_head_list[0](lres_input)[0, 0]
            
        logit.float().backward()
        print(f"Input gradient computed successfully! Shape: {dummy_input_ig.grad.shape}")
        vram_strat2 = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak VRAM for Strategy 2 (AMP Input Grad): {vram_strat2:.3f} GB")
    except Exception as e:
        print(f"Strategy 2 Failed: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    test_memory_and_gradients()
