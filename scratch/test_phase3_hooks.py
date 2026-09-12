import os
import sys
import torch
import torch.nn as nn

# Configure module paths
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import (
    ResEncoderUNet_two_seg_with_cls_modality
)

def test_hooks():
    print("=" * 60)
    print("PHASE 3: ARCHITECTURE & HOOK AUDIT TEST")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder=[1] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False
    ).to(device)
    model.eval()
    
    # Load frozen weights
    ckpt_path = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['network_weights']
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        print("Warning: Checkpoint not found at expected path, testing architecture with initialized weights.")
        
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check hooked modules
    activations = {}
    gradients = {}
    attention_weights_dict = {}
    
    # Hook bottleneck
    def forward_hook_bottleneck(module, input, output):
        activations['bottleneck'] = output
        output.retain_grad()
        
    def forward_hook_attn(module, input, output):
        # output is (attended, attention_weights)
        attention_weights_dict['presence_attn'] = output[1]
        
    hook1 = model.conv_encoder_blocks[5].register_forward_hook(forward_hook_bottleneck)
    hook2 = model.cls_head_list[0].pooling.cross_attention.register_forward_hook(forward_hook_attn)
    
    # Dummy input
    dummy_input = torch.randn(1, 1, 224, 224, 224, device=device, requires_grad=True)
    
    print("\nRunning forward pass (only_forward_cls=True)...")
    with torch.enable_grad():
        cls_preds = model(dummy_input, only_forward_cls=True)
        
        presence_logit = cls_preds[0][0, 0]
        location_logits = cls_preds[1]
        
        print(f"Presence logit shape: {cls_preds[0].shape}, value: {presence_logit.item():.4f}")
        print(f"Location logits shape: {location_logits.shape}")
        
        print(f"\nHooked bottleneck shape: {activations['bottleneck'].shape}")
        print(f"Hooked attention weights shape: {attention_weights_dict['presence_attn'].shape}")
        
        # Test backward pass
        print("\nRunning backward pass on presence_logit...")
        presence_logit.backward()
        
        bottleneck_grad = activations['bottleneck'].grad
        input_grad = dummy_input.grad
        
        print(f"Bottleneck grad shape: {bottleneck_grad.shape if bottleneck_grad is not None else None}")
        print(f"Input grad shape: {input_grad.shape if input_grad is not None else None}")
        print(f"Bottleneck grad is finite: {torch.isfinite(bottleneck_grad).all().item() if bottleneck_grad is not None else False}")
        print(f"Input grad is finite: {torch.isfinite(input_grad).all().item() if input_grad is not None else False}")
        
    hook1.remove()
    hook2.remove()
    print("\nHooks removed cleanly.")
    print("=" * 60)

if __name__ == "__main__":
    test_hooks()
