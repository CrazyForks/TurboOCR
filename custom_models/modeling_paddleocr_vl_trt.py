# Custom TRT-LLM PyTorch backend for PaddleOCRVLForConditionalGeneration.
#
# Architecture:
#   - Vision encoder: PaddleOCR SigLIP-style ViT (27 layers, hidden=1152) — run in plain FP16
#   - Projector: pre_norm → linear_1(4608→4608) → GELU → linear_2(4608→1024) — plain FP16
#   - Language model: Ernie-4.5-based GQA LLM (18 layers, hidden=1024, GQA 16/2 heads)
#     wired as a Qwen2/LlamaForCausalLM config shim so TRT-LLM's NVFP4 Linear is used natively
#
# Weight namespace in model.safetensors (NVFP4, modelopt 0.43.0):
#   model.language_model.layers.{i}.{...}
#   model.projector.{...}
#   model.visual.vision_model.{...}
#   lm_head.weight (FP16, tied with embed_tokens)
#
# Usage:
#   trtllm-serve pytorch <model_dir> \
#     --custom_module_dirs /path/to/custom_models \
#     --port 8005
#
# Mount: no engine build required — pytorch backend runs models as-is.

import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Minimal import shim — only available inside trtllm-serve container
# ---------------------------------------------------------------------------
try:
    from tensorrt_llm._torch.attention_backend.interface import (
        AttentionMetadata, PositionalEmbeddingParams, RopeParams)
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_utils import (
        DecoderModelForCausalLM, _load_weights_impl, filter_weights,
        register_auto_model)
    from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
    from tensorrt_llm._torch.models.modeling_utils import ModelConfig as MC
    from tensorrt_llm.functional import PositionEmbeddingType
    from tensorrt_llm.inputs.registry import (
        MultimodalPlaceholderMetadata,
        MultimodalPlaceholderPlacement,
        register_input_processor,
    )
    _TRTLLM_AVAILABLE = True
except ImportError:
    _TRTLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config shim: we present text_config as a flat LlamaConfig to TRT-LLM
# ---------------------------------------------------------------------------

def _make_text_model_config(parent_model_config: "ModelConfig") -> "ModelConfig":
    """Build a ModelConfig whose pretrained_config looks like a Qwen2/Llama config.

    TRT-LLM's LlamaForCausalLM only needs the flat text config attributes;
    we clone the parent ModelConfig and replace pretrained_config with the
    flattened text_config dict merged into it.
    """
    import copy
    mc = copy.deepcopy(parent_model_config)

    pc = parent_model_config.pretrained_config
    text_cfg = pc.text_config if hasattr(pc, 'text_config') else {}

    if isinstance(text_cfg, dict):
        text_dict = text_cfg
    else:
        text_dict = text_cfg.__dict__

    for k, v in text_dict.items():
        setattr(mc.pretrained_config, k, v)

    mc.pretrained_config.architectures = ["LlamaForCausalLM"]
    mc.pretrained_config.model_type = "llama"

    # Convert dtype string to torch.dtype so TRT-LLM Linear.create_weights gets
    # the right type (not a string).
    import torch as _torch
    _dtype_map = {
        "float16": _torch.float16, "fp16": _torch.float16,
        "bfloat16": _torch.bfloat16, "bf16": _torch.bfloat16,
        "float32": _torch.float32, "fp32": _torch.float32,
    }
    raw_dtype = text_dict.get("dtype", "float16")
    if isinstance(raw_dtype, str):
        torch_dtype = _dtype_map.get(raw_dtype, _torch.float16)
    else:
        torch_dtype = raw_dtype or _torch.float16
    mc.pretrained_config.dtype = torch_dtype
    mc.pretrained_config.torch_dtype = torch_dtype

    # rope_scaling: TRT-LLM valid types: none, linear, dynamic, longrope, llama3, yarn, mrope
    # "default" rope_type means standard RoPE (no special scaling) → set rope_scaling=None
    rope_params = text_dict.get('rope_parameters', {})
    if rope_params:
        rope_type = rope_params.get('rope_type', rope_params.get('type', 'default'))
        if rope_type in ('default', 'none', ''):
            # Standard RoPE without scaling - just set theta
            mc.pretrained_config.rope_scaling = None
            mc.pretrained_config.rope_theta = float(
                rope_params.get('rope_theta', 500000.0))
        else:
            rope_scaling = dict(rope_params)
            if 'type' not in rope_scaling and 'rope_type' in rope_scaling:
                rope_scaling['type'] = rope_scaling['rope_type']
            mc.pretrained_config.rope_scaling = rope_scaling
    else:
        mc.pretrained_config.rope_scaling = None

    # Required by LlamaConfig validation and KV cache sizing
    if not hasattr(mc.pretrained_config, 'attention_bias'):
        mc.pretrained_config.attention_bias = False
    if not hasattr(mc.pretrained_config, 'mlp_bias'):
        mc.pretrained_config.mlp_bias = False

    # Ensure num_key_value_heads is set (needed for KV cache size calculation)
    if not getattr(mc.pretrained_config, 'num_key_value_heads', None):
        mc.pretrained_config.num_key_value_heads = text_dict.get(
            'num_key_value_heads', text_dict.get('num_attention_heads', 2))

    # Ensure head_dim is set
    if not getattr(mc.pretrained_config, 'head_dim', None):
        mc.pretrained_config.head_dim = text_dict.get('head_dim', 128)

    # Ensure vocab_size, num_attention_heads, num_hidden_layers are explicit
    for attr in ('vocab_size', 'num_attention_heads', 'num_hidden_layers',
                 'hidden_size', 'intermediate_size', 'rms_norm_eps',
                 'max_position_embeddings', 'pad_token_id', 'bos_token_id',
                 'eos_token_id'):
        if not getattr(mc.pretrained_config, attr, None) and attr in text_dict:
            setattr(mc.pretrained_config, attr, text_dict[attr])

    return mc


# ---------------------------------------------------------------------------
# Vision-side dequantization helper (NVFP4 → FP16 on weight load)
# ---------------------------------------------------------------------------

def _dequant_nvfp4_weight(weight_u8: torch.Tensor,
                           weight_scale: torch.Tensor,
                           weight_scale_2: torch.Tensor) -> torch.Tensor:
    """Convert NVFP4 uint8 packed tensor to FP16.

    NVFP4 packing: packed = (q[..., 1::2] << 4) | q[..., 0::2]
    - Low nibble (bits 3-0) → even output indices
    - High nibble (bits 7-4) → odd output indices

    E2M1 lookup (indices 0-15):
      [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]

    Dequantization: w = e2m1[nibble] * per_block_scale * global_scale
    where per_block_scale = weight_scale (FP8 e4m3fn) * weight_scale_2 (FP32)
    """
    # Exact E2M1 lookup table (matches modelopt's e2m1_values)
    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )

    out_features, in_packed = weight_u8.shape

    # Unpack: low nibble at even positions [0::2], high at odd [1::2]
    # Reconstruct: shape [out_features, in_packed*2]
    unpacked = torch.empty(out_features, in_packed * 2, dtype=torch.float32)
    unpacked[:, 0::2] = (weight_u8 & 0x0F).float()   # low nibble
    unpacked[:, 1::2] = (weight_u8 >> 4).float()       # high nibble

    # Map through E2M1 lookup table
    idx = unpacked.long().clamp(0, 15)
    w = e2m1_lut[idx]  # [out_features, in_features]

    # Apply per-block scale (group_size=16, weight_scale is FP8 e4m3fn → float32)
    group_size = 16
    in_features = in_packed * 2
    num_groups = in_features // group_size
    # weight_scale shape: [out_features, num_groups]
    ws = weight_scale.to(torch.float32).view(out_features, num_groups)
    # per_block_scale = ws * global_scale
    per_block_scale = ws * weight_scale_2.float()
    # Apply per-group scale
    w = w.view(out_features, num_groups, group_size)
    w = w * per_block_scale.unsqueeze(-1)
    w = w.reshape(out_features, in_features)

    return w.to(torch.float16)


def _load_linear_dequant(linear: nn.Linear, prefix: str, weights: Dict) -> None:
    """Load NVFP4-quantized weight into a plain nn.Linear (dequantizing on load)."""
    w_key = f"{prefix}.weight"
    ws_key = f"{prefix}.weight_scale"
    ws2_key = f"{prefix}.weight_scale_2"
    b_key = f"{prefix}.bias"

    if w_key in weights and ws_key in weights:
        ws2_val = weights[ws2_key].reshape(()) if ws2_key in weights else torch.tensor(1.0)
        w_fp16 = _dequant_nvfp4_weight(
            weights[w_key][:].cpu(),
            weights[ws_key][:].cpu(),
            ws2_val.cpu(),
        )
        linear.weight.data.copy_(w_fp16)
    elif w_key in weights:
        linear.weight.data.copy_(weights[w_key][:].to(torch.float16).cpu())

    if b_key in weights:
        linear.bias.data.copy_(weights[b_key][:].to(torch.float16).cpu())


# ---------------------------------------------------------------------------
# Plain-PyTorch vision encoder (SigLIP-style, run in FP16, no TRT-LLM kernels)
# ---------------------------------------------------------------------------

class _VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class _VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _VisionEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = _VisionAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = _VisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _VisionEmbeddings(nn.Module):
    """Patch embedding that accepts PaddleOCR's flat-patch format.

    PaddleOCRVLImageProcessor returns patches of shape [N, C, P, P] where
    each patch is P×P pixels. The Conv2d(C, H, P, P) with stride=P applied
    to a P×P patch produces a 1×1 output → effectively a linear projection.
    Position is tracked via `packing_position_embedding` (sequential IDs 0..N-1).
    """

    def __init__(self, hidden_size: int, num_channels: int, patch_size: int,
                 image_size: int, max_packed_pos: int = 32768):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        # Legacy fixed-grid position embedding (loaded from checkpoint)
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, hidden_size)
        # Packing position embedding: arbitrary length sequential IDs (may not be in NVFP4 ckpt)
        self.packing_position_embedding = nn.Embedding(max_packed_pos, hidden_size)
        # Initialize to zero so we can detect "not loaded" state
        nn.init.zeros_(self.packing_position_embedding.weight)
        self._packing_pos_emb_loaded = False
        self.register_buffer(
            'position_ids',
            torch.arange(num_patches).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Accept flat patches [N, C, P, P] and return embeddings [N, E].

        Args:
            pixel_values: [N_patches, C, P, P]
            image_grid_thw: [n_images, 3] (t, h, w) grid dims, used for position interpolation
        """
        import math as _math
        N = pixel_values.shape[0]
        # Conv2d on P×P patch with kernel P → output shape [N, E, 1, 1]
        patch_embeds = self.patch_embedding(pixel_values)  # [N, E, 1, 1]
        embeddings = patch_embeds.flatten(2).squeeze(-1)   # [N, E]

        # Use packing_position_embedding if loaded (present in BF16 ckpt, absent in NVFP4)
        pos_ids = torch.arange(N, device=pixel_values.device)
        if self._packing_pos_emb_loaded and N < self.packing_position_embedding.num_embeddings:
            pos_emb = self.packing_position_embedding(pos_ids)
        else:
            # Interpolate the fixed-grid position embedding to the actual spatial grid
            num_orig = self.position_embedding.weight.shape[0]
            E = self.position_embedding.embedding_dim
            n_orig = int(_math.isqrt(num_orig))  # 27 for 729-entry table

            if image_grid_thw is not None:
                # Use actual h, w from grid to avoid non-integer sqrt issue
                t, h, w = int(image_grid_thw[0, 0]), int(image_grid_thw[0, 1]), int(image_grid_thw[0, 2])
            else:
                # Fallback: assume square grid
                n = int(_math.isqrt(N))
                t, h, w = 1, n, n

            if h == n_orig and w == n_orig and t == 1:
                pos_emb = self.position_embedding.weight  # [N, E] — exact match
            else:
                # Bilinear interpolation from n_orig×n_orig to h×w
                pe = self.position_embedding.weight.view(1, n_orig, n_orig, E).permute(0, 3, 1, 2)
                pe = nn.functional.interpolate(pe, size=(h, w), mode='bilinear', align_corners=False)
                # pe: [1, E, h, w] → [h*w, E]
                pe = pe.squeeze(0).permute(1, 2, 0).reshape(h * w, E)
                # Repeat for t temporal slices
                pos_emb = pe.repeat(t, 1)  # [t*h*w, E]

        return embeddings + pos_emb  # [N, E]


class PaddleOCRVisionEncoder(nn.Module):
    """Plain PyTorch vision encoder (weights dequantized from NVFP4 to FP16 on load)."""

    def __init__(self, vision_config):
        super().__init__()
        hidden_size = vision_config.get('hidden_size', 1152)
        intermediate_size = vision_config.get('intermediate_size', 4304)
        num_heads = vision_config.get('num_attention_heads', 16)
        num_layers = vision_config.get('num_hidden_layers', 27)
        patch_size = vision_config.get('patch_size', 14)
        image_size = vision_config.get('image_size', 384)
        layer_norm_eps = vision_config.get('layer_norm_eps', 1e-6)

        self.embeddings = _VisionEmbeddings(hidden_size, 3, patch_size, image_size)
        self.encoder_layers = nn.ModuleList([
            _VisionEncoderLayer(hidden_size, intermediate_size, num_heads, layer_norm_eps)
            for _ in range(num_layers)
        ])
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # pixel_values: [N_patches, C, P, P] — flat patches from PaddleOCR image processor
        x = self.embeddings(pixel_values, image_grid_thw)  # [N_patches, E]
        # Add batch dim for encoder layers (they expect [B, N, E])
        x = x.unsqueeze(0)                  # [1, N_patches, E]
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.post_layernorm(x)          # [1, N_patches, E]
        return x                            # [1, N_patches, E]

    def load_weights_from_dict(self, weights: Dict, prefix: str = "model.visual.vision_model") -> None:
        """Load NVFP4-quantized vision weights (dequantize on load)."""
        # Embeddings (FP16, not quantized)
        emb_prefix = f"{prefix}.embeddings"
        if f"{emb_prefix}.patch_embedding.weight" in weights:
            self.embeddings.patch_embedding.weight.data.copy_(
                weights[f"{emb_prefix}.patch_embedding.weight"][:])
        if f"{emb_prefix}.patch_embedding.bias" in weights:
            self.embeddings.patch_embedding.bias.data.copy_(
                weights[f"{emb_prefix}.patch_embedding.bias"][:])
        if f"{emb_prefix}.position_embedding.weight" in weights:
            pe_w = weights[f"{emb_prefix}.position_embedding.weight"][:]
            if pe_w.shape != self.embeddings.position_embedding.weight.shape:
                self.embeddings.position_embedding = nn.Embedding(
                    pe_w.shape[0], pe_w.shape[1]).to(pe_w.device)
            self.embeddings.position_embedding.weight.data.copy_(pe_w)
        # Packing position embedding (variable-length sequential IDs; absent in NVFP4 ckpt)
        ppack_key = f"{emb_prefix}.packing_position_embedding.weight"
        if ppack_key in weights:
            ppack_w = weights[ppack_key][:]
            if ppack_w.shape[0] != self.embeddings.packing_position_embedding.num_embeddings:
                self.embeddings.packing_position_embedding = nn.Embedding(
                    ppack_w.shape[0], ppack_w.shape[1])
            self.embeddings.packing_position_embedding.weight.data.copy_(ppack_w)
            self.embeddings._packing_pos_emb_loaded = True

        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            lp = f"{prefix}.encoder.layers.{i}"
            # Layer norms (FP16)
            for attr, key in [
                ('layer_norm1.weight', f"{lp}.layer_norm1.weight"),
                ('layer_norm1.bias', f"{lp}.layer_norm1.bias"),
                ('layer_norm2.weight', f"{lp}.layer_norm2.weight"),
                ('layer_norm2.bias', f"{lp}.layer_norm2.bias"),
            ]:
                if key in weights:
                    obj = layer
                    for part in attr.split('.'):
                        obj = getattr(obj, part)
                    obj.data.copy_(weights[key][:])

            # Attention projections (NVFP4)
            for proj_name, linear_obj in [
                ('q_proj', layer.self_attn.q_proj),
                ('k_proj', layer.self_attn.k_proj),
                ('v_proj', layer.self_attn.v_proj),
                ('out_proj', layer.self_attn.out_proj),
            ]:
                _load_linear_dequant(linear_obj, f"{lp}.self_attn.{proj_name}", weights)

            # MLP (NVFP4)
            _load_linear_dequant(layer.mlp.fc1, f"{lp}.mlp.fc1", weights)
            _load_linear_dequant(layer.mlp.fc2, f"{lp}.mlp.fc2", weights)

        # Post layernorm
        if f"{prefix}.post_layernorm.weight" in weights:
            self.post_layernorm.weight.data.copy_(
                weights[f"{prefix}.post_layernorm.weight"][:])
        if f"{prefix}.post_layernorm.bias" in weights:
            self.post_layernorm.bias.data.copy_(
                weights[f"{prefix}.post_layernorm.bias"][:])


# ---------------------------------------------------------------------------
# Projector (plain FP16, dequantized from NVFP4 on load)
# ---------------------------------------------------------------------------

class PaddleOCRProjector(nn.Module):
    """Vision-to-LLM projector: pre_norm → spatial_merge → linear_1 → GELU → linear_2.

    Matches PaddleOCRVLMultiModalProjector from the original HF model.
    Input: [N_patches, vision_hidden] flat sequence.
    Merge: rearrange "(t h p1 w p2) d -> (t h w) (p1 p2 d)" with p1=p2=spatial_merge.
    Output: [N_merged, text_hidden].
    """

    def __init__(self, vision_hidden: int, text_hidden: int, spatial_merge: int = 2):
        super().__init__()
        self.spatial_merge = spatial_merge
        merged_dim = vision_hidden * spatial_merge * spatial_merge
        self.pre_norm = nn.LayerNorm(vision_hidden, eps=1e-5)
        self.linear_1 = nn.Linear(merged_dim, merged_dim, bias=True)
        self.linear_2 = nn.Linear(merged_dim, text_hidden, bias=True)
        self.act = nn.GELU()

    def forward(
        self,
        vision_feats: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # vision_feats: [1, N_patches, vision_hidden] (batch dim added by encoder)
        x = vision_feats.squeeze(0)  # [N_patches, D]
        m = self.spatial_merge

        # Apply pre_norm before spatial merge (matches original)
        x = self.pre_norm(x)  # [N_patches, D]

        if image_grid_thw is not None:
            # Per-image spatial merge using actual grid dimensions
            t, h, w = image_grid_thw[0].tolist() if image_grid_thw.dim() > 1 else image_grid_thw.tolist()
            t, h, w = int(t), int(h), int(w)
            # Rearrange: (t h p1 w p2) d -> (t h w) (p1 p2 d)
            # x shape: [t*h*w, D]
            x = x.view(t, h // m, m, w // m, m, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [t, h//m, w//m, m, m, D]
            x = x.view(t * (h // m) * (w // m), m * m * x.shape[-1])
        else:
            # Fallback: assume square grid
            N, D = x.shape
            n = int(N ** 0.5)
            # Rearrange assuming t=1, h=n, w=n
            x = x.view(1, n // m, m, n // m, m, D)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.view((n // m) * (n // m), m * m * D)

        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x  # [N_merged, text_hidden]

    def load_weights_from_dict(self, weights: Dict, prefix: str = "model.projector") -> None:
        # pre_norm (FP16)
        for attr in ['weight', 'bias']:
            k = f"{prefix}.pre_norm.{attr}"
            if k in weights:
                getattr(self.pre_norm, attr).data.copy_(weights[k][:])
        # linear_1, linear_2 (NVFP4)
        _load_linear_dequant(self.linear_1, f"{prefix}.linear_1", weights)
        _load_linear_dequant(self.linear_2, f"{prefix}.linear_2", weights)


# ---------------------------------------------------------------------------
# Main registered model class
# ---------------------------------------------------------------------------

if _TRTLLM_AVAILABLE:
    from tensorrt_llm._torch.models.modeling_llama import LlamaForCausalLM
    try:
        from .input_processor_paddleocr_vl import PaddleOCRVLInputProcessor
    except ImportError:
        # Fallback for direct-file execution (not loaded as package)
        import importlib.util as _ilu
        import os as _os
        _proc_path = _os.path.join(_os.path.dirname(__file__), "input_processor_paddleocr_vl.py")
        _spec = _ilu.spec_from_file_location("input_processor_paddleocr_vl", _proc_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        PaddleOCRVLInputProcessor = _mod.PaddleOCRVLInputProcessor

    @register_auto_model("PaddleOCRVLForConditionalGeneration")
    @register_input_processor(
        PaddleOCRVLInputProcessor,
        model_type="paddleocr_vl",
        placeholder_metadata=MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<|IMAGE_PLACEHOLDER|>"},
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        ),
    )
    class PaddleOCRVLForConditionalGeneration(nn.Module):
        """TRT-LLM PyTorch backend model for PaddleOCR-VL-1.5 (NVFP4 quantized).

        LLM backbone runs via TRT-LLM's LlamaForCausalLM with NVFP4 Linear layers.
        Vision encoder and projector run as plain PyTorch FP16 modules (weights
        dequantized from NVFP4 on load). Only the LLM uses TRT-LLM's kernel path.
        """

        def __init__(self, model_config: "ModelConfig"):
            super().__init__()
            self.model_config = model_config
            # Required by TRT-LLM's BaseWeightMapper.init_model_and_config()
            self.config = model_config.pretrained_config
            pc = model_config.pretrained_config

            # Extract sub-configs
            text_cfg_dict = {}
            if hasattr(pc, 'text_config'):
                tc = pc.text_config
                text_cfg_dict = tc if isinstance(tc, dict) else tc.__dict__
            vision_cfg = {}
            if hasattr(pc, 'vision_config'):
                vc = pc.vision_config
                vision_cfg = vc if isinstance(vc, dict) else vc.__dict__

            text_hidden = text_cfg_dict.get('hidden_size', 1024)
            vision_hidden = vision_cfg.get('hidden_size', 1152)
            spatial_merge = vision_cfg.get('spatial_merge_size', 2)
            vocab_size = text_cfg_dict.get('vocab_size', 103424)

            # Promote key LLM attributes to top-level pretrained_config so that
            # TRT-LLM's resource_manager.get_cache_size_per_token() can read them.
            _promote = [
                'num_key_value_heads', 'num_attention_heads', 'num_hidden_layers',
                'hidden_size', 'head_dim', 'max_position_embeddings',
                'vocab_size', 'rms_norm_eps',
            ]
            for attr in _promote:
                if text_cfg_dict.get(attr) is not None:
                    setattr(pc, attr, text_cfg_dict[attr])

            # Vision encoder (plain FP16) — cast to FP16 immediately so LayerNorm
            # biases match the FP16 activations before weights are loaded.
            self.vision_encoder = PaddleOCRVisionEncoder(vision_cfg).to(torch.float16)
            self.projector = PaddleOCRProjector(vision_hidden, text_hidden, spatial_merge).to(torch.float16)

            # Language model via TRT-LLM's LlamaForCausalLM
            llm_config = _make_text_model_config(model_config)
            self.llm = LlamaForCausalLM(llm_config)

            self.logits_processor = LogitsProcessor()
            self.image_token_id = getattr(pc, 'image_token_id', 100295)
            self.vocab_size = vocab_size
            # OOV sentinel: same value input processor uses for image placeholders
            # (vocab_size + 1) — TRT-LLM's fuse_input_embeds filters tokens >= vocab_size
            self.tllm_mm_token_id = vocab_size + 1

        @property
        def vocab_size_padded(self) -> int:
            return self.llm.vocab_size_padded

        @property
        def multimodal_data_device_paths(self) -> List[str]:
            """Paths inside multimodal_data dict that should be moved to CUDA."""
            return ["pixel_values", "image_grid_thw"]

        def infer_max_seq_len(self) -> int:
            return self.llm.infer_max_seq_len()

        @torch.inference_mode()
        def forward(
            self,
            attn_metadata: "AttentionMetadata",
            input_ids: Optional[torch.IntTensor] = None,
            position_ids: Optional[torch.IntTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            return_context_logits: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            multimodal_params = kwargs.get("multimodal_params", [])
            from tensorrt_llm._torch.models.modeling_multimodal_utils import fuse_input_embeds

            # Only check for OOV tokens when there are actual multimodal params (context phase).
            # During CUDA graph capture/replay (generation phase, no mm params), skip this
            # to avoid .item() which is not permitted during stream capture.
            has_mm_params = bool(multimodal_params)
            is_cuda_graph = getattr(attn_metadata, 'is_cuda_graph', False)
            n_oov_in_chunk = 0
            # Only run CPU-sync operations when NOT in CUDA graph capture/replay mode
            if has_mm_params and not is_cuda_graph and input_ids is not None:
                n_oov_in_chunk = int((input_ids >= self.tllm_mm_token_id).sum().item())

            # Step 2: If this chunk has image tokens, compute or retrieve vision embeddings.
            mm_embeds = []
            if n_oov_in_chunk > 0:
                # Look for pixel_values in multimodal_params
                device = next(self.vision_encoder.parameters()).device
                all_embs = []
                for mm_param in multimodal_params:
                    data = mm_param.multimodal_data
                    if not data:
                        continue
                    pixel_values = data.get("pixel_values")
                    cached_emb = data.get("multimodal_embedding")
                    if cached_emb is not None:
                        all_embs.append(cached_emb)
                    elif pixel_values is not None:
                        pixel_values = pixel_values.to(dtype=torch.float16, device=device)
                        image_grid_thw = data.get("image_grid_thw")
                        if image_grid_thw is not None:
                            image_grid_thw = image_grid_thw.to(device)
                        vis_feats = self.vision_encoder(pixel_values, image_grid_thw)
                        img_emb = self.projector(vis_feats, image_grid_thw)  # [N_merged, E]
                        # Cache for potential re-use
                        data["multimodal_embedding"] = img_emb
                        all_embs.append(img_emb)
                if all_embs:
                    full_emb = torch.cat(all_embs, dim=0)  # [total_img_tokens, E]
                    # Only take the slice matching this chunk's OOV token count
                    mm_embeds = [full_emb[:n_oov_in_chunk]]

            # Step 3: Splice embeddings into the token sequence (fuse_input_embeds handles
            # the text embedding lookup and scatter-assign for image positions).
            fuse_result = fuse_input_embeds(
                self.llm.model.embed_tokens,
                input_ids,
                mm_embeds,
                mm_token_ids=(
                    torch.tensor([self.tllm_mm_token_id],
                                 device=input_ids.device,
                                 dtype=input_ids.dtype)
                    if mm_embeds else None
                ),
            )
            input_ids = fuse_result[0]
            inputs_embeds = fuse_result[1]

            return self.llm.forward(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_context_logits=return_context_logits,
                **{k: v for k, v in kwargs.items()
                   if k not in ("multimodal_params", "text_token_indices", "mm_token_indices")},
            )

        def load_weights(self, weights: Dict) -> None:
            """Load weights from the NVFP4 safetensors checkpoint.

            LLM weights use TRT-LLM's native load path (supports NVFP4 Linear).
            Vision + projector weights are dequantized to FP16.
            Also loads packing_position_embedding from the BF16 checkpoint
            (absent in NVFP4) so positional encoding matches training.
            """
            # Load vision encoder (dequantize NVFP4 → FP16)
            self.vision_encoder.load_weights_from_dict(weights)
            self.projector.load_weights_from_dict(weights)

            # Load packing_position_embedding from BF16 model if available.
            # The NVFP4 checkpoint lacks this weight (stripped during quantization).
            import os as _os
            _nvfp4_dir = _os.path.dirname(
                self.model_config.pretrained_config._name_or_path
                if hasattr(self.model_config.pretrained_config, '_name_or_path')
                else ""
            )
            # Try sibling directory: paddleocr_vl_1_5 → paddleocr_vl_1_5_nvfp4
            _base_dir = _os.path.dirname(_os.path.abspath(_nvfp4_dir or __file__))
            _sibling_dirs = [
                _os.path.join(_base_dir, "models", "vlm", "paddleocr_vl_1_5"),
                _os.path.join(_os.path.dirname(_base_dir), "models", "vlm", "paddleocr_vl_1_5"),
                "/home/nataell/code/epAiland/paddle-highspeed-cpp/models/vlm/paddleocr_vl_1_5",
            ]
            _pack_emb_loaded = False
            for _bf16_dir in _sibling_dirs:
                _bf16_model = _os.path.join(_bf16_dir, "model.safetensors")
                if _os.path.exists(_bf16_model):
                    try:
                        from safetensors import safe_open as _safe_open
                        with _safe_open(_bf16_model, framework="pt") as _f:
                            _ppack_key = "visual.vision_model.embeddings.packing_position_embedding.weight"
                            if _ppack_key in _f.keys():
                                _ppack_w = _f.get_tensor(_ppack_key).to(torch.float16)
                                if _ppack_w.shape[0] != self.vision_encoder.embeddings.packing_position_embedding.num_embeddings:
                                    self.vision_encoder.embeddings.packing_position_embedding = torch.nn.Embedding(
                                        _ppack_w.shape[0], _ppack_w.shape[1]).to(torch.float16)
                                self.vision_encoder.embeddings.packing_position_embedding.weight.data.copy_(_ppack_w)
                                self.vision_encoder.embeddings._packing_pos_emb_loaded = True
                                _pack_emb_loaded = True
                                print(f"[PaddleOCRVL] Loaded packing_position_embedding from: {_bf16_dir}", flush=True)
                                break
                    except Exception as _e:
                        print(f"[PaddleOCRVL] Error loading packing pos emb from {_bf16_dir}: {_e}", flush=True)
            if not _pack_emb_loaded:
                print("[PaddleOCRVL] WARNING: packing_position_embedding not found, using interpolated fallback", flush=True)

            # Load LLM via TRT-LLM native path.
            # LlamaForCausalLM.named_modules() returns paths like:
            #   "model.embed_tokens", "model.layers.0.self_attn.qkv_proj", "lm_head"
            # Our checkpoint has:
            #   "model.language_model.embed_tokens.weight"
            #   "model.language_model.layers.0.self_attn.q_proj.weight"
            #   "lm_head.weight"
            # Map: strip "model.language_model." → "model."
            llm_weights = {}
            src_prefix = "model.language_model."
            for k, v in weights.items():
                if k.startswith(src_prefix):
                    # Re-add "model." prefix so filter_weights can find them
                    new_key = "model." + k[len(src_prefix):]
                    llm_weights[new_key] = v
                elif k == "lm_head.weight":
                    llm_weights[k] = v

            _load_weights_impl(self.llm, llm_weights)

else:
    # Stub for offline inspection / testing outside TRT-LLM container
    class PaddleOCRVLForConditionalGeneration(nn.Module):
        """Stub: install tensorrt_llm to use."""
        def __init__(self, *a, **kw):
            raise RuntimeError("tensorrt_llm not available; run inside trtllm-serve container")
