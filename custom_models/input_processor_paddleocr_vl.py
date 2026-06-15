# TRT-LLM multimodal input processor for PaddleOCR-VL-1.5.
#
# Registered via register_input_processor applied to PaddleOCRVLForConditionalGeneration
# (see bottom of modeling_paddleocr_vl_trt.py where the decorator is applied).
#
# Pipeline:
#   1. Parse images from inputs["multi_modal_data"]["image"]
#   2. Run PaddleOCRVLImageProcessor._preprocess() per image to get patches + grid_thw
#   3. Expand <|IMAGE_PLACEHOLDER|> tokens in the prompt to N repetitions
#   4. Tokenize the expanded prompt
#   5. Return (token_ids, {"multimodal_data": {"pixel_values": tensor, "image_grid_thw": tensor}})
#
# The model forward() splices pixel_values into the LLM embedding at image_token_id=100295
# positions, so no separate vision encoder call is needed here.

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Minimal import shim — only available inside trtllm-serve container
# ---------------------------------------------------------------------------
try:
    from tensorrt_llm.inputs.registry import (
        BaseMultimodalInputProcessor,
        ExtraProcessedInputs,
        MultimodalPlaceholderMetadata,
        MultimodalPlaceholderPlacement,
        register_input_processor,
    )
    from tensorrt_llm.inputs.data import TextPrompt
    from tensorrt_llm.sampling_params import SamplingParams
    _TRTLLM_AVAILABLE = True
except ImportError:
    _TRTLLM_AVAILABLE = False

# Add model dir to path for the image processor
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "vlm", "paddleocr_vl_1_5_nvfp4",
)
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

try:
    from image_processing_paddleocr_vl import PaddleOCRVLImageProcessor
    _IMG_PROC_AVAILABLE = True
except ImportError as _e:
    import warnings
    warnings.warn(f"Could not import PaddleOCRVLImageProcessor: {_e}")
    _IMG_PROC_AVAILABLE = False


if _TRTLLM_AVAILABLE:

    class PaddleOCRVLInputProcessor(BaseMultimodalInputProcessor):
        """Multimodal input processor for PaddleOCR-VL-1.5.

        Converts raw PIL images to pixel_values tensors and expands
        <|IMAGE_PLACEHOLDER|> tokens in the prompt to the correct number
        of image tokens before tokenization.
        """

        # The placeholder token used in prompts for images
        IMAGE_PLACEHOLDER = "<|IMAGE_PLACEHOLDER|>"
        IMAGE_TOKEN_ID = 100295  # fixed by model vocab

        def __init__(
            self,
            model_path: str,
            config,
            tokenizer,
            trust_remote_code: bool = True,
            **kwargs,
        ):
            super().__init__(
                model_path=model_path,
                config=config,
                tokenizer=tokenizer,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            self._model_path = model_path
            self._config = config
            self._tokenizer = tokenizer
            self._dtype = torch.float16

            # Build image processor from config attrs or defaults
            vision_cfg = {}
            if hasattr(config, "vision_config"):
                vc = config.vision_config
                vision_cfg = vc if isinstance(vc, dict) else vc.__dict__

            patch_size = vision_cfg.get("patch_size", 14)
            merge_size = vision_cfg.get("spatial_merge_size", 2)

            if _IMG_PROC_AVAILABLE:
                self._image_processor = PaddleOCRVLImageProcessor(
                    patch_size=patch_size,
                    merge_size=merge_size,
                    temporal_patch_size=1,
                )
            else:
                self._image_processor = None

            # tllm OOV sentinel for image tokens (>= vocab_size so fuse_input_embeds detects them).
            # Use text_config.vocab_size (103424) not top-level vocab_size (which defaults to 32000).
            _text_vc = None
            if hasattr(config, "text_config"):
                _tc = config.text_config
                _text_vc = (
                    _tc.get("vocab_size") if isinstance(_tc, dict) else getattr(_tc, "vocab_size", None)
                )
            if _text_vc is None:
                _text_vc = getattr(config, "vocab_size", 32000)
                if _text_vc == 32000:
                    _text_vc = 103424  # known correct value for this model
            self.tllm_multimodal_token_id = _text_vc + 1  # = 103425

            # Processor stub (required by BaseMultimodalInputProcessor abstract prop)
            self._processor = self

        # ------------------------------------------------------------------
        # BaseMultimodalInputProcessor abstract properties
        # ------------------------------------------------------------------

        @property
        def config(self):
            return self._config

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def model_path(self) -> str:
            return self._model_path

        @property
        def processor(self):
            # We don't use AutoProcessor; return self as a compatible stub
            return self._processor

        @property
        def dtype(self) -> torch.dtype:
            return self._dtype

        def get_vocab_size(self) -> int:
            """Return actual text_config vocab_size, not the default 32000."""
            return self.tllm_multimodal_token_id - 1  # = 103424

        # ------------------------------------------------------------------
        # Token counting helpers (required by BaseMultimodalInputProcessor)
        # ------------------------------------------------------------------

        def get_num_tokens_per_image(self, *, image: Image.Image, **kwargs) -> int:
            """Compute how many LLM tokens one image produces after merging."""
            if self._image_processor is None:
                # Fallback: dummy square image at default resolution
                return 256
            vision_cfg = {}
            if hasattr(self._config, "vision_config"):
                vc = self._config.vision_config
                vision_cfg = vc if isinstance(vc, dict) else vc.__dict__

            patch_size = vision_cfg.get("patch_size", 14)
            merge_size = vision_cfg.get("spatial_merge_size", 2)
            min_pixels = self._image_processor.min_pixels
            max_pixels = self._image_processor.max_pixels

            from image_processing_paddleocr_vl import smart_resize
            h, w = image.height, image.width
            try:
                rh, rw = smart_resize(
                    h, w,
                    factor=patch_size * merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            except Exception:
                rh = rw = 448  # safe default
            grid_t, grid_h, grid_w = 1, rh // patch_size, rw // patch_size
            return (grid_t * grid_h * grid_w) // (merge_size * merge_size)

        # ------------------------------------------------------------------
        # Core preprocessing logic
        # ------------------------------------------------------------------

        def _preprocess_images(self, images: List[Image.Image]):
            """Run PaddleOCRVLImageProcessor on a list of PIL images.

            Returns:
                pixel_values: float16 tensor [total_patches, C, P, P]
                image_grid_thw: long tensor [N_images, 3]  (t, h, w)
            """
            all_patches = []
            all_grid_thws = []

            for img in images:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img

                if self._image_processor is not None:
                    patches, (grid_t, grid_h, grid_w) = (
                        self._image_processor._preprocess(
                            img,
                            do_resize=True,
                            resample=self._image_processor.resample,
                            do_rescale=True,
                            rescale_factor=self._image_processor.rescale_factor,
                            do_normalize=True,
                            image_mean=self._image_processor.image_mean,
                            image_std=self._image_processor.image_std,
                            do_convert_rgb=True,
                        )
                    )
                    # patches: [N_patches, C, P, P] numpy
                    all_patches.append(patches)
                    all_grid_thws.append((grid_t, grid_h, grid_w))
                else:
                    # Fallback: plain resize + normalize
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.Resize((448, 448)),
                        T.ToTensor(),
                        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                    [0.26862954, 0.26130258, 0.27577711]),
                    ])
                    pixel_values = transform(img.convert("RGB"))  # [C, H, W]
                    all_patches.append(pixel_values.unsqueeze(0).numpy())
                    all_grid_thws.append((1, 32, 32))

            if not all_patches:
                return None, None

            pixel_values = torch.from_numpy(
                np.concatenate(all_patches, axis=0)
            ).to(torch.float16)
            image_grid_thw = torch.tensor(all_grid_thws, dtype=torch.long)
            return pixel_values, image_grid_thw

        def _expand_image_placeholders(
            self, text: str, grid_thws: List[Tuple[int, int, int]]
        ) -> str:
            """Replace each IMAGE_PLACEHOLDER with N repetitions of the token.

            N = grid_t * grid_h * grid_w / merge_size^2
            (same formula as PaddleOCRVLProcessor.__call__)
            """
            vision_cfg = {}
            if hasattr(self._config, "vision_config"):
                vc = self._config.vision_config
                vision_cfg = vc if isinstance(vc, dict) else vc.__dict__
            merge_size = vision_cfg.get("spatial_merge_size", 2)

            idx = 0
            while self.IMAGE_PLACEHOLDER in text and idx < len(grid_thws):
                grid_t, grid_h, grid_w = grid_thws[idx]
                n_tokens = grid_t * grid_h * grid_w // (merge_size * merge_size)
                text = text.replace(
                    self.IMAGE_PLACEHOLDER,
                    self.IMAGE_PLACEHOLDER * n_tokens,
                    1,
                )
                idx += 1
            return text

        @torch.inference_mode()
        def __call__(
            self,
            inputs: "TextPrompt",
            sampling_params: "SamplingParams",
        ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
            text_prompt = inputs.get("prompt", "")
            mm_data = inputs.get("multi_modal_data", {})
            images = mm_data.get("image", [])

            if not isinstance(images, (list, tuple)):
                images = [images]

            # Filter to actual image objects (skip None / empty)
            images = [img for img in images if img is not None]

            multimodal_data: Dict[str, Any] = {}

            if images:
                pixel_values, image_grid_thw = self._preprocess_images(images)
                if pixel_values is not None:
                    multimodal_data["pixel_values"] = pixel_values
                    multimodal_data["image_grid_thw"] = image_grid_thw

                    # Expand placeholders in prompt text
                    grid_thws = [tuple(g.tolist()) for g in image_grid_thw]
                    text_prompt = self._expand_image_placeholders(text_prompt, grid_thws)

            # Tokenize the (possibly expanded) prompt
            if self._tokenizer is not None:
                try:
                    token_ids = self._tokenizer.encode(
                        text_prompt,
                        add_special_tokens=False,
                    )
                except Exception:
                    token_ids = self._tokenizer.encode(text_prompt)
            else:
                token_ids = []

            if not multimodal_data:
                return token_ids, None

            # Replace in-vocab image token IDs with out-of-vocab sentinel so TRT-LLM's
            # fuse_input_embeds() recognizes them as multimodal slots (filters on >= vocab_size).
            # The OOV sentinel is vocab_size + 1 (same pattern as Qwen2VL's tllm_multimodal_token_id).
            token_ids = [self.tllm_multimodal_token_id
                         if t == self.IMAGE_TOKEN_ID else t
                         for t in token_ids]

            return token_ids, {"multimodal_data": multimodal_data}

else:
    # Stub for offline inspection
    class PaddleOCRVLInputProcessor:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise RuntimeError("tensorrt_llm not available")
