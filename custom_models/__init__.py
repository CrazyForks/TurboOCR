# Register PaddleOCRVLForConditionalGeneration with TRT-LLM PyTorch backend.
from . import modeling_paddleocr_vl_trt

# Register paddleocr_vl in PLACEHOLDER_EXCEPTIONS so TRT-LLM's
# apply_chat_template passes the content as a structured list (not a flat string)
# to the chat template, allowing the template's image-block rendering to work.
# Also patch handle_placeholder_exceptions to reconstruct the list format.
try:
    import tensorrt_llm.inputs.utils as _tllm_utils
    if "paddleocr_vl" not in _tllm_utils.PLACEHOLDER_EXCEPTIONS:
        _tllm_utils.PLACEHOLDER_EXCEPTIONS.append("paddleocr_vl")

    _orig_hpe = _tllm_utils.handle_placeholder_exceptions

    def _patched_hpe(model_type, conversation, mm_placeholder_counts):
        if model_type == "paddleocr_vl":
            # Reconstruct content as list so the PaddleOCR-VL chat template
            # can find {"type": "image"} blocks and insert <|IMAGE_START|>...<|IMAGE_END|>
            for conv, mm_count in zip(conversation, mm_placeholder_counts):
                if not mm_count:
                    continue
                text = conv.get("content", "")
                content = [{"type": "text", "text": text}]
                n_images = mm_count.get("<|IMAGE_PLACEHOLDER|>", 0)
                content = [{"type": "image"}] * n_images + content
                conv["content"] = content
            return conversation
        return _orig_hpe(model_type, conversation, mm_placeholder_counts)

    _tllm_utils.handle_placeholder_exceptions = _patched_hpe
    # Also patch the reference inside inputs/__init__.py if it was already imported
    try:
        import tensorrt_llm.inputs as _tllm_inputs
        _tllm_inputs.handle_placeholder_exceptions = _patched_hpe
    except Exception:
        pass
except Exception as _e:
    import warnings
    warnings.warn(f"Could not patch PLACEHOLDER_EXCEPTIONS for paddleocr_vl: {_e}")

# Also register PaddleOCRVLConfig with transformers AutoConfig so that
# TRT-LLM's config loader (which calls AutoConfig.from_pretrained) can
# find the config class for model_type="paddleocr_vl".
try:
    from transformers import AutoConfig
    import sys, os
    # Add model dir to path so the config file can be imported
    _model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "vlm", "paddleocr_vl_1_5_nvfp4"
    )
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)
    from configuration_paddleocr_vl import PaddleOCRVLConfig
    AutoConfig.register("paddleocr_vl", PaddleOCRVLConfig)
except Exception as _e:
    import warnings
    warnings.warn(f"Could not register PaddleOCRVLConfig with AutoConfig: {_e}")

