"""Unit tests for the PP-FormulaNet-S sidecar LD_LIBRARY_PATH self-heal decision.

`_compute_ld_fix` is the pure core of the cu12-runtime-libs self-heal that keeps
the formula sidecar on the GPU (a cu13 TensorRT LD_LIBRARY_PATH otherwise shadows
onnxruntime-gpu's bundled cu12 libs and ORT silently falls back to CPU). It is
pure (no env, no exec) so we can test the prepend/no-op logic directly. We load
just the sidecar module with its heavy imports (cv2/numpy/PIL) stubbed, so this
test needs only stdlib + pytest.
"""

import importlib.util
import sys
import types
from pathlib import Path


def _load_sidecar():
    for name in ("cv2", "numpy"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        img = types.ModuleType("PIL.Image")
        pil.Image = img
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = img
    path = Path(__file__).resolve().parents[2] / "scripts" / "ppformulanet_s_sidecar.py"
    spec = importlib.util.spec_from_file_location("ppfns_sidecar_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SC = _load_sidecar()
compute = _SC._compute_ld_fix


def test_no_libdirs_is_noop():
    assert compute([], "") is None
    assert compute([], "/some/cu13/lib") is None


def test_prepends_when_empty_current():
    assert compute(["/a", "/b"], "") == "/a:/b"


def test_prepends_ahead_of_existing_cu13_path():
    # The cu12 dirs must come BEFORE the inherited cu13 TensorRT path.
    assert compute(["/a", "/b"], "/trt") == "/a:/b:/trt"


def test_noop_when_all_already_present():
    assert compute(["/a"], "/a:/trt") is None
    assert compute(["/a", "/b"], "/b:/a") is None


def test_prepends_on_partial_overlap():
    # If even one dir is missing, prepend the full set (cheap, idempotent next run).
    assert compute(["/a", "/b"], "/a") == "/a:/b:/a"
