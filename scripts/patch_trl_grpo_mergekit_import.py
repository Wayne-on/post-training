from __future__ import annotations

import importlib.util
from pathlib import Path


MERGEKIT_OLD = "from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf"

MERGEKIT_NEW = """\
try:
    from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf
except Exception:
    MergeConfig = object

    def merge_models(*args, **kwargs):
        raise ImportError("mergekit is unavailable; TRL merge callbacks are disabled.")

    def upload_model_to_hf(*args, **kwargs):
        raise ImportError("mergekit is unavailable; TRL merge callbacks are disabled.")
"""

LLM_BLENDER_OLD = """\
if is_llm_blender_available():
    import llm_blender
"""

LLM_BLENDER_NEW = """\
try:
    import llm_blender
except Exception:
    llm_blender = None
"""

PAIRRM_CHECK_OLD = "if not is_llm_blender_available():"
PAIRRM_CHECK_NEW = "if llm_blender is None:"


def patch_file(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"TRL {label} already patched: {path}")
        return False
    if old not in text:
        raise RuntimeError(f"Expected {label} block not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"TRL {label} patched: {path}")
    return True


def main() -> None:
    spec = importlib.util.find_spec("trl")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("trl is not installed")

    root = Path(next(iter(spec.submodule_search_locations)))

    callbacks = root / "trainer" / "callbacks.py"
    patch_file(callbacks, MERGEKIT_OLD, MERGEKIT_NEW, "mergekit import")

    judges = root / "trainer" / "judges.py"
    patch_file(judges, LLM_BLENDER_OLD, LLM_BLENDER_NEW, "llm_blender import")
    text = judges.read_text(encoding="utf-8")
    if PAIRRM_CHECK_NEW not in text:
        if PAIRRM_CHECK_OLD not in text:
            raise RuntimeError(f"Expected PairRM llm_blender check not found in {judges}")
        judges.write_text(text.replace(PAIRRM_CHECK_OLD, PAIRRM_CHECK_NEW), encoding="utf-8")
        print(f"TRL PairRM llm_blender check patched: {judges}")


if __name__ == "__main__":
    main()
