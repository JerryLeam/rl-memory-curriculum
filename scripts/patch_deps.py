"""
Patch third-party dependencies for transformers 5.x compatibility.

trl 0.22.2 uses transformers' _is_package_available which changed its return type
from bool to (bool, version) in transformers 5.x. This makes all availability checks
truthy, causing hard imports of optional packages (vllm, mergekit, etc.) to fire.

Run after `uv sync` to apply patches:
    uv run python scripts/patch_deps.py
"""
import pathlib
import site
import sys


def get_site_packages() -> pathlib.Path:
    """Find the site-packages directory for the current venv."""
    for p in site.getsitepackages():
        pp = pathlib.Path(p)
        if pp.exists() and (pp / "trl").exists():
            return pp
    # Fallback: search sys.path
    for p in sys.path:
        pp = pathlib.Path(p)
        if pp.name == "site-packages" and (pp / "trl").exists():
            return pp
    raise RuntimeError("Cannot find trl in site-packages")


def patch_trl_import_utils(sp: pathlib.Path) -> bool:
    """Wrap _is_package_available to return bool instead of tuple."""
    target = sp / "trl" / "import_utils.py"
    if not target.exists():
        print("  trl/import_utils.py not found, skipping")
        return False

    txt = target.read_text(encoding="utf-8")
    marker = "_orig_is_package_available"
    if marker in txt:
        print("  trl/import_utils.py already patched")
        return False

    old = "from transformers.utils.import_utils import _is_package_available"
    if old not in txt:
        print("  trl/import_utils.py: import line not found, skipping")
        return False

    new = """from transformers.utils.import_utils import _is_package_available as _orig_is_package_available

def _is_package_available(*args, **kwargs):
    result = _orig_is_package_available(*args, **kwargs)
    if "return_version" in kwargs and kwargs["return_version"]:
        return result  # Already a tuple, caller expects it
    if isinstance(result, tuple):
        return result[0]  # transformers 5.x returns (bool, version)
    return result"""

    txt = txt.replace(old, new)
    target.write_text(txt, encoding="utf-8")
    print("  Patched trl/import_utils.py")
    return True


def main():
    print("Patching dependencies for transformers 5.x compatibility...")
    sp = get_site_packages()
    print(f"Site-packages: {sp}")
    patched = patch_trl_import_utils(sp)
    if patched:
        print("Done. Patches applied.")
    else:
        print("Done. No patches needed.")


if __name__ == "__main__":
    main()
