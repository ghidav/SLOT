"""
Install SLOT patch into vLLM v1 (0.17.x).

Copies the slot_patch module into vLLM's package directory and patches
model_runner.py to import and apply it.

Usage:
  python install_slot_patch.py         # install
  python install_slot_patch.py --undo  # restore original
"""

import argparse
import os
import shutil

PATCH_IMPORT = """
# ── SLOT patch ──
import os as _os
if int(_os.environ.get("CHOT_STEPS", "0")) > 0:
    from vllm.v1.worker.gpu.slot_patch import apply_slot_patch as _apply_slot
    _SLOT_ENABLED = True
else:
    _SLOT_ENABLED = False
# ── end SLOT patch ──
"""

PATCH_HOOK = """        # ── SLOT: apply delta before sampling ──
        if _SLOT_ENABLED and self.execute_model_state is not None:
            from vllm.v1.worker.gpu.slot_patch import slot_optimize_hidden_states
            self.execute_model_state = slot_optimize_hidden_states(self)
        # ── end SLOT ──
"""


def find_vllm_v1_dir():
    import vllm
    return os.path.join(os.path.dirname(vllm.__file__), "v1", "worker", "gpu")


def install(vllm_dir):
    # 1. Copy slot_patch.py
    src = os.path.join(os.path.dirname(__file__), "vllm", "v1_model_runner_patch.py")
    dst = os.path.join(vllm_dir, "slot_patch.py")
    shutil.copy2(src, dst)
    print(f"Copied slot_patch.py -> {dst}")

    # 2. Patch model_runner.py
    mr_path = os.path.join(vllm_dir, "model_runner.py")
    backup = mr_path + ".orig"

    if not os.path.exists(backup):
        shutil.copy2(mr_path, backup)
        print(f"Backed up original -> {backup}")

    with open(mr_path, "r") as f:
        content = f.read()

    if "SLOT patch" in content:
        print("model_runner.py already patched, skipping.")
        return

    # Insert import after the existing imports
    # Find the last 'import' line before class definitions
    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1
        if line.startswith("class "):
            break

    lines.insert(insert_idx, PATCH_IMPORT)

    # Insert hook at the start of sample_tokens, after unpacking execute_model_state
    patched = "\n".join(lines)
    # Find "def sample_tokens" and inject after the None check
    marker = "self.execute_model_state = None"
    # Find the first occurrence in sample_tokens
    st_idx = patched.find("def sample_tokens")
    if st_idx == -1:
        print("ERROR: Could not find sample_tokens method")
        return
    marker_idx = patched.find(marker, st_idx)
    if marker_idx == -1:
        print("ERROR: Could not find execute_model_state = None in sample_tokens")
        return
    # Insert after that line
    end_of_line = patched.index("\n", marker_idx)
    patched = patched[:end_of_line + 1] + PATCH_HOOK + patched[end_of_line + 1:]

    with open(mr_path, "w") as f:
        f.write(patched)
    print(f"Patched {mr_path}")


def undo(vllm_dir):
    mr_path = os.path.join(vllm_dir, "model_runner.py")
    backup = mr_path + ".orig"
    slot_patch = os.path.join(vllm_dir, "slot_patch.py")

    if os.path.exists(backup):
        shutil.copy2(backup, mr_path)
        print(f"Restored {mr_path} from backup")
    if os.path.exists(slot_patch):
        os.remove(slot_patch)
        print(f"Removed {slot_patch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--undo", action="store_true", help="Restore original model_runner.py")
    args = parser.parse_args()

    vllm_dir = find_vllm_v1_dir()
    print(f"vLLM v1 GPU worker dir: {vllm_dir}")

    if args.undo:
        undo(vllm_dir)
    else:
        install(vllm_dir)


if __name__ == "__main__":
    main()
