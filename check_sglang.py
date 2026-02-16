import sglang
print(f"SGLang version: {sglang.__version__}")
try:
    import sglang.srt.patch_torch
    print("sglang.srt.patch_torch found")
except ImportError as e:
    print(f"sglang.srt.patch_torch not found: {e}")

try:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions
    print("monkey_patch_torch_reductions found")
except ImportError as e:
    print(f"monkey_patch_torch_reductions not found: {e}")
