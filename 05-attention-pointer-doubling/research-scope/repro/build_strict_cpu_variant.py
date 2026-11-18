"""Build a strict CPU-determinism version of the original LOCKED V2 model.

The model, data, hyperparameters, random seed and frozen test stay identical.
Only the CPU kernel / thread settings change, in a separate generated file.
This is a numerical reproducibility diagnostic on an already-consumed test,
NOT a new independent holdout or a new model comparison.
"""
from pathlib import Path
src=Path(__file__).with_name("train_composition_protocol.py").read_text()
needle="random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)"
assert src.count(needle)==1
src=src.replace(needle,"""random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled=False
torch.use_deterministic_algorithms(True)
print("STRICT_KERNEL_SETTINGS",dict(threads=torch.get_num_threads(),
    mkldnn=torch.backends.mkldnn.enabled,deterministic=torch.are_deterministic_algorithms_enabled()),flush=True)""")
out=Path(__file__).with_name("train_composition_strict_generated.py")
out.write_text(src)
print("GENERATED_STRICT_CPU_MODEL",len(src))
