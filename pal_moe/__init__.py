"""`pal_moe` is the former name of `cerata` (renamed 2026-09-26). New code imports
`cerata`; this package exists so that old code, old imports and old pickles still work.

Importing it registers, as the *same* module objects (never copies, never a second
load):

- every `cerata.X` module under `pal_moe.X`;
- the v1 paths the v3 restructure moved into `cerata.legacy` (`pal_moe.models`,
  `pal_moe.adaptation`, ..., `pal_moe.evaluation` -> `cerata.eval`) under those names.

So a checkpoint pickled by the stage1-final code, whose pickle names
`pal_moe.models.moe.DynamicMoE`, unpickles to `cerata.legacy.models.moe.DynamicMoE`
(`tests/test_v3_anchors.py`), and `from pal_moe.api import PalMoE` gives `Cerata`.
"""

import importlib
import pkgutil
import sys
import warnings

import cerata

warnings.warn(
    "`pal_moe` was renamed to `cerata`; import `cerata` instead",
    DeprecationWarning,
    stacklevel=2,
)

V1_PATHS = {
    "models": "cerata.legacy.models",
    "adaptation": "cerata.legacy.adaptation",
    "baselines": "cerata.legacy.baselines",
    "builder": "cerata.legacy.builder",
    "trigger": "cerata.legacy.trigger",
    "evaluation": "cerata.eval",
    "factory": "cerata.legacy.factory",
    "merge": "cerata.legacy.merge",
    "persistence": "cerata.legacy.persistence",
    "memory.prototype_memory": "cerata.legacy.memory.prototype_memory",
    "memory.generative": "cerata.legacy.memory.generative",
}


def _register(old: str, new: str) -> None:
    module = importlib.import_module(new)
    sys.modules[old] = module
    for info in pkgutil.walk_packages(getattr(module, "__path__", []), new + "."):
        try:
            sub = importlib.import_module(info.name)
        except ImportError:  # an optional dependency is missing: skip that module
            continue
        sys.modules[old + info.name[len(new) :]] = sub


for _info in pkgutil.walk_packages(cerata.__path__, "cerata."):
    try:
        sys.modules["pal_moe" + _info.name[len("cerata") :]] = importlib.import_module(
            _info.name
        )
    except ImportError:
        continue
for _old, _new in V1_PATHS.items():
    _register(f"pal_moe.{_old}", _new)


def _mixed_memory_package():
    """Before the rename `pal_moe.memory` held both the v3 FAST memory and the v1
    prototype memory; `cerata.memory` holds only the former. The old package name
    becomes a forwarding module that exposes both, so `import
    pal_moe.memory.prototype_memory as pm` and `from pal_moe.memory import
    PrototypeMemory` still work without adding v1 names to `cerata.memory`."""
    import types

    import cerata.legacy.memory.generative as generative
    import cerata.legacy.memory.prototype_memory as prototype_memory
    import cerata.memory as memory

    mod = types.ModuleType("pal_moe.memory", memory.__doc__)
    mod.__path__ = []  # a package, so its submodule names resolve via sys.modules
    mod.prototype_memory, mod.generative = prototype_memory, generative
    mod.Prototype = prototype_memory.Prototype
    mod.PrototypeMemory = prototype_memory.PrototypeMemory

    def forward(name):
        if name.startswith("__"):  # never borrow cerata.memory's identity (__file__)
            raise AttributeError(name)
        return getattr(memory, name)

    mod.__getattr__ = forward
    sys.modules["pal_moe.memory"] = mod


_mixed_memory_package()

# The v1 top-level names `pal_moe` exported before the rename.
from cerata.eval.metrics import ContinualEvaluator  # noqa: E402
from cerata.legacy.adaptation.ttt import ContinualTrainer, TestTimeAdapter  # noqa: E402
from cerata.legacy.builder.expert_builder import ExpertBuilder  # noqa: E402
from cerata.legacy.memory.prototype_memory import PrototypeMemory  # noqa: E402
from cerata.legacy.models.encoder import EMAEncoder, SharedEncoder  # noqa: E402
from cerata.legacy.models.expert import MLPExpert  # noqa: E402
from cerata.legacy.models.moe import DynamicMoE, PALMoE  # noqa: E402
from cerata.legacy.models.router import DynamicRouter  # noqa: E402
from cerata.legacy.trigger.expert_trigger import QuantitativeTrigger  # noqa: E402

__version__ = cerata.__version__


def __getattr__(name):
    mod = sys.modules.get(f"{__name__}.{name}")
    if mod is not None:
        return mod
    return getattr(cerata, name)


__all__ = [
    "PALMoE",
    "DynamicMoE",
    "SharedEncoder",
    "EMAEncoder",
    "DynamicRouter",
    "MLPExpert",
    "PrototypeMemory",
    "QuantitativeTrigger",
    "ExpertBuilder",
    "ContinualTrainer",
    "TestTimeAdapter",
    "ContinualEvaluator",
]
