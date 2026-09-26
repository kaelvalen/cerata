"""
Regenerate the v1 checkpoint fixtures used by tests/test_v3_anchors.py.

The fixtures only prove something if they are written by the **pre-v3** code, so
that the pickle names the old dotted paths (`pal_moe.models.*`,
`pal_moe.memory.prototype_memory`) the v3 alias shims must resolve. Run this file
against a checkout of the stage1-final tree (c43ac9c), not against the current one:

    git worktree add --detach /tmp/stage1 c43ac9c
    PYTHONPATH=/tmp/stage1 python tests/fixtures/make_v1_fixtures.py
    git worktree remove /tmp/stage1

It refuses to run if `pal_moe` resolves to a tree that already has the v3 layout.
"""

from pathlib import Path

import torch

import pal_moe
from pal_moe.memory.prototype_memory import Prototype, PrototypeMemory
from pal_moe.models.encoder import SharedEncoder
from pal_moe.models.expert import MLPExpert
from pal_moe.models.moe import DynamicMoE
from pal_moe.models.router import DynamicRouter
from pal_moe.persistence import save_checkpoint

OUT = Path(__file__).resolve().parent
MADE_BY = "stage1-final (c43ac9c)"

if DynamicMoE.__module__ != "pal_moe.models.moe":
    raise SystemExit(
        f"pal_moe resolves to {Path(pal_moe.__file__).parent} whose DynamicMoE lives in "
        f"{DynamicMoE.__module__}; point PYTHONPATH at the stage1-final tree"
    )

torch.manual_seed(0)
model = DynamicMoE(
    encoder=SharedEncoder(input_dim=16, hidden_dims=(8,), output_dim=4),
    router=DynamicRouter(input_dim=4, num_experts=2, top_k=1),
    experts=[
        MLPExpert(input_dim=4, hidden_dim=8, num_classes=3, expert_id=i)
        for i in range(2)
    ],
).eval()
memory = PrototypeMemory(feature_dim=4, store_raw=False)
memory.prototypes.append(
    Prototype(
        v_p=torch.randn(4),
        r_p=torch.softmax(torch.randn(2), dim=0),
        o_p=torch.randn(2, 3),
        task_id=0,
        owner_expert=0,
    )
)
x = torch.randn(5, 16)
with torch.no_grad():
    y = model(x)
ref = y[0] if isinstance(y, (tuple, list)) else y

torch.save(
    {"model": model, "memory": memory, "x": x, "ref": ref},
    OUT / "v1_whole_objects_stage1.pt",
)
save_checkpoint(
    str(OUT / "v1_checkpoint_stage1.pt"), model, memory, meta={"made_by": MADE_BY}
)
print(f"wrote {OUT}/v1_whole_objects_stage1.pt and v1_checkpoint_stage1.pt")
