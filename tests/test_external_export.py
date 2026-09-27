"""The external-bank exporters' plumbing (experiments/external/), without the official
repositories or timm: data mapping, row order, forced-expert predictions, log parsing.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments" / "external")
)

import common  # noqa: E402
import ease_export  # noqa: E402

from cerata.data import ptm_benchmarks as pb  # noqa: E402


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A 6-class shuffled ImageFolder benchmark, B2 Inc2 (three tasks)."""
    from PIL import Image

    monkeypatch.setitem(
        pb.BENCHMARKS, "tiny", pb.BenchmarkSpec("tiny", 6, "tiny", 2, 2)
    )
    rng = np.random.default_rng(0)
    for split, n in (("train", 3), ("test", 2)):
        for c in range(6):
            d = tmp_path / "tiny" / split / f"c{c}"
            d.mkdir(parents=True)
            for i in range(n):
                img = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                Image.fromarray(img).save(d / f"{i}.png")
    return tmp_path


def test_data_manager_holds_pinned_order_and_split(tiny):
    class DM:  # stands in for the official class; only __new__ is used
        pass

    dm = common.pilot_data_manager(
        DM, lambda train, args: [train], "tiny", tiny, 2, 2, {}
    )
    order = pb.class_order(6, True)
    pos = np.argsort(order)
    assert dm._class_order == order
    assert dm._increments == [2, 2, 2]
    assert dm.use_path and len(dm._train_data) == 18
    orig = np.array([int(Path(p).parent.name[1:]) for p in dm._train_data])
    assert (dm._train_targets == pos[orig]).all()
    assert dm._train_trsf == [True] and dm._test_trsf == [False]


def test_stream_loader_matches_the_runners_label_order(tiny):
    """The dump-order veto: y equals split_tasks' task-concatenated test labels."""
    loader, y = common.stream_test_loader("tiny", tiny, 2, 2, batch_size=4, workers=0)
    _, test = pb.image_datasets("tiny", tiny, None)
    y_orig = torch.tensor(test.targets)
    z = torch.zeros(len(y_orig), 1)
    tasks = pb.split_tasks((z, y_orig), (z, y_orig), pb.class_order(6, True), 2, 2)
    assert torch.equal(y, torch.cat([t["splits"]["test"][1] for t in tasks]))
    assert sum(x.size(0) for x, _ in loader) == len(y)
    assert torch.equal(
        common.task_of_class("tiny", 2, 2), torch.tensor([0, 0, 1, 1, 2, 2])
    )


class StubNet:
    """EaseNet's test interface: features = per-subspace [CLS] features concatenated."""

    def __init__(self, T, U, C=6, D=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.out_dim, self.use_init_ptm = D, bool(U)
        self.fc = SimpleNamespace(
            weight=torch.randn(C, (T + U) * D, generator=g), sigma=torch.ones(1)
        )
        self.T, self.U = T, U

    def eval(self):
        return self

    def __call__(self, x, test=True):
        feats = x.flatten(1)
        return {"logits": feats @ self.fc.weight.T, "features": feats}


@pytest.mark.parametrize("U", [0, 1])
def test_forced_expert_uses_only_its_subspace(U):
    T, D = 3, 4
    net = StubNet(T, U, D=D)
    x = torch.randn(10, (T + U) * D, generator=torch.Generator().manual_seed(1))
    loader = [(x[:6], None), (x[6:], None)]
    bounds = [(0, 2), (2, 4), (4, 6)]
    native, forced, within = ease_export.forced_predictions(
        net, T, bounds, loader, "cpu"
    )
    W = net.fc.weight
    assert torch.equal(native, (x @ W.T).argmax(-1))
    for t in range(T):
        s = slice((t + U) * D, (t + U + 1) * D)
        ref = (F.normalize(x[:, s], dim=1) @ F.normalize(W[:, s], dim=1).T).argmax(-1)
        assert torch.equal(forced[:, t], ref)
        lo, hi = bounds[t]
        assert torch.equal(within[:, t], (x @ W.T)[:, lo:hi].argmax(-1) + lo)


def test_forced_expert_rejects_a_mismatched_classifier():
    net = StubNet(3, 0)
    with pytest.raises(RuntimeError):
        ease_export.forced_predictions(net, 2, [(0, 3), (3, 6)], [], "cpu")


def test_timm_shim_pins_legacy_names():
    calls = []
    fake = SimpleNamespace(
        create_model=lambda tag, pretrained, **kw: calls.append((tag, pretrained))
        or tag,
        __version__="x",
    )
    shim = common.TimmShim(fake)
    shim.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    assert calls[-1] == ("vit_base_patch16_224.augreg_in21k_ft_in1k", True)
    shim.create_model("vit_base_patch16_224_in21k", pretrained=True)
    assert calls[-1] == ("vit_base_patch16_224.augreg_in21k", True)
    with pytest.raises(KeyError):
        shim.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k")
    assert common.TimmShim(fake, random_weights=True).create_model(
        "vit_base_patch16_224", pretrained=True
    )
    assert calls[-1][1] is False
    assert shim.__version__ == "x"


def test_official_final_accuracy_reads_the_last_curve(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "x => CNN top1 curve: [90.0, 80.5]\n"
        "x => CNN top5 curve: [99.0, 98.0]\n"
        "x => CNN top1 curve: [90.0, 80.5, 77.33]\n"
    )
    assert common.official_final_accuracy(log) == pytest.approx(0.7733)
    (tmp_path / "empty.log").write_text("nothing")
    with pytest.raises(ValueError):
        common.official_final_accuracy(tmp_path / "empty.log")


def test_dump_summary_and_roundtrip(tmp_path):
    y = torch.tensor([0, 1, 2, 3])
    toc = torch.tensor([0, 0, 1, 1])
    forced = torch.tensor([[0, 2], [0, 3], [1, 2], [0, 3]])
    dump = common.make_dump(y, toc, forced, torch.tensor([0, 1, 2, 0]), {"m": 1})
    s = common.dump_summary(dump)
    assert s["native_acc"] == 0.75 and s["oracle_acc"] == 0.75
    assert s["own_task_acc"] == [0.5, 1.0]
    path = tmp_path / common.dump_name("ease", "tiny", "in21k", 0)
    dump.save(path)
    assert path.name == "ease__tiny__in21k__seed0.npz"
    assert torch.equal(type(dump).load(path).expert_pred, forced)


def test_every_benchmark_has_an_official_config_name():
    assert set(ease_export.CONFIG_STEM) == set(pb.BENCHMARKS) - {"tiny"}


def test_micro_batches_reproduce_the_full_batch_step():
    """Accumulating over chunks gives the published batch's update (no dropout here)."""

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(5, 3)

        def forward(self, x, test=False):
            return {"logits": self.lin(x)}

    def run(micro):
        torch.manual_seed(0)
        net = Net()
        learner = SimpleNamespace(
            moni_adam=False,
            _cur_task=1,
            init_cls=3,
            inc=3,
            _known_classes=3,
            args={"init_epochs": 2, "later_epochs": 2},
            _network=net,
            _device="cpu",
        )
        g = torch.Generator().manual_seed(1)
        batches = [
            (
                None,
                torch.randn(7, 5, generator=g),
                torch.randint(3, 6, (7,), generator=g),
            )
            for _ in range(3)
        ]
        opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        ease_export.accumulating_init_train(learner, micro)(batches, None, opt, None)
        return torch.cat([p.detach().flatten() for p in net.parameters()])

    assert torch.allclose(run(7), run(2), atol=1e-6)
    torch.manual_seed(0)
    init = torch.cat([p.detach().flatten() for p in Net().parameters()])
    assert not torch.allclose(run(7), init)
