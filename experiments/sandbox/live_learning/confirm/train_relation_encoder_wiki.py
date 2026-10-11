"""SS77: relation encoders trained on a broad relation inventory (Wikidata properties).

1. The frozen base model writes four templates ("X" = entity) per item-valued Wikidata
   property; cached in results/live_learning/confirm/wikidata/templates.json.
2. Triplets: two templates of one property (anchor, positive) and the most similar
   template of another property under the untrained encoder (hard negative).
3. Encoder A: Wikidata triplets without any of the 34 CounterFact relations.
   Encoder B: the SS76 triplets plus Wikidata triplets without the 11 held-out relations.

    .venv/bin/python experiments/sandbox/live_learning/confirm/train_relation_encoder_wiki.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from router_v4_calibrate import _templates_by_relation, build_dev  # noqa: E402
from train_relation_encoder import BASE, build_triplets  # noqa: E402
from train_relation_encoder import SEED as SEED76  # noqa: E402

SEED = 77
CONFIRM = Path("results/live_learning/confirm")
WIKI = CONFIRM / "wikidata"
N_TEMPLATES = 4
SYSTEM = "You write prompts."
USER = (
    'Wikidata property: "{label}" ({desc}).\n'
    "Write four different incomplete sentences or questions about an entity named X that "
    'ask for the value of this property. Use exactly "X" for the entity. Write one per '
    "line, without numbering."
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_X = re.compile(r"\bX\b")


def load_properties():
    raw = json.loads((WIKI / "properties_raw.json").read_text())["results"]["bindings"]
    props = {}
    for b in raw:
        pid = b["p"]["value"].rsplit("/", 1)[-1]
        props.setdefault(
            pid,
            {"label": b["label"]["value"], "desc": b.get("desc", {}).get("value", "")},
        )
    return dict(sorted(props.items()))


def parse(text):
    out = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip().strip('"').strip()
        if line and _X.search(line) and line not in out:
            out.append(line)
        if len(out) == N_TEMPLATES:
            break
    return out


def generate_templates(props, batch=48):
    path = WIKI / "templates.json"
    if path.exists():
        return json.loads(path.read_text())
    import torch
    from routed_eval import MANIFEST_PATH
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = json.loads(MANIFEST_PATH.read_bytes())["models"]["base"]["id"]
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    ids = list(props)
    out = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(**props[p])},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id
            )
        texts = tok.batch_decode(
            gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for p, t in zip(chunk, texts):
            out[p] = {**props[p], "raw": t, "templates": parse(t)}
        print(f"templates {min(start + batch, len(ids))}/{len(ids)}", flush=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    del model
    torch.cuda.empty_cache()
    return out


def wiki_triplets(templates, excluded, encoder, rng):
    keep = {
        p: v["templates"]
        for p, v in templates.items()
        if p not in excluded and len(v["templates"]) >= 2
    }
    flat = [(p, t) for p, ts in keep.items() for t in ts]
    emb = encoder.encode(
        [t for _, t in flat], batch_size=256, normalize_embeddings=True
    )
    import numpy as np

    sims = emb @ emb.T
    triplets = []
    index = {}
    for i, (p, _) in enumerate(flat):
        index.setdefault(p, []).append(i)
    for idx in index.values():
        for k, i in enumerate(idx):
            j = idx[(k + 1) % len(idx)]
            row = sims[i].copy()
            row[idx] = -np.inf
            neg = int(np.argmax(row))
            triplets.append((flat[i][1], flat[j][1], flat[neg][1]))
    rng.shuffle(triplets)
    return triplets, len(keep)


def train(triplets, out_dir, seed, batch=32, lr=2e-5):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    model = SentenceTransformer(BASE, device="cuda")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = (len(triplets) + batch - 1) // batch
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then linear decay
        opt,
        lambda s: (
            (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
        ),
    )
    losses, t0 = [], time.time()
    for step in range(steps):
        chunk = triplets[step * batch : (step + 1) * batch]
        feats = model.tokenize([t for trip in chunk for t in trip])
        feats = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in feats.items()
        }
        emb = torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)
        a, p, n = emb[0::3], emb[1::3], emb[2::3]
        scores = 20.0 * a @ torch.cat([p, n]).T
        loss = torch.nn.functional.cross_entropy(
            scores, torch.arange(len(chunk), device=scores.device)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if (step + 1) % 50 == 0:
            print(
                f"{out_dir.name} step {step + 1}/{steps} loss {sum(losses[-50:]) / 50:.4f}",
                flush=True,
            )
    model.eval()
    model.save(str(out_dir))
    return {
        "steps": steps,
        "loss_first50": sum(losses[:50]) / min(50, len(losses)),
        "loss_last50": sum(losses[-50:]) / min(50, len(losses)),
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    out_a, out_b = (
        CONFIRM / "relation_encoder_wiki_A",
        CONFIRM / "relation_encoder_wiki_B",
    )
    for d in (out_a, out_b):
        if d.exists():
            sys.exit(f"refusing to overwrite {d}")
    from sentence_transformers import SentenceTransformer

    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    cf_relations = set(_templates_by_relation(raw_cf))
    heldout = set(
        json.loads((CONFIRM / "relation_encoder_mpnet/ss76_meta.json").read_text())[
            "heldout_relations"
        ]
    )

    templates = generate_templates(load_properties())
    n_t = [len(v["templates"]) for v in templates.values()]
    print(
        f"properties {len(templates)}, mean templates {sum(n_t) / len(n_t):.2f}, "
        f"with >= 2: {sum(k >= 2 for k in n_t)}",
        flush=True,
    )

    base = SentenceTransformer(BASE, device="cuda")
    trip_a, n_a = wiki_triplets(templates, cf_relations, base, random.Random(SEED))
    trip_wb, n_b = wiki_triplets(templates, heldout, base, random.Random(SEED + 1))
    del base

    tests = [
        json.loads((CONFIRM / f).read_text())["facts"]
        for f in ("external_counterfact_n1000.json", "external_zsre_n1000.json")
    ]
    test_subjects = {f["subject"] for fs in tests for f in fs}
    dev = build_dev(raw_cf, raw_zs, test_subjects)
    trip76, held76, *_ = build_triplets(
        raw_cf,
        raw_zs,
        test_subjects | {f["subject"] for f in dev},
        random.Random(SEED76),
    )
    assert set(held76) == heldout
    trip_b = trip76 + trip_wb
    random.Random(SEED + 2).shuffle(trip_b)

    meta = {
        "prereg": "SS77",
        "base": BASE,
        "n_properties": len(templates),
        "A": {
            "excluded": "all 34 CounterFact relations",
            "n_properties": n_a,
            "n_triplets": len(trip_a),
        },
        "B": {
            "excluded": sorted(heldout),
            "n_wiki_properties": n_b,
            "n_triplets": len(trip_b),
            "n_ss76_triplets": len(trip76),
        },
        "heldout_relations": sorted(heldout),
    }
    meta["A"].update(train(trip_a, out_a, SEED))
    meta["B"].update(train(trip_b, out_b, SEED))
    for d in (out_a, out_b):
        (d / "ss77_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
