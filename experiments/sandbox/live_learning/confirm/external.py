"""External subsets adapter for the confirmatory study (CPU-only; pinned in the prereg).

Downloads the ROME dumps (certifi CA bundle), adapts to the fact schema, filters by
target token count, samples n with a pinned seed and fabricates unseen-subject
distractors. Schema matches confirm/facts.py.

    python external.py --set counterfact --n 50
    python external.py --set zsre --n 50
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import make_subject  # noqa: E402

URLS = {
    "counterfact": "https://rome.baulab.info/data/dsets/counterfact.json",
    "zsre": "https://rome.baulab.info/data/dsets/zsre_mend_eval.json",
}


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=300, context=ctx) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def adapt_counterfact(item: dict) -> dict | None:
    rw = item.get("requested_rewrite") or {}
    prompt = rw.get("prompt") or ""
    subject = rw.get("subject") or ""
    target = (rw.get("target_new") or {}).get("str") or ""
    paras = item.get("paraphrase_prompts") or []
    if not (prompt and subject and target and paras):
        return None
    probe = prompt.replace("{}", subject).strip()
    para = paras[0].replace("{}", subject).strip()
    if para == probe or subject.lower() not in para.lower():
        return None
    return {
        "subject": subject,
        "relation": str(rw.get("relation_id") or "external"),
        "answer": target,
        "teach": f"{probe} {target}",
        "probe": probe,
        "paraphrase": para,
    }


def adapt_zsre(item: dict) -> dict | None:
    src = (item.get("src") or "").strip()
    answers = item.get("answers") or []
    subject = item.get("subject") or ""
    para = (item.get("rephrase") or item.get("alt") or "").strip()
    if not (src and answers and subject and para):
        return None
    if para == src or subject.lower() not in para.lower():
        return None
    return {
        "subject": subject,
        "relation": "external",
        "answer": answers[0],
        "teach": f"{src} {answers[0]}",
        "probe": src,
        "paraphrase": para,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(URLS), required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed = args.n if args.seed is None else args.seed

    raw = download(URLS[args.set], Path(f"results/live_learning/confirm/raw/{args.set}.json"))
    items = json.loads(raw.read_text())
    print(f"downloaded {raw} ({raw.stat().st_size / 1e6:.1f} MB), {len(items)} items")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    adapt = adapt_counterfact if args.set == "counterfact" else adapt_zsre

    adapted = [a for it in items if (a := adapt(it))]
    print(f"adapted: {len(adapted)}")
    for limit in (1, 2):
        pass_n = sum(
            1 for a in adapted if len(tok.encode(a["answer"], add_special_tokens=False)) <= limit
        )
        print(f"target <= {limit} token(s): {pass_n}")

    pool = [
        a
        for a in adapted
        if len(tok.encode(a["answer"], add_special_tokens=False)) <= args.max_tokens
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)
    facts = pool[: args.n]
    used = {f["subject"] for f in facts}
    unseen = [make_subject(rng, used) for _ in range(5)]
    for i, f in enumerate(facts):
        u = unseen[i % len(unseen)]
        if f["subject"] and f["subject"] in f["probe"]:
            f["distractor"] = f["probe"].replace(f["subject"], u)
        else:
            f["distractor"] = f"Who is {u}?"
        f["id"] = f"x{i:04d}"
    print(f"sampled: {len(facts)} (seed {seed}, max_tokens {args.max_tokens})")
    if facts:
        print("sample:", json.dumps(facts[0], ensure_ascii=False))
    out = Path(args.out or f"results/live_learning/confirm/external_{args.set}_n{args.n}.json")
    out.write_text(
        json.dumps({"set": args.set, "seed": seed, "max_tokens": args.max_tokens,
                    "facts": facts}, ensure_ascii=False, indent=1)
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
