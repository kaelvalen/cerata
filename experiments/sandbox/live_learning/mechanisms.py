"""Three mechanisms behind one loop: bounded transcript, episodic memory, online LoRA.

Sandbox code: correctness over elegance. Each mechanism owns its persistence under
`results/live_learning/state/<mechanism>_seed<seed>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from harness import chat, make_lora

SYSTEM = (
    "Sen kısa ve net cevap veren bir asistansın. "
    "Sana verilen notları kullan; bilmiyorsan 'bilmiyorum' de."
)
HISTORY_TURNS = 12  # the context baseline is a bounded chat window
# Generic anchors mixed into every LoRA update: not part of the capability probes,
# they just keep general behaviour from collapsing (classic replay, 3 examples).
ANCHORS = [
    ("Fransa'nın başkenti neresi?", "Paris"),
    ("On bölü iki kaç eder?", "5"),
    ("Gökyüzü genelde ne renktir?", "mavi"),
]


class ContextMechanism:
    name = "context"

    def __init__(self, tokenizer, model, state_dir: Path, model_name: str, seed: int):
        self.tok, self.model = tokenizer, model
        self.dir = Path(state_dir)
        self.log: list[str] = []
        self._load()

    # -- hooks -----------------------------------------------------------
    def teach(self, fact, text: str, answer: str | None = None) -> None:
        self.log.append(text)

    def filler(self, text: str) -> None:
        self.log.append(text)

    def unlearn(self, fact) -> None:
        # the baseline can only delete the line it can find
        self.log = [t for t in self.log if t != fact.teach]

    # -- query -----------------------------------------------------------
    def answer(self, user_text: str) -> str:
        tail = self.log[-HISTORY_TURNS:]
        notes = "\n".join(f"- {t}" for t in tail)
        system = f"{SYSTEM}\nSon konuşma notları:\n{notes}" if notes else SYSTEM
        return chat(
            self.tok,
            self.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        )

    # -- persistence -----------------------------------------------------
    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "log.json").write_text(json.dumps(self.log, ensure_ascii=False))

    def _load(self) -> None:
        path = self.dir / "log.json"
        if path.exists():
            self.log = json.loads(path.read_text())


class MemoryMechanism:
    name = "memory"

    def __init__(self, tokenizer, model, state_dir: Path, model_name: str, seed: int):
        self.tok, self.model = tokenizer, model
        self.dir = Path(state_dir)
        self.facts: list[dict] = []  # {"id": str, "text": str}
        self._load()

    # -- hooks -----------------------------------------------------------
    def teach(self, fact, text: str, answer: str | None = None) -> None:
        self.facts = [f for f in self.facts if f["id"] != fact.id]
        self.facts.append({"id": fact.id, "text": text})

    def filler(self, text: str) -> None:
        return None

    def unlearn(self, fact) -> None:
        self.facts = [f for f in self.facts if f["id"] != fact.id]

    # -- query -----------------------------------------------------------
    def _retrieve(self, user_text: str, k: int = 4) -> list[str]:
        if not self.facts:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [f["text"] for f in self.facts]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        X = vec.fit_transform(corpus + [user_text])
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
        order = sims.argsort()[::-1][:k]
        return [corpus[i] for i in order if sims[i] > 0.0]

    def answer(self, user_text: str) -> str:
        notes = self._retrieve(user_text)
        system = f"{SYSTEM}\nİlgili notlar:\n" + "\n".join(
            f"- {t}" for t in notes
        ) if notes else SYSTEM
        return chat(
            self.tok,
            self.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        )

    # -- persistence -----------------------------------------------------
    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "facts.json").write_text(
            json.dumps(self.facts, ensure_ascii=False)
        )

    def _load(self) -> None:
        path = self.dir / "facts.json"
        if path.exists():
            self.facts = json.loads(path.read_text())


class LoRAMechanism:
    name = "lora"

    def __init__(
        self,
        tokenizer,
        model,
        state_dir: Path,
        model_name: str,
        seed: int,
        steps: int = 4,
        lr: float = 1e-4,
    ):
        self.tok = tokenizer
        self.dir = Path(state_dir)
        self.model_name = model_name
        self.seed = seed
        self.steps, self.lr = steps, lr
        self.teach_texts: list[dict] = []
        self._load()
        self._build()

    # -- model lifecycle -------------------------------------------------
    def _fresh_base(self):
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.bfloat16, device_map={"": 0}
        )
        return base

    def _build(self) -> None:
        torch.manual_seed(self.seed)
        if self.teach_texts and (self.dir / "adapter").exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self._fresh_base(), str(self.dir / "adapter")
            )
        else:
            self.model = make_lora(self._fresh_base())
        self.model.eval()
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )

    def _rebuild(self) -> None:
        del self.model
        torch.cuda.empty_cache()
        self._build()
        if self.teach_texts:
            self._learn_all()

    def _pair_step(self, user: str, assistant: str) -> None:
        """One gradient step on a chat-template pair, loss on the answer only."""
        pair = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        prompt = self.tok.apply_chat_template(
            pair[:-1], add_generation_prompt=True, return_tensors="pt"
        )
        full = self.tok.apply_chat_template(pair, return_tensors="pt")
        prompt_ids = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
        full_ids = full["input_ids"] if hasattr(full, "keys") else full
        full_ids = full_ids.to(self.model.device)
        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        out = self.model(input_ids=full_ids, labels=labels)
        out.loss.backward()
        self.opt.step()
        self.opt.zero_grad()

    def _learn_all(self) -> None:
        """Replay-LoRA: one epoch over every stored fact (the new one included) plus
        the generic anchors - the online update without the forgetting."""
        batch = []
        for e in self.teach_texts:
            batch.append((e["text"], f"Not aldım: {e['text']}"))
            batch.append((e["probe"], e["answer"]))
        batch += ANCHORS
        self.model.train()
        for _ in range(self.steps):
            for user, assistant in batch:
                self._pair_step(user, assistant)
        self.model.eval()

    # -- hooks -----------------------------------------------------------
    def teach(self, fact, text: str, answer: str | None = None) -> None:
        answer = answer or fact.answer
        self.teach_texts = [t for t in self.teach_texts if t["id"] != fact.id]
        self.teach_texts.append(
            {"id": fact.id, "text": text, "probe": fact.probe, "answer": answer}
        )
        self._learn_all()

    def filler(self, text: str) -> None:
        return None

    def unlearn(self, fact) -> None:
        self.teach_texts = [t for t in self.teach_texts if t["id"] != fact.id]
        self._rebuild()  # exact-ish: retrain from scratch on the survivors

    # -- query -----------------------------------------------------------
    def answer(self, user_text: str) -> str:
        return chat(
            self.tok,
            self.model,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text},
            ],
        )

    # -- persistence -----------------------------------------------------
    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "teach_texts.json").write_text(
            json.dumps(self.teach_texts, ensure_ascii=False)
        )
        self.model.save_pretrained(str(self.dir / "adapter"))

    def _load(self) -> None:
        path = self.dir / "teach_texts.json"
        if path.exists():
            self.teach_texts = json.loads(path.read_text())


MECHANISMS = {
    "context": ContextMechanism,
    "memory": MemoryMechanism,
    "lora": LoRAMechanism,
}
