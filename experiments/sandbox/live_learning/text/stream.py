"""The general learning stream for the live-learning sandbox (sandbox, no prereg).

Deterministic given a seed. Bands: personal facts, world facts, procedures, one
update, one unlearn, plus capability probes. Every probe carries the token that must
(or must not) appear in the answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: str
    band: str  # personal | world | procedure
    teach: str
    probe: str
    answer: str  # must appear (case-insensitive) in the response


@dataclass(frozen=True)
class Event:
    kind: str  # teach | probe | filler | update | unlearn
    text: str = ""
    fact: Fact | None = None
    expect: str | None = None  # probe: token that must appear
    forbid: str | None = None  # probe: token that must NOT appear
    note: str = ""


FACTS = [
    Fact("p1", "personal", "Merhaba, benim adım Arel.", "Benim adım ne?", "Arel"),
    Fact("p2", "personal", "Köpeğimin adı Zeytin.", "Köpeğimin adı ne?", "Zeytin"),
    Fact(
        "p3",
        "personal",
        "Kahvemi sütsüz içerim.",
        "Kahvemi nasıl içerim?",
        "sütsüz",
    ),
    Fact("p4", "personal", "İzmir'de yaşıyorum.", "Nerede yaşıyorum?", "İzmir"),
    Fact(
        "p5",
        "personal",
        "Bir deprem simülasyonu projesinde çalışıyorum.",
        "Hangi projede çalışıyorum?",
        "deprem",
    ),
    Fact("p6", "personal", "Gitar çalıyorum.", "Hangi enstrümanı çalıyorum?", "gitar"),
    Fact(
        "w1",
        "world",
        "Zeta-9 uydusu 2031'de fırlatıldı.",
        "Zeta-9 ne zaman fırlatıldı?",
        "2031",
    ),
    Fact(
        "w2",
        "world",
        "Marnix deresi yılın iki ayında kurur.",
        "Marnix deresi yılın kaç ayında kurur?",
        "iki",
    ),
    Fact(
        "w3",
        "world",
        "Kavra protokolü üç anahtarla açılır.",
        "Kavra protokolü kaç anahtarla açılır?",
        "üç",
    ),
    Fact("w4", "world", "Tilos madeni 22 gramdır.", "Tilos madeni kaç gramdır?", "22"),
    Fact(
        "w5",
        "world",
        "Berta kütüphanesi salı günleri kapalıdır.",
        "Berta kütüphanesi hangi gün kapalıdır?",
        "salı",
    ),
    Fact(
        "w6",
        "world",
        "Runa bölgesinde yedi köy vardır.",
        "Runa bölgesinde kaç köy vardır?",
        "yedi",
    ),
    Fact(
        "c1",
        "procedure",
        "Kırmızı dosyayı açmak için 4711 kodunu gir.",
        "Kırmızı dosya hangi kodla açılır?",
        "4711",
    ),
    Fact(
        "c2",
        "procedure",
        "Siparişi iptal etmek için önce 'kilit' komutunu çalıştır.",
        "Siparişi iptal etmeden önce hangi komutu çalıştırmalıyım?",
        "kilit",
    ),
    Fact(
        "c3",
        "procedure",
        "Motoru durdurmak için kolu saat yönünde çevir.",
        "Motor nasıl durdurulur?",
        "saat yönünde",
    ),
    Fact(
        "c4",
        "procedure",
        "Yedek almak için mavi düğmeye iki kez bas.",
        "Yedek almak için hangi düğmeye basılır?",
        "mavi",
    ),
    Fact(
        "c5",
        "procedure",
        "Acil çıkış kapısı üçüncü kattadır.",
        "Acil çıkış hangi katta?",
        "üçüncü",
    ),
    Fact(
        "c6",
        "procedure",
        "Kalibrasyon için sensörü 30 saniye beklet.",
        "Sensör kalibrasyon için kaç saniye bekletilir?",
        "30",
    ),
]

FILLERS = [
    "Bugün hava nasıl görünüyor?",
    "Bana kısa bir motivasyon cümlesi yaz.",
    "İki artı üç kaç eder?",
    "Deniz kenarında yürümeyi sever misin?",
    "Kısa bir bilmece sor.",
    "En sevdiğin renk hangisi?",
    "Bana bir tavsiye ver.",
    "Bugün gündemde ne var?",
    "Kısa bir şiir yaz.",
    "Bir kitap öner.",
    "Sabah rutini nasıl olmalı?",
    "Kısa bir espri yap.",
]

UPDATES = [
    ("p3", "Bu arada, kahvemi artık sütlü içiyorum.", "Kahvemi nasıl içerim?", "sütlü"),
]

CAPABILITY = [
    ("cap1", "Türkiye'nin başkenti neresi?", "Ankara"),
    ("cap2", "Beş kere altı kaç eder?", "30"),
    ("cap3", "Suyun kimyasal formülü nedir?", "H2O"),
    ("cap4", "Bir yılda kaç ay vardır?", "12"),
]

UNLEARN = ("p2", "Köpeğimin adını unut lütfen.", "Köpeğimin adı ne?", "Zeytin")


def build_events(seed: int = 0):
    """(session1, session2). Session 2 is only probes: the cross-session test."""
    rng = random.Random(seed)
    s1: list[Event] = []
    fillers = FILLERS[:]
    rng.shuffle(fillers)
    fi = 0

    def filler() -> Event:
        nonlocal fi
        text = fillers[fi % len(fillers)]
        fi += 1
        return Event("filler", text=text, note="s1")

    for i, fact in enumerate(FACTS):
        s1.append(Event("teach", text=fact.teach, fact=fact, note="s1"))
        s1.append(
            Event("probe", text=fact.probe, expect=fact.answer, fact=fact, note="s1-imm")
        )
        if i % 2 == 1:
            s1.append(filler())

    # delayed probes: eight fillers between the last teach and the sample
    for _ in range(8):
        s1.append(filler())
    delayed = rng.sample(FACTS, 6)
    for fact in delayed:
        s1.append(
            Event("probe", text=fact.probe, expect=fact.answer, fact=fact, note="s1-del")
        )

    # update: a corrected fact must win
    for fid, teach, probe, expect in UPDATES:
        s1.append(Event("update", text=teach, fact=next(f for f in FACTS if f.id == fid), note="s1"))
        for _ in range(2):
            s1.append(filler())
        s1.append(Event("probe", text=probe, expect=expect, note="s1-upd"))
    for fact in delayed[:2]:
        s1.append(
            Event("probe", text=fact.probe, expect=fact.answer, fact=fact, note="s1-intf")
        )

    # unlearn: the token must vanish, neighbours and capability must survive
    fid, turn, probe, forbidden = UNLEARN
    s1.append(Event("unlearn", text=turn, fact=next(f for f in FACTS if f.id == fid), note="s1"))
    s1.append(
        Event("probe", text=probe, forbid=forbidden, fact=None, note="s1-unlearn")
    )
    s1.append(
        Event("probe", text="Benim adım ne?", expect="Arel", note="s1-unlearn-neigh")
    )
    for _cid, q, a in CAPABILITY:
        s1.append(Event("probe", text=q, expect=a, note="s1-cap"))

    s2: list[Event] = [filler()]
    for fact in FACTS:
        if fact.id == fid:
            continue  # unlearned
        s2.append(
            Event("probe", text=fact.probe, expect=fact.answer, fact=fact, note="s2")
        )
        s2.append(filler())
    for _cid, q, a in CAPABILITY:
        s2.append(Event("probe", text=q, expect=a, note="s2-cap"))
    return s1, s2
