"""
Download, unpack and verify the PTM-CIL benchmark images (docs/PTM_CIL_PREREG.md).

The processed splits are the ones linked from the RevisitingCIL README
(github.com/zhoudw-zdw/RevisitingCIL). Target layout, which
`cerata.data.ptm_benchmarks` reads:

    data/ptm/cub/{train,test}/            data/ptm/objectnet/{train,test}/
    data/ptm/imagenet-r/{train,test}/     data/ptm/omnibenchmark/{train,test}/
    data/ptm/imagenet-a/{train,test}/     data/ptm/vtab-cil/vtab/{train,test}/

CIFAR-100 needs nothing here (torchvision downloads it).

1. Download. Five archives come from Google Drive through `gdown` (the `ptm` extra).
   ObjectNet is on OneDrive only: download it in a browser from the README's link and
   put the archive in `data/ptm/downloads/` (any name starting with "objectnet"). A
   Drive archive that fails (quota, rate limit) can be placed there by hand the same
   way; an archive already in `downloads/` is never downloaded again.
2. Unpack each archive and move the directory that holds `train/` and `test/` to its
   target, whatever the archive's own top-level layout.
3. Verify: the class count, identical train/test class lists, and the image counts
   (reported; the reference counts are not published with checksums, so a count
   mismatch is a warning to investigate, not an error), and each archive's md5 against
   the sums the maintainers published in RevisitingCIL issue #5 (`PUBLISHED_MD5`,
   copied from the issue; a mismatch fails, an archive with no published sum is
   reported as unchecked). `data/ptm/MANIFEST.json` records the md5, the sha256 and the
   per-split counts of every archive. Keep that file with the results.

    python experiments/prepare_ptm_data.py                  # all six
    python experiments/prepare_ptm_data.py --only cub,vtab
    python experiments/prepare_ptm_data.py --verify_only
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cerata.data.ptm_benchmarks import BENCHMARKS  # noqa: E402

# name -> (Google Drive file id or None, target directory under data/ptm)
SOURCES = {
    "cub": ("1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb", "cub"),
    "imagenet_r": ("1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR", "imagenet-r"),
    "imagenet_a": ("19l52ua_vvTtttgVRziCZJjal0TPE9f2p", "imagenet-a"),
    "omnibenchmark": ("1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u", "omnibenchmark"),
    "vtab": ("1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_", "vtab-cil/vtab"),
    "objectnet": (None, "objectnet"),  # OneDrive only: place the archive by hand
}
ONEDRIVE_OBJECTNET = (
    "https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/"
    "EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy"
)
# Train / test image counts as the PTM-CIL papers tabulate them (APER, EASE). Written
# from the literature without a checksum to back them: a mismatch is reported, and
# the owner decides whether it matters.
REFERENCE_COUNTS = {
    "cub": (9430, 2358),
    "imagenet_r": (24000, 6000),
    "imagenet_a": (5981, 1519),
    "objectnet": (26509, 6628),
    "omnibenchmark": (89697, 5985),
    "vtab": (1796, 8619),
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


# md5 sums published by the maintainers in github.com/zhoudw-zdw/RevisitingCIL/issues/5,
# keyed by archive file name. Copied from the issue by hand; empty until then.
PUBLISHED_MD5: dict[str, str] = {}


def digests(path: Path, chunk: int = 1 << 20) -> tuple[str, str]:
    """(md5, sha256) of a file in one pass."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def find_archive(downloads: Path, name: str) -> Path | None:
    stems = {name, name.replace("_", "-"), SOURCES[name][1].split("/")[0]}
    for p in sorted(downloads.iterdir()) if downloads.exists() else []:
        low = p.name.lower()
        if (
            p.is_file()
            and low.endswith(ARCHIVE_EXT)
            and any(low.startswith(s) for s in stems)
        ):
            return p
    return None


def download(name: str, downloads: Path) -> Path:
    found = find_archive(downloads, name)
    if found is not None:
        print(f"[{name}] using {found}")
        return found
    file_id = SOURCES[name][0]
    if file_id is None:
        raise SystemExit(
            f"[{name}] OneDrive only. Download it in a browser from\n  {ONEDRIVE_OBJECTNET}\n"
            f"and put the archive in {downloads}/ (name starting with 'objectnet')."
        )
    import gdown

    downloads.mkdir(parents=True, exist_ok=True)
    out = gdown.download(
        id=file_id, output=str(downloads) + "/", quiet=False, resume=True
    )
    if not out:
        raise SystemExit(
            f"[{name}] Google Drive refused the download (quota or rate limit). Download "
            f"https://drive.google.com/file/d/{file_id} in a browser into {downloads}/ "
            "and run this again."
        )
    return Path(out)


def split_root(tree: Path) -> Path:
    """The shallowest directory under `tree` that has both train/ and test/."""
    candidates = [p.parent for p in tree.rglob("train") if p.is_dir()]
    candidates = [c for c in candidates if (c / "test").is_dir()]
    if not candidates:
        raise SystemExit(f"no directory with train/ and test/ inside {tree}")
    return min(candidates, key=lambda c: len(c.parts))


SOURCE_MARKER = ".cerata_source_sha256"


def unpack(name: str, archive: Path, root: Path, archive_sha: str) -> Path:
    """Unpack unless the target was unpacked from this exact archive (by sha256)."""
    target = root / SOURCES[name][1]
    marker = target / SOURCE_MARKER
    if (
        (target / "train").is_dir()
        and (target / "test").is_dir()
        and marker.is_file()
        and marker.read_text().strip() == archive_sha
    ):
        print(f"[{name}] already unpacked from this archive at {target}")
        return target
    if target.exists():
        print(f"[{name}] {target} is missing or from another archive: re-unpacking")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        print(f"[{name}] unpacking {archive.name} ...")
        shutil.unpack_archive(str(archive), tmp)
        src = split_root(Path(tmp))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(src), str(target))
    marker.write_text(archive_sha + "\n")
    return target


def count_split(split: Path) -> tuple[list[str], int]:
    classes = sorted(p.name for p in split.iterdir() if p.is_dir())
    n = sum(
        1
        for c in classes
        for f in (split / c).rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXT
    )
    return classes, n


def verify(name: str, target: Path) -> dict:
    train_cls, n_train = count_split(target / "train")
    test_cls, n_test = count_split(target / "test")
    expected_c = BENCHMARKS[name].num_classes
    ref = REFERENCE_COUNTS.get(name)
    rep = {
        "path": str(target),
        "classes": len(train_cls),
        "n_train": n_train,
        "n_test": n_test,
        "class_count_ok": len(train_cls) == expected_c,
        "train_test_classes_match": train_cls == test_cls,
        "reference_counts": ref,
        "counts_match_reference": ref == (n_train, n_test) if ref else None,
    }
    status = (
        "OK" if rep["class_count_ok"] and rep["train_test_classes_match"] else "FAIL"
    )
    note = ""
    if ref and not rep["counts_match_reference"]:
        note = f"  (reference {ref[0]}/{ref[1]}: check)"
    print(
        f"[{name}] {status}: {len(train_cls)} classes (expected {expected_c}), "
        f"train {n_train}, test {n_test}{note}"
    )
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ptm")
    ap.add_argument("--only", default=",".join(SOURCES))
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failed = []
    for name in args.only.split(","):
        if name not in SOURCES:
            raise SystemExit(
                f"unknown benchmark {name!r}; choose from {sorted(SOURCES)}"
            )
        entry = manifest.get(name, {})
        if not args.verify_only:
            archive = download(name, downloads)
            print(f"[{name}] md5 / sha256 of {archive.name} ...")
            # Always recomputed: an archive replaced under the same name is caught.
            archive_md5, archive_sha = digests(archive)
            published = PUBLISHED_MD5.get(archive.name)
            md5_ok = None if published is None else archive_md5 == published
            entry.update(
                archive=archive.name,
                archive_md5=archive_md5,
                archive_sha256=archive_sha,
                published_md5=published,
                md5_matches_published=md5_ok,
            )
            if md5_ok is False:
                print(f"[{name}] md5 MISMATCH: {archive_md5} != published {published}")
                failed.append(name)
                manifest[name] = entry
                continue
            print(
                f"[{name}] md5 {archive_md5} "
                + (
                    "matches the published sum"
                    if md5_ok
                    else "(no published sum to check)"
                )
            )
            unpack(name, archive, root, archive_sha)
        rep = verify(name, root / SOURCES[name][1])
        entry.update(rep, verified_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        manifest[name] = entry
        if not (rep["class_count_ok"] and rep["train_test_classes_match"]):
            failed.append(name)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {manifest_path}")
    if failed:
        raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()
