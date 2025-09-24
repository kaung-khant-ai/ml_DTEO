"""
scan_videos.py

Purpose
-------
Recursively scan a folder of videos and append a single row per video to a
unified dataset manifest CSV (data/manifests/manifest.csv by default).

Each row contains:
clip_id, source, subset, label, subtype, duration_s, fps, width, height,
sha256, rel_path, notes

Why this exists
---------------
- You will mix multiple datasets (FF++, DFDC, Celeb-DF, Kinetics/UCF, curated
  AI-generated clips). A single manifest keeps everything consistent.
- The SHA-256 hash helps deduplicate and verify files.
- Storing path + basic metadata lets later steps (preprocess/train) just read
  the manifest—no dataset-specific code paths.

Usage (examples)
----------------
# Deepfake clips (AI-generated class)
python scripts/scan_videos.py data/raw/ffpp/manip ffpp ai_generated deepfake_face_swap train

# Fully synthetic (text-to-video) clips
python scripts/scan_videos.py data/external/curated_genai t2v ai_generated t2v_full_synthetic train

# Real human-recorded clips
python scripts/scan_videos.py data/raw/kinetics clips kinetics real recorded_human train
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2  # pip install opencv-python


# Default location for the unified manifest
DEFAULT_MANIFEST = Path("data/manifests/manifest.csv")

# We’ll look for these video extensions by default
DEFAULT_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def ensure_header(manifest_path: Path) -> None:
    """
    Make sure the manifest file exists and has a header row.
    - Creates parent folder(s) if needed.
    - Writes header if file doesn't exist or is empty.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "clip_id",
                    "source",
                    "subset",
                    "label",
                    "subtype",
                    "duration_s",
                    "fps",
                    "width",
                    "height",
                    "sha256",
                    "rel_path",
                    "notes",
                ]
            )


def sha256sum(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute SHA-256 hash of a file in streaming chunks (memory-safe).
    - Used for deduplication and verifying integrity later.
    """
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def video_meta(file_path: Path) -> Tuple[float, int, int, float]:
    """
    Extract basic metadata using OpenCV without reading the whole file:
      - fps (frames per second)
      - width, height (pixels)
      - duration_s (seconds) = frame_count / fps
    Returns (fps, width, height, duration_s). If metadata is missing, returns 0s.
    """
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        # OpenCV failed to open the file (corrupt or unsupported)
        return 0.0, 0, 0, 0.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()

    duration_s = (frame_count / fps) if fps > 0 else 0.0
    return round(fps, 3), width, height, round(duration_s, 3)


def read_existing_hashes(manifest_path: Path) -> set[str]:
    """
    Load existing SHA-256 values from the manifest to avoid duplicate rows.
    If the file doesn't exist yet, returns an empty set.
    """
    if not manifest_path.exists():
        return set()
    hashes: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = (row.get("sha256") or "").strip()
            if h:
                hashes.add(h)
    return hashes


def iter_video_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    """
    Yield all files under 'root' whose extension is in 'exts' (case-insensitive).
    Uses rglob for recursive search.
    """
    exts_lower = tuple(e.lower() for e in exts)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            yield p


def build_rows(
    files: Iterable[Path],
    source: str,
    label: str,
    subtype: str,
    subset: str,
) -> List[List[str]]:
    """
    Build manifest rows for each video file.
    - clip_id: filename stem (no extension). If collisions are possible across sources,
      the (source, sha256) pair still keeps rows unique.
    - rel_path: POSIX-style path (uses forward slashes on Windows too).
    """
    rows: List[List[str]] = []
    for p in files:
        fps, w, h, dur = video_meta(p)
        file_hash = sha256sum(p)
        clip_id = p.stem  # you can change to f"{source}_{p.stem}" if you fear name clashes
        rel_path = p.as_posix()

        rows.append(
            [
                clip_id,
                source,
                subset,
                label,
                subtype,
                f"{dur:.3f}",
                f"{fps:.3f}",
                str(w),
                str(h),
                file_hash,
                rel_path,
                "",  # notes (left blank for now)
            ]
        )
    return rows


def append_rows(manifest_path: Path, rows: List[List[str]], dedupe_hashes: set[str]) -> int:
    """
    Append new rows to the manifest, skipping any whose sha256 already exists.
    Returns the count of rows actually written.
    """
    written = 0
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            if r[9] in dedupe_hashes:  # sha256 column index = 9
                continue
            writer.writerow(r)
            dedupe_hashes.add(r[9])
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Scan a folder of videos and append metadata rows to a unified manifest CSV."
    )
    parser.add_argument("root", help="Folder to scan (recursively)")
    parser.add_argument("source", help="Dataset/source tag (e.g., ffpp, celebdf, dfdc, kinetics, t2v)")
    parser.add_argument("label", choices=["real", "ai_generated"], help="Top-level label")
    parser.add_argument("subtype", help="Subtype, e.g., deepfake_face_swap, t2v_full_synthetic, recorded_human")
    parser.add_argument("subset", choices=["train", "val", "test"], help="Split label to assign")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to manifest CSV (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--exts",
        nargs="*",
        default=list(DEFAULT_EXTS),
        help="Video extensions to include (e.g., .mp4 .webm .mov)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    # 1) Ensure manifest exists and has a header
    ensure_header(args.manifest)

    # 2) Read existing hashes for deduplication
    existing = read_existing_hashes(args.manifest)

    # 3) Find all video files under 'root' with allowed extensions
    files = list(iter_video_files(root, args.exts))
    if not files:
        print(f"[scan] No videos found under: {root}")
        return

    # 4) Build candidate rows (one per video)
    rows = build_rows(files, args.source, args.label, args.subtype, args.subset)

    # 5) Append rows that aren't duplicates (by sha256)
    written = append_rows(args.manifest, rows, existing)

    print(
        f"[scan] Found={len(files)} | New rows written={written} | "
        f"Skipped as duplicates={len(files) - written} | Manifest={args.manifest.as_posix()}"
    )


if __name__ == "__main__":
    main()
