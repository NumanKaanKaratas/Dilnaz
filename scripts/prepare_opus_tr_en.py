from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpusCorpus:
    name: str
    url: str
    stem: str


CORPORA = (
    OpusCorpus(
        "Tatoeba",
        "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-tr.txt.zip",
        "Tatoeba.en-tr",
    ),
    OpusCorpus(
        "TED2020",
        "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-tr.txt.zip",
        "TED2020.en-tr",
    ),
    OpusCorpus(
        "QED",
        "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-tr.txt.zip",
        "QED.en-tr",
    ),
    OpusCorpus(
        "GlobalVoices",
        "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-tr.txt.zip",
        "GlobalVoices.en-tr",
    ),
    OpusCorpus(
        "Bianet",
        "https://object.pouta.csc.fi/OPUS-Bianet/v1/moses/en-tr.txt.zip",
        "Bianet.en-tr",
    ),
    OpusCorpus(
        "WikiMatrix",
        "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-tr.txt.zip",
        "WikiMatrix.en-tr",
    ),
    OpusCorpus(
        "wikimedia",
        "https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/en-tr.txt.zip",
        "wikimedia.en-tr",
    ),
    OpusCorpus(
        "OpenSubtitles",
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-tr.txt.zip",
        "OpenSubtitles.en-tr",
    ),
    OpusCorpus(
        "NLLB",
        "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-tr.txt.zip",
        "NLLB.en-tr",
    ),
)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"exists {path}", flush=True)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"download {url}", flush=True)
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp.replace(path)
    print(f"saved {path}", flush=True)


def extract(zip_path: Path, extract_root: Path, corpus: OpusCorpus) -> tuple[Path, Path]:
    corpus_dir = extract_root / corpus.name
    en_path = corpus_dir / f"{corpus.stem}.en"
    tr_path = corpus_dir / f"{corpus.stem}.tr"
    if en_path.exists() and tr_path.exists():
        print(f"extracted {corpus.name}", flush=True)
        return en_path, tr_path
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(corpus_dir)
    if not en_path.exists() or not tr_path.exists():
        names = ", ".join(sorted(item.name for item in corpus_dir.iterdir()))
        raise FileNotFoundError(f"{corpus.name} missing {en_path.name}/{tr_path.name}; files={names}")
    print(f"extract {zip_path} -> {corpus_dir}", flush=True)
    return en_path, tr_path


def valid_pair(en: str, tr: str, max_chars: int) -> bool:
    if not en or not tr:
        return False
    if len(en) > max_chars or len(tr) > max_chars:
        return False
    return en != tr


def append_pairs(en_path: Path, tr_path: Path, output, max_chars: int) -> tuple[int, int]:
    written = 0
    skipped = 0
    with en_path.open("r", encoding="utf-8", errors="replace") as en_file, tr_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as tr_file:
        for en_line, tr_line in zip(en_file, tr_file, strict=False):
            en = en_line.strip()
            tr = tr_line.strip()
            if not valid_pair(en, tr, max_chars):
                skipped += 1
                continue
            output.write(json.dumps({"tr": tr, "en": en}, ensure_ascii=False))
            output.write("\n")
            written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OPUS TR-EN corpora and merge them into one JSONL file.")
    parser.add_argument("--download-dir", type=Path, default=Path("TrainDatas/opus_raw"))
    parser.add_argument("--extract-dir", type=Path, default=Path("TrainDatas/opus_extracted"))
    parser.add_argument("--output", type=Path, default=Path("TrainDatas/opus_tr_en_all.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1000)
    args = parser.parse_args()

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[OpusCorpus, Path, Path]] = []
    for corpus in CORPORA:
        zip_path = args.download_dir / f"{corpus.stem}.txt.zip"
        download(corpus.url, zip_path)
        en_path, tr_path = extract(zip_path, args.extract_dir, corpus)
        pairs.append((corpus, en_path, tr_path))

    total_written = 0
    total_skipped = 0
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8", newline="\n") as output:
        for corpus, en_path, tr_path in pairs:
            written, skipped = append_pairs(en_path, tr_path, output, args.max_chars)
            total_written += written
            total_skipped += skipped
            print(f"merge {corpus.name} written={written} skipped={skipped}", flush=True)
    tmp_output.replace(args.output)
    print(f"output={args.output} written={total_written} skipped={total_skipped}", flush=True)


if __name__ == "__main__":
    main()
