"""Retrieval evaluation: how often does the right chunk land in the top-k?

Runs fully in memory (no database), so different chunk sizes and embedding
models can be compared quickly on the same questions.

Example:
    python3 -m eval.run --doc article.txt
    python3 -m eval.run --doc article.txt --chunk-sizes 200,300,500,800 --overlaps 0,80
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from app.config import settings
from app.services.chunking import chunk_text

_models: dict[str, TextEmbedding] = {}


def get_model(name: str) -> TextEmbedding:
    if name not in _models:
        _models[name] = TextEmbedding(model_name=name, cache_dir=".models")
    return _models[name]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def prefixes_for(model_name: str) -> tuple[str, str]:
    # E5 models were trained with these prefixes and perform poorly without them
    if "e5" in model_name.lower():
        return "query: ", "passage: "
    return "", ""


def embed_normalized(model: TextEmbedding, texts: list[str], prefix: str) -> np.ndarray:
    vectors = np.array(list(model.embed([prefix + t for t in texts])), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def evaluate(text: str, data: dict, model_name: str, chunk_size: int, overlap: int) -> dict:
    model = get_model(model_name)
    q_prefix, p_prefix = prefixes_for(model_name)

    chunks = chunk_text(text, chunk_size, overlap)
    normalized = [normalize(c) for c in chunks]
    chunk_vecs = embed_normalized(model, chunks, p_prefix)

    answerable = data["answerable"]
    q_vecs = embed_normalized(model, [a["question"] for a in answerable], q_prefix)
    distances = 1.0 - q_vecs @ chunk_vecs.T  # cosine distance: smaller is closer
    order = np.argsort(distances, axis=1)

    details, unmatched = [], []
    for row, item in enumerate(answerable):
        expected = normalize(item["expected"])
        relevant = {i for i, c in enumerate(normalized) if expected in c}
        if not relevant:
            unmatched.append(item["question"])
            continue
        rank = next(r for r, idx in enumerate(order[row], start=1) if idx in relevant)
        top = int(order[row][0])
        details.append({
            "question": item["question"],
            "expected": item["expected"],
            "rank": rank,
            "best_relevant_dist": float(min(distances[row][i] for i in relevant)),
            "top1_dist": float(distances[row][top]),
            "top1_preview": chunks[top][:120],
        })

    unanswerable = data.get("unanswerable", [])
    if unanswerable:
        u_vecs = embed_normalized(model, unanswerable, q_prefix)
        unanswerable_best = (1.0 - u_vecs @ chunk_vecs.T).min(axis=1)
    else:
        unanswerable_best = np.array([])

    return {
        "chunks": len(chunks),
        "details": details,
        "unmatched": unmatched,
        "unanswerable_best": unanswerable_best,
    }


def summarize(result: dict, ks: list[int]) -> dict:
    ranks = [d["rank"] for d in result["details"]]
    best = [d["best_relevant_dist"] for d in result["details"]]
    n = len(ranks)
    worst_answerable = max(best)
    min_unanswerable = (
        float(result["unanswerable_best"].min()) if len(result["unanswerable_best"]) else float("nan")
    )
    return {
        "hits": {k: sum(r <= k for r in ranks) / n for k in ks},
        "mrr": sum(1 / r for r in ranks) / n,
        "worst_answerable": worst_answerable,
        "min_unanswerable": min_unanswerable,
        # positive gap: one threshold can separate on-topic from off-topic questions
        "gap": min_unanswerable - worst_answerable,
    }


def print_table(rows: list[dict], ks: list[int]) -> None:
    hit_cols = " ".join(f"{'hit@' + str(k):>6}" for k in ks)
    print(f"\n{'model':<36} {'size':>5} {'ovl':>4} {'chunks':>6} {hit_cols} "
          f"{'MRR':>5} {'worst_ans':>9} {'min_unans':>9} {'gap':>6}")
    print("-" * (36 + 24 + 7 * len(ks) + 40))
    for r in rows:
        s = r["summary"]
        hits = " ".join(f"{s['hits'][k]:>6.0%}" for k in ks)
        print(f"{r['model'].split('/')[-1][:36]:<36} {r['size']:>5} {r['overlap']:>4} "
              f"{r['chunks']:>6} {hits} {s['mrr']:>5.2f} "
              f"{s['worst_answerable']:>9.3f} {s['min_unanswerable']:>9.3f} {s['gap']:>6.3f}")


def print_thresholds(result: dict) -> None:
    best = np.array([d["best_relevant_dist"] for d in result["details"]])
    off = result["unanswerable_best"]
    print("\n  threshold   on-topic kept   off-topic rejected")
    for t in np.arange(0.30, 0.96, 0.05):
        kept = float(np.mean(best <= t))
        rejected = float(np.mean(off > t)) if len(off) else float("nan")
        print(f"     {t:.2f}        {kept:>6.0%}           {rejected:>6.0%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc", required=True, help="path to a .txt/.md document")
    p.add_argument("--questions", default="eval/questions.json")
    p.add_argument("--models", default=settings.embedding_model, help="comma-separated model names")
    p.add_argument("--chunk-sizes", default=str(settings.chunk_size))
    p.add_argument("--overlaps", default=str(settings.chunk_overlap))
    p.add_argument("--ks", default="1,3,5")
    p.add_argument("--miss-k", type=int, default=3, help="print questions whose chunk ranks worse than this")
    p.add_argument("--show-misses", action="store_true")
    p.add_argument("--thresholds", action="store_true", help="print a threshold sweep per configuration")
    args = p.parse_args()

    text = Path(args.doc).read_text(encoding="utf-8-sig")
    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    models = [m.strip() for m in args.models.split(",")]
    sizes = [int(x) for x in args.chunk_sizes.split(",")]
    overlaps = [int(x) for x in args.overlaps.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    rows, warnings = [], []
    for model_name, size, overlap in itertools.product(models, sizes, overlaps):
        print(f"evaluating {model_name.split('/')[-1]} size={size} overlap={overlap} ...")
        try:
            result = evaluate(text, data, model_name, size, overlap)
        except ValueError as e:
            print(f"  skipped: {e}")
            continue
        if not result["details"]:
            raise SystemExit("None of the expected phrases were found in the document. Check questions.json.")
        if result["unmatched"]:
            warnings.append((model_name, size, overlap, result["unmatched"]))
        rows.append({
            "model": model_name, "size": size, "overlap": overlap,
            "chunks": result["chunks"], "summary": summarize(result, ks), "result": result,
        })

        if args.show_misses:
            for d in result["details"]:
                if d["rank"] > args.miss_k:
                    print(f"  MISS rank={d['rank']}  Q: {d['question']}")
                    print(f"       expected: {d['expected']}")
                    print(f"       top-1 (dist {d['top1_dist']:.3f}): {d['top1_preview']!r}")
        if args.thresholds:
            print_thresholds(result)

    print_table(rows, ks)

    for model_name, size, overlap, questions in warnings:
        print(f"\nWARNING ({model_name.split('/')[-1]}, size={size}, overlap={overlap}): "
              f"expected phrase not found in any chunk for {len(questions)} question(s), "
              f"excluded from the metrics. The phrase is probably split by a chunk boundary "
              f"or is not an exact copy of the document text:")
        for q in questions:
            print(f"  - {q}")


if __name__ == "__main__":
    main()