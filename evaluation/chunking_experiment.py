"""Run a deterministic comparison of NoteBot chunking configurations."""

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from app.ingestion.chunker import ChunkingConfig, split_text


SAMPLE_TEXT = """INTRODUCTION
Retrieval augmented generation combines search with language model generation.
The retriever selects relevant passages from a document before the model answers.

CHUNKING
Chunk size controls how much context is included in each passage. Overlap keeps
important sentences from being lost at boundaries. Heading-aware splitting keeps
sections together when headings are present.

EVALUATION
Chunking quality should be measured using retrieval recall, context precision,
and the number of chunks produced for representative documents."""


def run_experiment() -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for strategy in ("recursive", "heading"):
        for size in (400, 800, 1200):
            for overlap in (50, 100, 200):
                chunks = split_text(SAMPLE_TEXT, ChunkingConfig(size, overlap, strategy))
                lengths = [len(chunk) for chunk in chunks]
                results.append(
                    {
                        "strategy": strategy,
                        "chunk_size": size,
                        "overlap": overlap,
                        "chunk_count": len(chunks),
                        "average_characters": round(mean(lengths), 2) if lengths else 0,
                        "min_characters": min(lengths) if lengths else 0,
                        "max_characters": max(lengths) if lengths else 0,
                    }
                )
    return results


if __name__ == "__main__":
    output = Path(__file__).with_name("chunking_experiment_results.json")
    output.write_text(json.dumps(run_experiment(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
