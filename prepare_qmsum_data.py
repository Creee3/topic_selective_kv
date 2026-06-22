"""
Prepare QMSum into a normalized JSONL format for routing experiments.

Why this script exists:
- The current `data/qmsum/*.txt` files are flattened text prompts.
- For routing simulation we need structured fields:
  - meeting transcripts
  - specific queries
  - relevant text spans

Usage:
  python prepare_qmsum_data.py

  Or explicitly:
  python prepare_qmsum_data.py \
      --input_dir ../reference_repos/QMSum-main/data/ALL/jsonl \
      --output_dir data/qmsum_structured
"""

import argparse
import json
import os
from typing import Any, Dict, List


def normalize_spans(spans: List[List[str]]) -> List[List[int]]:
    normalized = []
    for span in spans:
        if len(span) != 2:
            continue
        normalized.append([int(span[0]), int(span[1])])
    return normalized


def normalize_meeting(sample: Dict[str, Any], split: str, meeting_idx: int) -> Dict[str, Any]:
    transcripts = sample.get("meeting_transcripts", [])
    specific_queries = sample.get("specific_query_list", [])
    general_queries = sample.get("general_query_list", [])
    topic_list = sample.get("topic_list", [])

    normalized_transcripts = [
        {
            "turn_idx": turn_idx,
            "speaker": turn.get("speaker", "").strip(),
            "content": turn.get("content", "").strip(),
            # A simulator-friendly text field for later prompt construction.
            "text": f"{turn.get('speaker', '').strip()}: {turn.get('content', '').strip()}".strip(),
        }
        for turn_idx, turn in enumerate(transcripts)
    ]

    normalized_specific_queries = [
        {
            "query_idx": query_idx,
            "query": q.get("query", "").strip(),
            "answer": q.get("answer", "").strip(),
            "relevant_text_span": normalize_spans(q.get("relevant_text_span", [])),
        }
        for query_idx, q in enumerate(specific_queries)
    ]

    normalized_general_queries = [
        {
            "query_idx": query_idx,
            "query": q.get("query", "").strip(),
            "answer": q.get("answer", "").strip(),
        }
        for query_idx, q in enumerate(general_queries)
    ]

    normalized_topics = [
        {
            "topic_idx": topic_idx,
            "topic": topic.get("topic", "").strip(),
            "relevant_text_span": normalize_spans(topic.get("relevant_text_span", [])),
        }
        for topic_idx, topic in enumerate(topic_list)
    ]

    meeting_id = sample.get("meeting_id")
    if not meeting_id:
        meeting_id = f"{split}_{meeting_idx:05d}"

    return {
        "dataset": "qmsum",
        "split": split,
        "meeting_idx": meeting_idx,
        "meeting_id": meeting_id,
        "num_turns": len(normalized_transcripts),
        "meeting_transcripts": normalized_transcripts,
        "specific_query_list": normalized_specific_queries,
        "general_query_list": normalized_general_queries,
        "topic_list": normalized_topics,
    }


def convert_split(infile: str, outfile: str, split: str) -> int:
    count = 0
    with open(infile, "r", encoding="utf-8") as fin, open(
        outfile, "w", encoding="utf-8"
    ) as fout:
        for meeting_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            normalized = normalize_meeting(sample, split, meeting_idx)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_input_dir(explicit_input_dir: str | None) -> str:
    if explicit_input_dir:
        return explicit_input_dir

    candidate_dirs = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "reference_repos",
                "QMSum-main",
                "data",
                "ALL",
                "jsonl",
            )
        ),
        os.path.join(os.path.dirname(__file__), "data", "qmsum"),
    ]

    for candidate in candidate_dirs:
        if os.path.exists(os.path.join(candidate, "train.jsonl")):
            return candidate

    raise FileNotFoundError(
        "Could not find QMSum input_dir automatically. "
        "Please pass --input_dir explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare structured QMSum JSONL files")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "qmsum_structured"),
    )
    args = parser.parse_args()

    args.input_dir = resolve_input_dir(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    print("=" * 70)
    print("Preparing QMSum structured files")
    print(f"  input_dir:  {args.input_dir}")
    print(f"  output_dir: {args.output_dir}")
    print("=" * 70)

    for split in splits:
        infile = os.path.join(args.input_dir, f"{split}.jsonl")
        outfile = os.path.join(args.output_dir, f"{split}.jsonl")
        if not os.path.exists(infile):
            raise FileNotFoundError(f"Input file not found: {infile}")
        count = convert_split(infile, outfile, split)
        print(f"  {split:>5}: wrote {count} meetings -> {outfile}")

    print("\nDone.")


if __name__ == "__main__":
    main()
