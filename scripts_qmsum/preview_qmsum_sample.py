"""
Preview one structured QMSum sample after preparation.

Usage:
  python preview_qmsum_sample.py --split train --doc_id 0
"""

import argparse
import json
import os


def load_jsonl_line(filepath: str, doc_id: int):
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == doc_id:
                return json.loads(line)
    raise IndexError(f"doc_id={doc_id} out of range for {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview one prepared QMSum sample")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--doc_id", type=int, default=0)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "qmsum_structured"),
    )
    args = parser.parse_args()

    infile = os.path.join(args.data_dir, f"{args.split}.jsonl")
    sample = load_jsonl_line(infile, args.doc_id)

    transcripts = sample.get("meeting_transcripts", [])
    specific_queries = sample.get("specific_query_list", [])
    topics = sample.get("topic_list", [])

    print("=" * 70)
    print("QMSum Prepared Sample Preview")
    print(f"  split:       {sample.get('split')}")
    print(f"  meeting_id:  {sample.get('meeting_id')}")
    print(f"  meeting_idx: {sample.get('meeting_idx')}")
    print(f"  num_turns:   {sample.get('num_turns')}")
    print(f"  queries:     {len(specific_queries)} specific / {len(sample.get('general_query_list', []))} general")
    print(f"  topics:      {len(topics)}")
    print("=" * 70)

    if topics:
        print("\nFirst topic:")
        print(f"  topic: {topics[0].get('topic', '')}")
        print(f"  spans: {topics[0].get('relevant_text_span', [])}")

    if specific_queries:
        print("\nFirst specific query:")
        print(f"  query:  {specific_queries[0].get('query', '')}")
        print(f"  answer: {specific_queries[0].get('answer', '')[:240]}")
        print(f"  spans:  {specific_queries[0].get('relevant_text_span', [])}")

    if transcripts:
        print("\nFirst 3 transcript turns:")
        for turn in transcripts[:3]:
            print(f"  [{turn['turn_idx']}] {turn['speaker']}: {turn['content'][:180]}")


if __name__ == "__main__":
    main()
