"""Extract a small, reproducible output-length histogram from Azure's trace.

Reads 16 evenly spaced 64-KiB byte ranges (~1 MiB of a 1.1-GB trace). This is
a stratified byte-window sample, NOT a uniform request sample or a chat replay.
Only positive output lengths are retained. No prompt text or user IDs are read.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/"
       "dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv")
CARD = "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md"


def get_range(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}",
                                               "User-Agent": "turn-boundary-research/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        if response.status != 206:
            raise RuntimeError("server ignored byte range; refusing full download")
        content_range = response.headers["Content-Range"]
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"unexpected content range: {content_range}")
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError("incomplete/oversized byte-range response")
        return data, int(content_range.rsplit("/", 1)[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/azure_output_histogram.json")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--window-bytes", type=int, default=65536)
    args = ap.parse_args()
    if args.windows < 2 or args.window_bytes < 1024:
        ap.error("use at least 2 windows and 1024 bytes/window")
    head, size = get_range(0, 1023)
    header = head.splitlines()[0].decode().split(",")
    if header != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
        raise ValueError(f"unexpected schema: {header}")
    counts, ranges, skipped = Counter(), [], 0
    for i in range(args.windows):
        start = int(i * (size - args.window_bytes) / (args.windows - 1))
        data, current_size = get_range(start, start + args.window_bytes - 1)
        if current_size != size:
            raise RuntimeError("dataset changed during sampling")
        # Exclude header/first partial line and final partial line in every window.
        rows = list(csv.reader(io.StringIO(b"\n".join(data.splitlines()[1:-1]).decode())))
        for row in rows:
            if len(row) != 3:
                raise ValueError("malformed CSV row")
            length = int(row[2])
            if length > 0:
                counts[length] += 1
            else:
                skipped += 1
        ranges.append({"start": start, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                       "first_timestamp": rows[0][0], "last_timestamp": rows[-1][0]})
        print(f"window {i + 1}/{args.windows}: {sum(counts.values())} positive outputs", flush=True)
    result = {
        "source": URL, "dataset_card": CARD, "source_bytes": size,
        "license": "CC-BY-4.0 (AzurePublicDataset LICENSE)",
        "attribution": "Stojkovic et al., DynamoLLM, HPCA 2025, https://arxiv.org/abs/2408.00741",
        "sampling": "evenly spaced byte windows; partial edge records excluded; not uniform over requests",
        "think_time": "unavailable: invocation timestamps have no session IDs or response-end times",
        "records": sum(counts.values()), "nonpositive_outputs_excluded": skipped,
        "ranges": ranges, "output_token_histogram": sorted(counts.items()),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
