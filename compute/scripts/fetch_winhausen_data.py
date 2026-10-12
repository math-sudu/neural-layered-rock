"""Download the original Winhausen et al. (2023) records from their OSF repository."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import Request, urlopen


DATA_ROOT = Path(__file__).resolve().parents[1] / "results/data/winhausen2023"


def fetch(item):
    target = DATA_ROOT / "raw" / item["name"]
    if target.is_file() and target.stat().st_size == item["bytes"]:
        return f"Present: {item['name']}"
    request = Request(item["download"], headers={"User-Agent": "neural-layered-rock/1.0"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) != item["bytes"]:
        raise ValueError(f"Unexpected download size for {item['name']}: {len(data)} bytes")
    target.write_bytes(data)
    return f"Downloaded: {item['name']}"


def main():
    source = json.loads((DATA_ROOT / "source.json").read_text(encoding="utf-8"))
    (DATA_ROOT / "raw").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for message in pool.map(fetch, source["files"]):
            print(message, flush=True)


if __name__ == "__main__":
    main()
