"""Flag gross screenshot problems for manual visual review.

Image statistics cannot prove algorithmic correctness, clipping, or legibility.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "qa"
inventory = json.loads((QA / "catalog-inventory.json").read_text(encoding="utf-8"))
topics = {row["topic_id"]: row for row in inventory}
shots = QA / "screenshots" / "visualizations"
rows = []
fingerprints = defaultdict(list)

for topic_id, topic in topics.items():
    path = shots / f"{topic_id}.png"
    row = {"topic_id": topic_id, "name": topic["name"], "width": "", "height": "",
           "body_stddev": "", "body_colors": "", "flags": ""}
    flags = []
    if not path.exists():
        flags.append("missing")
    else:
        try:
            with Image.open(path) as original:
                image = original.convert("RGB")
                row["width"], row["height"] = image.size
                if image.width < 500 or image.height < 900:
                    flags.append("unexpected_size")
                body = image.crop((int(.04 * image.width), int(.21 * image.height),
                                   int(.96 * image.width), int(.88 * image.height)))
                thumbnail = body.resize((96, 128))
                stddev = sum(ImageStat.Stat(thumbnail).stddev) / 3
                colors = len(thumbnail.quantize(colors=64).getcolors(96 * 128) or [])
                row["body_stddev"] = f"{stddev:.1f}"
                row["body_colors"] = colors
                if stddev < 10 or colors < 8:
                    flags.append("possibly_blank_body")
                # Similar bodies can be legitimate teaching-family reuse; inspect them.
                fingerprint = hashlib.sha256(thumbnail.resize((16, 16)).tobytes()).hexdigest()
                fingerprints[fingerprint].append(topic_id)
        except OSError:
            flags.append("unreadable_png")
    row["flags"] = ";".join(flags)
    rows.append(row)

duplicates = {topic for group in fingerprints.values() if len(group) > 1 for topic in group}
for row in rows:
    if row["topic_id"] in duplicates:
        row["flags"] = ";".join(filter(None, (row["flags"], "same_body_fingerprint")))

destination = QA / "screenshot-screening.csv"
with destination.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Screened {len(rows)} topic-ID images; {sum(bool(row['flags']) for row in rows)} flagged for review: {destination}")
