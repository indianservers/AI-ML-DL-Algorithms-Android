"""Render every saved topic screenshot into review pages with stable topic labels."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
records = json.loads((ROOT / "qa/catalog-inventory.json").read_text(encoding="utf-8"))
unique = list({record["topic_id"]: record for record in reversed(records)}.values())[::-1]
out = ROOT / "qa/contact"
out.mkdir(parents=True, exist_ok=True)
columns, rows, thumb_w, thumb_h, caption = 4, 3, 300, 650, 54
per_page = columns * rows
for start in range(0, len(unique), per_page):
    page = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + caption)), "#0c172d")
    draw = ImageDraw.Draw(page)
    for local, record in enumerate(unique[start:start + per_page]):
        x = (local % columns) * thumb_w
        y = (local // columns) * (thumb_h + caption)
        path = ROOT / "qa/screenshots/visualizations" / f"{record['topic_id']}.png"
        if path.exists():
            with Image.open(path) as source:
                preview = ImageOps.contain(source.convert("RGB"), (thumb_w, thumb_h))
                page.paste(preview, (x + (thumb_w - preview.width) // 2, y + caption))
        draw.text((x + 5, y + 4), f"{start + local + 1}. {record['name'][:29]}", fill="white")
        draw.text((x + 5, y + 27), record["category"][:31], fill="#8bd7ed")
    page.save(out / f"page-{start // per_page + 1:02d}.jpg", quality=85)
print(f"{len(unique)} topics on {(len(unique) + per_page - 1) // per_page} pages")
