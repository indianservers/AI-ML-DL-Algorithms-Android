"""Summarize observed emulator evidence without promoting attempted taps to semantic passes."""
import collections
import csv
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "qa"
inventory = json.loads((QA / "catalog-inventory.json").read_text(encoding="utf-8"))
by_id = {row["topic_id"]: row for row in inventory}
deep = {}
sources = [QA / "pilot-linear-five.csv", QA / "pilot-agent.csv", QA / "pilot-cross-family.csv"]
sources.extend(sorted(QA.glob("final-emulator-*.csv")))
for source in sources:
    if not source.exists():
        continue
    with source.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["opened"] == "True" and row["rendered"] == "True" and row["scrolled"] == "True" and row["crash"] == "False" and not row["error"]:
                deep[row["topic_id"]] = row

categories = collections.OrderedDict()
for row in inventory:
    categories.setdefault(row["category"], set()).add(row["topic_id"])
screenshots = {topic_id for topic_id in by_id if (QA / "screenshots/visualizations" / f"{topic_id}.png").exists()}
contact_pages = sorted((QA / "contact").glob("page-*.jpg"))
screening = {}
if (QA / "screenshot-screening.csv").exists():
    with (QA / "screenshot-screening.csv").open(encoding="utf-8", newline="") as stream:
        screening = {row["topic_id"]: row["flags"] for row in csv.DictReader(stream)}
blank_images = {topic_id for topic_id, flags in screening.items() if "possibly_blank_body" in flags}
with_controls = {topic_id for topic_id, row in deep.items() if int(row["controls_attempted"]) > 0}
canvas_changed = {topic_id for topic_id, row in deep.items() if row.get("canvas_tap_changed") == "True"}

lines = [
    "# Final visualization coverage",
    "",
    f"Updated {datetime.datetime.now().isoformat(timespec='minutes')} from the current catalog inventory and saved emulator CSV files.",
    "",
    "## Evidence and acceptance state",
    "",
    f"- Catalog route registrations: **{len(inventory)}/{len(inventory)} placements** have an explicit native route (unit-test assertion).",
    f"- Saved topic-ID screenshot files: **{len(screenshots)}/{len(by_id)}**. Legacy captures may carry the wrong topic where title-based filenames collided; fresh topic-ID captures are counted in the deeper passes below.",
    f"- Gross image screening flags **{len(blank_images)} blank body capture(s)** and **{sum(bool(flags) for flags in screening.values())} images for review** (many are legitimate shared or alias views)." + (" Blank captures must be replaced before visual acceptance." if blank_images else ""),
    f"- Deeper emulator passes: **{len(deep)}/{len(by_id)} unique IDs** opened, rendered, scrolled, and had no observed fatal crash.",
    f"- At least one visible control attempted: **{len(with_controls)}/{len(by_id)} unique IDs**. An attempted tap or slider move is not yet a confirmed state transition.",
    f"- Vision canvas tap changed a selected-state readout: **{len(canvas_changed)} unique IDs**.",
    f"- Contact sheets reviewed at overview resolution: **{len(contact_pages)}/27 pages**; this caught visible layout and copy problems but does not substitute for full-resolution per-screen inspection.",
    "- Full per-control semantic validation, per-topic numerical checks, visual inspection, and performance profiling: **pending**.",
    "",
    "## Per-category decision",
    "",
    "`BLOCKED` means the final acceptance criteria remain unproven; it does not mean the screen failed to open.",
    "",
    "| Category | Unique IDs | Screenshot | Deeper emulator pass | Control attempted | Final status |",
    "|---|---:|---:|---:|---:|---|",
]
for category, ids in categories.items():
    lines.append(f"| {category} | {len(ids)} | {len(ids & screenshots)} | {len(ids & deep.keys())} | {len(ids & with_controls)} | BLOCKED |")
lines += [
    "",
    "## Confirmed corrections in this hardening pass",
    "",
    "- Seq2Seq example target words are identified as teacher-forced teaching data; decoder hidden states now depend on context and the prior target token.",
    "- Unsupported Train & Inference tabs state that a training engine is not connected, instead of presenting a generic fake trainer.",
    "- Polynomial training starts with a polynomial dataset, and regularized Train defaults match the Visualization defaults.",
    "- CNN matrix cards summarize dimensions and value range rather than squeezing unreadable raw arrays beneath the grid.",
    "- SHAP waterfall now includes every feature contribution before the final prediction marker.",
    "- Segmentation masks and selected pixel readouts now use the same explicit per-pixel logits and softmax instead of unrelated decorative ovals and fixed confidence text.",
    "- Multinomial and Bernoulli Naive Bayes open with meaningful token evidence and show the per-token terms that sum to each class score.",
    "- Association learning places its algorithm state above the long transaction list; students can expand that list when needed.",
    "- Dark system bars use light icons for readable status information.",
    "- Language, vision, forecasting, evolution, and recommendation screens now open with concise mechanism-specific guidance instead of repeated generic subtitles.",
    "- Adam-family update labels and the numerical engine now place the stabilizer ε outside the square root in the adaptive denominator.",
    "- The sequence fundamentals lab identifies manually set recurrent weights and gate activations as teaching overrides.",
    "- Short NLP example sets use two columns so the computed matrix is visible sooner; longer text remains full width.",
    "- Genetic Programming now draws operator and operand nodes with branches in its expression tree instead of using a small ASCII diagram.",
    "- Autoencoder reconstruction and denoising now decode only the retained code. The former clean-pixel blend was removed; the view identifies its fixed orthogonal teaching basis and separates the denoising topic from the basic bottleneck example.",
    "- The diffusion teaching timeline now subtracts a noise estimate computed from the noisy image. Its prior implementation interpolated directly toward the clean target and derived the prediction from known true noise. The screen labels the fixed predictor and display stages honestly.",
    "- The GAN progression is labeled as an analytic teaching generator and discriminator, so changing its slider is not mistaken for live network training.",
    "- The displayed GAN discriminator loss now averages real and generated binary cross-entropy terms; it previously included only generated examples.",
    "- Tree and SVM screens group less common settings behind an Advanced controls toggle while keeping the defining controls visible.",
    "- K-Means++ shows its computed distance-squared seed probabilities on the first frame, making the seeding mechanism visible before students press Next.",
    "- Supervised and clustering plots now claim drag gestures only when they begin near a sample, allowing vertical swipes on empty chart space to scroll to controls below.",
    "- On the current debug APK, a Simple Linear Regression sample moved after a point-origin drag and the plot changed; an empty-plot swipe exposed the metrics below. SVM and both K-Means variants also reached controls below their charts in the emulator recapture.",
    "- Recurrent and Seq2Seq hidden-state strips use thicker values, endpoints, and a baseline so the computed state is easier to inspect.",
    "",
    "## Remaining acceptance work",
    "",
    "- Verify each visible control produces the expected state and metric change; the generic sweep only attempts controls.",
    "- Compare every related algorithm family against `VisualizationDifferentiationMatrix.md` in the running app.",
    "- Add numerical assertions for each defining mechanism and validate plotted values against the engine output.",
    "- Inspect each unique screenshot for clipping and readability, including after interaction and rotation or alternate screen sizes.",
    "- Replace every legacy title-named capture with a fresh topic-ID capture; K-Means and K-Means++ demonstrate why title slug collisions cannot be trusted.",
    "- Profile recomposition, frame time, memory, and long-running interaction on representative heavy screens.",
    "- Repeat frame timing on a hardware-accelerated emulator or device. The software-rendered emulator reported 45/49 janky frames during an otherwise idle Simple Linear Regression interaction and 32/32 on Home; these samples do not isolate app cost from renderer cost. Raw `gfxinfo` and `meminfo` files are saved under `qa/`.",
    "- Extend live model training beyond the eight connected topics if full Train & Inference capability is required for all algorithms.",
    "",
]
if "deep-learning-transformers-transformer" in blank_images:
    lines.insert(-1, "- Recapture `deep-learning-transformers-transformer`: its saved PNG is an all-white transient screen rather than an inspected visualization.")
(ROOT / "docs/FinalVisualizationCoverage.md").write_text("\n".join(lines), encoding="utf-8")
print(f"deeper passes {len(deep)}/{len(by_id)}, controls attempted {len(with_controls)}/{len(by_id)}, screenshots {len(screenshots)}/{len(by_id)}")
