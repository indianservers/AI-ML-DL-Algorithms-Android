"""Build a source-aligned hardening inventory from the runtime catalog test output."""
import collections
import csv
import datetime
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "app/build/test-results/testDebugUnitTest"
OUT = ROOT / "qa"
SHOTS = OUT / "screenshots/visualizations"
DOC = ROOT / "docs/FinalAlgorithmVisualizationAudit.md"
OUT.mkdir(exist_ok=True)
SHOTS.mkdir(parents=True, exist_ok=True)


def audit_rows(marker):
    found = []
    for source in RESULTS.glob("TEST-*.xml"):
        output = ET.parse(source).getroot().findtext("system-out") or ""
        found.extend(line.split("\t")[1:] for line in output.splitlines()
                     if line.startswith(marker + "\t"))
    return found


placements = audit_rows("VIS_PLACE")
unique = audit_rows("VIS_AUDIT")
assert len(placements) == 327 and len(unique) == 320
assert len({row[2] for row in placements}) == 320
by_id = {row[0]: row for row in unique}
assert len(by_id) == 320


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def screenshot_for(row):
    category, section, topic_id, title, source_domain, source_section, route, *_ = row
    canonical = by_id[topic_id]
    options = [
        f"{slug(category + '-' + section + '-' + title)}-top.png",
        f"{slug(section + '-' + title)}-top.png",
        f"{slug(title)}-top.png",
        f"{slug(canonical[2] + '-' + canonical[3] + '-' + canonical[1])}-top.png",
        f"{slug(canonical[3] + '-' + canonical[1])}-top.png",
    ]
    matches = [path for candidate in options
               for path in (ROOT / "outputs").glob("*verification/" + candidate)]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def engine_for(row):
    category, section, _id, title, source_domain, source_section, route, implementation, *_ = row
    if route == "SupervisedNative":
        if any(word in title for word in ("Forest", "Tree", "Extra Trees")) and "Boost" not in title:
            return "SupervisedTreeEngine"
        if any(word in title for word in ("Boost", "XGBoost", "LightGBM", "CatBoost")):
            return "SupervisedBoostingEngine / SupervisedBoostingVariants"
        if title == "Logistic Regression":
            return "fitLogistic"
        if title in {"Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression",
                     "Ridge Regression", "Lasso Regression", "Elastic Net Regression", "K-Nearest Neighbors"}:
            return "PhaseOneEngines"
        return "SupervisedStatisticalEngine / PhaseOneEngines"
    if route == "StageTwoNative":
        return {
            "Clustering": "StageTwoClusteringEngine",
            "Dimensionality Reduction": "StageTwoReductionEngine",
            "Association Learning": "StageTwoAssociationEngine",
            "Anomaly Detection": "StageTwoAnomalyEngine",
            "Methods": "StageTwoSemiSupervisedEngine",
            "Bagging": "StageTwoEnsembleEngine",
            "Boosting": "StageTwoEnsembleEngine",
            "Combining Models": "StageTwoEnsembleEngine",
        }.get(section, "StageTwoVisualizationScreen")
    if route == "StageThreeForecastNative":
        return "StageThreeForecastEngine"
    if route == "StageThreeLanguageNative":
        return "StageThreeLanguageEngine"
    if route == "StageThreeNeuralLanguageNative":
        return "StageThreeNeuralLanguageScreen / PhaseSeven-Eight engines"
    if route == "StageThreeVisionNative":
        return "StageThreeVisionEngine"
    if route == "StageThreeDeepNative":
        return "StageThreeDeepScreen section dispatch / PhaseFive-Nine engines"
    if route == "StageFourNative":
        return {
            "Reinforcement Learning": "StageFourRlEngine",
            "Probabilistic & Bayesian Learning": "StageFourProbabilityEngine",
            "Optimization Algorithms": "StageFourOptimizationEngine",
            "Evolutionary Algorithms": "StageFourEvolutionEngine",
            "Recommendation Algorithms": "StageFourRecommendationEngine",
            "Explainable AI": "StageFourExplanationEngine",
            "AI Algorithms": "StageFourSearchEngine / RL / Probability / Evolution engines",
        }.get(source_domain, "StageFourVisualizationScreen")
    if route == "ExistingNative":
        prefix = implementation.split("/")[0]
        match = re.search(r"Phase(\w+)AlgorithmLab", prefix)
        return f"{prefix} / Phase{match.group(1)}Engines" if match else prefix
    return "UNRESOLVED"


concepts = {
    ("Reinforcement Learning", title) for title in
    ("Agent", "Environment", "State", "Action", "Reward", "Policy", "Value Function", "Markov Decision Process")
} | {
    ("Deep Learning", title) for title in
    ("Artificial Neuron", "Activation Functions", "Loss Functions", "Weight Initialization",
     "Batch Normalization", "Layer Normalization", "Dropout", "Regularization", "Positional Encoding")
}
live_ids = {
    row[2] for row in placements if row[0] == "Supervised Learning" and row[3] in {
        "Simple Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression",
        "Elastic Net Regression", "Logistic Regression", "K-Nearest Neighbors"
    }
}
assert len(live_ids) == 8

saved = {}
for row in placements:
    topic_id = row[2]
    if topic_id in saved:
        continue
    target = SHOTS / f"{topic_id}.png"
    if target.exists():
        saved[topic_id] = str(target.relative_to(ROOT)).replace("\\", "/")
        continue
    source = screenshot_for(row)
    if source:
        shutil.copy2(source, target)
        saved[topic_id] = str(target.relative_to(ROOT)).replace("\\", "/")

deep_checks = {}
latest_errors = {}
for result_file in (OUT / "pilot-linear-five.csv", OUT / "pilot-agent.csv", OUT / "pilot-cross-family.csv"):
    if result_file.exists():
        with result_file.open(encoding="utf-8", newline="") as stream:
            for check in csv.DictReader(stream):
                if check["opened"] == "True" and check["rendered"] == "True" and check["crash"] == "False" and not check["error"]:
                    deep_checks[check["topic_id"]] = check
for result_file in OUT.glob("final-emulator-*.csv"):
    with result_file.open(encoding="utf-8", newline="") as stream:
        for check in csv.DictReader(stream):
            if check["opened"] == "True" and check["rendered"] == "True" and check["crash"] == "False" and not check["error"]:
                deep_checks[check["topic_id"]] = check
                latest_errors.pop(check["topic_id"], None)
            elif check["error"]:
                latest_errors[check["topic_id"]] = check["error"]

screening = {}
screening_file = OUT / "screenshot-screening.csv"
if screening_file.exists():
    with screening_file.open(encoding="utf-8", newline="") as stream:
        screening = {check["topic_id"]: check["flags"] for check in csv.DictReader(stream)}

records = []
for row in placements:
    category, section, topic_id, title, source_domain, source_section, route, implementation, controls, dataset, status = row
    metadata = by_id[topic_id]
    role = ("Live-trainable" if topic_id in live_ids else
            "Concept-only" if (source_domain, title) in concepts else "Visualization-only in current build")
    alias = category != source_domain or section != source_section or title != metadata[1]
    checked = deep_checks.get(topic_id)
    finding = ""
    repair = ""
    if title in {"Seq2Seq", "Sequence-to-Sequence"}:
        finding = "Fixed example targets were described as generated output"
        repair = "Teacher-forced target label and context-dependent decoder state added"
    elif category == "Deep Learning" and title == "CNN":
        finding = "Raw matrix text was unreadable below pixel grids"
        repair = "Compact dimension and range summary added"
    elif category == "Explainable AI" and title == "SHAP":
        finding = "Waterfall omitted final feature contribution row"
        repair = "Row mapping includes all features before prediction"
    elif category == "Computer Vision" and title in {"Semantic Segmentation", "U-Net", "Instance Segmentation", "Mask R-CNN"}:
        finding = "Mask colors and selected pixel labels came from unrelated fixed drawings"
        repair = "Shared toy pixel logits/softmax now drive mask and readout"
    elif category == "Supervised Learning" and title in {"Multinomial Naive Bayes", "Bernoulli Naive Bayes"}:
        finding = "Default all-zero features left the visual nearly empty"
        repair = "Meaningful initial tokens and per-token log-likelihood contributions added"
    elif category == "Optimization Algorithms" and title in {"AdamW", "Nadam"}:
        finding = "Displayed update equation omitted the stabilizing epsilon denominator term"
        repair = "Equation and engine now put epsilon outside the square root"
    elif category == "Evolutionary Algorithms" and title == "Genetic Programming":
        finding = "Small ASCII expression tree was difficult to read"
        repair = "Native node-and-branch expression tree diagram added"
    elif category == "Deep Learning" and title in {"Basic Autoencoder", "Denoising Autoencoder"}:
        finding = "Reconstruction blended clean input pixels, bypassing the latent bottleneck"
        repair = "Fixed orthogonal code is decoded without input leakage; denoising has a focused noisy-input view"
    elif category == "Deep Learning" and title == "Diffusion Models":
        finding = "Predicted noise and reverse timeline used known true noise and clean target"
        repair = "Noise estimate derives from noisy input; timeline subtracts that estimate and labels the fixed predictor"
    elif category == "Deep Learning" and title == "GAN":
        finding = "Discriminator loss omitted real examples and an analytic progression appeared to be live training"
        repair = "Balanced real/fake BCE displayed and analytic teaching progression labeled honestly"
    elif category == "Unsupervised Learning" and title == "K-Means++":
        finding = "First frame hid distance-squared seeding and resembled ordinary K-Means"
        repair = "Normalized D² probabilities appear as rings on the initial frame"
    elif category == "Supervised Learning" and title in {"Support Vector Machine", "Support Vector Regression", "Decision Tree", "Decision Tree Regression", "Random Forest", "Random Forest Regression", "Extra Trees", "Extra Trees Regression"}:
        finding = "Advanced hyperparameters competed with the core teaching controls"
        repair = "Less common controls moved behind an Advanced controls toggle"
    records.append({
        "category": category, "subcategory": section, "topic_id": topic_id, "name": title,
        "canonical_name": metadata[1], "alias": alias, "role": role,
        "route": route, "implementation": implementation, "family": metadata[5],
        "engine": engine_for(row), "dataset": dataset, "controls": controls,
        "interaction": metadata[8], "animation": metadata[9], "graphic": metadata[10],
        "math_status": "Engine-backed; per-topic numerical audit pending",
        "emulator_status": (f"Deeper emulator pass: {checked['controls_attempted']} control attempts, canvas tap changed={checked.get('canvas_tap_changed', 'unmeasured')}, scrolled, no crash"
                            if checked else f"Deeper retry required: {latest_errors[topic_id]}" if topic_id in latest_errors
                            else "Earlier open/render/scroll smoke test passed; controls pending"),
        "visual_qa_status": ("Blank capture flagged; recapture required" if "possibly_blank_body" in screening.get(topic_id, "") else
                             "Fresh topic-ID capture saved; visual review pending" if checked else
                             "Legacy capture saved; topic identity and visual QA pending") if topic_id in saved else "Screenshot missing",
        "performance_status": "Individual profile pending",
        "issues_found": finding, "issues_fixed": repair,
        "screenshot": saved.get(topic_id, ""),
    })

assert len(records) == 327
with (OUT / "catalog-inventory.json").open("w", encoding="utf-8") as handle:
    json.dump(records, handle, ensure_ascii=False, indent=2)

categories = collections.OrderedDict()
for row in records:
    categories.setdefault(row["category"], []).append(row)
repeated = collections.Counter(row["topic_id"] for row in records)
alias_rows = [row for row in records if row["alias"]]
missing_shots = [row for row in records if not row["screenshot"]]

def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")

lines = [
    "# Final Algorithm Visualization Audit",
    "",
    f"Generated {datetime.datetime.now().isoformat(timespec='minutes')} from the current `LearnCatalog.kt` runtime test output. This is a hardening inventory, not a completion claim.",
    "",
    f"**Catalog:** {len(categories)} categories, {sum(len({r['subcategory'] for r in rows}) for rows in categories.values())} subcategories, {len(records)} visible placements, {len(by_id)} unique IDs, {sum(n > 1 for n in repeated.values())} repeated IDs, {len(alias_rows)} alias placements.",
    f"**Capability:** {len(live_ids)} unique topics have a connected live trainer; concept topics have a simulation; the remaining topics have Visualization only in the current build.",
    f"**Screenshot files:** {len(saved)}/{len(by_id)} topic-ID filenames exist; legacy captures can be misidentified where title-based filenames collide. Fresh topic-ID capture and identity verification is complete only for deeper emulator passes. {len(missing_shots)} placements lack a file.",
    f"**Image screening:** {sum('possibly_blank_body' in flags for flags in screening.values())} saved capture(s) have a blank body and require recapture. Same-body fingerprints may reflect legitimate aliases or stale title-based captures; review individually.",
    "",
    "## Category inventory",
    "",
    "| Category | Subcategories | Placements | Unique IDs | Screenshot present |",
    "|---|---:|---:|---:|---:|",
]
for name, rows in categories.items():
    lines.append(f"| {cell(name)} | {len({r['subcategory'] for r in rows})} | {len(rows)} | "
                 f"{len({r['topic_id'] for r in rows})} | {sum(bool(r['screenshot']) for r in rows)} |")

lines += [
    "",
    "## Reused IDs and aliases",
    "",
    "| Placement | Topic ID | Canonical topic | Placement route |",
    "|---|---|---|---|",
]
for row in alias_rows:
    lines.append("| " + " | ".join(map(cell, [
        f"{row['category']} / {row['subcategory']} / {row['name']}", row["topic_id"],
        row["canonical_name"], row["route"]])) + " |")

lines += [
    "",
    "## Per-placement audit",
    "",
    "`Pending` fields require the deeper control, numerical, visual, and performance review in this pass. Prior screenshots and smoke tests alone do not prove every interaction.",
    "",
    "| Category | Subcategory | Topic ID | Name | Role | Route | Implementation | Family | Engine | Dataset/scenario | Controls | Interaction | Math status | Emulator status | Visual QA | Performance | Issues found | Issues fixed |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for row in records:
    keys = ("category", "subcategory", "topic_id", "name", "role", "route", "implementation",
            "family", "engine", "dataset", "controls", "interaction", "math_status", "emulator_status",
            "visual_qa_status", "performance_status", "issues_found", "issues_fixed")
    lines.append("| " + " | ".join(cell(row[key]) for key in keys) + " |")
DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(records)} placements, {len(by_id)} IDs, {len(alias_rows)} aliases, {len(saved)} screenshots, {len(missing_shots)} missing screenshot placements")
for row in missing_shots:
    print("MISSING", row["category"], row["subcategory"], row["name"])
