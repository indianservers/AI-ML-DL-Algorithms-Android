"""Exercise each catalog visualization through the app on an Android emulator.

This is a debug QA harness, not part of the APK. It records what it could observe;
an attempted tap is not automatically reported as a successful semantic change.
"""
import argparse
import csv
import io
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
ADB = r"C:\Users\saisa\AppData\Local\Android\Sdk\platform-tools\adb.exe"
PACKAGE = "com.indianservers.ai_ml_dl_algorithms"
ACTIVITY = PACKAGE + "/.MainActivity"
INVENTORY = json.loads((ROOT / "qa/catalog-inventory.json").read_text(encoding="utf-8"))
EXCLUDE = {"Learn", "Visualization", "Train & Inference", "Quiz", "Home", "  Home", "‹", "Back",
           "Load CSV", "Import CSV", "Export", "Share", "Profile", "Settings"}


def center(bounds):
    x0, y0, x1, y1 = map(int, re.findall(r"\d+", bounds))
    return (x0 + x1) // 2, (y0 + y1) // 2


class Device:
    def __init__(self, serial, out):
        self.serial = serial
        self.out = out

    def adb(self, *args, read=False):
        result = subprocess.run([ADB, "-s", self.serial, *args], check=True,
                                stdout=subprocess.PIPE if read else subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        return result.stdout

    def hierarchy(self):
        for attempt in range(12):
            try:
                self.adb("shell", "rm", "-f", "/sdcard/final-qa.xml")
                self.adb("shell", "uiautomator", "dump", "/sdcard/final-qa.xml")
                return ET.fromstring(self.adb("shell", "cat", "/sdcard/final-qa.xml", read=True))
            except (subprocess.CalledProcessError, ET.ParseError, RuntimeError):
                if attempt == 11:
                    raise
                time.sleep(.5)

    def tap(self, x, y):
        self.adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x0=500, y0=1850, x1=500, y1=620, duration=320):
        self.adb("shell", "input", "swipe", str(x0), str(y0), str(x1), str(y1), str(duration))

    def screenshot(self, path, verify=True):
        data = self.adb("exec-out", "screencap", "-p", read=True)
        if verify:
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert("RGB")
                body = image.crop((int(.04 * image.width), int(.21 * image.height),
                                   int(.96 * image.width), int(.88 * image.height))).resize((96, 128))
                if sum(ImageStat.Stat(body).stddev) / 3 < 10:
                    raise RuntimeError("Emulator screenshot body is blank or graphics-corrupted")
        path.write_bytes(data)

    def nodes(self, root, value):
        return [node for node in root.iter("node") if node.get("text") == value]

    def search_result(self, root, record):
        for node in root.iter("node"):
            if node.get("clickable") != "true":
                continue
            visible_text = [child.get("text") for child in node.iter("node")]
            if record["name"] in visible_text and record["subcategory"] in visible_text:
                location = center(node.get("bounds", ""))
                # Keep the search field itself out of result matching when a
                # topic and its subcategory share the same text (Bagging).
                if 900 < location[1] < 2200:
                    return location
        return None

    def open_topic(self, record):
        self.adb("shell", "am", "force-stop", PACKAGE)
        self.adb("shell", "am", "start", "-n", ACTIVITY)
        for attempt in range(8):
            time.sleep(.35)
            root = self.hierarchy()
            matches = self.nodes(root, "Search algorithms, topics, or concepts")
            if matches:
                self.tap(*center(matches[0].get("bounds")))
                break
        else:
            raise RuntimeError("Home search unavailable")
        title = record["name"]
        query = ("Time-Series" if record["category"] == "Time-Series Algorithms" else
                 title.split("'")[0].split("–")[0].strip())
        query = query.encode("ascii", errors="ignore").decode().strip()
        self.adb("shell", "input", "text", query.replace(" ", "%s"))
        self.adb("shell", "input", "keyevent", "4")
        # Some emulator IME builds occasionally duplicate the first injected
        # character. Verify what Compose actually received before searching.
        for correction in range(3):
            root = self.hierarchy()
            fields = [node for node in root.iter("node") if node.get("class") == "android.widget.EditText"]
            actual = fields[0].get("text", "") if fields else ""
            if actual == query:
                break
            if actual == query[0] + query:
                self.tap(*center(fields[0].get("bounds")))
                self.adb("shell", "input", "keyevent", "122")  # move to start
                self.adb("shell", "input", "keyevent", "112")  # forward delete
            elif actual == "":
                self.tap(*center(fields[0].get("bounds")))
                self.adb("shell", "input", "text", query.replace(" ", "%s"))
            else:
                raise RuntimeError(f"Search input mismatch: expected {query!r}, got {actual!r}")
            self.adb("shell", "input", "keyevent", "4")
        else:
            raise RuntimeError(f"Search input could not be corrected: {actual!r}")
        target = None
        for attempt in range(12):
            root = self.hierarchy()
            target = self.search_result(root, record)
            if target:
                break
            self.swipe(960, 1900, 960, 1050, 290)
            time.sleep(.12)
        if target is None:
            raise RuntimeError("section-specific search result unavailable")
        self.tap(*target)
        time.sleep(.25)
        tabs = []
        for attempt in range(6):
            root = self.hierarchy()
            tabs = self.nodes(root, "Visualization")
            if tabs:
                break
            # Some IMEs consume the first result tap solely to dismiss the
            # keyboard. Retry the same visible result after hiding the IME.
            if attempt == 2:
                self.adb("shell", "input", "keyevent", "4")
                root = self.hierarchy()
                retry_target = self.search_result(root, record)
                if retry_target:
                    self.tap(*retry_target)
            time.sleep(.2)
        if not tabs:
            raise RuntimeError("Visualization tab unavailable")
        self.tap(*center(tabs[0].get("bounds")))
        for attempt in range(6):
            time.sleep(.25)
            root = self.hierarchy()
            if self.visualization_visible(root, title):
                break
        else:
            raise RuntimeError("Wrong topic or lesson still displayed")
        if record["route"] == "StageTwoNative" and record["subcategory"] == "Clustering":
            if not self.nodes(root, record["graphic"]):
                raise RuntimeError("Clustering body does not match the selected algorithm")
        return root

    def controls(self, root):
        found = []
        for node in root.iter("node"):
            if node.get("package") != PACKAGE or node.get("class") == "android.widget.EditText":
                continue
            bounds = node.get("bounds", "")
            if not bounds:
                continue
            x, y = center(bounds)
            if not (35 < x < 1045 and 420 < y < 2100):
                continue
            kind = node.get("class", "")
            if kind == "android.widget.SeekBar":
                found.append(("slider", node.get("content-desc") or node.get("text") or "slider", bounds))
                continue
            if node.get("clickable") != "true":
                continue
            if any(child.get("clickable") == "true" for child in node.iter("node") if child is not node):
                continue
            label = node.get("text") or node.get("content-desc") or " ".join(
                part.get("text", "") for part in node.iter("node") if part.get("text"))
            label = " ".join(label.split())
            if (not label or any(excluded in label for excluded in EXCLUDE)
                    or label.startswith("Load ") or label.startswith("Export ")):
                continue
            found.append(("tap", label, bounds))
        return found

    def visualization_visible(self, root, title):
        return (bool(self.nodes(root, title)) and not self.nodes(root, "5-Page Lesson")
                and len(self.nodes(root, "Train & Inference")) == 1)

    def crash(self):
        log = self.adb("logcat", "-d", "-t", "500", "AndroidRuntime:E", "*:S", read=True)
        return b"FATAL EXCEPTION" in log and PACKAGE.encode() in log


def run_record(device, record, capture, max_frames, reopen):
    start = time.perf_counter()
    result = {key: record[key] for key in ("topic_id", "name", "category", "subcategory")}
    result.update(opened=False, rendered=False, scrolled=False, reopened=False, controls_seen=0,
                  controls_attempted=0, control_labels="", slider_count=0, canvas_drag=False,
                  canvas_tap_attempted=False, canvas_tap_changed=False, crash=False, seconds=0.0, error="")
    labels = []
    try:
        device.adb("logcat", "-c")
        root = device.open_topic(record)
        result["opened"] = result["rendered"] = True
        if capture:
            device.screenshot(capture)
        if record["category"] == "Computer Vision":
            before = [node.get("text", "") for node in root.iter("node") if node.get("text", "").startswith("Selected ")]
            device.tap(830, 900)
            result["canvas_tap_attempted"] = True
            after_root = device.hierarchy()
            after = [node.get("text", "") for node in after_root.iter("node") if node.get("text", "").startswith("Selected ")]
            result["canvas_tap_changed"] = before != after
        elif record["subcategory"] == "Graph Neural Networks":
            before = [node.get("text", "") for node in root.iter("node")
                      if node.get("text", "").startswith("Node ") and "receives" in node.get("text", "")]
            device.tap(216, 1000)
            result["canvas_tap_attempted"] = True
            after_root = device.hierarchy()
            after = [node.get("text", "") for node in after_root.iter("node")
                     if node.get("text", "").startswith("Node ") and "receives" in node.get("text", "")]
            result["canvas_tap_changed"] = before != after
        signatures = set()
        for frame in range(max_frames):
            root = device.hierarchy()
            if not device.visualization_visible(root, record["name"]):
                raise RuntimeError(f"Visualization was left before frame {frame}")
            signature = tuple(node.get("text") for node in root.iter("node") if node.get("text"))
            if signature in signatures:
                break
            signatures.add(signature)
            controls = device.controls(root)
            result["controls_seen"] += len(controls)
            for kind, label, bounds in controls:
                try:
                    x0, y0, x1, y1 = map(int, re.findall(r"\d+", bounds))
                    if kind == "slider":
                        device.tap(x0 + int((x1-x0)*.73), (y0+y1)//2)
                        result["slider_count"] += 1
                    else:
                        device.tap((x0+x1)//2, (y0+y1)//2)
                    labels.append(label)
                    result["controls_attempted"] += 1
                    time.sleep(.05)
                except subprocess.CalledProcessError:
                    labels.append("FAILED:" + label)
            if frame == 0 and ("drag" in record["interaction"].lower() or "samples" in record["controls"].lower()):
                device.swipe(420, 1330, 620, 1250, 400)
                result["canvas_drag"] = True
            device.swipe()
            result["scrolled"] = True
            time.sleep(.1)
            root = device.hierarchy()
            if not device.visualization_visible(root, record["name"]):
                raise RuntimeError(f"Visualization was left after frame {frame}; controls: {labels[-len(controls):]}")
        if capture:
            device.screenshot(capture.with_name(capture.stem + "-bottom.png"), verify=False)
        if reopen:
            device.adb("shell", "input", "keyevent", "4")
            time.sleep(.18)
            device.adb("shell", "input", "keyevent", "4")
            time.sleep(.18)
            root = device.open_topic(record)
            result["reopened"] = bool(device.nodes(root, record["name"]))
        result["crash"] = device.crash()
    except Exception as exc:
        result["error"] = str(exc)
        try:
            result["crash"] = device.crash()
        except Exception:
            pass
    result["control_labels"] = "; ".join(labels[:120])
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=320)
    parser.add_argument("--out", default="qa/final-emulator-results.csv")
    parser.add_argument("--title", action="append", default=[])
    parser.add_argument("--id", action="append", default=[], help="Run exact topic IDs, for failed-case retries")
    parser.add_argument("--failed", action="store_true", help="Retry IDs with a saved error and no saved passing deep run")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--no-reopen", action="store_true", help="Skip second app launch; navigation was covered by the earlier route sweep")
    args = parser.parse_args()
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical = list(dict((record["topic_id"], record) for record in INVENTORY[::-1]).values())[::-1]
    assert len(canonical) == 320
    selected = canonical[args.start:args.start+args.limit]
    if args.title:
        selected = [record for record in canonical if record["name"] in args.title]
    if args.id:
        selected = [record for record in canonical if record["topic_id"] in args.id]
    if args.failed:
        failed, passed = set(), set()
        for source in (ROOT / "qa").glob("final-emulator-*.csv"):
            with source.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["opened"] == "True" and row["rendered"] == "True" and row["scrolled"] == "True" and row["crash"] == "False" and not row["error"]:
                        passed.add(row["topic_id"])
                    elif row["error"]:
                        failed.add(row["topic_id"])
        selected = [record for record in canonical if record["topic_id"] in failed - passed]
    device = Device(args.serial, destination.parent)
    rows = []
    for index, record in enumerate(selected, 1):
        capture = ROOT / "qa/screenshots/visualizations" / f"{record['topic_id']}.png"
        result = run_record(device, record, capture, args.frames, not args.no_reopen)
        rows.append(result)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=result.keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"{index}/{len(selected)} {record['category']} / {record['name']}: "
              f"open={result['opened']} controls={result['controls_attempted']} "
              f"reopen={result['reopened']} crash={result['crash']} error={result['error']}", flush=True)


if __name__ == "__main__":
    main()
