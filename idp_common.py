"""
Shared helpers for the IDP streaming pipelines (direct-API route and ai_query route).
Pure Python + Spark helpers only; no work happens at import time except installing pypdf if it is missing.
"""
import datetime
import heapq
import io
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future

from PIL import Image
from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType, StructField, StructType

try:
    import pypdf
except ImportError:                                   # also listed in the job environment dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdf"])
    import pypdf

GAP_PX = 20                      # composite rule from the customer job: pieces within 20 px are one figure
MAX_IMAGE_SIDE = 1568            # downscale crops before sending to the vision model
TARGET_SECONDS = 300


# ----------------------------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------------------------
def uuid7() -> str:
    millis = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (millis << 80) | (0x7 << 76) | (((rand >> 68) & 0xFFF) << 64) | (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


def clean(text) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def local_path(path: str) -> str:
    return path[len("dbfs:"):] if path.startswith("dbfs:") else path


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}

    def add(self, key, amount=1):
        with self.lock:
            self.values[key] = self.values.get(key, 0) + amount

    def get(self, key):
        return self.values.get(key, 0)


class TokenBucket:
    """Paces requests by OUTPUT tokens per minute; shared by every document in the process."""

    def __init__(self, tokens_per_minute: int):
        self.rate = tokens_per_minute / 60.0
        self.capacity = tokens_per_minute / 4.0          # 15-second burst
        self.tokens = self.capacity
        self.updated = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def take(self, amount: float) -> float:
        started = time.time()
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= amount or self.tokens >= self.capacity:
                    self.tokens -= amount
                    return time.time() - started
                needed = (amount - self.tokens) / self.rate
            time.sleep(min(max(needed, 0.05), 2.0))

    def correct(self, delta: float):
        with self.lock:
            self.tokens -= delta


class PriorityPool:
    """Thread pool whose queue is ordered by a priority key (we use file-arrival time: oldest document first)."""

    def __init__(self, workers: int, name: str):
        self.heap, self.counter = [], itertools.count()
        self.cv = threading.Condition()
        for index in range(workers):
            threading.Thread(target=self._worker, name=f"{name}-{index}", daemon=True).start()

    def submit(self, priority, fn, *args) -> Future:
        future = Future()
        with self.cv:
            heapq.heappush(self.heap, (priority, next(self.counter), future, fn, args))
            self.cv.notify()
        return future

    def _worker(self):
        while True:
            with self.cv:
                while not self.heap:
                    self.cv.wait()
                _, _, future, fn, args = heapq.heappop(self.heap)
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as failure:
                future.set_exception(failure)


# ----------------------------------------------------------------------------------------------
# Prompts and response schemas
# ----------------------------------------------------------------------------------------------
JSON_RULE = ('Return ONLY a JSON object (no prose, no code fences). Inside string values escape every double quote '
             'as \\" and every line break as \\n; prefer single quotes for quoted labels.')

FIGURE_PROMPT = """You are reviewing a figure cut from a model validation document for a model risk team.
Context: page {page}; section "{section}"; caption: "{caption}"; composite of {pieces} panel(s).

""" + JSON_RULE + """ Use exactly these keys:
- "aiInsight": markdown string with three parts:
    "## PART 1" - what the visual shows (type, axes, units, legend, labels, title).
    "## PART 2" - the data as a markdown table; prefix estimated values with "~"; say "no data" for non-chart images.
    "## PART 3" - 2-4 sentences interpreting it from a model-validation standpoint (trends, outliers, traceability gaps).
  Keep the whole aiInsight under 350 words.
- "aiSummary": one sentence describing the figure.
- "chartType": one of "bar_chart","line_chart","scatter_plot","pie_chart","histogram","heatmap","table_image","diagram","flowchart","photo","other".
- "classification": one of "CHART","DIAGRAM","TABLE_IMAGE","IMAGE","OTHER".
- "aiClassifyLabel": exactly one of [{labels}].
- "extractionConfidence": number 0-1 for how reliably you read the values.
- "confidenceLevel": "high", "medium" or "low".
"""

TABLE_PROMPT = """You are reviewing a table extracted (as HTML) from a model validation document for a model risk team.
Context: page {page}; section "{section}"; caption: "{caption}".

TABLE HTML:
{html}

""" + JSON_RULE + """ Use exactly these keys:
- "aiInsight": 2-5 sentences on what the table contains and what stands out (key rows/values, empty columns, anomalies).
- "aiSummary": one sentence describing the table.
- "aiClassifyLabel": exactly one of [{labels}].
- "extractionConfidence": number 0-1.
- "confidenceLevel": "high", "medium" or "low".
"""

DOC_PROMPT = """You are classifying a model validation document for a model risk team.
Title: "{title}". Pages: {pages}. Section headers: {sections}
Figure and table captions: {captions}
Opening text:
{text}

""" + JSON_RULE + """ Use exactly these keys:
- "aiSummary": 2-3 sentences summarizing the document.
- "aiInsight": 4-6 sentences on purpose, methods, key results and notable gaps a validator would flag.
- "aiClassifyLabel": exactly one of [{labels}].
- "documentType": short label such as "model_validation_report", "research_paper", "model_documentation", "other".
- "keyTopics": list of up to 8 short topic strings.
- "extractionConfidence": number 0-1.
- "confidenceLevel": "high", "medium" or "low".
"""


def _schema(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def make_schemas(labels):
    conf = {"type": "string", "enum": ["high", "medium", "low"]}
    label = {"type": "string", "enum": list(labels)}
    figure = _schema({
        "aiInsight": {"type": "string"}, "aiSummary": {"type": "string"},
        "chartType": {"type": "string", "enum": ["bar_chart", "line_chart", "scatter_plot", "pie_chart", "histogram",
                                                 "heatmap", "table_image", "diagram", "flowchart", "photo", "other"]},
        "classification": {"type": "string", "enum": ["CHART", "DIAGRAM", "TABLE_IMAGE", "IMAGE", "OTHER"]},
        "aiClassifyLabel": label, "extractionConfidence": {"type": "number"}, "confidenceLevel": conf})
    table = _schema({"aiInsight": {"type": "string"}, "aiSummary": {"type": "string"}, "aiClassifyLabel": label,
                     "extractionConfidence": {"type": "number"}, "confidenceLevel": conf})
    document = _schema({"aiSummary": {"type": "string"}, "aiInsight": {"type": "string"}, "aiClassifyLabel": label,
                        "documentType": {"type": "string"}, "keyTopics": {"type": "array", "items": {"type": "string"}},
                        "extractionConfidence": {"type": "number"}, "confidenceLevel": conf})
    return {"figure": figure, "table": table, "document": document}


def figure_prompt(job, label_text):
    return FIGURE_PROMPT.format(page=job["page"], section=job["section"][1] or "", caption=job["caption"],
                                pieces=job["pieces"], labels=label_text)


def table_prompt(job, label_text):
    return TABLE_PROMPT.format(page=job["page"], section=job["section"][1] or "", caption=job["caption"],
                               html=job["html"], labels=label_text)


def document_prompt(ctx, file_name, label_text):
    return DOC_PROMPT.format(title=ctx["title"] or file_name, pages=ctx["pages"], sections=json.dumps(ctx["sections"][:60]),
                             captions=json.dumps(ctx["captions"][:60]), text=ctx["text"][:6000], labels=label_text)


# ----------------------------------------------------------------------------------------------
# Tolerant JSON parsing of model replies
# ----------------------------------------------------------------------------------------------
def strip_reply(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)        # reasoning models
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[start:end + 1] if start >= 0 and end > start else cleaned


def repair_by_keys(text: str, keys) -> dict:
    found = []
    for key in keys:
        match = re.search(r'"%s"\s*:' % re.escape(key), text)
        if match:
            found.append((match.start(), match.end(), key))
    found.sort()
    result = {}
    for index, (_, value_start, key) in enumerate(found):
        value_end = found[index + 1][0] if index + 1 < len(found) else text.rfind("}")
        raw = text[value_start:value_end].strip().rstrip(",").strip()
        try:
            result[key] = json.loads(raw)
            continue
        except ValueError:
            pass
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            result[key] = raw[1:-1].replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
        else:
            result[key] = raw
    if len(result) < max(2, len(keys) // 2):
        raise ValueError(f"could not repair JSON; recovered keys {sorted(result)}")
    return result


def parse_reply(text: str, keys, stats: Stats = None) -> dict:
    cleaned = strip_reply(text)
    if not cleaned:
        raise ValueError("empty reply")
    try:
        return json.loads(cleaned)
    except ValueError:
        repaired = repair_by_keys(cleaned, keys)
        if stats:
            stats.add("json_repaired")
        return repaired


# ----------------------------------------------------------------------------------------------
# Split, parse normalisation, cut
# ----------------------------------------------------------------------------------------------
class ParseFailure(Exception):
    pass


def split_pdf(path, chunk_dir, pages_per_chunk, name_fn):
    """Writes chunk PDFs; returns (pdf_page_count, [(chunk_index, first_page_index, page_count, chunk_path)])."""
    reader = pypdf.PdfReader(path)
    total = len(reader.pages)
    os.makedirs(chunk_dir, exist_ok=True)
    chunks = []
    for index, first in enumerate(range(0, total, pages_per_chunk)):
        count = min(pages_per_chunk, total - first)
        chunk_path = f"{chunk_dir}/{name_fn(index, first, count)}"
        if total <= pages_per_chunk:
            shutil.copyfile(path, chunk_path)
        else:
            writer = pypdf.PdfWriter()
            for page_index in range(first, first + count):
                writer.add_page(reader.pages[page_index])
            with open(chunk_path, "wb") as handle:
                writer.write(handle)
        chunks.append((index, first, count, chunk_path))
    return total, chunks


def normalize_parsed(parsed_text, index, first_page, page_count):
    """ai_parse_document output for one chunk -> (elements, pages) with document-wide page ids; raises ParseFailure."""
    if not parsed_text:
        raise ParseFailure(f"chunk {index}: ai_parse_document returned nothing")
    parsed = json.loads(parsed_text)
    errors = [e for e in (parsed.get("error_status") or []) if e]
    document = parsed.get("document") or {}
    elements = document.get("elements") or []
    pages = document.get("pages") or []
    span = f"pages {first_page + 1}-{first_page + page_count}"
    if not elements:
        raise ParseFailure(f"chunk {index} ({span}): no elements returned"
                           + (f"; error_status {json.dumps(errors)[:400]}" if errors else ""))
    if page_count and len(pages) < page_count:
        raise ParseFailure(f"chunk {index} ({span}): parsed {len(pages)} of {page_count} pages")
    for page in pages:
        page["id"] = page["id"] + first_page
    for element in elements:
        element["key"] = (index, element.get("id", 0))
        for box in element.get("bbox") or []:
            box["page_id"] = box.get("page_id", 0) + first_page
    elements.sort(key=lambda e: e["key"])
    return elements, {page["id"]: page for page in pages}, errors


def section_context(elements, start=(None, None)):
    section_root, section_title = start
    context = {}
    for element in elements:
        content = clean(element.get("content"))
        if element.get("type") == "section_header" and content:
            section_title = content
            if section_root is None or (re.match(r"^\s*(\d+|[IVX]+)[.)]?\s", content) and not re.match(r"^\s*\d+\.\d", content)):
                section_root = content
        context[element["key"]] = (section_root, section_title)
    return context, (section_root, section_title)


def captions_on_pages(elements):
    captions = {}
    for element in elements:
        if element.get("type") == "caption" and element.get("bbox"):
            box = element["bbox"][0]
            captions.setdefault(box["page_id"], []).append((box["coord"], clean(element.get("content"))))
    return captions


def cut_chunk(elements, pages, artifact_dir):
    """Crops figures (merging pieces within GAP_PX) and collects tables for one chunk."""
    context, _ = section_context(elements)
    captions_by_page = captions_on_pages(elements)

    def nearest_caption(page_id, coord, below=True):
        best, best_distance = "", 10 ** 9
        for caption_coord, caption_text in captions_by_page.get(page_id, []):
            distance = (caption_coord[1] - coord[3]) if below else (coord[1] - caption_coord[3])
            if -GAP_PX <= distance <= 400 and distance < best_distance:
                best, best_distance = caption_text, distance
        return best

    def close(a, b):
        return not (a[2] + GAP_PX < b[0] or b[2] + GAP_PX < a[0] or a[3] + GAP_PX < b[1] or b[3] + GAP_PX < a[1])

    figure_pieces = {}
    for element in elements:
        if element.get("type") == "figure" and element.get("bbox"):
            box = element["bbox"][0]
            x0, y0, x1, y1 = box["coord"]
            if x1 > x0 and y1 > y0:
                figure_pieces.setdefault(box["page_id"], []).append(element)

    figure_jobs = []
    for page_id, pieces in sorted(figure_pieces.items()):
        page = pages.get(page_id)
        if not page or not page.get("image_uri"):
            continue
        boxes = [p["bbox"][0]["coord"] for p in pieces]
        parent = list(range(len(pieces)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if close(boxes[i], boxes[j]):
                    parent[find(i)] = find(j)
        groups = {}
        for index in range(len(pieces)):
            groups.setdefault(find(index), []).append(index)
        with Image.open(local_path(page["image_uri"])) as page_image:
            for number, members in enumerate(sorted(groups.values(), key=lambda m: min(boxes[i][1] for i in m)), start=1):
                merged = (min(boxes[i][0] for i in members), min(boxes[i][1] for i in members),
                          max(boxes[i][2] for i in members), max(boxes[i][3] for i in members))
                crop = page_image.crop(merged).convert("RGB")
                crop.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
                name = f"page_{page_id + 1:03d}_img_{number:03d}.png"
                with open(f"{artifact_dir}/{name}", "wb") as handle:
                    handle.write(buffer.getvalue())
                first = pieces[min(members)]
                confidences = [pieces[i].get("confidence") for i in members if pieces[i].get("confidence") is not None]
                figure_jobs.append({
                    "kind": "figure", "name": name, "page": page_id + 1, "bytes": buffer.getvalue(),
                    "pieces": len(members), "key": first["key"], "section": context.get(first["key"], (None, None)),
                    "caption": nearest_caption(page_id, merged, below=True),
                    "parse_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
                })

    table_jobs = []
    for element in elements:
        if element.get("type") == "table" and element.get("content"):
            box = (element.get("bbox") or [{"coord": [0, 0, 0, 0], "page_id": 0}])[0]
            table_jobs.append({
                "kind": "table", "page": box["page_id"] + 1, "html": element["content"][:30000], "key": element["key"],
                "section": context.get(element["key"], (None, None)),
                "caption": nearest_caption(box["page_id"], box["coord"], below=False) or nearest_caption(box["page_id"], box["coord"], below=True),
                "parse_confidence": element.get("confidence"),
            })
    return figure_jobs, table_jobs, sum(len(p) for p in figure_pieces.values())


def document_context(all_elements, page_total):
    title = next((clean(e.get("content")) for e in all_elements if e.get("type") == "title" and clean(e.get("content"))), None)
    sections = [clean(e.get("content")) for e in all_elements if e.get("type") == "section_header" and clean(e.get("content"))]
    captions = [c for page_captions in captions_on_pages(all_elements).values() for _, c in page_captions]
    text_parts = []
    for e in all_elements:
        if e.get("type") in ("text", "title", "section_header") and sum(len(p) for p in text_parts) < 6000:
            text_parts.append(clean(e.get("content")))
    return {"title": title, "sections": sections, "captions": captions, "text": "\n".join(text_parts), "pages": page_total}


# ----------------------------------------------------------------------------------------------
# Assemble IDP records
# ----------------------------------------------------------------------------------------------
def build_records(job_id, job_artifact_id, rel_root, file_name, size_bytes, page_total, doc_title,
                  figure_jobs, table_jobs, results, doc_ai, model_tags, all_elements):
    global_context, _ = section_context(all_elements)
    for job in figure_jobs + table_jobs:
        job["section"] = global_context.get(job["key"], job["section"])

    def base(name, fmt, mime, size):
        return {"jobId": job_id, "jobArtifactID": job_artifact_id, "dbxArtifactID": uuid7(),
                "parentArtifact": job_artifact_id, "artifactName": name, "artifactType": {"type": "OUTPUT"},
                "artifactFormat": {"format": fmt, "mimeType": mime}, "sizeInBytes": size}

    records = []
    doc_record = base(file_name, "DOCUMENT", "application/pdf", size_bytes)
    doc_record["metaData"] = {
        "pageNumber": None, "pageCount": page_total, "sectionRoot": doc_title, "sectionTitle": doc_title,
        "aiInsight": doc_ai.get("aiInsight"), "aiSummary": doc_ai.get("aiSummary"),
        "aiClassifyLabel": doc_ai.get("aiClassifyLabel"), "documentType": doc_ai.get("documentType"),
        "keyTopics": doc_ai.get("keyTopics"), "figureCount": len(figure_jobs), "tableCount": len(table_jobs),
        "parseConfidence": None, "extractionConfidence": doc_ai.get("extractionConfidence"),
        "confidenceLevel": doc_ai.get("confidenceLevel"), "extractionModel": model_tags["document"],
        "sourceDocument": file_name, "error": doc_ai.get("error"),
    }
    doc_record["artifactPath"] = f"{rel_root}/{file_name}"
    records.append(doc_record)

    ordered = sorted(figure_jobs + table_jobs, key=lambda j: (j["page"], j["kind"] != "table", j.get("name", "")))
    for job in ordered:
        ai = results.get(job["task_id"], {})
        if job["kind"] == "figure":
            record = base(job["name"], "IMAGE", "image/png", len(job["bytes"]))
            record["metaData"] = {
                "pageNumber": job["page"], "sectionRoot": job["section"][0], "sectionTitle": job["section"][1],
                "aiInsight": ai.get("aiInsight"), "aiSummary": ai.get("aiSummary"), "chartType": ai.get("chartType"),
                "classification": ai.get("classification"), "aiClassifyLabel": ai.get("aiClassifyLabel"),
                "isCompositePanel": job["pieces"] > 1, "parseConfidence": job["parse_confidence"],
                "extractionConfidence": ai.get("extractionConfidence"), "extractionModel": model_tags["figure"],
                "confidenceLevel": ai.get("confidenceLevel"), "sourceDocument": file_name, "error": ai.get("error"),
            }
            record["artifactPath"] = f"{rel_root}/{job['name']}"
        else:
            record = base(file_name, "DOCUMENT", "application/pdf", size_bytes)
            record["metaData"] = {
                "pageNumber": job["page"], "sectionRoot": job["section"][0], "sectionTitle": job["section"][1],
                "aiInsight": ai.get("aiInsight"), "aiSummary": ai.get("aiSummary"),
                "aiClassifyLabel": ai.get("aiClassifyLabel"), "parseConfidence": job["parse_confidence"],
                "extractionConfidence": ai.get("extractionConfidence"), "extractionModel": model_tags["table"],
                "confidenceLevel": ai.get("confidenceLevel"), "sourceDocument": file_name, "error": ai.get("error"),
            }
            record["artifactPath"] = f"{rel_root}/{file_name}"
        records.append(record)
    return records


# ----------------------------------------------------------------------------------------------
# Delta writes (explicit schemas)
# ----------------------------------------------------------------------------------------------
ARTIFACT_SCHEMA = StructType([
    StructField("artifact_format", StringType()), StructField("artifact_name", StringType()),
    StructField("job_artifact_id", StringType()), StructField("page_number", LongType()),
    StructField("record_json", StringType()), StructField("source_document", StringType()),
    StructField("pipeline", StringType())])

METRIC_TYPES = {
    "processed_at": StringType(), "pipeline": StringType(), "job_id": StringType(), "run_id": StringType(),
    "job_artifact_id": StringType(), "source_document": StringType(), "size_bytes": LongType(),
    "pdf_page_count": LongType(), "pages": LongType(), "chunks": LongType(), "pages_per_chunk": LongType(),
    "parse_status": StringType(), "parse_error": StringType(), "chunk_timings": StringType(),
    "figure_pieces": LongType(), "figures": LongType(), "tables": LongType(),
    "vision_endpoint": StringType(), "text_endpoint": StringType(), "vision_calls": LongType(), "text_calls": LongType(),
    "model_calls": LongType(), "attempts": LongType(), "rate_limited_429": LongType(), "timeouts": LongType(),
    "server_errors": LongType(), "network_errors": LongType(), "failed_items": LongType(),
    "json_repaired": LongType(), "json_retried": LongType(), "structured_output": BooleanType(),
    "input_tokens": LongType(), "output_tokens": LongType(), "vision_output_tokens": LongType(),
    "text_output_tokens": LongType(), "avg_otpm_during_enrich": LongType(), "threads": LongType(),
    "otpm_budget": LongType(), "docs_in_flight": LongType(),
    "trigger_wait_s": DoubleType(), "queue_wait_s": DoubleType(), "intake_s": DoubleType(), "split_s": DoubleType(),
    "parse_s": DoubleType(), "cut_s": DoubleType(), "enrich_s": DoubleType(), "vision_enrich_s": DoubleType(),
    "text_enrich_s": DoubleType(), "assemble_s": DoubleType(), "parse_to_last_call_s": DoubleType(),
    "processing_s": DoubleType(), "end_to_end_from_arrival_s": DoubleType(), "met_5_min_processing": BooleanType(),
    "met_5_min_from_arrival": BooleanType(), "output_json": StringType(), "artifacts_written": LongType(),
    "persist_error": StringType()}
METRIC_SCHEMA = StructType([StructField(name, kind) for name, kind in METRIC_TYPES.items()])

CALL_LOG_SCHEMA = StructType([
    StructField("pipeline", StringType()), StructField("job_artifact_id", StringType()),
    StructField("source_document", StringType()), StructField("kind", StringType()),
    StructField("artifact_name", StringType()), StructField("page_number", LongType()),
    StructField("endpoint", StringType()), StructField("submitted_at", DoubleType()),
    StructField("started_at", DoubleType()), StructField("ended_at", DoubleType()),
    StructField("queue_s", DoubleType()), StructField("pacer_wait_s", DoubleType()), StructField("call_s", DoubleType()),
    StructField("attempts", LongType()), StructField("status", StringType()), StructField("input_tokens", LongType()),
    StructField("output_tokens", LongType()), StructField("error", StringType())])

_CASTS = {LongType: int, DoubleType: float, BooleanType: bool, StringType: str}


def _row(schema, values: dict):
    out = []
    for field in schema.fields:
        value = values.get(field.name)
        out.append(None if value is None else _CASTS[type(field.dataType)](value))
    return tuple(out)


_RETRYABLE = ("DELTA_METADATA_CHANGED", "MetadataChangedException", "ConcurrentAppend", "DELTA_CONCURRENT",
              "ConcurrentTransaction", "ProtocolChangedException")


def _with_retry(action, what, attempts=6):
    """Two jobs (and several documents) write the same Delta tables; retry commit conflicts with backoff."""
    for attempt in range(attempts):
        try:
            return action()
        except Exception as failure:
            text = f"{type(failure).__name__}: {failure}"
            if attempt == attempts - 1 or not any(marker in text for marker in _RETRYABLE):
                raise
            wait = min(2 ** attempt, 20) + random.uniform(0, 1)
            print(f"[{what}] commit conflict, retrying in {wait:.1f}s ({text[:120]})")
            time.sleep(wait)


def write_metrics(spark, table, metrics):
    frame = spark.createDataFrame([_row(METRIC_SCHEMA, metrics)], schema=METRIC_SCHEMA)
    _with_retry(lambda: frame.write.mode("append").option("mergeSchema", "true").saveAsTable(table), table)


def write_artifacts(spark, table, records, job_artifact_id, file_name, pipeline):
    rows = [(r["artifactFormat"]["format"], r["artifactName"], job_artifact_id, r["metaData"].get("pageNumber"),
             json.dumps(r), file_name, pipeline) for r in records]
    frame = spark.createDataFrame(rows, schema=ARTIFACT_SCHEMA)
    _with_retry(lambda: frame.write.mode("append").option("mergeSchema", "true").saveAsTable(table), table)
    return len(rows)


def write_call_log(spark, table, rows):
    if rows:
        frame = spark.createDataFrame([_row(CALL_LOG_SCHEMA, r) for r in rows], schema=CALL_LOG_SCHEMA)
        _with_retry(lambda: frame.write.mode("append").option("mergeSchema", "true").saveAsTable(table), table)


# ----------------------------------------------------------------------------------------------
# File discovery: Auto Loader with Trigger.AvailableNow in a loop (serverless does not allow ProcessingTime)
# ----------------------------------------------------------------------------------------------
def claim(path, processing_dir):
    """Move a new file out of incoming/ so it is processed exactly once. Returns (claimed_path, arrived_at) or None."""
    source = local_path(path)
    try:
        arrived_at = os.stat(source).st_mtime
    except FileNotFoundError:
        return None
    job_artifact_id = uuid7()
    target = f"{processing_dir}/{job_artifact_id}__{os.path.basename(source)}"
    try:
        shutil.move(source, target)
    except FileNotFoundError:
        return None
    return {"path": target, "file_name": os.path.basename(source), "arrived_at": arrived_at,
            "claimed_at": time.time(), "job_artifact_id": job_artifact_id}


def recover_processing(processing_dir):
    """Files left in processing/ by a previous run (crash/restart) are queued again."""
    claimed = []
    for name in sorted(os.listdir(processing_dir)):
        if "__" in name and name.lower().endswith(".pdf"):
            job_artifact_id, file_name = name.split("__", 1)
            path = f"{processing_dir}/{name}"
            claimed.append({"path": path, "file_name": file_name, "arrived_at": os.stat(path).st_mtime,
                            "claimed_at": time.time(), "job_artifact_id": job_artifact_id})
    return claimed


def discover_forever(spark, incoming, checkpoint, processing_dir, on_claimed, poll_seconds, queue_table=None,
                     mode="autoloader", max_files_per_trigger=200):
    """Find new PDFs and hand them to on_claimed(list_of_claims) in THIS (driver) process.

    mode="autoloader": Auto Loader (Trigger.AvailableNow, repeated every poll_seconds) records new files in a Delta
    queue table; the driver reads rows it has not seen yet and claims them. No foreachBatch is used: on serverless
    (Spark Connect) a foreachBatch function is pickled and runs in a separate server-side Python process, so it cannot
    reach the driver's thread pools or import idp_common - the likely reason v1 of this loop failed at start-up.
    mode="listing": plain directory listing of incoming/ every poll_seconds (fallback, no Spark streaming involved).
    """
    if mode == "listing":
        print(f"[discovery/listing] watching {incoming}")
        while True:
            started = time.time()
            names = sorted(n for n in os.listdir(incoming) if n.lower().endswith(".pdf"))
            claimed = [c for c in (claim(f"{incoming}/{n}", processing_dir) for n in names) if c]
            if claimed:
                print(f"[discovery/listing] claimed {len(claimed)} file(s)")
                on_claimed(claimed)
            time.sleep(max(0.0, poll_seconds - (time.time() - started)))

    from pyspark.sql import functions as F
    if not queue_table:
        raise ValueError("queue_table is required for mode='autoloader'")
    stream = (spark.readStream.format("cloudFiles")
              .option("cloudFiles.format", "binaryFile")
              .option("pathGlobFilter", "*.pdf")
              .option("cloudFiles.includeExistingFiles", "true")
              .option("cloudFiles.maxFilesPerTrigger", str(max_files_per_trigger))
              .load(incoming)
              .select("path", "modificationTime", "length")      # metadata only; bytes are read later
              .withColumn("discovered_at", F.current_timestamp()))
    watermark, seen = None, set()
    print(f"[discovery/autoloader] queue table {queue_table}, checkpoint {checkpoint}")
    while True:
        started = time.time()
        query = (stream.writeStream.format("delta")
                 .option("checkpointLocation", checkpoint)
                 .trigger(availableNow=True)
                 .toTable(queue_table))
        query.awaitTermination()
        frame = spark.read.table(queue_table)
        if watermark is not None:
            frame = frame.where(F.col("discovered_at") > F.lit(watermark))
        rows = frame.select("path", "discovered_at").collect()
        if rows:
            watermark = max(r.discovered_at for r in rows)
        fresh = [r.path for r in rows if r.path not in seen]
        seen.update(fresh)
        claimed = [c for c in (claim(path, processing_dir) for path in fresh) if c]
        if claimed:
            print(f"[discovery/autoloader] claimed {len(claimed)} file(s)")
            on_claimed(claimed)
        time.sleep(max(0.0, poll_seconds - (time.time() - started)))
