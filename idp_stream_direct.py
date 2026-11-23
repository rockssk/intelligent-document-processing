"""
IDP - DIRECT API route, continuously running (v3, 2026-11-20).

* Always-on: Auto Loader (Trigger.AvailableNow every --poll_seconds) claims new PDFs from <volume>/incoming/ into
  processing/ and hands them to a document pool, so up to --max_parallel_docs documents are processed at once.
* Models: figures -> --vision_endpoint (Claude Sonnet 4.6); tables + the document-level call -> --text_endpoint
  (Qwen3.5 122B A10B, reasoning off). Each endpoint has ONE workspace-wide output-token pacer shared by all documents.
* One shared call pool, ordered oldest-file-first, so a big document cannot starve documents that arrived earlier.
* Parse: PDF split into --pages_per_chunk page chunks, parsed --parse_parallelism at a time (shared across documents);
  a chunk's figures/tables go to the models as soon as that chunk is parsed. Per-chunk timings are recorded.
* Every model call is logged to <catalog>.<schema>.idp_call_log; per-document metrics to idp_run_metrics.

Run as a serverless spark_python_task in a job with a CONTINUOUS schedule (the job restarts it if it stops).
"""
import argparse
import base64
import datetime
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--volume", default="/Volumes/<catalog>/<schema>/<volume>")
parser.add_argument("--catalog", default="<catalog>")
parser.add_argument("--schema", default="<schema>")
parser.add_argument("--code_dir", default="/Workspace/Shared/idp-sample")
parser.add_argument("--vision_endpoint", default="databricks-claude-sonnet-4-6")
parser.add_argument("--text_endpoint", default="databricks-qwen35-122b-a10b")
parser.add_argument("--text_reasoning_effort", default="none", help="'none' turns Qwen thinking off; '' to omit")
parser.add_argument("--vision_otpm", type=int, default=60000, help="output tokens/min for the vision endpoint (whole job)")
parser.add_argument("--text_otpm", type=int, default=60000, help="output tokens/min for the text endpoint (whole job)")
parser.add_argument("--threads", type=int, default=64, help="concurrent model calls across all documents")
parser.add_argument("--max_parallel_docs", type=int, default=8)
parser.add_argument("--pages_per_chunk", type=int, default=10)
parser.add_argument("--parse_parallelism", type=int, default=32, help="chunks parsed at once across all documents")
parser.add_argument("--poll_seconds", type=int, default=5)
parser.add_argument("--discovery", default="autoloader", choices=["autoloader", "listing"])
parser.add_argument("--call_timeout", type=int, default=60)
parser.add_argument("--max_attempts", type=int, default=6)
parser.add_argument("--structured", default="true")
parser.add_argument("--labels", default="stress_testing,benchmark_testing,model_outputs,other")
parser.add_argument("--job_id", default="")
parser.add_argument("--run_id", default="")
args, _ = parser.parse_known_args()

sys.path.insert(0, args.code_dir)
import idp_common as C                                     # noqa: E402
from databricks.sdk import WorkspaceClient                      # noqa: E402
from pyspark.sql import SparkSession                            # noqa: E402

PIPELINE = "direct_stream"
VOL = args.volume.rstrip("/")
INCOMING, PROCESSING, FAILED = f"{VOL}/incoming", f"{VOL}/processing", f"{VOL}/failed"
ARTIFACTS, OUTPUT, PROCESSED = f"{VOL}/artifacts", f"{VOL}/output", f"{VOL}/processed"
CHECKPOINT = f"{VOL}/_checkpoints/direct_stream_v2"
TABLE = lambda name: f"{args.catalog}.{args.schema}.{name}"     # noqa: E731
LABELS = [label.strip() for label in args.labels.split(",") if label.strip()]
LABEL_TEXT = ", ".join(f'"{label}"' for label in LABELS)
SCHEMAS = C.make_schemas(LABELS)
MODEL_TAGS = {"figure": f"{args.vision_endpoint} (direct API)", "table": f"{args.text_endpoint} (direct API)",
              "document": f"{args.text_endpoint} (direct API)"}
FIGURE_MAX_TOKENS, TABLE_MAX_TOKENS, DOC_MAX_TOKENS = 3500, 1500, 2000
EXPECTED = {"figure": 1500, "table": 500, "document": 800}

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
HOST = workspace.config.host.rstrip("/")
for folder in (INCOMING, PROCESSING, FAILED, ARTIFACTS, OUTPUT, PROCESSED, CHECKPOINT):
    os.makedirs(folder, exist_ok=True)

ENDPOINTS = {
    "vision": {"name": args.vision_endpoint, "bucket": C.TokenBucket(args.vision_otpm), "structured": args.structured.lower() == "true",
               "extra": {}},
    "text": {"name": args.text_endpoint, "bucket": C.TokenBucket(args.text_otpm), "structured": args.structured.lower() == "true",
             "extra": {"reasoning_effort": args.text_reasoning_effort} if args.text_reasoning_effort else {}},
}
CALL_POOL = C.PriorityPool(args.threads, "call")
PARSE_POOL = ThreadPoolExecutor(max_workers=args.parse_parallelism, thread_name_prefix="parse")
DOC_POOL = ThreadPoolExecutor(max_workers=args.max_parallel_docs, thread_name_prefix="doc")
IN_FLIGHT = {"count": 0, "lock": threading.Lock()}
_local = threading.local()


def http_session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def reply_text(body):
    content = body["choices"][0]["message"].get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content
                          if isinstance(p, dict) and p.get("type", "text") == "text")
    return content or ""


def call_model(route, messages, max_tokens, expected, stats, schema, log):
    """One logical call (own retry policy). Fills `log` with timings; returns reply text."""
    endpoint = ENDPOINTS[route]
    url = f"{HOST}/serving-endpoints/{endpoint['name']}/invocations"
    log.update(endpoint=endpoint["name"], started_at=time.time(), attempts=0, pacer_wait_s=0.0)
    stats.add(f"{route}_calls")
    last_error, attempt = "no attempt made", 0
    while attempt < args.max_attempts:
        log["pacer_wait_s"] += endpoint["bucket"].take(expected)
        log["attempts"] += 1
        stats.add("attempts")
        wait = min(2 ** attempt, 20)
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, **endpoint["extra"]}
        use_schema = schema is not None and endpoint["structured"]
        if use_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "artifact", "schema": schema, "strict": True}}
        try:
            response = http_session().post(url, headers={**workspace.config.authenticate(), "Content-Type": "application/json"},
                                           json=body, timeout=(10, args.call_timeout))
        except requests.Timeout:
            stats.add("timeouts")
            last_error, wait = f"timeout after {args.call_timeout}s", 1
        except requests.RequestException as failure:
            stats.add("network_errors")
            last_error = f"{type(failure).__name__}: {str(failure)[:200]}"
        else:
            if response.status_code == 200:
                payload = response.json()
                usage = payload.get("usage") or {}
                out_tokens, in_tokens = int(usage.get("completion_tokens") or 0), int(usage.get("prompt_tokens") or 0)
                stats.add("output_tokens", out_tokens)
                stats.add("input_tokens", in_tokens)
                stats.add(f"{route}_output_tokens", out_tokens)
                endpoint["bucket"].correct(out_tokens - expected)
                log.update(status="ok", input_tokens=in_tokens, output_tokens=out_tokens)
                return reply_text(payload)
            text = response.text
            if response.status_code == 400 and use_schema and re.search(r"response_format|json_schema|schema", text, re.I):
                endpoint["structured"] = False
                print(f"[{endpoint['name']}] structured output rejected, using plain JSON prompts: {text[:200]}")
                endpoint["bucket"].correct(-expected)
                continue
            if response.status_code == 400 and endpoint["extra"] and re.search(r"reasoning", text, re.I):
                endpoint["extra"] = {}
                print(f"[{endpoint['name']}] reasoning_effort rejected, sending without it: {text[:200]}")
                endpoint["bucket"].correct(-expected)
                continue
            if response.status_code == 429:
                stats.add("rate_limited")
                retry_after = response.headers.get("Retry-After")
                if not retry_after:
                    try:
                        retry_after = (response.json().get("error") or {}).get("retry_after")
                    except ValueError:
                        retry_after = None
                wait = float(retry_after) if retry_after else min(2 ** attempt, 20)
                last_error = "429 rate limited"
            elif response.status_code in (500, 502, 503, 504):
                stats.add("server_errors")
                last_error = f"HTTP {response.status_code}"
            else:
                log.update(status=f"http_{response.status_code}")
                raise RuntimeError(f"HTTP {response.status_code}: {text[:300]}")
        attempt += 1
        if time.time() - log["started_at"] + wait > 240:
            break
        time.sleep(min(wait, 30) + random.uniform(0, 1))
    log.update(status="gave_up")
    raise RuntimeError(f"gave up after {log['attempts']} attempts: {last_error}")


def ask_json(route, messages, max_tokens, kind, stats, log):
    schema = SCHEMAS[kind]
    keys = list(schema["properties"])
    text = call_model(route, messages, max_tokens, EXPECTED[kind], stats, schema, log)
    try:
        return C.parse_reply(text, keys, stats)
    except ValueError as first_error:
        stats.add("json_retried")
        retry = messages + [{"role": "assistant", "content": (text or "")[:8000] or "(empty)"},
                            {"role": "user", "content": "That reply was not valid JSON (" + str(first_error)[:120] +
                             "). Send the same answer again as one valid JSON object with keys " + json.dumps(keys) +
                             ". Escape double quotes inside strings as \\\" and line breaks as \\n. No other text."}]
        text = call_model(route, retry, max_tokens, EXPECTED[kind], stats, schema, log)
        return C.parse_reply(text, keys, stats)


def run_task(job, stats, ctx_holder):
    log = {"kind": job["kind"], "artifact_name": job.get("name") or job["kind"], "page_number": job.get("page"),
           "submitted_at": job["submitted_at"], "status": "error"}
    try:
        if job["kind"] == "figure":
            messages = [{"role": "user", "content": [
                {"type": "text", "text": C.figure_prompt(job, LABEL_TEXT)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(job["bytes"]).decode()}}]}]
            result = ask_json("vision", messages, FIGURE_MAX_TOKENS, "figure", stats, log)
        elif job["kind"] == "table":
            result = ask_json("text", [{"role": "user", "content": C.table_prompt(job, LABEL_TEXT)}], TABLE_MAX_TOKENS,
                              "table", stats, log)
        else:
            result = ask_json("text", [{"role": "user", "content": C.document_prompt(ctx_holder["ctx"], ctx_holder["file_name"], LABEL_TEXT)}],
                              DOC_MAX_TOKENS, "document", stats, log)
        return result, log
    except Exception as failure:
        log["error"] = f"{type(failure).__name__}: {str(failure)[:300]}"
        raise RuntimeError(log["error"]) from failure
    finally:
        log["ended_at"] = time.time()
        log["call_s"] = round(log["ended_at"] - log.get("started_at", log["ended_at"]), 3)
        log["queue_s"] = round(log.get("started_at", log["ended_at"]) - log["submitted_at"], 3)
        job["log"] = log


def parse_one(chunk, pages_dir):
    index, first, count, chunk_path = chunk
    started = time.time()
    image_dir = f"{pages_dir}/chunk_{index:03d}"
    os.makedirs(image_dir, exist_ok=True)
    row = spark.sql(f"""
        SELECT CAST(ai_parse_document(content, map('version', '2.0', 'imageOutputPath', '{image_dir}')) AS STRING) AS parsed
        FROM read_files('{chunk_path}', format => 'binaryFile')""").first()
    elements, pages, warnings = C.normalize_parsed(row["parsed"] if row else None, index, first, count)
    return index, elements, pages, {"chunk": index, "first_page": first + 1, "pages": count,
                                    "start_s": started, "end_s": time.time(), "warnings": len(warnings)}


def process_document(claimed):
    with IN_FLIGHT["lock"]:
        IN_FLIGHT["count"] += 1
        in_flight = IN_FLIGHT["count"]
    try:
        return _process_document(claimed, in_flight)
    finally:
        with IN_FLIGHT["lock"]:
            IN_FLIGHT["count"] -= 1


def _process_document(claimed, in_flight):
    started = time.time()
    stats = C.Stats()
    path, file_name, arrived_at = claimed["path"], claimed["file_name"], claimed["arrived_at"]
    job_artifact_id, job_id = claimed["job_artifact_id"], args.job_id or "manual"
    size_bytes = os.stat(path).st_size
    metrics = {"pipeline": PIPELINE, "job_id": job_id, "run_id": args.run_id, "job_artifact_id": job_artifact_id,
               "source_document": file_name, "size_bytes": size_bytes, "threads": args.threads,
               "otpm_budget": args.vision_otpm, "pages_per_chunk": args.pages_per_chunk, "docs_in_flight": in_flight,
               "vision_endpoint": args.vision_endpoint, "text_endpoint": args.text_endpoint,
               "trigger_wait_s": round(claimed["claimed_at"] - arrived_at, 1),
               "queue_wait_s": round(started - claimed["claimed_at"], 1)}
    if "'" in path:
        raise ValueError("File names with single quotes are not supported.")

    # intake + split
    t = time.time()
    rel_root = f"IDP/DEFAULT/DEFAULT/{datetime.date.today():%Y/%m/%d}/{job_artifact_id}"
    artifact_dir = f"{ARTIFACTS}/{rel_root}"
    pages_dir = f"{artifact_dir}/_pages"
    os.makedirs(pages_dir, exist_ok=True)
    shutil.copyfile(path, f"{artifact_dir}/{file_name}")
    metrics["intake_s"] = round(time.time() - t, 1)
    t = time.time()
    try:
        pdf_page_count, chunks = C.split_pdf(path, f"{artifact_dir}/_chunks", args.pages_per_chunk,
                                             lambda i, f, n: f"chunk_{i:03d}.pdf")
    except Exception as failure:
        print(f"[{file_name}] could not split ({failure}); parsing as one piece")
        pdf_page_count, chunks = None, [(0, 0, 0, path)]
    metrics.update(split_s=round(time.time() - t, 1), pdf_page_count=pdf_page_count, chunks=len(chunks))

    # parse (shared pool) + cut per chunk + submit calls as each chunk lands
    t_parse = time.time()
    chunk_results, chunk_timings, cut_seconds, piece_count = {}, [], 0.0, 0
    figure_jobs, table_jobs, futures = [], [], []
    ctx_holder = {"file_name": file_name}
    parse_error = None
    try:
        for parse_future in as_completed([PARSE_POOL.submit(parse_one, c, pages_dir) for c in chunks]):
            index, elements, pages, timing = parse_future.result()
            chunk_results[index] = (elements, pages)
            chunk_timings.append(timing)
            t_cut = time.time()
            chunk_figures, chunk_tables, pieces = C.cut_chunk(elements, pages, artifact_dir)
            cut_seconds += time.time() - t_cut
            piece_count += pieces
            for job in chunk_figures + chunk_tables:
                job["task_id"], job["submitted_at"] = f"{job['kind']}:{len(figure_jobs) + len(table_jobs)}:{job['page']}:{job.get('name', '')}", time.time()
                futures.append((job, CALL_POOL.submit(arrived_at, run_task, job, stats, ctx_holder)))
                (figure_jobs if job["kind"] == "figure" else table_jobs).append(job)
    except Exception as failure:
        parse_error = f"{type(failure).__name__}: {str(failure)[:500]}"
    metrics["parse_s"] = round(time.time() - t_parse, 1)
    metrics["cut_s"] = round(cut_seconds, 1)
    metrics["chunk_timings"] = json.dumps([{**c, "start_s": round(c["start_s"] - t_parse, 1), "end_s": round(c["end_s"] - t_parse, 1)}
                                           for c in sorted(chunk_timings, key=lambda c: c["chunk"])])
    all_elements = [e for i in sorted(chunk_results) for e in chunk_results[i][0]]
    all_pages = {pid: p for i in sorted(chunk_results) for pid, p in chunk_results[i][1].items()}
    if parse_error is None and pdf_page_count and len(all_pages) < pdf_page_count:
        parse_error = f"ParseFailure: parsed {len(all_pages)} of {pdf_page_count} pages"

    if parse_error:
        for _, future in futures:
            future.cancel()
        shutil.move(path, f"{FAILED}/{job_artifact_id}__{file_name}")
        metrics.update(processed_at=datetime.datetime.utcnow().isoformat(), parse_status="failed", parse_error=parse_error,
                       pages=len(all_pages), processing_s=round(time.time() - started, 1),
                       end_to_end_from_arrival_s=round(time.time() - arrived_at, 1), met_5_min_processing=False,
                       met_5_min_from_arrival=False, artifacts_written=0)
        C.write_metrics(spark, TABLE("idp_run_metrics"), metrics)
        print(f"[{file_name}] PARSE FAILED: {parse_error}")
        return metrics

    # document-level call (text model) once every chunk is parsed
    ctx_holder["ctx"] = C.document_context(all_elements, len(all_pages))
    doc_job = {"kind": "document", "task_id": "document", "page": None, "submitted_at": time.time()}
    futures.append((doc_job, CALL_POOL.submit(arrived_at, run_task, doc_job, stats, ctx_holder)))
    first_submit = min(job["submitted_at"] for job, _ in futures)

    results, logs = {}, []
    for job, future in futures:
        try:
            results[job["task_id"]], log = future.result()
        except Exception as failure:
            stats.add("failed_items")
            results[job["task_id"]] = {"error": str(failure)[:300]}
            log = job.get("log", {})
        logs.append({**log, "pipeline": PIPELINE, "job_artifact_id": job_artifact_id, "source_document": file_name})
    last_done = max([l.get("ended_at", time.time()) for l in logs] or [time.time()])

    def window(kind_filter):
        rows = [l for l in logs if kind_filter(l.get("kind")) and l.get("started_at")]
        return round(max(l["ended_at"] for l in rows) - min(l["submitted_at"] for l in rows), 1) if rows else 0.0

    metrics.update(enrich_s=round(last_done - first_submit, 1), parse_to_last_call_s=round(last_done - t_parse, 1),
                   vision_enrich_s=window(lambda k: k == "figure"), text_enrich_s=window(lambda k: k in ("table", "document")))

    # assemble + deliver JSON
    t = time.time()
    records = C.build_records(job_id, job_artifact_id, rel_root, file_name, size_bytes, len(all_pages),
                              ctx_holder["ctx"]["title"], figure_jobs, table_jobs, results,
                              results.get("document", {}), MODEL_TAGS, all_elements)
    output_path = f"{OUTPUT}/{os.path.splitext(file_name)[0]}_{job_artifact_id}.json"
    with open(output_path, "w") as handle:
        json.dump(records, handle, indent=2)
    metrics["assemble_s"] = round(time.time() - t, 1)
    processing_seconds, end_to_end = time.time() - started, time.time() - arrived_at

    # persist
    shutil.move(path, f"{PROCESSED}/{job_artifact_id}__{file_name}")
    v = stats.values
    enrich_minutes = max(metrics["enrich_s"] / 60.0, 1e-6)
    metrics.update({
        "processed_at": datetime.datetime.utcnow().isoformat(), "parse_status": "ok", "pages": len(all_pages),
        "figure_pieces": piece_count, "figures": len(figure_jobs), "tables": len(table_jobs),
        "vision_calls": v.get("vision_calls", 0), "text_calls": v.get("text_calls", 0),
        "model_calls": v.get("vision_calls", 0) + v.get("text_calls", 0), "attempts": v.get("attempts", 0),
        "rate_limited_429": v.get("rate_limited", 0), "timeouts": v.get("timeouts", 0),
        "server_errors": v.get("server_errors", 0), "network_errors": v.get("network_errors", 0),
        "failed_items": v.get("failed_items", 0), "json_repaired": v.get("json_repaired", 0),
        "json_retried": v.get("json_retried", 0),
        "structured_output": ENDPOINTS["vision"]["structured"] and ENDPOINTS["text"]["structured"],
        "input_tokens": v.get("input_tokens", 0), "output_tokens": v.get("output_tokens", 0),
        "vision_output_tokens": v.get("vision_output_tokens", 0), "text_output_tokens": v.get("text_output_tokens", 0),
        "avg_otpm_during_enrich": round(v.get("output_tokens", 0) / enrich_minutes),
        "processing_s": round(processing_seconds, 1), "end_to_end_from_arrival_s": round(end_to_end, 1),
        "met_5_min_processing": processing_seconds <= C.TARGET_SECONDS,
        "met_5_min_from_arrival": end_to_end <= C.TARGET_SECONDS, "output_json": output_path})
    errors = []
    try:
        metrics["artifacts_written"] = C.write_artifacts(spark, TABLE("idp_artifacts"), records, job_artifact_id, file_name, PIPELINE)
    except Exception as failure:
        metrics["artifacts_written"] = 0
        errors.append(f"idp_artifacts: {type(failure).__name__}: {str(failure)[:300]}")
    try:
        C.write_call_log(spark, TABLE("idp_call_log"), logs)
    except Exception as failure:
        errors.append(f"idp_call_log: {type(failure).__name__}: {str(failure)[:300]}")
    metrics["persist_error"] = "; ".join(errors) or None
    try:
        C.write_metrics(spark, TABLE("idp_run_metrics"), metrics)
    except Exception as failure:
        errors.append(f"idp_run_metrics: {type(failure).__name__}: {str(failure)[:300]}")
        print(f"[{file_name}] METRICS NOT SAVED: {json.dumps(metrics, default=str)}")
    print(f"[{file_name}] {len(all_pages)} pages / {len(chunks)} chunks | figures {len(figure_jobs)} (vision) | tables {len(table_jobs)} (text) | "
          f"parse {metrics['parse_s']}s | enrich {metrics['enrich_s']}s | processing {metrics['processing_s']}s | "
          f"from arrival {metrics['end_to_end_from_arrival_s']}s -> {'PASS' if metrics['met_5_min_from_arrival'] else 'FAIL'} | "
          f"429s {metrics['rate_limited_429']} | failed {metrics['failed_items']}" + (f" | PERSIST ERRORS {errors}" if errors else ""))
    return metrics


def safe_process(claimed):
    try:
        process_document(claimed)
    except Exception:
        print(f"[{claimed['file_name']}] FAILED:\n{traceback.format_exc()}")
        try:
            shutil.move(claimed["path"], f"{FAILED}/{claimed['job_artifact_id']}__{claimed['file_name']}")
        except Exception:
            pass


def on_claimed(claimed_files):
    for claimed in sorted(claimed_files, key=lambda c: c["arrived_at"]):
        DOC_POOL.submit(safe_process, claimed)


if __name__ == "__main__":
    leftovers = C.recover_processing(PROCESSING)
    if leftovers:
        print(f"Recovering {len(leftovers)} file(s) left in processing/")
        on_claimed(leftovers)
    print(f"Watching {INCOMING} every {args.poll_seconds}s | docs in parallel {args.max_parallel_docs} | "
          f"vision {args.vision_endpoint} @ {args.vision_otpm} OTPM | text {args.text_endpoint} @ {args.text_otpm} OTPM")
    C.discover_forever(spark, INCOMING, CHECKPOINT, PROCESSING, on_claimed, args.poll_seconds,
                       queue_table=TABLE("idp_discovered_direct"), mode=args.discovery)
