"""
IDP - ai_query route, continuously running (v1, 2026-11-20).

Same input/output contract as the direct-API route, but every model call goes through ai_query, so retries, backoff
and throttling against the endpoints are handled by AI Functions (no pacer, no thread pool, no retry code here).

* Always-on: Auto Loader (Trigger.AvailableNow every --poll_seconds) claims new PDFs from <volume>/incoming_aiq/.
  Each discovery batch becomes one work batch; up to --max_parallel_batches batches run at the same time, so files
  that arrive while a large batch is running start immediately in their own batch.
* Per batch:
    1. split every PDF into --pages_per_chunk page chunks (driver, pypdf)
    2. ONE query parses every chunk of every document with ai_parse_document (repartitioned: one task per chunk)
    3. cut figures / collect tables (driver)
    4. THREE ai_query queries run at the same time:
         figures   -> --vision_endpoint (Claude Sonnet 4.6) with files => image bytes
         tables    -> --text_endpoint (Qwen3.5 122B A10B, reasoning off)
         documents -> --text_endpoint
       all with responseFormat json_schema and failOnError => false (one row's failure never fails the batch)
    5. assemble IDP JSON per document -> <volume>/output_aiq/, Delta rows (pipeline = 'ai_query_stream')
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--volume", default="/Volumes/<catalog>/<schema>/<volume>")
parser.add_argument("--catalog", default="<catalog>")
parser.add_argument("--schema", default="<schema>")
parser.add_argument("--code_dir", default="/Workspace/Shared/idp-sample")
parser.add_argument("--vision_endpoint", default="databricks-claude-sonnet-4-6")
parser.add_argument("--text_endpoint", default="databricks-qwen35-122b-a10b")
parser.add_argument("--text_reasoning_effort", default="none")
parser.add_argument("--pages_per_chunk", type=int, default=10)
parser.add_argument("--max_parallel_batches", type=int, default=4)
parser.add_argument("--poll_seconds", type=int, default=5)
parser.add_argument("--discovery", default="autoloader", choices=["autoloader", "listing"])
parser.add_argument("--labels", default="stress_testing,benchmark_testing,model_outputs,other")
parser.add_argument("--job_id", default="")
parser.add_argument("--run_id", default="")
args, _ = parser.parse_known_args()

sys.path.insert(0, args.code_dir)
import idp_common as C                                     # noqa: E402
from pyspark.sql import SparkSession                            # noqa: E402

PIPELINE = "ai_query_stream"
VOL = args.volume.rstrip("/")
INCOMING, PROCESSING, FAILED = f"{VOL}/incoming_aiq", f"{VOL}/processing_aiq", f"{VOL}/failed_aiq"
ARTIFACTS, OUTPUT, PROCESSED, WORK = f"{VOL}/artifacts", f"{VOL}/output_aiq", f"{VOL}/processed_aiq", f"{VOL}/_aiq_work"
CHECKPOINT = f"{VOL}/_checkpoints/ai_query_stream_v2"
TABLE = lambda name: f"{args.catalog}.{args.schema}.{name}"     # noqa: E731
LABELS = [label.strip() for label in args.labels.split(",") if label.strip()]
LABEL_TEXT = ", ".join(f'"{label}"' for label in LABELS)
SCHEMAS = C.make_schemas(LABELS)
MODEL_TAGS = {"figure": f"{args.vision_endpoint} (ai_query)", "table": f"{args.text_endpoint} (ai_query)",
              "document": f"{args.text_endpoint} (ai_query)"}
MAX_TOKENS = {"figure": 3500, "table": 1500, "document": 2000}

spark = SparkSession.builder.getOrCreate()
for folder in (INCOMING, PROCESSING, FAILED, ARTIFACTS, OUTPUT, PROCESSED, WORK, CHECKPOINT):
    os.makedirs(folder, exist_ok=True)
BATCH_POOL = ThreadPoolExecutor(max_workers=args.max_parallel_batches, thread_name_prefix="batch")
IN_FLIGHT = {"count": 0, "lock": threading.Lock()}


def sql_str(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def response_format(kind):
    return sql_str(json.dumps({"type": "json_schema", "json_schema": {"name": kind, "schema": SCHEMAS[kind], "strict": True}}))


def model_params(kind, text_model):
    params = f"'max_tokens', {MAX_TOKENS[kind]}, 'temperature', 0.0"
    if text_model and args.text_reasoning_effort:
        params += f", 'reasoning_effort', {sql_str(args.text_reasoning_effort)}"
    return f"named_struct({params})"


def run_ai_query(kind, rows):
    """rows: list of (task_id, prompt[, image bytes]). Returns {task_id: (dict|None, error|None)}, seconds."""
    if not rows:
        return {}, 0.0
    started = time.time()
    if kind == "figure":
        frame = spark.createDataFrame([(r[0], r[1], bytearray(r[2])) for r in rows], "task_id string, prompt string, content binary")
        expr = (f"ai_query('{args.vision_endpoint}', prompt, files => content, responseFormat => {response_format(kind)}, "
                f"modelParameters => {model_params(kind, False)}, failOnError => false) AS r")
    else:
        frame = spark.createDataFrame(rows, "task_id string, prompt string")
        expr = (f"ai_query('{args.text_endpoint}', prompt, responseFormat => {response_format(kind)}, "
                f"modelParameters => {model_params(kind, True)}, failOnError => false) AS r")
    keys = list(SCHEMAS[kind]["properties"])
    out = {}
    for row in frame.repartition(max(1, min(len(rows), 200))).selectExpr("task_id", expr).collect():
        result, error = (row.r.result, row.r.errorMessage) if row.r is not None else (None, "null response")
        if error or not result:
            out[row.task_id] = (None, error or "empty reply")
            continue
        try:
            out[row.task_id] = (C.parse_reply(result, keys), None)
        except ValueError as failure:
            out[row.task_id] = (None, f"JSON: {failure}")
    return out, time.time() - started


def process_batch(claimed_files):
    with IN_FLIGHT["lock"]:
        IN_FLIGHT["count"] += 1
        in_flight = IN_FLIGHT["count"]
    try:
        _process_batch(claimed_files, in_flight)
    except Exception:
        print(f"[batch] FAILED:\n{traceback.format_exc()}")
        for claimed in claimed_files:
            if os.path.exists(claimed["path"]):
                shutil.move(claimed["path"], f"{FAILED}/{claimed['job_artifact_id']}__{claimed['file_name']}")
    finally:
        with IN_FLIGHT["lock"]:
            IN_FLIGHT["count"] -= 1


def _process_batch(claimed_files, in_flight):
    batch_started = time.time()
    batch_id = C.uuid7()
    chunk_dir, pages_dir = f"{WORK}/{batch_id}/chunks", f"{WORK}/{batch_id}/pages"
    os.makedirs(pages_dir, exist_ok=True)
    job_id = args.job_id or "manual"
    docs = {}

    # 1. intake + split -----------------------------------------------------------------------
    t = time.time()
    for claimed in claimed_files:
        doc_id, file_name = claimed["job_artifact_id"], claimed["file_name"]
        if "'" in claimed["path"]:
            raise ValueError("File names with single quotes are not supported.")
        rel_root = f"IDP/DEFAULT/DEFAULT/{datetime.date.today():%Y/%m/%d}/{doc_id}"
        artifact_dir = f"{ARTIFACTS}/{rel_root}"
        os.makedirs(artifact_dir, exist_ok=True)
        shutil.copyfile(claimed["path"], f"{artifact_dir}/{file_name}")
        doc = {"claimed": claimed, "rel_root": rel_root, "artifact_dir": artifact_dir, "chunks": {}, "error": None,
               "size_bytes": os.stat(claimed["path"]).st_size, "pdf_page_count": None}
        try:
            total, chunks = C.split_pdf(claimed["path"], chunk_dir, args.pages_per_chunk,
                                        lambda i, f, n, d=doc_id: f"{d}__{i:03d}__{f:05d}__{n:04d}.pdf")
            doc["pdf_page_count"] = total
        except Exception as failure:
            print(f"[{file_name}] could not split ({failure}); parsing as one piece")
            single = f"{chunk_dir}/{doc_id}__000__00000__0000.pdf"
            os.makedirs(chunk_dir, exist_ok=True)
            shutil.copyfile(claimed["path"], single)
            chunks = [(0, 0, 0, single)]
        doc["chunk_count"] = len(chunks)
        docs[doc_id] = doc
    split_s = time.time() - t

    # 2. parse every chunk of every document in one query ---------------------------------------
    t = time.time()
    chunk_total = sum(d["chunk_count"] for d in docs.values())
    parsed_rows = (spark.read.format("binaryFile").load(chunk_dir)
                   .repartition(max(1, chunk_total))
                   .selectExpr("path", f"CAST(ai_parse_document(content, map('version', '2.0', 'imageOutputPath', '{pages_dir}')) AS STRING) AS parsed")
                   .collect())
    parse_s = time.time() - t
    for row in parsed_rows:
        doc_id, index, first, count = os.path.basename(C.local_path(row.path))[:-4].split("__")
        doc = docs[doc_id]
        try:
            elements, pages, _ = C.normalize_parsed(row.parsed, int(index), int(first), int(count))
            doc["chunks"][int(index)] = (elements, pages)
        except Exception as failure:
            doc["error"] = doc["error"] or f"{type(failure).__name__}: {str(failure)[:400]}"

    # 3. cut ----------------------------------------------------------------------------------
    t = time.time()
    figure_rows, table_rows, doc_rows = [], [], []
    for doc_id, doc in docs.items():
        elements_all = [e for i in sorted(doc["chunks"]) for e in doc["chunks"][i][0]]
        pages_all = {pid: p for i in sorted(doc["chunks"]) for pid, p in doc["chunks"][i][1].items()}
        if not doc["error"] and doc["pdf_page_count"] and len(pages_all) < doc["pdf_page_count"]:
            doc["error"] = f"ParseFailure: parsed {len(pages_all)} of {doc['pdf_page_count']} pages"
        doc.update(elements=elements_all, pages=pages_all, figures=[], tables=[], pieces=0)
        if doc["error"]:
            continue
        for index in sorted(doc["chunks"]):
            figures, tables, pieces = C.cut_chunk(*doc["chunks"][index], doc["artifact_dir"])
            doc["figures"] += figures
            doc["tables"] += tables
            doc["pieces"] += pieces
        for n, job in enumerate(doc["figures"] + doc["tables"]):
            job["task_id"] = f"{doc_id}|{job['kind']}|{n}"
            if job["kind"] == "figure":
                figure_rows.append((job["task_id"], C.figure_prompt(job, LABEL_TEXT), job["bytes"]))
            else:
                table_rows.append((job["task_id"], C.table_prompt(job, LABEL_TEXT)))
        doc["ctx"] = C.document_context(elements_all, len(pages_all))
        doc_rows.append((f"{doc_id}|document", C.document_prompt(doc["ctx"], doc["claimed"]["file_name"], LABEL_TEXT)))
    cut_s = time.time() - t

    # 4. enrich: three ai_query queries at the same time ----------------------------------------
    t = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fig_future = pool.submit(run_ai_query, "figure", figure_rows)
        tab_future = pool.submit(run_ai_query, "table", table_rows)
        doc_future = pool.submit(run_ai_query, "document", doc_rows)
        (fig_out, fig_s), (tab_out, tab_s), (doc_out, doc_s) = fig_future.result(), tab_future.result(), doc_future.result()
    enrich_s = time.time() - t
    answers = {**fig_out, **tab_out, **doc_out}

    # 5. assemble + persist per document ---------------------------------------------------------
    for doc_id, doc in docs.items():
        claimed, file_name = doc["claimed"], doc["claimed"]["file_name"]
        metrics = {"pipeline": PIPELINE, "job_id": job_id, "run_id": args.run_id, "job_artifact_id": doc_id,
                   "source_document": file_name, "size_bytes": doc["size_bytes"], "pdf_page_count": doc["pdf_page_count"],
                   "chunks": doc["chunk_count"], "pages_per_chunk": args.pages_per_chunk, "docs_in_flight": len(docs),
                   "vision_endpoint": args.vision_endpoint, "text_endpoint": args.text_endpoint,
                   "trigger_wait_s": round(claimed["claimed_at"] - claimed["arrived_at"], 1),
                   "queue_wait_s": round(batch_started - claimed["claimed_at"], 1), "split_s": round(split_s, 1),
                   "parse_s": round(parse_s, 1), "cut_s": round(cut_s, 1), "enrich_s": round(enrich_s, 1),
                   "vision_enrich_s": round(fig_s, 1), "text_enrich_s": round(max(tab_s, doc_s), 1),
                   "structured_output": True, "pages": len(doc["pages"])}
        if doc["error"]:
            shutil.move(claimed["path"], f"{FAILED}/{doc_id}__{file_name}")
            metrics.update(processed_at=datetime.datetime.utcnow().isoformat(), parse_status="failed", parse_error=doc["error"],
                           processing_s=round(time.time() - batch_started, 1),
                           end_to_end_from_arrival_s=round(time.time() - claimed["arrived_at"], 1),
                           met_5_min_processing=False, met_5_min_from_arrival=False, artifacts_written=0)
            C.write_metrics(spark, TABLE("idp_run_metrics"), metrics)
            print(f"[{file_name}] PARSE FAILED: {doc['error']}")
            continue
        t = time.time()
        results, failed = {}, 0
        for job in doc["figures"] + doc["tables"]:
            value, error = answers.get(job["task_id"], (None, "missing"))
            results[job["task_id"]] = value if value is not None else {"error": error}
            failed += value is None
        doc_value, doc_error = answers.get(f"{doc_id}|document", (None, "missing"))
        failed += doc_value is None
        records = C.build_records(job_id, doc_id, doc["rel_root"], file_name, doc["size_bytes"], len(doc["pages"]),
                                  doc["ctx"]["title"], doc["figures"], doc["tables"], results,
                                  doc_value if doc_value is not None else {"error": doc_error}, MODEL_TAGS, doc["elements"])
        output_path = f"{OUTPUT}/{os.path.splitext(file_name)[0]}_{doc_id}.json"
        with open(output_path, "w") as handle:
            json.dump(records, handle, indent=2)
        processing_s, end_to_end = time.time() - batch_started, time.time() - claimed["arrived_at"]
        shutil.move(claimed["path"], f"{PROCESSED}/{doc_id}__{file_name}")
        metrics.update(processed_at=datetime.datetime.utcnow().isoformat(), parse_status="ok", figure_pieces=doc["pieces"],
                       figures=len(doc["figures"]), tables=len(doc["tables"]), vision_calls=len(doc["figures"]),
                       text_calls=len(doc["tables"]) + 1, model_calls=len(doc["figures"]) + len(doc["tables"]) + 1,
                       failed_items=failed, assemble_s=round(time.time() - t, 1), processing_s=round(processing_s, 1),
                       end_to_end_from_arrival_s=round(end_to_end, 1),
                       met_5_min_processing=processing_s <= C.TARGET_SECONDS,
                       met_5_min_from_arrival=end_to_end <= C.TARGET_SECONDS, output_json=output_path)
        errors = []
        try:
            metrics["artifacts_written"] = C.write_artifacts(spark, TABLE("idp_artifacts"), records, doc_id, file_name, PIPELINE)
        except Exception as failure:
            metrics["artifacts_written"] = 0
            errors.append(f"idp_artifacts: {type(failure).__name__}: {str(failure)[:300]}")
        metrics["persist_error"] = "; ".join(errors) or None
        try:
            C.write_metrics(spark, TABLE("idp_run_metrics"), metrics)
        except Exception as failure:
            print(f"[{file_name}] METRICS NOT SAVED ({type(failure).__name__}): {json.dumps(metrics, default=str)}")
        print(f"[{file_name}] batch of {len(docs)} | {len(doc['pages'])} pages | figures {len(doc['figures'])} | tables {len(doc['tables'])} | "
              f"parse {metrics['parse_s']}s | enrich {metrics['enrich_s']}s (vision {metrics['vision_enrich_s']}s, text {metrics['text_enrich_s']}s) | "
              f"from arrival {metrics['end_to_end_from_arrival_s']}s -> {'PASS' if metrics['met_5_min_from_arrival'] else 'FAIL'} | failed {failed}")
    shutil.rmtree(f"{WORK}/{batch_id}/chunks", ignore_errors=True)


def on_claimed(claimed_files):
    BATCH_POOL.submit(process_batch, sorted(claimed_files, key=lambda c: c["arrived_at"]))


if __name__ == "__main__":
    leftovers = C.recover_processing(PROCESSING)
    if leftovers:
        print(f"Recovering {len(leftovers)} file(s) left in processing_aiq/")
        on_claimed(leftovers)
    print(f"Watching {INCOMING} every {args.poll_seconds}s | parallel batches {args.max_parallel_batches} | "
          f"vision {args.vision_endpoint} | text {args.text_endpoint} (ai_query)")
    C.discover_forever(spark, INCOMING, CHECKPOINT, PROCESSING, on_claimed, args.poll_seconds,
                       queue_table=TABLE("idp_discovered_aiq"), mode=args.discovery)
