# IDP sample: PDF parse + AI enrichment with a per-file latency target

Two always-on Databricks serverless pipelines that pick up PDFs dropped into a Unity Catalog volume, parse them with
`ai_parse_document`, enrich figures (vision model) and tables (text model), and write one artifact-style JSON per document.

| Route | Script | Watches | Throttling |
|---|---|---|---|
| Direct API | `idp_stream_direct.py` | `<volume>/incoming/` | Built-in token-bucket pacer per model, retries, backoff, oldest-file-first call queue |
| ai_query | `idp_stream_aiquery.py` | `<volume>/incoming_aiq/` | Handled by AI Functions |

Shared helpers (chunking, figure cropping, prompts/schemas, tolerant JSON parsing, Delta writers with commit-conflict
retry, Auto Loader discovery loop) live in `idp_common.py`.

## Setup
1. Copy the three `.py` files to a workspace folder, e.g. `/Workspace/Shared/idp-sample`.
2. Create a Unity Catalog volume and replace `<catalog>`, `<schema>`, `<volume>` in `jobs/*.json`
   (and `--code_dir` if you used another folder).
3. Create the jobs: `databricks jobs create --json @jobs/job_direct_stream.json` (and the ai_query one).
   Both use a **continuous** schedule on serverless (environment v4, deps: pillow, requests, pypdf).
4. Drop PDFs into `incoming/` (direct) or `incoming_aiq/` (ai_query). Outputs: `output/` / `output_aiq/`,
   tables `<catalog>.<schema>.idp_run_metrics`, `idp_artifacts`, `idp_call_log` (direct route only).

## Key settings (direct route)
| Flag | Default | Notes |
|---|---|---|
| `--vision_endpoint` | `databricks-claude-sonnet-4-6` | Figures only |
| `--text_endpoint` | `databricks-qwen35-122b-a10b` | Tables + document summary; `--text_reasoning_effort none` |
| `--vision_otpm`, `--text_otpm` | 60000 | Output-token budget per model for the whole job; keep at 60-80% of the workspace limit |
| `--threads` | 64 | Concurrent model calls across all documents |
| `--max_parallel_docs` | 8 | Documents processed at once |
| `--pages_per_chunk` / `--parse_parallelism` | 10 / 32 | `ai_parse_document` max 500 pages / 100 MB per call |
| `--poll_seconds` | 5 | Auto Loader `AvailableNow` loop (serverless does not support time-based triggers) |
| `--discovery` | `autoloader` | `listing` = simple directory polling fallback |

## Notes
- Sample code, provided as-is for evaluation; validate output quality, cost and limits for your workload.
- Labels (`--labels`) are placeholders; replace with your classification list.
- Measured in testing: 16 pages in ~102 s, 450 pages in ~348 s from file drop (direct route); parsing dominates.

## Guidance
See [`docs/recommendations.md`](docs/recommendations.md) for when to use `ai_query` vs direct calls, test findings and pacer/OTPM tuning.
