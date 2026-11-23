# Document AI Enrichment at Scale: ai_query vs Direct API Calls
## Recommendations from an intelligent document processing simulation, November 2026

**Bottom line:** if each file must be enriched within minutes of being dropped, call the model endpoints directly from an always-on pipeline. Use `ai_query` for high-volume batch work where throughput matters more than how long any one file takes. In our tests the direct route was faster on every document, with no failed items and no rate-limit errors. The remaining bottleneck is document parsing, not the model calls.

### When to use which
| | **`ai_query` (AI Functions)** | **Direct API calls (REST)** |
|---|---|---|
| Best for | Batch and backfill jobs, scheduled runs, large volumes where hours are acceptable | A per-file deadline (minutes), event-driven processing, many users dropping files at once |
| Throttling and retries | Handled automatically by AI Functions | Built into your code: pacer, retries, backoff, timeouts |
| Control over ordering | None (runs as a batch) | Full: oldest file first, overlapping parsing with model calls, separate limits per model |
| Visibility | Stage-level timings | Timing for every call: queue, rate-limit wait, latency, tokens |
| Model support | Only models supported for batch inference (GLM 5.3 and DeepSeek V4.1 Flash were rejected) | Any pay-per-token or provisioned endpoint |
| Effort | Low: SQL or DataFrame code | Medium: a small amount of Python orchestration |

### What we tested
Two always-on serverless jobs processed the same three PDFs dropped at the same time. Both used Claude Sonnet 4.6 for figures and Qwen3.5 122B A10B for tables and the document summary.

| Document | Direct API (s from drop) | ai_query (s from drop) |
|---|---|---|
| 16 pages | **102** | 192 |
| 450 pages (21 MB) | **348** | 1,034 |
| 1,180 pages | **985** | 1,456 |

- **Reliability:** across both routes, about 1,350 model calls completed with **0 failed items and 0 rate-limit (429) errors**. Getting the model to return a fixed JSON format removed the invalid-JSON failures we saw earlier (11 -> 0).
- **Call latency (direct route):** Sonnet figure calls took a median of **7.9 s**. Qwen table calls took **1.8 s**, about 4x faster.
- **Parsing takes most of the time:** about 0.8 s per page for very large files, even with 32 chunks parsed at once. The direct route parsed faster because it ran many parse queries at the same time and sent figures and tables to the models as each chunk finished. `ai_query` waits until the whole batch is parsed.
- **Files over 500 pages:** `ai_parse_document` accepts at most 500 pages or 100 MB per call. A 1,180-page file sent in one piece came back as 1 page. Split into chunks, all 1,180 pages were parsed.

### Recommendations
1. **Split large PDFs into chunks of 10-25 pages and parse them in parallel.** Send each chunk's figures and tables to the models as soon as that chunk is parsed. This is the biggest single gain in speed.
2. **Pick the model by content type.** Keep the vision model for images. Send tables and document summaries to a fast text model: it's 4x faster per call and frees most of the vision model's rate limit.
3. **Turn reasoning off for extraction** (`reasoning_effort: none` on Qwen). With reasoning on and a short token limit, Qwen returned empty answers.
4. **Use structured output** (a `json_schema` response format), with a tolerant JSON parser and one corrective retry as a backup.
5. **Plan model capacity early.** Claude Sonnet 4.6 is pay-per-token only: 100K output tokens/min per workspace, with no provisioned throughput and no priority tier. Ten 450-page documents arriving together need about 450K output tokens/min. Either request a higher limit, or move text work to a model that supports provisioned throughput, such as Qwen3.5 122B A10B (Public Preview).
6. **Share one rate pacer per model across the whole pipeline**, and process the oldest file first, so large uploads don't delay small ones. See "Staying within the output-token limit" below.
7. **Process files continuously instead of starting a job per file.** File-arrival triggers waited 51-585 s before starting. A continuous serverless job that checks for new files every 5 s with Auto Loader (`AvailableNow`) picked files up in 7-16 s.
8. **Build in safeguards:**
   - check the parsed page count against the PDF's page count
   - use fixed schemas for Delta tables
   - retry Delta commit conflicts when several writers share a table
9. **Set the target by document size.** Five minutes is realistic up to about 300-400 pages. For very large documents, deliver the document summary and the earliest figures and tables first.

### Staying within the output-token limit
The direct route paces every call against an **output-tokens-per-minute (OTPM) budget for each model**. It's a token bucket shared by every document and thread in the job. In our tests, all 230 calls on the 16- and 450-page documents succeeded first time with **0 rate-limit errors**, and the pacer added only **3.7 s of total wait** across 90 Sonnet calls.

| Setting | Sample value | What it does |
|---|---|---|
| `--vision_otpm` / `--text_otpm` | 60,000 each | Budget per model, set to **60% of the 100K workspace limit** to leave room for other workloads |
| Burst allowance | 15 s of budget (OTPM / 4) | Lets a new document start at once without sending a burst of requests |
| Up-front estimate per call | Figure 1,500 / table 500 / summary 800 tokens | Taken from the budget before each call, then **corrected to the actual tokens used** |
| `max_tokens` per call | Figure 3,500 / table 1,500 / summary 2,000 | Hard cap on any single call's output |
| `--threads` | 64 | Model calls running at once across all documents. The pacer, not this setting, limits the token rate |
| Retries | Up to 6 attempts; follows `Retry-After` on 429s; backoff 1-20 s with jitter on 5xx errors; 60 s timeout per call | Recovers from errors without making them worse |

**Tips**
- **Start the budget at 60-80% of the workspace limit.** All jobs in a workspace share one limit per model, so split the budget if more than one pipeline calls the same model.
- **Size throughput from tokens per document.** Sonnet used about **7.8K output tokens for a 16-page document, 64K for a 450-page one and 312K for a 1,180-page one**. At 60K OTPM that's about **7.7, 0.9 and 0.2 documents per minute** respectively.
- **Cut output tokens before raising limits.** Keep prompts tight (for example, a 350-word cap on each figure's summary), skip decorative images, and send text-only work to a text model.
- **Watch input tokens too.** The 1,180-page document sent about 861K input tokens to Sonnet, close to the 1M input-tokens-per-minute limit if it arrived in a single minute. Add an input-token budget for very large or bursty workloads.
- **Monitor** 429s, pacer wait and output tokens per minute in each run's metrics. A steady rise in pacer wait is the signal to request more capacity.

### Next steps
- Run a test with many users: 10-20 small and medium files dropped at once.
- Measure where parsing stops getting faster as more chunks run at once.
- Compare table quality between Qwen and Sonnet on representative documents.
- Decide whether to request a higher Claude limit or reserve provisioned throughput for the text model.
