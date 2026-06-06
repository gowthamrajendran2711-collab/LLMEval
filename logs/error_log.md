# LLMEval Error Log

## [ERR-001] DeepEval Pytest plugin conflict
**Date:** 2024-02-28 | **Severity:** Medium | **Status:** Resolved

**Description:** deepeval pytest plugin overrode our custom JSON reporter, breaking CI output parsing.
**Root Cause:** Both deepeval and pytest-json-report register `pytest_runtest_logreport` hooks.
**Fix:** Disabled deepeval's pytest plugin with `DEEPEVAL_TELEMETRY_OPT_OUT=YES`. Used deepeval API directly without plugin.
**Impact:** CI report generation works correctly. Test run time unchanged.

---

## [ERR-002] MMLU dataset loading fails for certain subtasks
**Date:** 2024-03-10 | **Severity:** Low | **Status:** Resolved

**Description:** `load_dataset("lukaemon/mmlu", "professional_medicine")` raised `FileNotFoundError` intermittently.
**Root Cause:** HuggingFace Hub rate limiting caused partial downloads, corrupting the local cache.
**Fix:** Added retry logic with cache invalidation on HF_HUB_DOWNLOAD_TIMEOUT exceeded errors.
**Impact:** MMLU loading 100% reliable across 50 CI runs.

---

## [ERR-003] NLI model GPU memory error in batch scoring
**Date:** 2024-04-05 | **Severity:** Medium | **Status:** Resolved

**Description:** Batch hallucination scoring with batch_size=64 caused CUDA OOM on T4 GPU.
**Root Cause:** NLI model (DeBERTa-v3) input sequences exceeded 512 tokens after concatenation.
**Fix:** Added input truncation to max 384 tokens per side + reduced batch_size to 16 with gradient checkpointing.
**Impact:** Throughput stable at 340 samples/min with zero OOM errors.
