# Methodology Notes

Reusable methodology references for the factor research pipeline. Each note is
vendor-neutral: it documents the *method*, not the tooling of any particular data
provider, and points at the module in this repository that implements it.

| Note | Scope |
| --- | --- |
| [`factor-ic-research.md`](factor-ic-research.md) | Testing whether a factor predicts forward returns — IC / IR, decile long-short backtest, IC decay. |
| [`execution-model.md`](execution-model.md) | Execution-cost modelling — slippage, VWAP/TWAP slicing, market impact, Kyle lambda. |
| [`strategy-optimizer.md`](strategy-optimizer.md) | Parameter search with in-sample / out-of-sample split, walk-forward validation, overfitting detection. |

## Literature search

`tools/arxiv_fetch.py` queries the arXiv API and emits one JSON document on
stdout. It is dependency-free (standard library only) and throttles itself to at
most one request every 3 seconds, per arXiv's API etiquette.

```bash
# Search by query
python3 tools/arxiv_fetch.py --query "factor mining machine learning" --max_results 10

# Most recent submissions first
python3 tools/arxiv_fetch.py --query "asset pricing neural network" --sort_by submittedDate --sort_order descending

# Resolve specific arXiv IDs
python3 tools/arxiv_fetch.py --id_list 2305.10601,1706.03762
```

Typical output:

```json
{
  "status": "success",
  "results_count": 1,
  "papers": [
    {
      "id": "2305.10601v1",
      "title": "...",
      "summary": "...",
      "published": "2023-05-17T...",
      "authors": ["..."],
      "pdf_url": "http://arxiv.org/pdf/2305.10601v1",
      "primary_category": "cs.CL"
    }
  ]
}
```

On failure it prints `{"status": "error", "message": "..."}` and exits non-zero,
so it composes cleanly into the G001 literature-research stage.

## Attribution

The three methodology notes above are adapted from the `longbridge-quant` skill
pack's `references/` documents, which are distributed under the MIT License.
Vendor-specific CLI instructions and vendor error-handling tables have been
removed, and data-access steps have been re-pointed at this repository's own
modules.

`tools/arxiv_fetch.py` is adapted from the `literature-search-arxiv` skill's
`scripts/search_arxiv.py`, which is Copyright 2026 Google LLC and distributed
under the Apache License, Version 2.0. Changes: the external `polite-http`
dependency was removed in favour of a standard-library request throttle, and a
printing bug that emitted a partial JSON document per paper was fixed. The full
license header is preserved at the top of the file.
