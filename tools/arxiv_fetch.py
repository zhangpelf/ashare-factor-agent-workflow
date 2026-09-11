# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------------------------
# Adapted from the `literature-search-arxiv` skill's `scripts/search_arxiv.py`
# (Apache-2.0, Copyright 2026 Google LLC).
#
# Changes made when landing this tool in the repository:
#   * Dropped the external `polite-http` dependency. Request throttling is
#     reimplemented with the standard library only, capped at 1 request / 3s
#     as arXiv's API etiquette requires.
#   * Fixed the result printing, which sat inside the per-entry loop and
#     therefore emitted a partial JSON document for every paper instead of one
#     final document.
#   * Normalised whitespace in `title` / `summary` (the Atom feed wraps long
#     values across lines with deep indentation, which previously survived into
#     the JSON output as runs of spaces).
#   * Added a network/timeout error path so failures produce a JSON error
#     envelope instead of a traceback, to keep it composable in G001.
# ---------------------------------------------------------------------------
"""Searches the arXiv API and returns results in a clean JSON format.

Usage:
    python3 tools/arxiv_fetch.py --query "factor mining machine learning" --max_results 10
    python3 tools/arxiv_fetch.py --query "asset pricing neural network" \
        --sort_by submittedDate --sort_order descending
    python3 tools/arxiv_fetch.py --id_list 2305.10601,1706.03762

Output is a single JSON object on stdout:
    {"status": "success", "results_count": N, "papers": [ ... ]}

No third-party dependencies — standard library only (Python >= 3.10).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_BASE_URL = "http://export.arxiv.org/api/query?"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_MIN_INTERVAL_SECONDS = 3.0  # arXiv asks for <= 1 request per 3 seconds
_USER_AGENT = "ashare-factor-agent-workflow/arxiv_fetch"

_last_request_ts = 0.0


def _throttled_fetch(url: str) -> bytes:
  """Fetches `url`, enforcing arXiv's minimum interval between requests."""
  global _last_request_ts
  elapsed = time.monotonic() - _last_request_ts
  if elapsed < _MIN_INTERVAL_SECONDS:
    time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
  try:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
      return response.read()
  finally:
    _last_request_ts = time.monotonic()


def parse_args(argv=None) -> argparse.Namespace:
  """Parses command-line arguments for the arXiv search script."""
  parser = argparse.ArgumentParser(
      description="Search the arXiv API and return clean JSON"
  )
  parser.add_argument(
      "--query",
      type=str,
      help="Search query string (e.g. 'au:einstein AND ti:relativity')",
  )
  parser.add_argument(
      "--id_list", type=str, help="Comma-separated list of arXiv IDs"
  )
  parser.add_argument("--start", type=int, default=0, help="Pagination offset")
  parser.add_argument(
      "--max_results", type=int, default=10, help="Number of results to return"
  )
  parser.add_argument(
      "--sort_by",
      type=str,
      choices=["relevance", "lastUpdatedDate", "submittedDate"],
      help="Sort by",
  )
  parser.add_argument(
      "--sort_order",
      type=str,
      choices=["ascending", "descending"],
      help="Sort order",
  )
  return parser.parse_args(argv)


def strip_namespace(tag: str) -> str:
  """Removes the `{namespace}` prefix from an ElementTree tag."""
  if tag.startswith("{"):
    return tag.split("}", 1)[1]
  return tag


def clean_text(text: str | None) -> str:
  """Collapses all whitespace runs (newlines, tabs, indentation) to one space."""
  if not text:
    return ""
  return " ".join(text.split())


def build_url(args: argparse.Namespace) -> str:
  """Builds the arXiv API URL from parsed arguments."""
  params = {"start": args.start, "max_results": args.max_results}
  if args.query:
    params["search_query"] = args.query
  if args.id_list:
    params["id_list"] = args.id_list
  if args.sort_by:
    params["sortBy"] = args.sort_by
  if args.sort_order:
    params["sortOrder"] = args.sort_order

  # quote_plus so that spaces are encoded as '+' as the arXiv API expects.
  query_string = urllib.parse.urlencode(
      params, quote_via=urllib.parse.quote_plus
  )
  return _BASE_URL + query_string


def parse_feed(xml_data: bytes) -> list[dict]:
  """Parses an arXiv Atom feed into a list of paper dictionaries."""
  root = ET.fromstring(xml_data)
  results = []

  for entry in root.findall(f"{_ATOM_NS}entry"):
    paper: dict = {}
    authors = []

    for child in entry:
      tag = strip_namespace(child.tag)
      if tag == "id":
        # Keep just the identifier, e.g. .../abs/2305.10601v1 -> 2305.10601v1
        if child.text:
          paper["id"] = child.text.split("/abs/")[-1]
      elif tag == "title":
        paper["title"] = clean_text(child.text)
      elif tag == "summary":
        paper["summary"] = clean_text(child.text)
      elif tag == "published":
        paper["published"] = child.text
      elif tag == "author":
        for name_node in child.findall(f"{_ATOM_NS}name"):
          authors.append(name_node.text)
      elif tag == "link":
        if child.get("title") == "pdf":
          paper["pdf_url"] = child.get("href")
      elif tag == "primary_category":
        paper["primary_category"] = child.get("term")
      elif tag in {"doi", "journal_ref", "comment"}:
        paper[tag] = child.text

    paper["authors"] = authors
    results.append(paper)

  return results


def search_arxiv(args: argparse.Namespace) -> int:
  """Runs the search and prints one JSON document. Returns an exit code."""
  url = build_url(args)

  try:
    xml_data = _throttled_fetch(url)
    papers = parse_feed(xml_data)
  except ET.ParseError as exc:
    print(
        json.dumps(
            {"status": "error", "message": f"malformed arXiv response: {exc}"},
            indent=2,
        )
    )
    return 1
  except (urllib.error.URLError, TimeoutError, OSError) as exc:
    # OSError covers URLError and socket timeouts; list explicitly for clarity.
    print(
        json.dumps(
            {"status": "error", "message": f"network error: {exc}"}, indent=2
        )
    )
    return 1

  print(
      json.dumps(
          {
              "status": "success",
              "results_count": len(papers),
              "papers": papers,
          },
          indent=2,
          ensure_ascii=False,
      )
  )
  return 0


def main(argv=None) -> int:
  """Entry point."""
  args = parse_args(argv)
  if not args.query and not args.id_list:
    print(
        json.dumps(
            {
                "status": "error",
                "message": "Must provide either --query or --id_list",
            },
            indent=2,
        )
    )
    return 1
  return search_arxiv(args)


if __name__ == "__main__":
  sys.exit(main())
