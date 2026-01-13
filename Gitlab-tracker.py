#!/usr/bin/env python3
"""
gitlab_mlops_adoption.py

Track MLOps adoption and contributions on GitLab at:
- Project level
- Workspace (Group) level (aggregated across projects)
- User level (commits, MRs, comments + contribution signals from MR diffs)

Outputs:
- CSV files per metric family
- A JSON summary file

Requires:
- Python 3.9+
- requests

Auth:
- GitLab Personal Access Token (PAT) with API scope

Example:
  python gitlab_mlops_adoption.py \
    --base-url https://gitlab.example.com \
    --token $GITLAB_TOKEN \
    --group-id 12345 \
    --function-forge-names function_forge,function-forge \
    --outdir ./out \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------
# Logging
# -----------------------------

LOGGER = logging.getLogger("gitlab_mlops_adoption")


def setup_logger(level: str) -> None:
    """Configure logging with a readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------
# Date helpers (monthly windows)
# -----------------------------

def first_day_of_month(d: dt.date) -> dt.date:
    """Return the first day of the month for a given date."""
    return dt.date(d.year, d.month, 1)


def add_months(d: dt.date, months: int) -> dt.date:
    """Add months to a date, keeping day=1 safe for month windows."""
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    return dt.date(year, month, 1)


def month_windows_last_full_months(n_months: int, today: Optional[dt.date] = None) -> List[Tuple[dt.date, dt.date]]:
    """
    Return a list of (start_date, end_date) for the last n full months.
    end_date is inclusive.
    """
    today = today or dt.date.today()
    this_month_start = first_day_of_month(today)
    last_month_start = add_months(this_month_start, -1)

    windows: List[Tuple[dt.date, dt.date]] = []
    cursor_start = add_months(last_month_start, -(n_months - 1))

    for i in range(n_months):
        start = add_months(cursor_start, i)
        next_start = add_months(start, 1)
        end = next_start - dt.timedelta(days=1)
        windows.append((start, end))
    return windows


def iso_datetime_range(start: dt.date, end: dt.date) -> Tuple[str, str]:
    """Convert date range to ISO datetimes (GitLab API expects timestamps)."""
    start_dt = dt.datetime.combine(start, dt.time.min).isoformat()
    end_dt = dt.datetime.combine(end, dt.time.max).isoformat()
    return start_dt, end_dt


# -----------------------------
# GitLab API client
# -----------------------------

class GitLabClient:
    """
    Minimal GitLab REST client with pagination + retries.
    """

    def __init__(self, base_url: str, token: str, timeout_s: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({"PRIVATE-TOKEN": token})

        retries = Retry(
            total=6,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST", "PUT", "DELETE"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v4{path}"

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET a GitLab API resource and return JSON."""
        url = self._url(path)
        resp = self.session.get(url, params=params or {}, timeout=self.timeout_s)
        if resp.status_code >= 400:
            LOGGER.warning("GET %s failed: %s %s", url, resp.status_code, resp.text[:300])
            resp.raise_for_status()
        return resp.json()

    def get_paginated(self, path: str, params: Optional[Dict[str, Any]] = None) -> Generator[Any, None, None]:
        """
        Yield items across all pages (GitLab uses X-Next-Page headers).
        """
        url = self._url(path)
        page = 1
        params = dict(params or {})
        params.setdefault("per_page", 100)

        while True:
            params["page"] = page
            resp = self.session.get(url, params=params, timeout=self.timeout_s)
            if resp.status_code >= 400:
                LOGGER.warning("GET(paginated) %s failed: %s %s", url, resp.status_code, resp.text[:300])
                resp.raise_for_status()

            items = resp.json()
            if not isinstance(items, list):
                raise ValueError(f"Expected list response for {path}, got: {type(items)}")

            for item in items:
                yield item

            next_page = resp.headers.get("X-Next-Page")
            if not next_page:
                break
            page = int(next_page)

    # ---- group / projects ----

    def list_group_projects(self, group_id: int) -> List[Dict[str, Any]]:
        """List projects for a group (workspace)."""
        LOGGER.info("Listing projects for group_id=%s", group_id)
        projects = list(
            self.get_paginated(
                f"/groups/{group_id}/projects",
                params={"include_subgroups": True, "archived": False, "simple": False, "order_by": "path", "sort": "asc"},
            )
        )
        LOGGER.info("Found %d projects", len(projects))
        return projects

    # ---- repo tree / files ----

    def list_repository_tree(self, project_id: int, ref: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """List repository tree entries for a project at a ref."""
        params = {"ref": ref, "recursive": recursive}
        return list(self.get_paginated(f"/projects/{project_id}/repository/tree", params=params))

    def get_file(self, project_id: int, file_path: str, ref: str) -> Optional[str]:
        """
        Fetch file content as text. Returns None on errors (e.g., binary or too large).
        """
        encoded_path = requests.utils.quote(file_path, safe="")
        try:
            data = self.get(f"/projects/{project_id}/repository/files/{encoded_path}", params={"ref": ref})
            content_b64 = data.get("content", "")
            if not content_b64:
                return None
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return decoded
        except Exception as e:
            LOGGER.debug("Failed to fetch file %s@%s: %s", file_path, ref, e)
            return None

    def get_project(self, project_id: int) -> Dict[str, Any]:
        """Get project metadata."""
        return self.get(f"/projects/{project_id}")

    # ---- pipelines ----

    def list_pipelines(self, project_id: int, ref: Optional[str], updated_after: str, updated_before: str) -> List[Dict[str, Any]]:
        """List pipelines in a time range."""
        params = {
            "updated_after": updated_after,
            "updated_before": updated_before,
            "per_page": 100,
        }
        if ref:
            params["ref"] = ref
        return list(self.get_paginated(f"/projects/{project_id}/pipelines", params=params))

    def get_pipeline(self, project_id: int, pipeline_id: int) -> Dict[str, Any]:
        """Get a pipeline (includes coverage if configured)."""
        return self.get(f"/projects/{project_id}/pipelines/{pipeline_id}")

    # ---- merge requests ----

    def list_merge_requests(self, project_id: int, created_after: str, created_before: str, state: str = "all") -> List[Dict[str, Any]]:
        """List merge requests by creation date range."""
        params = {
            "created_after": created_after,
            "created_before": created_before,
            "state": state,
            "per_page": 100,
            "order_by": "created_at",
            "sort": "asc",
        }
        return list(self.get_paginated(f"/projects/{project_id}/merge_requests", params=params))

    def list_merge_requests_merged_in_range(self, project_id: int, merged_after: str, merged_before: str) -> List[Dict[str, Any]]:
        """List merge requests merged within a range (best-effort via updated_after + filter)."""
        # GitLab doesn't provide merged_after/merged_before universally across all editions in same way.
        # We'll pull updated in range and then filter by merged_at in Python.
        params = {
            "updated_after": merged_after,
            "updated_before": merged_before,
            "state": "merged",
            "per_page": 100,
            "order_by": "updated_at",
            "sort": "asc",
        }
        mrs = list(self.get_paginated(f"/projects/{project_id}/merge_requests", params=params))
        out = []
        for mr in mrs:
            merged_at = mr.get("merged_at")
            if not merged_at:
                continue
            merged_dt = parse_gitlab_dt(merged_at)
            if merged_dt is None:
                continue
            if merged_dt >= parse_gitlab_dt(merged_after) and merged_dt <= parse_gitlab_dt(merged_before):
                out.append(mr)
        return out

    def get_mr_changes(self, project_id: int, mr_iid: int) -> Dict[str, Any]:
        """Get MR changes (includes per-file diffs)."""
        return self.get(f"/projects/{project_id}/merge_requests/{mr_iid}/changes")

    def list_mr_commits(self, project_id: int, mr_iid: int) -> List[Dict[str, Any]]:
        """List commits for a merge request (proxy for 'commits before MR raised')."""
        return list(self.get_paginated(f"/projects/{project_id}/merge_requests/{mr_iid}/commits"))

    def list_mr_notes(self, project_id: int, mr_iid: int) -> List[Dict[str, Any]]:
        """List notes/comments for a merge request."""
        return list(self.get_paginated(f"/projects/{project_id}/merge_requests/{mr_iid}/notes"))

    # ---- commits (user volumes) ----

    def list_commits(self, project_id: int, since: str, until: str, author: Optional[str] = None) -> List[Dict[str, Any]]:
        """List commits in a time range (optionally filtered by author)."""
        params = {"since": since, "until": until, "per_page": 100}
        if author:
            params["author"] = author
        return list(self.get_paginated(f"/projects/{project_id}/repository/commits", params=params))


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_gitlab_dt(s: str) -> Optional[dt.datetime]:
    """Parse GitLab ISO timestamps robustly."""
    if not s:
        return None
    try:
        # GitLab examples: "2025-12-31T12:34:56.123Z" or with offset
        s2 = s.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s2)
    except Exception:
        return None


# -----------------------------
# Static repo analysis (AST)
# -----------------------------

@dataclass
class RepoSnapshotMetrics:
    # structure
    has_configs: int
    has_docs: int
    has_notebooks: int
    has_requirements: int
    has_src: int
    has_tests: int
    has_tests_integration: int

    # code
    volume_functions: int
    volume_functions_with_docstrings: int
    pct_functions_with_docstrings: float

    volume_function_calls: int
    volume_function_forge_calls: int
    pct_function_forge_calls: float

    # artifacts
    volume_notebooks: int
    volume_scripts: int
    pct_scripts_over_notebooks_plus_scripts: float

    # CI / lifecycle (snapshot-ish, filled elsewhere)
    latest_pipeline_coverage: Optional[float]


@dataclass
class MonthlyProjectMetrics:
    project_id: int
    project_path: str
    month_start: str
    month_end: str

    pipelines_count: int
    avg_pipeline_coverage: Optional[float]

    merged_mrs_count: int
    avg_time_to_merge_hours: Optional[float]
    avg_mr_commits: Optional[float]

    # adoption contribution signals from MR diffs (net changes merged in month)
    net_notebooks: int
    net_scripts: int
    net_docstring_additions: int
    net_tests_files_added: int
    net_function_forge_imports_added: int


@dataclass
class MonthlyUserMetrics:
    user_id: int
    user_name: str
    user_username: str
    month_start: str
    month_end: str

    commits_count: int
    mrs_authored_count: int
    comments_count: int

    # contribution signals from MRs authored+merged in month
    notebooks_reduced: int
    docstrings_added: int
    tests_added: int
    function_forge_added: int


# -----------------------------
# Heuristics for diffs (MLOps contributions)
# -----------------------------

_DIFF_HUNK_RE = re.compile(r"^@@", re.MULTILINE)
_DEF_LINE_RE = re.compile(r"^\+\s*def\s+\w+\s*\(", re.MULTILINE)
_DOCSTRING_LINE_RE = re.compile(r'^\+\s*(?:r|u|f|fr|rf)?("""|\'\'\')', re.MULTILINE)
_FUNCTION_FORGE_IMPORT_RE = re.compile(r"^\+\s*(from\s+function_forge\b|import\s+function_forge\b)", re.MULTILINE)

def classify_script_path(path: str) -> bool:
    """
    Heuristic: treat "scripts" as files under scripts/ OR top-level runnable python,
    excluding src/ and tests/ and notebooks.
    """
    p = path.lower()
    if p.endswith(".py"):
        if p.startswith("src/") or p.startswith("tests/") or p.startswith("test/") or p.startswith("notebooks/"):
            return False
        if p.startswith("scripts/"):
            return True
        # top-level python file
        if "/" not in p:
            return True
        # other folders - leave as False by default
    if p.endswith(".sh") and p.startswith("scripts/"):
        return True
    return False


def count_docstring_additions_in_patch(patch: str) -> int:
    """
    Best-effort docstring additions:
    - count added def lines and nearby added triple-quote lines.
    This is approximate but works well in practice.
    """
    if not patch:
        return 0
    # Simple: count added triple quote lines (docstrings or multi-line strings)
    # but constrain by also requiring at least one added 'def' in the patch.
    defs = len(_DEF_LINE_RE.findall(patch))
    if defs == 0:
        return 0
    doc_lines = len(_DOCSTRING_LINE_RE.findall(patch))
    return min(doc_lines, defs) if doc_lines else 0


def count_function_forge_imports_added(patch: str) -> int:
    """Count added lines importing function_forge."""
    if not patch:
        return 0
    return len(_FUNCTION_FORGE_IMPORT_RE.findall(patch))


# -----------------------------
# Repo analyzer
# -----------------------------

class RepoAnalyzer:
    """
    Analyze repository snapshot at default branch for structure + code metrics.
    Uses GitLab repository tree + file API (no git clone).
    """

    def __init__(
        self,
        gl: GitLabClient,
        function_forge_names: List[str],
        max_python_files: int = 1500,
        max_file_bytes: int = 1_000_000,
        exclude_dirs: Optional[List[str]] = None,
    ) -> None:
        self.gl = gl
        self.function_forge_names = [n.strip() for n in function_forge_names if n.strip()]
        self.max_python_files = max_python_files
        self.max_file_bytes = max_file_bytes
        self.exclude_dirs = [d.strip().lower().strip("/") for d in (exclude_dirs or ["venv", ".venv", ".git", "dist", "build", "__pycache__", ".mypy_cache", ".ruff_cache"])]

    def snapshot_metrics(self, project_id: int, ref: str) -> RepoSnapshotMetrics:
        """Compute adoption snapshot metrics on a given ref."""
        LOGGER.info("Analyzing repo snapshot project_id=%s ref=%s", project_id, ref)
        tree = self.gl.list_repository_tree(project_id, ref=ref, recursive=True)

        paths = [t["path"] for t in tree if "path" in t]
        paths_lower = [p.lower() for p in paths]

        has_configs = int(any(p.startswith("configs/") or p == "configs" for p in paths_lower))
        has_docs = int(any(p.startswith("docs/") or p == "docs" for p in paths_lower))
        has_notebooks_folder = int(any(p.startswith("notebooks/") or p == "notebooks" for p in paths_lower))
        has_src = int(any(p.startswith("src/") or p == "src" for p in paths_lower))
        has_tests = int(any(p.startswith("tests/") or p == "tests" for p in paths_lower))
        has_tests_integration = int(any(p.startswith("tests_integration/") or p == "tests_integration" for p in paths_lower))

        has_requirements = int(
            any(p in ("requirements.txt", "pyproject.toml", "setup.cfg", "setup.py") for p in paths_lower)
            or any(p.startswith("requirements/") or p == "requirements" for p in paths_lower)
        )

        notebook_paths = [p for p in paths if p.lower().endswith(".ipynb")]
        python_paths = [p for p in paths if p.lower().endswith(".py")]

        volume_notebooks = len(notebook_paths)
        volume_scripts = sum(1 for p in paths if classify_script_path(p))

        # AST-based metrics (best-effort, bounded)
        volume_functions = 0
        volume_functions_with_docstrings = 0
        volume_calls = 0
        volume_ff_calls = 0

        python_paths_filtered = self._filter_paths(python_paths)
        if len(python_paths_filtered) > self.max_python_files:
            LOGGER.warning("Too many python files (%d). Truncating to %d for analysis.",
                           len(python_paths_filtered), self.max_python_files)
            python_paths_filtered = python_paths_filtered[: self.max_python_files]

        for i, file_path in enumerate(python_paths_filtered, 1):
            if i % 100 == 0:
                LOGGER.info("... analyzed %d/%d python files", i, len(python_paths_filtered))

            content = self.gl.get_file(project_id, file_path, ref=ref)
            if content is None:
                continue
            if len(content.encode("utf-8", errors="ignore")) > self.max_file_bytes:
                continue

            f, f_doc, calls, ff_calls = self._analyze_python_source(content)
            volume_functions += f
            volume_functions_with_docstrings += f_doc
            volume_calls += calls
            volume_ff_calls += ff_calls

        pct_doc = (100.0 * volume_functions_with_docstrings / volume_functions) if volume_functions else 0.0
        pct_ff = (100.0 * volume_ff_calls / volume_calls) if volume_calls else 0.0
        denom = (volume_notebooks + volume_scripts)
        pct_scripts = (100.0 * volume_scripts / denom) if denom else 0.0

        return RepoSnapshotMetrics(
            has_configs=has_configs,
            has_docs=has_docs,
            has_notebooks=has_notebooks_folder,
            has_requirements=has_requirements,
            has_src=has_src,
            has_tests=has_tests,
            has_tests_integration=has_tests_integration,
            volume_functions=volume_functions,
            volume_functions_with_docstrings=volume_functions_with_docstrings,
            pct_functions_with_docstrings=round(pct_doc, 2),
            volume_function_calls=volume_calls,
            volume_function_forge_calls=volume_ff_calls,
            pct_function_forge_calls=round(pct_ff, 2),
            volume_notebooks=volume_notebooks,
            volume_scripts=volume_scripts,
            pct_scripts_over_notebooks_plus_scripts=round(pct_scripts, 2),
            latest_pipeline_coverage=None,  # filled elsewhere
        )

    def _filter_paths(self, paths: List[str]) -> List[str]:
        """Exclude typical vendor/cache folders."""
        out = []
        for p in paths:
            pl = p.lower()
            if any(pl.startswith(d + "/") for d in self.exclude_dirs):
                continue
            out.append(p)
        return out

    def _analyze_python_source(self, src: str) -> Tuple[int, int, int, int]:
        """
        AST-based metrics:
        - number of functions
        - number of functions with docstrings
        - number of function calls
        - number of calls attributable to function_forge (best-effort via imports)
        """
        import ast

        try:
            tree = ast.parse(src)
        except SyntaxError:
            return 0, 0, 0, 0

        function_defs: List[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_defs.append(node)

        functions = len(function_defs)
        functions_with_doc = 0
        for fn in function_defs:
            doc = ast.get_docstring(fn)  # type: ignore[arg-type]
            if doc:
                functions_with_doc += 1

        # import map for function_forge
        module_aliases: set[str] = set()
        imported_funcs: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.function_forge_names:
                        module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in self.function_forge_names:
                    for alias in node.names:
                        imported_funcs.add(alias.asname or alias.name)

        calls = 0
        ff_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                calls += 1
                # module alias call: ff.something(...)
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in module_aliases:
                        ff_calls += 1
                # direct imported function call: foo(...)
                elif isinstance(node.func, ast.Name):
                    if node.func.id in imported_funcs:
                        ff_calls += 1

        return functions, functions_with_doc, calls, ff_calls


# -----------------------------
# Metrics extraction
# -----------------------------

def safe_mean(xs: List[float]) -> Optional[float]:
    """Compute mean with None when empty."""
    if not xs:
        return None
    return sum(xs) / len(xs)


def project_monthly_metrics(
    gl: GitLabClient,
    project: Dict[str, Any],
    month_start: dt.date,
    month_end: dt.date,
) -> Tuple[MonthlyProjectMetrics, Dict[int, MonthlyUserMetrics]]:
    """
    Compute monthly project metrics + accumulate per-user metrics from merged MRs.
    Returns:
      - MonthlyProjectMetrics
      - dict[user_id] -> MonthlyUserMetrics (partial; aggregated by caller across projects)
    """
    project_id = project["id"]
    project_path = project.get("path_with_namespace", project.get("path", str(project_id)))

    start_iso, end_iso = iso_datetime_range(month_start, month_end)

    # Pipelines
    pipelines = gl.list_pipelines(project_id, ref=None, updated_after=start_iso, updated_before=end_iso)
    coverages: List[float] = []
    for p in pipelines[:200]:  # guard rails
        pid = p.get("id")
        if pid is None:
            continue
        try:
            pfull = gl.get_pipeline(project_id, int(pid))
            cov = pfull.get("coverage")
            if cov is not None:
                try:
                    coverages.append(float(cov))
                except Exception:
                    pass
        except Exception:
            continue

    avg_cov = safe_mean(coverages)

    # MRs merged in month
    merged_mrs = gl.list_merge_requests_merged_in_range(project_id, merged_after=start_iso, merged_before=end_iso)

    time_to_merge_hours: List[float] = []
    mr_commits_counts: List[float] = []

    net_notebooks = 0
    net_scripts = 0
    net_docstrings = 0
    net_tests_files_added = 0
    net_ff_imports_added = 0

    user_bucket: Dict[int, MonthlyUserMetrics] = {}

    for mr in merged_mrs:
        mr_iid = mr.get("iid")
        if mr_iid is None:
            continue

        created_at = parse_gitlab_dt(mr.get("created_at", ""))
        merged_at = parse_gitlab_dt(mr.get("merged_at", ""))
        if created_at and merged_at:
            delta_h = (merged_at - created_at).total_seconds() / 3600.0
            if delta_h >= 0:
                time_to_merge_hours.append(delta_h)

        # commits in MR (proxy)
        try:
            commits = gl.list_mr_commits(project_id, int(mr_iid))
            mr_commits_counts.append(float(len(commits)))
        except Exception:
            pass

        # Notes (comments) for user metrics
        try:
            notes = gl.list_mr_notes(project_id, int(mr_iid))
        except Exception:
            notes = []

        # Changes/diffs for contribution signals
        try:
            changes = gl.get_mr_changes(project_id, int(mr_iid))
            change_list = changes.get("changes", []) or []
        except Exception:
            change_list = []

        # Identify author
        author = mr.get("author") or {}
        user_id = author.get("id")
        if user_id is None:
            continue

        if user_id not in user_bucket:
            user_bucket[user_id] = MonthlyUserMetrics(
                user_id=int(user_id),
                user_name=author.get("name", ""),
                user_username=author.get("username", ""),
                month_start=str(month_start),
                month_end=str(month_end),
                commits_count=0,            # filled later via commits endpoint (caller)
                mrs_authored_count=0,
                comments_count=0,
                notebooks_reduced=0,
                docstrings_added=0,
                tests_added=0,
                function_forge_added=0,
            )

        user_bucket[user_id].mrs_authored_count += 1

        # Count comments by author in this MR (notes)
        for n in notes:
            na = n.get("author") or {}
            nid = na.get("id")
            if nid is None:
                continue
            if nid not in user_bucket:
                user_bucket[nid] = MonthlyUserMetrics(
                    user_id=int(nid),
                    user_name=na.get("name", ""),
                    user_username=na.get("username", ""),
                    month_start=str(month_start),
                    month_end=str(month_end),
                    commits_count=0,
                    mrs_authored_count=0,
                    comments_count=0,
                    notebooks_reduced=0,
                    docstrings_added=0,
                    tests_added=0,
                    function_forge_added=0,
                )
            user_bucket[nid].comments_count += 1

        # Diff-based contribution heuristics
        for ch in change_list:
            new_path = (ch.get("new_path") or ch.get("old_path") or "").strip()
            old_path = (ch.get("old_path") or "").strip()
            deleted = bool(ch.get("deleted_file"))
            new_file = bool(ch.get("new_file"))
            patch = ch.get("diff") or ""

            path = new_path or old_path
            pl = path.lower()

            # notebooks net
            if pl.endswith(".ipynb"):
                if new_file:
                    net_notebooks += 1
                if deleted:
                    net_notebooks -= 1

            # scripts net
            if classify_script_path(path):
                if new_file:
                    net_scripts += 1
                if deleted:
                    net_scripts -= 1

            # tests files added
            if new_file and (pl.startswith("tests/") or pl.startswith("test/") or pl.startswith("tests_integration/")):
                net_tests_files_added += 1

            # docstring additions (approx)
            ds_added = count_docstring_additions_in_patch(patch)
            net_docstrings += ds_added

            # function_forge imports added
            ff_added = count_function_forge_imports_added(patch)
            net_ff_imports_added += ff_added

            # Update author contribution bucket (only attribute “improvements” to author)
            # Notebook reduction: count deletions of ipynb as reduction signal
            if pl.endswith(".ipynb") and deleted:
                user_bucket[user_id].notebooks_reduced += 1
            user_bucket[user_id].docstrings_added += ds_added
            user_bucket[user_id].tests_added += int(new_file and (pl.startswith("tests/") or pl.startswith("test/") or pl.startswith("tests_integration/")))
            user_bucket[user_id].function_forge_added += ff_added

    metrics = MonthlyProjectMetrics(
        project_id=int(project_id),
        project_path=str(project_path),
        month_start=str(month_start),
        month_end=str(month_end),
        pipelines_count=len(pipelines),
        avg_pipeline_coverage=(round(avg_cov, 2) if avg_cov is not None else None),
        merged_mrs_count=len(merged_mrs),
        avg_time_to_merge_hours=(round(safe_mean(time_to_merge_hours), 2) if time_to_merge_hours else None),
        avg_mr_commits=(round(safe_mean(mr_commits_counts), 2) if mr_commits_counts else None),
        net_notebooks=net_notebooks,
        net_scripts=net_scripts,
        net_docstring_additions=net_docstrings,
        net_tests_files_added=net_tests_files_added,
        net_function_forge_imports_added=net_ff_imports_added,
    )
    return metrics, user_bucket


def fill_user_commit_counts(
    gl: GitLabClient,
    projects: List[Dict[str, Any]],
    user_metrics: Dict[int, MonthlyUserMetrics],
    start_iso: str,
    end_iso: str,
) -> None:
    """
    Fill commits_count for users in the bucket.
    Best-effort: GitLab commit endpoint supports author filtering by string.
    We use username as author filter when available.
    """
    # index username -> user_id
    user_by_username = {um.user_username: uid for uid, um in user_metrics.items() if um.user_username}

    if not user_by_username:
        return

    for proj in projects:
        pid = proj["id"]
        for username, uid in user_by_username.items():
            try:
                commits = gl.list_commits(pid, since=start_iso, until=end_iso, author=username)
                user_metrics[uid].commits_count += len(commits)
            except Exception:
                continue


# -----------------------------
# Writers
# -----------------------------

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write list of dicts to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        LOGGER.warning("No rows to write: %s", path)
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: str, obj: Any) -> None:
    """Write JSON safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Track MLOps adoption on GitLab (project, workspace, user).")
    parser.add_argument("--base-url", required=True, help="GitLab base URL, e.g. https://gitlab.example.com")
    parser.add_argument("--token", required=True, help="GitLab personal access token (API scope)")
    parser.add_argument("--group-id", type=int, required=True, help="Workspace / group ID")
    parser.add_argument("--outdir", default="./out", help="Output directory")
    parser.add_argument("--months", type=int, default=3, help="Number of last full months to report (default 3)")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--function-forge-names", default="function_forge",
                        help="Comma-separated module names to treat as function_forge (e.g. function_forge,function-forge)")
    parser.add_argument("--exclude-dirs", default="venv,.venv,.git,dist,build,__pycache__,.mypy_cache,.ruff_cache",
                        help="Comma-separated directory prefixes to exclude from analysis")
    parser.add_argument("--max-python-files", type=int, default=1500, help="Max python files analyzed per project")
    parser.add_argument("--default-ref", default="", help="Override ref (branch/tag). Default: project default branch")
    args = parser.parse_args()

    setup_logger(args.log_level)

    gl = GitLabClient(base_url=args.base_url, token=args.token)
    analyzer = RepoAnalyzer(
        gl=gl,
        function_forge_names=args.function_forge_names.split(","),
        max_python_files=args.max_python_files,
        exclude_dirs=args.exclude_dirs.split(","),
    )

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Load projects
    projects = gl.list_group_projects(args.group_id)
    if not projects:
        LOGGER.error("No projects found for group_id=%s", args.group_id)
        return 2

    # -----------------------------
    # 1) Current snapshot (per project + workspace aggregate)
    # -----------------------------
    snapshot_rows: List[Dict[str, Any]] = []
    workspace_agg: Dict[str, float] = {
        "n_projects": 0,
        "has_configs_sum": 0,
        "has_docs_sum": 0,
        "has_notebooks_sum": 0,
        "has_requirements_sum": 0,
        "has_src_sum": 0,
        "has_tests_sum": 0,
        "has_tests_integration_sum": 0,
        "volume_functions_sum": 0,
        "volume_functions_with_docstrings_sum": 0,
        "volume_function_calls_sum": 0,
        "volume_function_forge_calls_sum": 0,
        "volume_notebooks_sum": 0,
        "volume_scripts_sum": 0,
    }

    LOGGER.info("Computing CURRENT snapshot metrics for %d projects", len(projects))
    for proj in projects:
        pid = proj["id"]
        ppath = proj.get("path_with_namespace", proj.get("path", str(pid)))

        try:
            default_branch = args.default_ref or proj.get("default_branch") or "main"
            snap = analyzer.snapshot_metrics(pid, ref=default_branch)

            # Latest pipeline coverage on default branch (best effort)
            today = dt.date.today()
            # look back 30 days for "latest"
            start_iso, end_iso = iso_datetime_range(today - dt.timedelta(days=30), today)
            pipelines = gl.list_pipelines(pid, ref=default_branch, updated_after=start_iso, updated_before=end_iso)
            latest_cov = None
            for p in pipelines[:10]:
                try:
                    pf = gl.get_pipeline(pid, int(p["id"]))
                    cov = pf.get("coverage")
                    if cov is not None:
                        latest_cov = float(cov)
                        break
                except Exception:
                    continue

            snap.latest_pipeline_coverage = round(latest_cov, 2) if latest_cov is not None else None

            row = {"project_id": pid, "project_path": ppath, "ref": default_branch, **asdict(snap)}
            snapshot_rows.append(row)

            workspace_agg["n_projects"] += 1
            workspace_agg["has_configs_sum"] += snap.has_configs
            workspace_agg["has_docs_sum"] += snap.has_docs
            workspace_agg["has_notebooks_sum"] += snap.has_notebooks
            workspace_agg["has_requirements_sum"] += snap.has_requirements
            workspace_agg["has_src_sum"] += snap.has_src
            workspace_agg["has_tests_sum"] += snap.has_tests
            workspace_agg["has_tests_integration_sum"] += snap.has_tests_integration
            workspace_agg["volume_functions_sum"] += snap.volume_functions
            workspace_agg["volume_functions_with_docstrings_sum"] += snap.volume_functions_with_docstrings
            workspace_agg["volume_function_calls_sum"] += snap.volume_function_calls
            workspace_agg["volume_function_forge_calls_sum"] += snap.volume_function_forge_calls
            workspace_agg["volume_notebooks_sum"] += snap.volume_notebooks
            workspace_agg["volume_scripts_sum"] += snap.volume_scripts

        except Exception as e:
            LOGGER.exception("Snapshot analysis failed for %s: %s", ppath, e)

    # derive workspace percentages
    wf = workspace_agg["volume_functions_sum"]
    wfd = workspace_agg["volume_functions_with_docstrings_sum"]
    wcalls = workspace_agg["volume_function_calls_sum"]
    wff = workspace_agg["volume_function_forge_calls_sum"]
    wnb = workspace_agg["volume_notebooks_sum"]
    wsc = workspace_agg["volume_scripts_sum"]

    workspace_snapshot = {
        "n_projects": int(workspace_agg["n_projects"]),
        "folder_presence_rate_configs_pct": round(100.0 * workspace_agg["has_configs_sum"] / max(1, workspace_agg["n_projects"]), 2),
        "folder_presence_rate_docs_pct": round(100.0 * workspace_agg["has_docs_sum"] / max(1, workspace_agg["n_projects"]), 2),
        "folder_presence_rate_notebooks_pct": round(100.0 * workspace_agg["has_notebooks_sum"] / max(1, workspace_agg["n_projects"]), 2),
        "folder_presence_rate_requirements_pct": round(100.0 * workspace_agg["has_requirements_sum"] / max(1, workspace_agg["n_projects"]), 2),
        "folder_presence_rate_src_pct": round(100.0 * workspace_agg["has_src_sum"] / max(1, workspace_agg["n_projects"]), 2),
        "folder_presence_rate_tests_pct": round(100.0 * workspace_agg["has_tests_sum"] / max(1, workspace_agg["n_projects"]), 2),
        "folder_presence_rate_tests_integration_pct": round(100.0 * workspace_agg["has_tests_integration_sum"] / max(1, workspace_agg["n_projects"]), 2),
        "volume_functions": int(wf),
        "pct_functions_with_docstrings": round(100.0 * wfd / wf, 2) if wf else 0.0,
        "volume_function_calls": int(wcalls),
        "pct_function_forge_calls": round(100.0 * wff / wcalls, 2) if wcalls else 0.0,
        "volume_notebooks": int(wnb),
        "volume_scripts": int(wsc),
        "pct_scripts_over_notebooks_plus_scripts": round(100.0 * wsc / (wnb + wsc), 2) if (wnb + wsc) else 0.0,
    }

    write_csv(os.path.join(outdir, "current_project_snapshot.csv"), snapshot_rows)
    write_json(os.path.join(outdir, "current_workspace_snapshot.json"), workspace_snapshot)

    # -----------------------------
    # 2) Monthly metrics (last quarter / last N full months)
    # -----------------------------
    windows = month_windows_last_full_months(args.months)
    monthly_project_rows: List[Dict[str, Any]] = []
    monthly_workspace_rows: List[Dict[str, Any]] = []
    monthly_user_rows: List[Dict[str, Any]] = []

    LOGGER.info("Computing MONTHLY metrics for %d months (last full months)", len(windows))

    for (m_start, m_end) in windows:
        LOGGER.info("Month window: %s -> %s", m_start, m_end)
        start_iso, end_iso = iso_datetime_range(m_start, m_end)

        # per-month workspace aggregation
        ws_pipelines = 0
        ws_mrs = 0
        ws_time_to_merge: List[float] = []
        ws_mr_commits: List[float] = []
        ws_covs: List[float] = []

        ws_net_notebooks = 0
        ws_net_scripts = 0
        ws_net_docstrings = 0
        ws_net_tests = 0
        ws_net_ff = 0

        # user aggregation bucket for this month across all projects
        user_bucket_month: Dict[int, MonthlyUserMetrics] = {}

        # project loop
        for proj in projects:
            try:
                p_metrics, p_user_bucket = project_monthly_metrics(gl, proj, m_start, m_end)
                monthly_project_rows.append(asdict(p_metrics))

                ws_pipelines += p_metrics.pipelines_count
                ws_mrs += p_metrics.merged_mrs_count
                if p_metrics.avg_pipeline_coverage is not None:
                    ws_covs.append(float(p_metrics.avg_pipeline_coverage))
                if p_metrics.avg_time_to_merge_hours is not None:
                    ws_time_to_merge.append(float(p_metrics.avg_time_to_merge_hours))
                if p_metrics.avg_mr_commits is not None:
                    ws_mr_commits.append(float(p_metrics.avg_mr_commits))

                ws_net_notebooks += p_metrics.net_notebooks
                ws_net_scripts += p_metrics.net_scripts
                ws_net_docstrings += p_metrics.net_docstring_additions
                ws_net_tests += p_metrics.net_tests_files_added
                ws_net_ff += p_metrics.net_function_forge_imports_added

                # merge per-project user buckets into month bucket
                for uid, um in p_user_bucket.items():
                    if uid not in user_bucket_month:
                        user_bucket_month[uid] = um
                    else:
                        user_bucket_month[uid].mrs_authored_count += um.mrs_authored_count
                        user_bucket_month[uid].comments_count += um.comments_count
                        user_bucket_month[uid].notebooks_reduced += um.notebooks_reduced
                        user_bucket_month[uid].docstrings_added += um.docstrings_added
                        user_bucket_month[uid].tests_added += um.tests_added
                        user_bucket_month[uid].function_forge_added += um.function_forge_added

            except Exception as e:
                LOGGER.exception("Monthly metrics failed for project %s: %s", proj.get("path_with_namespace"), e)

        # fill commit counts for users (best-effort)
        fill_user_commit_counts(gl, projects, user_bucket_month, start_iso, end_iso)

        # write month’s user rows
        for uid, um in user_bucket_month.items():
            monthly_user_rows.append(asdict(um))

        # workspace month row
        ws_row = {
            "month_start": str(m_start),
            "month_end": str(m_end),
            "pipelines_count": ws_pipelines,
            "avg_pipeline_coverage": round(safe_mean(ws_covs), 2) if ws_covs else None,
            "merged_mrs_count": ws_mrs,
            "avg_time_to_merge_hours": round(safe_mean(ws_time_to_merge), 2) if ws_time_to_merge else None,
            "avg_mr_commits": round(safe_mean(ws_mr_commits), 2) if ws_mr_commits else None,
            "net_notebooks": ws_net_notebooks,
            "net_scripts": ws_net_scripts,
            "net_docstring_additions": ws_net_docstrings,
            "net_tests_files_added": ws_net_tests,
            "net_function_forge_imports_added": ws_net_ff,
        }
        monthly_workspace_rows.append(ws_row)

    write_csv(os.path.join(outdir, "monthly_project_metrics.csv"), monthly_project_rows)
    write_csv(os.path.join(outdir, "monthly_workspace_metrics.csv"), monthly_workspace_rows)
    write_csv(os.path.join(outdir, "monthly_user_metrics.csv"), monthly_user_rows)

    # -----------------------------
    # 3) Summary JSON
    # -----------------------------
    summary = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "base_url": args.base_url,
        "group_id": args.group_id,
        "n_projects": len(projects),
        "months": args.months,
        "outputs": {
            "current_project_snapshot": "current_project_snapshot.csv",
            "current_workspace_snapshot": "current_workspace_snapshot.json",
            "monthly_project_metrics": "monthly_project_metrics.csv",
            "monthly_workspace_metrics": "monthly_workspace_metrics.csv",
            "monthly_user_metrics": "monthly_user_metrics.csv",
        },
        "workspace_snapshot": workspace_snapshot,
    }
    write_json(os.path.join(outdir, "summary.json"), summary)

    LOGGER.info("Done. Outputs in: %s", os.path.abspath(outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
