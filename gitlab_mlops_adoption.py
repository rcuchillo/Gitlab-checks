#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import re
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import ast
import pandas as pd
import gitlab
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta


# ----------------------------
# Logging
# ----------------------------
LOG = logging.getLogger("gitlab-mlops-adoption")


# ----------------------------
# Date helpers
# ----------------------------
def parse_dt(x: Optional[str]) -> Optional[datetime]:
    if not x:
        return None
    dt = dtparser.isoparse(x)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)


def month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def iter_month_windows(since: datetime, until: datetime) -> List[Tuple[str, datetime, datetime]]:
    """Returns list of (YYYY-MM, start_dt, end_dt_inclusive)."""
    windows = []
    cur = month_start(since)
    end_month = month_start(until)
    while cur <= end_month:
        nxt = cur + relativedelta(months=1)
        end = min(until, nxt - relativedelta(seconds=1))
        windows.append((month_key(cur), cur, end))
        cur = nxt
    return windows


def safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return (sum(vals) / len(vals)) if vals else None


def safe_median(xs: List[Optional[float]]) -> Optional[float]:
    vals = sorted([x for x in xs if x is not None and not math.isnan(x)])
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2


# ----------------------------
# GitLab client
# ----------------------------
class GL:
    def __init__(self, url: str, token: str, per_page: int = 100):
        self.gl = gitlab.Gitlab(url=url, private_token=token, per_page=per_page)
        self.gl.auth()

    def group_projects(self, group_path: str, include_subgroups: bool) -> List[Any]:
        g = self.gl.groups.get(group_path)
        ps = g.projects.list(include_subgroups=include_subgroups, all=True)
        out = []
        for p in ps:
            try:
                out.append(self.gl.projects.get(p.id))
            except gitlab.exceptions.GitlabGetError:
                continue
        return out

    def group_members(self, group_path: str) -> List[Tuple[int, str, str]]:
        g = self.gl.groups.get(group_path)
        members = g.members.list(all=True)
        out = []
        for m in members:
            out.append((m.id, getattr(m, "username", "") or "", getattr(m, "name", "") or ""))
        return out

    def latest_sha_before(self, project, ref: str, until: datetime) -> Optional[str]:
        """Latest commit sha on ref at or before 'until'."""
        try:
            commits = project.commits.list(ref_name=ref, until=until.isoformat(), per_page=1, get_all=False)
            return commits[0].id if commits else None
        except Exception:
            return None

    def repo_archive(self, project_id: int, sha_or_ref: str) -> bytes:
        """Repository archive tar.gz for ref/sha."""
        return self.gl.http_get(
            f"/projects/{project_id}/repository/archive",
            query_data={"sha": sha_or_ref},
            raw=True,
        )

    def mrs_updated_after(self, project, since: datetime) -> List[Any]:
        # We bucket ourselves by created_at / merged_at / closed_at
        try:
            return project.mergerequests.list(
                state="all",
                scope="all",
                updated_after=since.isoformat(),
                per_page=100,
                get_all=True,
            )
        except Exception:
            return []

    def mr_notes(self, mr) -> List[Any]:
        try:
            return mr.notes.list(get_all=True)
        except Exception:
            return []

    def mr_changes(self, project_id: int, mr_iid: int) -> Optional[Dict[str, Any]]:
        try:
            return self.gl.http_get(f"/projects/{project_id}/merge_requests/{mr_iid}/changes")
        except Exception:
            return None

    def mr_pipelines(self, project_id: int, mr_iid: int) -> List[Dict[str, Any]]:
        """MR pipelines list (may be empty if not enabled)."""
        try:
            data = self.gl.http_get(f"/projects/{project_id}/merge_requests/{mr_iid}/pipelines")
            return data or []
        except Exception:
            return []

    def pipelines_on_ref(self, project, ref: str, since: datetime) -> List[Any]:
        try:
            return project.pipelines.list(ref=ref, updated_after=since.isoformat(), per_page=100, get_all=True)
        except Exception:
            return []

    def pipeline_details(self, project, pipeline_id: int) -> Dict[str, Any]:
        try:
            p = project.pipelines.get(pipeline_id)
            return {
                "status": getattr(p, "status", None),
                "coverage": getattr(p, "coverage", None),
                "duration": getattr(p, "duration", None),
                "created_at": getattr(p, "created_at", None),
                "ref": getattr(p, "ref", None),
            }
        except Exception:
            return {}

    def user_events(self, user_id: int, since: datetime, until: datetime, action: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"after": since.date().isoformat(), "before": until.date().isoformat(), "per_page": 100}
        if action:
            params["action"] = action
        events = []
        page = 1
        while True:
            params["page"] = page
            batch = self.gl.http_get(f"/users/{user_id}/events", query_data=params) or []
            if not batch:
                break
            events.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return events


# ----------------------------
# Snapshot scanning helpers
# ----------------------------

REQ_FILES = [
    "requirements.txt",
    "requirements-dev.txt",
    "requirements_dev.txt",
    "pyproject.toml",
]

CI_FILE = ".gitlab-ci.yml"
IMPORT_LINE_RE = re.compile(r"^\s*(from\s+([A-Za-z_]\w*)\b|import\s+([A-Za-z_]\w*)\b)")


@dataclass
class SnapshotStats:
    # 0/1 ints (as requested)
    has_ci_config: int = 0
    has_configs: int = 0
    has_docs: int = 0
    has_notebooks_dir: int = 0
    has_src: int = 0
    has_tests: int = 0
    has_tests_integration: int = 0
    has_scripts_dir: int = 0
    has_requirements_file: int = 0

    notebooks_count: int = 0
    scripts_count: int = 0  # scripts/*.py
    notebook_ratio: Optional[float] = None  # notebooks/(notebooks+scripts)

    # docstring coverage of functions
    functions_total: int = 0
    functions_with_docstring: int = 0
    docstring_coverage_pct: Optional[float] = None

    # Function Forge import share
    import_symbols_total: int = 0
    forge_import_symbols: int = 0
    forge_import_symbols_pct: Optional[float] = None


def extract_python_sources_from_archive(
    tar_bytes: bytes,
    include_prefixes: Tuple[str, ...] = ("src/", "scripts/"),
    exclude_prefixes: Tuple[str, ...] = ("tests/", "tests_integration/", ".venv/", "venv/", "dist/", "build/"),
    max_files: int = 8000,
    max_file_bytes: int = 2_000_000,
) -> Dict[str, str]:
    """
    Reads .py files from a tar.gz archive into memory.
    Focused on src/ and scripts/ by default.
    """
    py_sources: Dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if len(py_sources) >= max_files:
                break
            if not m.isfile():
                continue
            name = m.name
            if not name.endswith(".py"):
                continue
            if m.size and m.size > max_file_bytes:
                continue

            # Strip archive top folder
            norm = name.split("/", 1)[1] if "/" in name else name

            if include_prefixes and not any(norm.startswith(p) for p in include_prefixes):
                continue
            if exclude_prefixes and any(norm.startswith(p) for p in exclude_prefixes):
                continue

            f = tf.extractfile(m)
            if not f:
                continue
            try:
                src = f.read().decode("utf-8", errors="replace")
            except Exception:
                continue
            py_sources[norm] = src

    return py_sources


def scan_snapshot(tar_bytes: bytes, forge_prefixes: List[str]) -> SnapshotStats:
    st = SnapshotStats()
    forge_prefixes = [p.strip() for p in forge_prefixes if p.strip()]

    # First pass: presence + counts
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            norm = name.split("/", 1)[1] if "/" in name else name

            if norm == CI_FILE:
                st.has_ci_config = 1

            if norm.startswith("configs/"):
                st.has_configs = 1
            if norm.startswith("docs/"):
                st.has_docs = 1
            if norm.startswith("notebooks/"):
                st.has_notebooks_dir = 1
            if norm.startswith("src/"):
                st.has_src = 1
            if norm.startswith("tests/"):
                st.has_tests = 1
            if norm.startswith("tests_integration/"):
                st.has_tests_integration = 1
            if norm.startswith("scripts/"):
                st.has_scripts_dir = 1

            if any(norm == rf for rf in REQ_FILES):
                st.has_requirements_file = 1

            if norm.endswith(".ipynb"):
                st.notebooks_count += 1
            if norm.startswith("scripts/") and norm.endswith(".py"):
                st.scripts_count += 1

    denom = st.notebooks_count + st.scripts_count
    st.notebook_ratio = (st.notebooks_count / denom) if denom > 0 else None

    # Second pass: AST for docstring coverage + import share
    py_sources = extract_python_sources_from_archive(tar_bytes, include_prefixes=("src/", "scripts/"))

    import_symbols_total = 0
    forge_symbols = 0
    fn_total = 0
    fn_doc = 0

    for _, src in py_sources.items():
        try:
            tree = ast.parse(src)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_total += 1
                if ast.get_docstring(node, clean=False) is not None:
                    fn_doc += 1

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_symbols_total += 1
                    top = (alias.name or "").split(".", 1)[0]
                    if top in forge_prefixes:
                        forge_symbols += 1
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".", 1)[0] if mod else ""
                for _alias in node.names:
                    import_symbols_total += 1
                    if top in forge_prefixes:
                        forge_symbols += 1

    st.functions_total = fn_total
    st.functions_with_docstring = fn_doc
    st.docstring_coverage_pct = (100.0 * fn_doc / fn_total) if fn_total > 0 else None

    st.import_symbols_total = import_symbols_total
    st.forge_import_symbols = forge_symbols
    st.forge_import_symbols_pct = (100.0 * forge_symbols / import_symbols_total) if import_symbols_total > 0 else None

    return st


# ----------------------------
# MR diff contributions (merged MRs)
# ----------------------------

TRIPLE_RE = re.compile(r"'''|\"\"\"")


@dataclass
class DiffContribAcc:
    added_docstring_lines: int = 0
    test_files_touched: int = 0
    test_lines_added: int = 0

    notebooks_added_files: int = 0
    notebooks_deleted_files: int = 0

    files_touched_py: int = 0
    import_total_added: int = 0
    forge_imports_added: int = 0
    forge_files_hit: int = 0

    # CI (MR pipeline) attribution
    mr_pipelines_total: int = 0
    mr_pipelines_success: int = 0
    mr_pipelines_failed: int = 0
    mr_pipelines_canceled: int = 0
    mrs_merged_with_mr_pipeline: int = 0
    mr_pipeline_durations: List[float] = None
    mr_pipeline_coverages: List[float] = None

    def __post_init__(self):
        if self.mr_pipeline_durations is None:
            self.mr_pipeline_durations = []
        if self.mr_pipeline_coverages is None:
            self.mr_pipeline_coverages = []


def analyze_added_lines(lines: List[str], forge_prefixes: List[str]) -> Tuple[int, int, int]:
    """
    Returns: added_docstring_lines, import_total, forge_imports
    Docstring is heuristic from triple quotes toggling in added lines.
    """
    forge_prefixes = [p.strip() for p in forge_prefixes if p.strip()]

    doc_lines = 0
    import_total = 0
    forge_imports = 0
    in_doc = False

    for ln in lines:
        if not ln.strip():
            continue

        if TRIPLE_RE.search(ln):
            doc_lines += 1
            in_doc = not in_doc
        elif in_doc:
            doc_lines += 1

        m = IMPORT_LINE_RE.match(ln)
        if m:
            import_total += 1
            top = m.group(2) or m.group(3) or ""
            if top in forge_prefixes:
                forge_imports += 1

    return doc_lines, import_total, forge_imports


def compute_diff_contrib(changes_payload: Dict[str, Any], forge_prefixes: List[str]) -> DiffContribAcc:
    acc = DiffContribAcc()
    changes = changes_payload.get("changes") or []

    for ch in changes:
        path = ch.get("new_path") or ch.get("old_path") or ""
        new_file = bool(ch.get("new_file"))
        deleted_file = bool(ch.get("deleted_file"))

        # notebooks delta
        if path.endswith(".ipynb"):
            if new_file:
                acc.notebooks_added_files += 1
            if deleted_file:
                acc.notebooks_deleted_files += 1

        # tests touched
        is_test = path.startswith("tests/") or path.startswith("tests_integration/")
        if is_test:
            acc.test_files_touched += 1

        diff = ch.get("diff") or ""
        if not diff:
            continue

        if not path.endswith(".py"):
            continue

        added_lines = []
        for line in diff.splitlines():
            if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
                continue
            if line.startswith("+") and not line.startswith("++"):
                added_lines.append(line[1:])

        if not added_lines:
            continue

        acc.files_touched_py += 1

        doc_lines, import_total, forge_imports = analyze_added_lines(added_lines, forge_prefixes)
        acc.added_docstring_lines += doc_lines
        acc.import_total_added += import_total
        acc.forge_imports_added += forge_imports
        if forge_imports > 0:
            acc.forge_files_hit += 1

        if is_test:
            nonblank = sum(1 for ln in added_lines if ln.strip())
            acc.test_lines_added += nonblank

    return acc


def diff_forge_reuse_score(acc: DiffContribAcc) -> Optional[float]:
    """0-100 score based on file hit % and import statement share in added lines."""
    if acc.files_touched_py <= 0:
        return None
    file_pct = acc.forge_files_hit / acc.files_touched_py
    stmt_pct = (acc.forge_imports_added / acc.import_total_added) if acc.import_total_added > 0 else 0.0
    return 100.0 * (0.5 * file_pct + 0.5 * stmt_pct)


def pipeline_bucket_status(status: Optional[str]) -> str:
    if status == "success":
        return "success"
    if status == "failed":
        return "failed"
    if status == "canceled":
        return "canceled"
    return "other"


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="GitLab group path, e.g. gcqa/mlops")
    ap.add_argument("--since", required=True, help="ISO datetime, e.g. 2026-01-01T00:00:00Z")
    ap.add_argument("--until", required=True, help="ISO datetime, e.g. 2026-12-31T23:59:59Z")
    ap.add_argument("--outdir", default="out", help="Output directory for CSVs")
    ap.add_argument("--include-subgroups", action="store_true", default=True)
    ap.add_argument("--forge-prefixes", default="function_forge",
                    help="Comma-separated top-level import names for Function Forge (default: function_forge)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--max-projects", type=int, default=10_000, help="Safety limit on number of projects processed")
    ap.add_argument("--max-mrs-per-project", type=int, default=50_000,
                    help="Safety limit on MRs pulled per project (updated_after filter still applies)")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    url = os.getenv("GITLAB_URL")
    token = os.getenv("GITLAB_TOKEN")
    if not url or not token:
        LOG.error("Set GITLAB_URL and GITLAB_TOKEN env vars.")
        return 2

    since = parse_dt(args.since)
    until = parse_dt(args.until)
    if not since or not until or until < since:
        LOG.error("Invalid since/until.")
        return 2

    forge_prefixes = [x.strip() for x in args.forge_prefixes.split(",") if x.strip()]
    month_windows = iter_month_windows(since, until)
    months = [m for (m, _, _) in month_windows]

    gl = GL(url, token)

    LOG.info("Loading group projects and members...")
    projects = gl.group_projects(args.group, include_subgroups=args.include_subgroups)[: args.max_projects]
    members = gl.group_members(args.group)
    LOG.info("Found %d projects, %d members", len(projects), len(members))

    # ----------------------------
    # Seed rows
    # ----------------------------
    project_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for p in projects:
        for (m, _, _) in month_windows:
            project_rows[(p.id, m)] = {
                "month": m,
                "project_id": p.id,
                "project": getattr(p, "path_with_namespace", ""),
                "default_branch": getattr(p, "default_branch", "") or "main",

                # snapshot structure (0/1) + success flag
                "snapshot_ok": 0,
                "has_ci_config": 0,
                "has_configs": 0,
                "has_docs": 0,
                "has_notebooks_dir": 0,
                "has_src": 0,
                "has_tests": 0,
                "has_tests_integration": 0,
                "has_scripts_dir": 0,
                "has_requirements_file": 0,

                # snapshot content metrics
                "functions_total": None,
                "functions_with_docstring": None,
                "docstring_coverage_pct": None,
                "forge_import_symbols_pct": None,
                "notebooks_count": None,
                "scripts_count": None,
                "notebook_ratio": None,

                # activity volumes
                "mrs_created": 0,
                "mrs_merged": 0,
                "mr_notes_total": 0,

                # NEW: MR close metrics
                "mrs_closed": 0,
                "mr_time_to_close_hours_median": None,
                "mr_time_to_close_hours_mean": None,

                # CI usage: default branch pipelines
                "default_branch_pipelines_total": 0,
                "default_branch_pipelines_success": 0,
                "default_branch_pipelines_failed": 0,
                "default_branch_pipelines_canceled": 0,
                "default_branch_pipeline_success_rate": None,
                "default_branch_pipeline_duration_median_seconds": None,
                "default_branch_pipeline_duration_mean_seconds": None,
                "default_branch_pipeline_coverage_median": None,
                "default_branch_pipeline_coverage_mean": None,

                # MR diff contributions aggregated for project
                "mr_added_docstring_lines": 0,
                "mr_test_files_touched": 0,
                "mr_test_lines_added": 0,
                "mr_notebooks_added_files": 0,
                "mr_notebooks_deleted_files": 0,
                "mr_notebooks_net_files": 0,
                "mr_forge_reuse_score": None,

                # CI usage: MR pipelines (attributed to merge month)
                "mr_pipelines_total": 0,
                "mr_pipelines_success": 0,
                "mr_pipelines_failed": 0,
                "mr_pipelines_canceled": 0,
                "mr_pipeline_success_rate": None,
                "mrs_merged_with_mr_pipeline": 0,
                "mr_pipeline_adoption_rate": None,
                "mr_pipeline_duration_median_seconds": None,
                "mr_pipeline_duration_mean_seconds": None,
                "mr_pipeline_coverage_median": None,
                "mr_pipeline_coverage_mean": None,
            }

    user_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for (uid, username, name) in members:
        for (m, _, _) in month_windows:
            user_rows[(uid, m)] = {
                "month": m,
                "user_id": uid,
                "username": username,
                "name": name,

                # volumes
                "commits_estimated": 0,
                "mrs_created": 0,
                "mrs_merged": 0,
                "mr_notes_written": 0,   # strict: MR notes authored
                "comment_events": 0,     # broad: events action=commented

                # NEW: MR close metrics (user)
                "mrs_closed": 0,
                "mr_time_to_close_hours_median": None,
                "mr_time_to_close_hours_mean": None,

                # MR diff contributions (for merged MRs)
                "mr_added_docstring_lines": 0,
                "mr_test_files_touched": 0,
                "mr_test_lines_added": 0,
                "mr_notebooks_added_files": 0,
                "mr_notebooks_deleted_files": 0,
                "mr_notebooks_net_files": 0,
                "mr_forge_reuse_score": None,

                # CI usage: MR pipelines attributed to merged MRs
                "mr_pipelines_total": 0,
                "mr_pipelines_success": 0,
                "mr_pipelines_failed": 0,
                "mr_pipelines_canceled": 0,
                "mr_pipeline_success_rate": None,
                "mrs_merged_with_mr_pipeline": 0,
                "mr_pipeline_adoption_rate": None,
                "mr_pipeline_duration_median_seconds": None,
                "mr_pipeline_duration_mean_seconds": None,
                "mr_pipeline_coverage_mean": None,
            }

    diff_acc_proj: Dict[Tuple[int, str], DiffContribAcc] = {}
    diff_acc_user: Dict[Tuple[int, str], DiffContribAcc] = {}

    # NEW: MR close time accumulators
    close_times_proj: Dict[Tuple[int, str], List[float]] = {}
    close_times_user: Dict[Tuple[int, str], List[float]] = {}

    # ----------------------------
    # User events volumes
    # ----------------------------
    LOG.info("Collecting user events for commit/comment volumes...")
    for (uid, _, _) in members:
        pushed = gl.user_events(uid, since, until, action="pushed")
        for ev in pushed:
            t = parse_dt(ev.get("created_at"))
            if not t:
                continue
            mk = month_key(t)
            if (uid, mk) not in user_rows:
                continue
            push_data = ev.get("push_data") or {}
            cc = push_data.get("commit_count")
            if isinstance(cc, int):
                user_rows[(uid, mk)]["commits_estimated"] += cc

        commented = gl.user_events(uid, since, until, action="commented")
        for ev in commented:
            t = parse_dt(ev.get("created_at"))
            if not t:
                continue
            mk = month_key(t)
            if (uid, mk) in user_rows:
                user_rows[(uid, mk)]["comment_events"] += 1

    # ----------------------------
    # Per-project: MRs, notes, MR close time, MR diffs + MR pipeline CI attribution
    # ----------------------------
    LOG.info("Collecting MR volumes, notes, close time, MR diff contributions, and MR pipeline CI attribution...")
    for idx, project in enumerate(projects, start=1):
        pid = project.id
        pname = getattr(project, "path_with_namespace", str(pid))
        LOG.info("[%d/%d] Project %s", idx, len(projects), pname)

        mrs = gl.mrs_updated_after(project, since)[: args.max_mrs_per_project]
        LOG.info("  MRs pulled: %d (updated after %s)", len(mrs), since.date().isoformat())

        for mr in mrs:
            author = (getattr(mr, "author", {}) or {})
            author_id = author.get("id")
            mr_iid = mr.iid
            state = getattr(mr, "state", None)

            created_at = parse_dt(getattr(mr, "created_at", None))
            merged_at = parse_dt(getattr(mr, "merged_at", None))
            closed_at = parse_dt(getattr(mr, "closed_at", None))

            # MR created volume
            if created_at:
                mk = month_key(created_at)
                if (pid, mk) in project_rows:
                    project_rows[(pid, mk)]["mrs_created"] += 1
                if author_id and (author_id, mk) in user_rows:
                    user_rows[(author_id, mk)]["mrs_created"] += 1

            # MR merged volume
            merged_month = None
            if state == "merged" and merged_at:
                mk = month_key(merged_at)
                if (pid, mk) in project_rows:
                    project_rows[(pid, mk)]["mrs_merged"] += 1
                if author_id and (author_id, mk) in user_rows:
                    user_rows[(author_id, mk)]["mrs_merged"] += 1
                merged_month = mk

            # NEW: time to close (merged or closed), bucket by close timestamp month
            close_ts = None
            if state == "merged" and merged_at:
                close_ts = merged_at
            elif state == "closed" and closed_at:
                close_ts = closed_at

            if created_at and close_ts:
                mk = month_key(close_ts)
                hrs = (close_ts - created_at).total_seconds() / 3600.0
                if (pid, mk) in project_rows:
                    project_rows[(pid, mk)]["mrs_closed"] += 1
                    close_times_proj.setdefault((pid, mk), []).append(hrs)
                if author_id and (author_id, mk) in user_rows:
                    user_rows[(author_id, mk)]["mrs_closed"] += 1
                    close_times_user.setdefault((author_id, mk), []).append(hrs)

            # Notes volume (bucket by note timestamp)
            notes = gl.mr_notes(mr)
            for n in notes:
                if getattr(n, "system", False):
                    continue
                nt = parse_dt(getattr(n, "created_at", None))
                if not nt:
                    continue
                mk = month_key(nt)
                if (pid, mk) in project_rows:
                    project_rows[(pid, mk)]["mr_notes_total"] += 1

                na = (getattr(n, "author", {}) or {}).get("id")
                if na and (na, mk) in user_rows:
                    user_rows[(na, mk)]["mr_notes_written"] += 1

            # MR diff contributions + MR pipeline CI (only for merged MRs, attributed to merge month)
            if merged_month and author_id and (pid, merged_month) in project_rows and (author_id, merged_month) in user_rows:
                # 1) MR diff contributions
                changes = gl.mr_changes(pid, mr_iid)
                if changes:
                    acc = compute_diff_contrib(changes, forge_prefixes)

                    diff_acc_proj.setdefault((pid, merged_month), DiffContribAcc())
                    diff_acc_user.setdefault((author_id, merged_month), DiffContribAcc())

                    for dst in (diff_acc_proj[(pid, merged_month)], diff_acc_user[(author_id, merged_month)]):
                        dst.added_docstring_lines += acc.added_docstring_lines
                        dst.test_files_touched += acc.test_files_touched
                        dst.test_lines_added += acc.test_lines_added
                        dst.notebooks_added_files += acc.notebooks_added_files
                        dst.notebooks_deleted_files += acc.notebooks_deleted_files
                        dst.files_touched_py += acc.files_touched_py
                        dst.import_total_added += acc.import_total_added
                        dst.forge_imports_added += acc.forge_imports_added
                        dst.forge_files_hit += acc.forge_files_hit

                # 2) MR pipeline CI attribution (latest MR pipeline)
                mr_pipes = gl.mr_pipelines(pid, mr_iid)
                if mr_pipes:
                    mr_pipes_sorted = sorted(
                        mr_pipes,
                        key=lambda x: parse_dt(x.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
                        reverse=True,
                    )
                    chosen = mr_pipes_sorted[0]
                    pipeline_id = chosen.get("id")
                    if pipeline_id:
                        details = gl.pipeline_details(project, int(pipeline_id))
                        status = pipeline_bucket_status(details.get("status"))
                        duration = details.get("duration")
                        coverage = details.get("coverage")

                        for dst in (diff_acc_proj.setdefault((pid, merged_month), DiffContribAcc()),
                                    diff_acc_user.setdefault((author_id, merged_month), DiffContribAcc())):
                            dst.mr_pipelines_total += 1
                            dst.mrs_merged_with_mr_pipeline += 1
                            if status == "success":
                                dst.mr_pipelines_success += 1
                            elif status == "failed":
                                dst.mr_pipelines_failed += 1
                            elif status == "canceled":
                                dst.mr_pipelines_canceled += 1

                            if isinstance(duration, (int, float)):
                                dst.mr_pipeline_durations.append(float(duration))
                            try:
                                if coverage is not None:
                                    dst.mr_pipeline_coverages.append(float(coverage))
                            except Exception:
                                pass

    # Finalize MR close time mean/median
    LOG.info("Finalizing MR close-time metrics...")
    for (pid, mk), vals in close_times_proj.items():
        project_rows[(pid, mk)]["mr_time_to_close_hours_median"] = safe_median(vals)
        project_rows[(pid, mk)]["mr_time_to_close_hours_mean"] = safe_mean(vals)

    for (uid, mk), vals in close_times_user.items():
        user_rows[(uid, mk)]["mr_time_to_close_hours_median"] = safe_median(vals)
        user_rows[(uid, mk)]["mr_time_to_close_hours_mean"] = safe_mean(vals)

    # Write diff aggregates + MR pipeline stats into rows
    LOG.info("Finalizing diff-based contribution metrics and MR pipeline CI stats...")
    for (pid, mk), acc in diff_acc_proj.items():
        row = project_rows[(pid, mk)]
        row["mr_added_docstring_lines"] = acc.added_docstring_lines
        row["mr_test_files_touched"] = acc.test_files_touched
        row["mr_test_lines_added"] = acc.test_lines_added
        row["mr_notebooks_added_files"] = acc.notebooks_added_files
        row["mr_notebooks_deleted_files"] = acc.notebooks_deleted_files
        row["mr_notebooks_net_files"] = acc.notebooks_added_files - acc.notebooks_deleted_files
        row["mr_forge_reuse_score"] = diff_forge_reuse_score(acc)

        row["mr_pipelines_total"] = acc.mr_pipelines_total
        row["mr_pipelines_success"] = acc.mr_pipelines_success
        row["mr_pipelines_failed"] = acc.mr_pipelines_failed
        row["mr_pipelines_canceled"] = acc.mr_pipelines_canceled
        row["mrs_merged_with_mr_pipeline"] = acc.mrs_merged_with_mr_pipeline
        row["mr_pipeline_success_rate"] = (acc.mr_pipelines_success / acc.mr_pipelines_total) if acc.mr_pipelines_total else None
        merged = row["mrs_merged"]
        row["mr_pipeline_adoption_rate"] = (acc.mrs_merged_with_mr_pipeline / merged) if merged else None
        row["mr_pipeline_duration_median_seconds"] = safe_median(acc.mr_pipeline_durations)
        row["mr_pipeline_duration_mean_seconds"] = safe_mean(acc.mr_pipeline_durations)
        row["mr_pipeline_coverage_median"] = safe_median(acc.mr_pipeline_coverages)
        row["mr_pipeline_coverage_mean"] = safe_mean(acc.mr_pipeline_coverages)

    for (uid, mk), acc in diff_acc_user.items():
        row = user_rows[(uid, mk)]
        row["mr_added_docstring_lines"] = acc.added_docstring_lines
        row["mr_test_files_touched"] = acc.test_files_touched
        row["mr_test_lines_added"] = acc.test_lines_added
        row["mr_notebooks_added_files"] = acc.notebooks_added_files
        row["mr_notebooks_deleted_files"] = acc.notebooks_deleted_files
        row["mr_notebooks_net_files"] = acc.notebooks_added_files - acc.notebooks_deleted_files
        row["mr_forge_reuse_score"] = diff_forge_reuse_score(acc)

        row["mr_pipelines_total"] = acc.mr_pipelines_total
        row["mr_pipelines_success"] = acc.mr_pipelines_success
        row["mr_pipelines_failed"] = acc.mr_pipelines_failed
        row["mr_pipelines_canceled"] = acc.mr_pipelines_canceled
        row["mrs_merged_with_mr_pipeline"] = acc.mrs_merged_with_mr_pipeline
        row["mr_pipeline_success_rate"] = (acc.mr_pipelines_success / acc.mr_pipelines_total) if acc.mr_pipelines_total else None
        merged = row["mrs_merged"]
        row["mr_pipeline_adoption_rate"] = (acc.mrs_merged_with_mr_pipeline / merged) if merged else None
        row["mr_pipeline_duration_median_seconds"] = safe_median(acc.mr_pipeline_durations)
        row["mr_pipeline_duration_mean_seconds"] = safe_mean(acc.mr_pipeline_durations)
        row["mr_pipeline_coverage_mean"] = safe_mean(acc.mr_pipeline_coverages)

    # ----------------------------
    # Project snapshots + default-branch CI metrics per month
    # ----------------------------
    LOG.info("Computing monthly project snapshots + default-branch CI metrics...")
    for idx, project in enumerate(projects, start=1):
        pid = project.id
        pname = getattr(project, "path_with_namespace", str(pid))
        default_branch = getattr(project, "default_branch", "") or "main"
        LOG.info("[%d/%d] Snapshot & default-branch pipelines for %s", idx, len(projects), pname)

        # Pipelines on default branch (bucket by created_at)
        pipelines = gl.pipelines_on_ref(project, default_branch, since)

        dur_by_month: Dict[str, List[float]] = {m: [] for m in months}
        cov_by_month: Dict[str, List[float]] = {m: [] for m in months}
        status_by_month: Dict[str, Dict[str, int]] = {m: {"success": 0, "failed": 0, "canceled": 0, "other": 0} for m in months}
        total_by_month: Dict[str, int] = {m: 0 for m in months}

        for p in pipelines:
            created = parse_dt(getattr(p, "created_at", None))
            if not created or created < since or created > until:
                continue
            mk = month_key(created)
            if mk not in total_by_month:
                continue

            details = gl.pipeline_details(project, p.id)
            status_bucket = pipeline_bucket_status(details.get("status"))
            total_by_month[mk] += 1
            status_by_month[mk][status_bucket] += 1

            d = details.get("duration")
            if isinstance(d, (int, float)):
                dur_by_month[mk].append(float(d))

            cov = details.get("coverage")
            try:
                if cov is not None:
                    cov_by_month[mk].append(float(cov))
            except Exception:
                pass

        # Fill into project rows
        for mk in months:
            row = project_rows[(pid, mk)]
            row["default_branch_pipelines_total"] = total_by_month[mk]
            row["default_branch_pipelines_success"] = status_by_month[mk]["success"]
            row["default_branch_pipelines_failed"] = status_by_month[mk]["failed"]
            row["default_branch_pipelines_canceled"] = status_by_month[mk]["canceled"]

            total = total_by_month[mk]
            row["default_branch_pipeline_success_rate"] = (status_by_month[mk]["success"] / total) if total else None
            row["default_branch_pipeline_duration_median_seconds"] = safe_median(dur_by_month[mk])
            row["default_branch_pipeline_duration_mean_seconds"] = safe_mean(dur_by_month[mk])
            row["default_branch_pipeline_coverage_median"] = safe_median(cov_by_month[mk])
            row["default_branch_pipeline_coverage_mean"] = safe_mean(cov_by_month[mk])

        # Repo snapshots per month (end-of-month sha)
        for (mk, _mstart, mend) in month_windows:
            sha = gl.latest_sha_before(project, default_branch, mend)
            if not sha:
                continue
            try:
                tar_bytes = gl.repo_archive(pid, sha)
                st = scan_snapshot(tar_bytes, forge_prefixes)

                row = project_rows[(pid, mk)]
                row["snapshot_ok"] = 1  # success
                row["has_ci_config"] = st.has_ci_config
                row["has_configs"] = st.has_configs
                row["has_docs"] = st.has_docs
                row["has_notebooks_dir"] = st.has_notebooks_dir
                row["has_src"] = st.has_src
                row["has_tests"] = st.has_tests
                row["has_tests_integration"] = st.has_tests_integration
                row["has_scripts_dir"] = st.has_scripts_dir
                row["has_requirements_file"] = st.has_requirements_file

                row["functions_total"] = st.functions_total
                row["functions_with_docstring"] = st.functions_with_docstring
                row["docstring_coverage_pct"] = st.docstring_coverage_pct
                row["forge_import_symbols_pct"] = st.forge_import_symbols_pct
                row["notebooks_count"] = st.notebooks_count
                row["scripts_count"] = st.scripts_count
                row["notebook_ratio"] = st.notebook_ratio

            except Exception as e:
                # keep 0/1 folder flags at their default 0, but mark snapshot_ok=0 (already)
                LOG.warning("  Snapshot failed for %s %s: %s", pname, mk, str(e))

    # ----------------------------
    # Workspace aggregation (namespace-level)
    # ----------------------------
    LOG.info("Aggregating workspace-level (namespace) metrics...")
    df_proj = pd.DataFrame(list(project_rows.values()))
    df_user = pd.DataFrame(list(user_rows.values()))

    # Folder flags are already 0/1; workspace % = mean * 100
    folder_cols = [
        "has_ci_config", "has_configs", "has_docs", "has_notebooks_dir", "has_src", "has_tests",
        "has_tests_integration", "has_scripts_dir", "has_requirements_file",
    ]

    workspace = df_proj.groupby("month", as_index=False).agg({
        "project_id": "count",
        "snapshot_ok": "sum",
        "docstring_coverage_pct": "mean",
        "forge_import_symbols_pct": "mean",
        "notebooks_count": "sum",
        "scripts_count": "sum",
        "notebook_ratio": "mean",  # mean of project ratios

        "mrs_created": "sum",
        "mrs_merged": "sum",
        "mr_notes_total": "sum",

        "mrs_closed": "sum",
        "mr_time_to_close_hours_median": "mean",  # mean of medians across projects
        "mr_time_to_close_hours_mean": "mean",

        # default branch pipelines totals
        "default_branch_pipelines_total": "sum",
        "default_branch_pipelines_success": "sum",
        "default_branch_pipelines_failed": "sum",
        "default_branch_pipelines_canceled": "sum",
        "default_branch_pipeline_duration_median_seconds": "mean",
        "default_branch_pipeline_duration_mean_seconds": "mean",
        "default_branch_pipeline_coverage_median": "mean",
        "default_branch_pipeline_coverage_mean": "mean",

        # MR pipeline totals
        "mr_pipelines_total": "sum",
        "mr_pipelines_success": "sum",
        "mr_pipelines_failed": "sum",
        "mr_pipelines_canceled": "sum",
        "mrs_merged_with_mr_pipeline": "sum",
        "mr_pipeline_duration_median_seconds": "mean",
        "mr_pipeline_duration_mean_seconds": "mean",
        "mr_pipeline_coverage_median": "mean",
        "mr_pipeline_coverage_mean": "mean",

        # diff contribution totals
        "mr_added_docstring_lines": "sum",
        "mr_test_files_touched": "sum",
        "mr_test_lines_added": "sum",
        "mr_notebooks_net_files": "sum",
    }).rename(columns={"project_id": "projects_count"})

    # workspace notebook ratio across totals
    def _ws_ratio(row):
        denom = (row["notebooks_count"] + row["scripts_count"])
        return (row["notebooks_count"] / denom) if denom and denom > 0 else None

    workspace["workspace_notebook_ratio"] = workspace.apply(_ws_ratio, axis=1)

    # Weighted workspace pipeline success rate
    workspace["workspace_default_branch_pipeline_success_rate"] = workspace.apply(
        lambda r: (r["default_branch_pipelines_success"] / r["default_branch_pipelines_total"])
        if r["default_branch_pipelines_total"] else None,
        axis=1,
    )
    workspace["workspace_mr_pipeline_success_rate"] = workspace.apply(
        lambda r: (r["mr_pipelines_success"] / r["mr_pipelines_total"])
        if r["mr_pipelines_total"] else None,
        axis=1,
    )
    workspace["workspace_mr_pipeline_adoption_rate"] = workspace.apply(
        lambda r: (r["mrs_merged_with_mr_pipeline"] / r["mrs_merged"])
        if r["mrs_merged"] else None,
        axis=1,
    )

    # Folder adoption percentages (only meaningful where snapshot_ok>0)
    # If you want to exclude failed snapshots from % calculations, filter df_proj[df_proj.snapshot_ok==1] here.
    for c in folder_cols:
        workspace[f"pct_projects_{c}"] = 100.0 * df_proj.groupby("month")[c].mean().values

    # ----------------------------
    # Write CSVs (folder flags remain 0/1)
    # ----------------------------
    os.makedirs(args.outdir, exist_ok=True)

    proj_path = os.path.join(args.outdir, "project_monthly.csv")
    user_path = os.path.join(args.outdir, "user_monthly.csv")
    ws_path = os.path.join(args.outdir, "workspace_monthly.csv")

    df_proj_out = df_proj.sort_values(["month", "project"]).copy()
    df_user_out = df_user.sort_values(["month", "username"]).copy()

    df_proj_out.to_csv(proj_path, index=False)
    df_user_out.to_csv(user_path, index=False)
    workspace.sort_values(["month"]).to_csv(ws_path, index=False)

    LOG.info("Wrote: %s", proj_path)
    LOG.info("Wrote: %s", user_path)
    LOG.info("Wrote: %s", ws_path)
    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
