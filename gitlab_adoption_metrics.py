#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import math
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import gitlab
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta
import ast
import tokenize


# ----------------------------
# Helpers: datetime + bucketing
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

def iter_months(since: datetime, until: datetime) -> List[str]:
    cur = month_start(since)
    end = month_start(until)
    out = []
    while cur <= end:
        out.append(month_key(cur))
        cur = cur + relativedelta(months=1)
    return out

def clamp_month(dt: datetime, months: List[str]) -> Optional[str]:
    mk = month_key(dt)
    return mk if mk in months else None

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
# Repo snapshot analysis (Python)
# - comment lines
# - docstring lines
# - reuse import scoring
# ----------------------------

@dataclass
class PySnapshotStats:
    py_files: int = 0
    py_nonblank_lines: int = 0
    comment_lines: int = 0
    docstring_lines: int = 0
    import_total: int = 0

    forge_files_with_import: int = 0
    forge_import_count: int = 0

    src_files_with_import: int = 0
    src_import_count: int = 0


def _count_docstring_lines_via_ast(source: str) -> int:
    """
    Counts lines in module/class/function docstrings using AST lineno/end_lineno.
    Best effort; returns 0 if parsing fails.
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return 0

    total = 0

    def doc_expr_span(body: List[ast.stmt]) -> Optional[Tuple[int, int]]:
        if not body:
            return None
        first = body[0]
        if isinstance(first, ast.Expr):
            val = first.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                # Python 3.8+ has end_lineno
                start = getattr(first, "lineno", None)
                end = getattr(first, "end_lineno", None)
                if start and end:
                    return (start, end)
                # fallback: approximate by counting lines in string literal
                if start:
                    return (start, start + max(0, len(val.value.splitlines()) - 1))
        return None

    # module docstring
    span = doc_expr_span(tree.body)
    if span:
        total += (span[1] - span[0] + 1)

    # class/function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            span = doc_expr_span(node.body)
            if span:
                total += (span[1] - span[0] + 1)

    return total


def _count_comment_lines_via_tokenize(source: str) -> int:
    """
    Counts unique line numbers that contain a COMMENT token.
    """
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    except Exception:
        return 0

    lines = set()
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            lines.add(tok.start[0])
    return len(lines)


def _iter_import_nodes(source: str) -> Iterable[Tuple[str, str]]:
    """
    Yields tuples describing imports:
      ("import", "module") for `import x` (module is the top-level name)
      ("from", "module") for `from x.y import z` (module is the full module)
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name or ""
                top = name.split(".")[0] if name else ""
                out.append(("import", top))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0] if mod else ""
            out.append(("from", top))
    return out


def analyze_python_snapshot(py_sources: Dict[str, str],
                            forge_prefixes: List[str],
                            src_prefixes: List[str]) -> PySnapshotStats:
    stats = PySnapshotStats()
    forge_prefixes = [p.strip() for p in forge_prefixes if p.strip()]
    src_prefixes = [p.strip() for p in src_prefixes if p.strip()]

    for path, src in py_sources.items():
        stats.py_files += 1
        nonblank = sum(1 for ln in src.splitlines() if ln.strip())
        stats.py_nonblank_lines += nonblank

        doc_lines = _count_docstring_lines_via_ast(src)
        com_lines = _count_comment_lines_via_tokenize(src)
        stats.docstring_lines += doc_lines
        stats.comment_lines += com_lines

        imports = list(_iter_import_nodes(src))
        stats.import_total += len(imports)

        # Forge/src reuse: count import statements matching prefix lists
        forge_hit = 0
        src_hit = 0
        for kind, top in imports:
            if any(top == p for p in forge_prefixes):
                forge_hit += 1
            if any(top == p for p in src_prefixes):
                src_hit += 1

        if forge_hit > 0:
            stats.forge_files_with_import += 1
            stats.forge_import_count += forge_hit
        if src_hit > 0:
            stats.src_files_with_import += 1
            stats.src_import_count += src_hit

    return stats


def reuse_score(file_hits: int, total_files: int, import_hits: int, import_total: int) -> Optional[float]:
    """
    Score in [0, 100].
      file_pct = files_with_matching_import / total_files
      stmt_pct = matching_import_statements / total_import_statements
      score = 50*file_pct + 50*stmt_pct, scaled to 100
    """
    if total_files <= 0:
        return None
    file_pct = file_hits / total_files
    stmt_pct = (import_hits / import_total) if import_total > 0 else 0.0
    return 100.0 * (0.5 * file_pct + 0.5 * stmt_pct)


def doc_ratio(numerator: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return numerator / denom


# ----------------------------
# MR diff analysis (added lines)
# - comment lines added
# - docstring lines added (heuristic)
# - reuse import scoring from diffs
# ----------------------------

IMPORT_RE = re.compile(r"^\s*(from\s+([A-Za-z_][\w]*)|import\s+([A-Za-z_][\w]*))")

def analyze_added_lines_for_docs_and_imports(
    added_lines: List[str],
    forge_prefixes: List[str],
    src_prefixes: List[str],
) -> Dict[str, Any]:
    forge_prefixes = [p.strip() for p in forge_prefixes if p.strip()]
    src_prefixes = [p.strip() for p in src_prefixes if p.strip()]

    nonblank = 0
    comment_lines = 0

    # docstring heuristic: count lines between triple quotes toggles
    docstring_lines = 0
    in_doc = False
    triple_pat = re.compile(r"'''|\"\"\"")

    import_total = 0
    forge_imports = 0
    src_imports = 0

    for ln in added_lines:
        if not ln.strip():
            continue
        nonblank += 1

        stripped = ln.lstrip()
        if stripped.startswith("#"):
            comment_lines += 1

        # docstring heuristic
        if triple_pat.search(ln):
            # count this line as docstring-ish
            docstring_lines += 1
            # toggle (might toggle twice on same line; still ok as heuristic)
            in_doc = not in_doc
        elif in_doc:
            docstring_lines += 1

        m = IMPORT_RE.match(ln)
        if m:
            import_total += 1
            top = m.group(2) or m.group(3) or ""
            if any(top == p for p in forge_prefixes):
                forge_imports += 1
            if any(top == p for p in src_prefixes):
                src_imports += 1

    return {
        "added_nonblank": nonblank,
        "added_comment_lines": comment_lines,
        "added_docstring_lines": docstring_lines,
        "import_total": import_total,
        "forge_imports": forge_imports,
        "src_imports": src_imports,
    }


# ----------------------------
# GitLab extractor
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
                pass
        return out

    def group_members(self, group_path: str) -> List[Tuple[int, str, str]]:
        g = self.gl.groups.get(group_path)
        members = g.members.list(all=True)
        out = []
        for m in members:
            out.append((m.id, getattr(m, "username", "") or "", getattr(m, "name", "") or ""))
        return out

    def project_commits_count(self, project, ref: str, since: datetime, until: datetime) -> int:
        # commits API supports since/until
        try:
            commits = project.commits.list(ref_name=ref, since=since.isoformat(), until=until.isoformat(),
                                          per_page=100, get_all=True)
            return len(commits)
        except Exception:
            return 0

    def project_commits_latest_sha_before(self, project, ref: str, until: datetime) -> Optional[str]:
        """
        Return sha of latest commit on ref at or before 'until'.
        """
        try:
            commits = project.commits.list(ref_name=ref, until=until.isoformat(), per_page=1, get_all=False)
            if commits:
                return commits[0].id
        except Exception:
            pass
        return None

    def project_repo_archive_tar_gz(self, project_id: int, sha_or_ref: str) -> bytes:
        """
        Download repository archive (tar.gz) for a ref/sha.
        """
        # /projects/:id/repository/archive?sha=...
        return self.gl.http_get(
            f"/projects/{project_id}/repository/archive",
            query_data={"sha": sha_or_ref},
            raw=True,
        )

    def merged_mrs_in_window(self, project, since: datetime, until: datetime) -> List[Any]:
        # Filter by updated_after and then merged_at window
        try:
            mrs = project.mergerequests.list(
                state="merged", scope="all",
                updated_after=since.isoformat(),
                per_page=100, get_all=True
            )
        except Exception:
            return []
        out = []
        for mr in mrs:
            ma = parse_dt(getattr(mr, "merged_at", None))
            if ma and since <= ma <= until:
                out.append(mr)
        return out

    def mrs_updated_after(self, project, since: datetime) -> List[Any]:
        try:
            return project.mergerequests.list(
                state="all", scope="all",
                updated_after=since.isoformat(),
                per_page=100, get_all=True
            )
        except Exception:
            return []

    def mr_notes(self, mr) -> List[Any]:
        try:
            return mr.notes.list(get_all=True)
        except Exception:
            return []

    def mr_changes(self, project_id: int, mr_iid: int) -> Optional[Dict[str, Any]]:
        # /projects/:id/merge_requests/:iid/changes
        try:
            return self.gl.http_get(f"/projects/{project_id}/merge_requests/{mr_iid}/changes")
        except Exception:
            return None

    def mr_pipelines(self, project_id: int, mr_iid: int) -> List[Dict[str, Any]]:
        # /projects/:id/merge_requests/:iid/pipelines
        try:
            data = self.gl.http_get(f"/projects/{project_id}/merge_requests/{mr_iid}/pipelines")
            return data or []
        except Exception:
            return []

    def pipeline_details(self, project, pipeline_id: int) -> Dict[str, Any]:
        try:
            p = project.pipelines.get(pipeline_id)
            return {
                "status": getattr(p, "status", None),
                "duration": getattr(p, "duration", None),
                "coverage": getattr(p, "coverage", None),
                "created_at": getattr(p, "created_at", None),
                "ref": getattr(p, "ref", None),
            }
        except Exception:
            return {}

    def pipelines_on_ref_in_window(self, project, ref: str, since: datetime, until: datetime) -> List[Any]:
        try:
            pipes = project.pipelines.list(ref=ref, updated_after=since.isoformat(), per_page=100, get_all=True)
        except Exception:
            return []
        out = []
        for p in pipes:
            ca = parse_dt(getattr(p, "created_at", None))
            if ca and since <= ca <= until:
                out.append(p)
        return out

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
# Archive extraction: read python files from tar.gz
# ----------------------------

def extract_python_sources_from_archive(tar_gz_bytes: bytes, max_files: int = 10000, max_file_bytes: int = 2_000_000) -> Dict[str, str]:
    """
    Returns dict path->source for .py files. Skips very large files.
    """
    py_sources: Dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_gz_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if len(py_sources) >= max_files:
                break
            if not member.isfile():
                continue
            name = member.name
            if not name.endswith(".py"):
                continue
            if member.size and member.size > max_file_bytes:
                continue
            f = tf.extractfile(member)
            if not f:
                continue
            try:
                b = f.read()
                src = b.decode("utf-8", errors="replace")
            except Exception:
                continue
            py_sources[name] = src
    return py_sources


# ----------------------------
# Main compute
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="Group path, e.g. gcqa/mlops")
    ap.add_argument("--since", required=True, help="ISO datetime, e.g. 2026-01-01T00:00:00Z")
    ap.add_argument("--until", required=True, help="ISO datetime, e.g. 2026-12-31T23:59:59Z")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--include-subgroups", action="store_true", default=True)

    ap.add_argument("--forge-prefixes", default="function_forge",
                    help="Comma-separated top-level import names for Function Forge (default: function_forge)")
    ap.add_argument("--src-prefixes", default="src",
                    help="Comma-separated top-level import names for src reuse (default: src)")

    ap.add_argument("--max-projects", type=int, default=10_000)
    ap.add_argument("--snapshot-every-month", action="store_true",
                    help="If set, download a repo snapshot each month (more accurate, more API calls). "
                         "If not set, uses a single snapshot at --until for all months (faster).")

    args = ap.parse_args()

    url = os.getenv("GITLAB_URL")
    token = os.getenv("GITLAB_TOKEN")
    if not url or not token:
        print("ERROR: set GITLAB_URL and GITLAB_TOKEN", file=sys.stderr)
        return 2

    since = parse_dt(args.since)
    until = parse_dt(args.until)
    if not since or not until or until < since:
        print("ERROR: invalid since/until", file=sys.stderr)
        return 2

    forge_prefixes = [x.strip() for x in args.forge_prefixes.split(",") if x.strip()]
    src_prefixes = [x.strip() for x in args.src_prefixes.split(",") if x.strip()]

    months = iter_months(since, until)

    gl = GL(url, token)
    members = gl.group_members(args.group)
    projects = gl.group_projects(args.group, include_subgroups=args.include_subgroups)[: args.max_projects]

    # Seed rows so every user & project appears each month
    user_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for (uid, username, name) in members:
        for m in months:
            user_rows[(uid, m)] = {
                "month": m, "user_id": uid, "username": username, "name": name,
                # volumes
                "commits_estimated": 0,
                "mrs_created": 0,
                "mrs_merged": 0,
                "mr_notes_written": 0,          # strict (MR notes authored)
                "comment_events": 0,            # broad (events API)
                # coverage attributed to user's merged MRs
                "mr_pipeline_coverage_mean": None,
                # docs from MR diffs (added lines)
                "mr_added_nonblank": 0,
                "mr_added_comment_lines": 0,
                "mr_added_docstring_lines": 0,
                "mr_comment_ratio_added": None,
                "mr_docstring_ratio_added": None,
                # reuse from MR diffs (added imports)
                "mr_forge_reuse_score": None,
                "mr_src_reuse_score": None,
            }

    proj_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for p in projects:
        for m in months:
            proj_rows[(p.id, m)] = {
                "month": m, "project_id": p.id, "project": getattr(p, "path_with_namespace", ""),
                "default_branch": getattr(p, "default_branch", "") or "main",
                # volumes
                "commits_count_default_branch": 0,
                "mrs_created": 0,
                "mrs_merged": 0,
                "mr_notes_total": 0,
                # coverage (default branch)
                "default_branch_coverage_median": None,
                # snapshot docs & reuse
                "snapshot_py_files": None,
                "snapshot_py_nonblank_lines": None,
                "snapshot_comment_ratio": None,
                "snapshot_docstring_ratio": None,
                "snapshot_forge_reuse_score": None,
                "snapshot_src_reuse_score": None,
                # MR diff docs & reuse (aggregate added lines)
                "mr_added_nonblank": 0,
                "mr_added_comment_lines": 0,
                "mr_added_docstring_lines": 0,
                "mr_comment_ratio_added": None,
                "mr_docstring_ratio_added": None,
                "mr_forge_reuse_score": None,
                "mr_src_reuse_score": None,
                # MR pipeline coverage (optional aggregate)
                "mr_pipeline_coverage_median": None,
            }

    # --- User commit & comment volumes from Events ---
    for (uid, username, name) in members:
        pushed = gl.user_events(uid, since, until, action="pushed")
        for ev in pushed:
            t = parse_dt(ev.get("created_at"))
            if not t:
                continue
            mk = clamp_month(t, months)
            if not mk:
                continue
            row = user_rows[(uid, mk)]
            push_data = ev.get("push_data") or {}
            cc = push_data.get("commit_count")
            if isinstance(cc, int):
                row["commits_estimated"] += cc

        commented = gl.user_events(uid, since, until, action="commented")
        for ev in commented:
            t = parse_dt(ev.get("created_at"))
            if not t:
                continue
            mk = clamp_month(t, months)
            if not mk:
                continue
            user_rows[(uid, mk)]["comment_events"] += 1

    # Accumulators for coverage attributed to users / projects via MR pipelines
    user_cov_acc: Dict[Tuple[int, str], List[float]] = {}
    proj_cov_acc: Dict[Tuple[int, str], List[float]] = {}

    # Accumulators for MR-diff reuse scoring per user/project (imports + files)
    # We'll compute score from: files_with_hit / total_files_touched, import_hits / import_total
    @dataclass
    class ReuseAcc:
        files_touched: int = 0
        forge_files_hit: int = 0
        src_files_hit: int = 0
        import_total: int = 0
        forge_imports: int = 0
        src_imports: int = 0

    user_reuse_acc: Dict[Tuple[int, str], ReuseAcc] = {}
    proj_reuse_acc: Dict[Tuple[int, str], ReuseAcc] = {}

    def acc_reuse(acc: ReuseAcc, file_has_any_added: bool, file_has_forge: bool, file_has_src: bool,
                  import_total: int, forge_imports: int, src_imports: int):
        if file_has_any_added:
            acc.files_touched += 1
            if file_has_forge:
                acc.forge_files_hit += 1
            if file_has_src:
                acc.src_files_hit += 1
        acc.import_total += import_total
        acc.forge_imports += forge_imports
        acc.src_imports += src_imports

    # --- Per project: MRs, notes, MR diffs docs+reuse, MR pipelines coverage ---
    for project in projects:
        pid = project.id
        default_branch = getattr(project, "default_branch", "") or "main"

        # MRs updated after since (covers created/merged in window; we bucket ourselves)
        mrs = gl.mrs_updated_after(project, since)

        # Pipelines on default branch for coverage bucketing
        pipes = gl.pipelines_on_ref_in_window(project, default_branch, since, until)
        by_month_cov: Dict[str, List[float]] = {}
        for p in pipes:
            created = parse_dt(getattr(p, "created_at", None))
            if not created:
                continue
            mk = clamp_month(created, months)
            if not mk:
                continue
            details = gl.pipeline_details(project, p.id)
            cov = details.get("coverage")
            try:
                if cov is not None:
                    by_month_cov.setdefault(mk, []).append(float(cov))
            except Exception:
                pass

        for m, vals in by_month_cov.items():
            proj_rows[(pid, m)]["default_branch_coverage_median"] = safe_median(vals)

        for mr in mrs:
            author = (getattr(mr, "author", {}) or {})
            author_id = author.get("id")
            mr_iid = mr.iid

            created_at = parse_dt(getattr(mr, "created_at", None))
            merged_at = parse_dt(getattr(mr, "merged_at", None))
            state = getattr(mr, "state", None)

            # MR created
            if created_at:
                mk = clamp_month(created_at, months)
                if mk:
                    proj_rows[(pid, mk)]["mrs_created"] += 1
                    if author_id and (author_id, mk) in user_rows:
                        user_rows[(author_id, mk)]["mrs_created"] += 1

            # MR merged
            merged_month = None
            if state == "merged" and merged_at:
                merged_month = clamp_month(merged_at, months)
                if merged_month:
                    proj_rows[(pid, merged_month)]["mrs_merged"] += 1
                    if author_id and (author_id, merged_month) in user_rows:
                        user_rows[(author_id, merged_month)]["mrs_merged"] += 1

            # Notes volume (bucket by note created_at)
            notes = gl.mr_notes(mr)
            for n in notes:
                if getattr(n, "system", False):
                    continue
                nt = parse_dt(getattr(n, "created_at", None))
                if not nt:
                    continue
                mk = clamp_month(nt, months)
                if not mk:
                    continue
                proj_rows[(pid, mk)]["mr_notes_total"] += 1

                n_author = (getattr(n, "author", {}) or {}).get("id")
                if n_author and (n_author, mk) in user_rows:
                    user_rows[(n_author, mk)]["mr_notes_written"] += 1

            # MR diffs & MR pipeline coverage attribution (only for merged MRs in window)
            if merged_month and author_id:
                # MR pipelines: attribute coverage to author and project in merge month (latest pipeline)
                mr_pipes = gl.mr_pipelines(pid, mr_iid)
                if mr_pipes:
                    # choose most recent by created_at
                    mr_pipes_sorted = sorted(
                        mr_pipes,
                        key=lambda x: parse_dt(x.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
                        reverse=True,
                    )
                    chosen = mr_pipes_sorted[0]
                    pipeline_id = chosen.get("id")
                    if pipeline_id:
                        details = gl.pipeline_details(project, int(pipeline_id))
                        cov = details.get("coverage")
                        try:
                            if cov is not None:
                                c = float(cov)
                                user_cov_acc.setdefault((author_id, merged_month), []).append(c)
                                proj_cov_acc.setdefault((pid, merged_month), []).append(c)
                        except Exception:
                            pass

                # MR changes (diffs) to compute added comment/docstring + reuse from added import lines
                changes = gl.mr_changes(pid, mr_iid)
                if changes and isinstance(changes.get("changes"), list):
                    # aggregate per MR: only added lines from python files
                    for ch in changes["changes"]:
                        path = ch.get("new_path") or ch.get("old_path") or ""
                        if not path.endswith(".py"):
                            continue
                        diff = ch.get("diff") or ""
                        added_lines = []
                        for line in diff.splitlines():
                            if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
                                continue
                            if line.startswith("+") and not line.startswith("++"):
                                added_lines.append(line[1:])

                        if not added_lines:
                            continue

                        res = analyze_added_lines_for_docs_and_imports(added_lines, forge_prefixes, src_prefixes)

                        # project MR-diff docs totals
                        pr = proj_rows[(pid, merged_month)]
                        pr["mr_added_nonblank"] += res["added_nonblank"]
                        pr["mr_added_comment_lines"] += res["added_comment_lines"]
                        pr["mr_added_docstring_lines"] += res["added_docstring_lines"]

                        # user MR-diff docs totals
                        ur = user_rows[(author_id, merged_month)]
                        ur["mr_added_nonblank"] += res["added_nonblank"]
                        ur["mr_added_comment_lines"] += res["added_comment_lines"]
                        ur["mr_added_docstring_lines"] += res["added_docstring_lines"]

                        # reuse accumulators per file touched (for scoring later)
                        uacc = user_reuse_acc.setdefault((author_id, merged_month), ReuseAcc())
                        pacc = proj_reuse_acc.setdefault((pid, merged_month), ReuseAcc())

                        file_has_any = res["added_nonblank"] > 0
                        file_has_forge = res["forge_imports"] > 0
                        file_has_src = res["src_imports"] > 0

                        acc_reuse(uacc, file_has_any, file_has_forge, file_has_src,
                                  res["import_total"], res["forge_imports"], res["src_imports"])
                        acc_reuse(pacc, file_has_any, file_has_forge, file_has_src,
                                  res["import_total"], res["forge_imports"], res["src_imports"])

    # Finalize MR pipeline coverage per user/project
    for (uid, m), vals in user_cov_acc.items():
        if (uid, m) in user_rows:
            user_rows[(uid, m)]["mr_pipeline_coverage_mean"] = safe_mean(vals)
    for (pid, m), vals in proj_cov_acc.items():
        if (pid, m) in proj_rows:
            proj_rows[(pid, m)]["mr_pipeline_coverage_median"] = safe_median(vals)

    # Finalize MR-diff doc ratios
    for row in user_rows.values():
        denom = row["mr_added_nonblank"]
        row["mr_comment_ratio_added"] = doc_ratio(row["mr_added_comment_lines"], denom) if denom else None
        row["mr_docstring_ratio_added"] = doc_ratio(row["mr_added_docstring_lines"], denom) if denom else None

    for row in proj_rows.values():
        denom = row["mr_added_nonblank"]
        row["mr_comment_ratio_added"] = doc_ratio(row["mr_added_comment_lines"], denom) if denom else None
        row["mr_docstring_ratio_added"] = doc_ratio(row["mr_added_docstring_lines"], denom) if denom else None

    # Finalize MR-diff reuse scores
    for (uid, m), acc in user_reuse_acc.items():
        score_forge = reuse_score(acc.forge_files_hit, acc.files_touched, acc.forge_imports, acc.import_total)
        score_src = reuse_score(acc.src_files_hit, acc.files_touched, acc.src_imports, acc.import_total)
        if (uid, m) in user_rows:
            user_rows[(uid, m)]["mr_forge_reuse_score"] = score_forge
            user_rows[(uid, m)]["mr_src_reuse_score"] = score_src

    for (pid, m), acc in proj_reuse_acc.items():
        score_forge = reuse_score(acc.forge_files_hit, acc.files_touched, acc.forge_imports, acc.import_total)
        score_src = reuse_score(acc.src_files_hit, acc.files_touched, acc.src_imports, acc.import_total)
        if (pid, m) in proj_rows:
            proj_rows[(pid, m)]["mr_forge_reuse_score"] = score_forge
            proj_rows[(pid, m)]["mr_src_reuse_score"] = score_src

    # --- Project commit volume and repo snapshot metrics ---
    # Commits per month on default branch
    for project in projects:
        pid = project.id
        default_branch = getattr(project, "default_branch", "") or "main"

        # Monthly commit counts (can be slow on huge repos; but straightforward)
        # We'll compute by month windows.
        cur = month_start(since)
        while cur <= month_start(until):
            nxt = cur + relativedelta(months=1)
            m = month_key(cur)
            count = gl.project_commits_count(project, default_branch, cur, min(nxt - relativedelta(seconds=1), until))
            proj_rows[(pid, m)]["commits_count_default_branch"] = count
            cur = nxt

        # Repo snapshots:
        if args.snapshot_every_month:
            # One snapshot per month (end of month)
            cur = month_start(since)
            while cur <= month_start(until):
                nxt = cur + relativedelta(months=1)
                month_end = min(nxt - relativedelta(seconds=1), until)
                sha = gl.project_commits_latest_sha_before(project, default_branch, month_end)
                if sha:
                    try:
                        tarbytes = gl.project_repo_archive_tar_gz(pid, sha)
                        py_sources = extract_python_sources_from_archive(tarbytes)
                        stats = analyze_python_snapshot(py_sources, forge_prefixes, src_prefixes)

                        row = proj_rows[(pid, month_key(cur))]
                        row["snapshot_py_files"] = stats.py_files
                        row["snapshot_py_nonblank_lines"] = stats.py_nonblank_lines
                        row["snapshot_comment_ratio"] = doc_ratio(stats.comment_lines, stats.py_nonblank_lines)
                        row["snapshot_docstring_ratio"] = doc_ratio(stats.docstring_lines, stats.py_nonblank_lines)
                        row["snapshot_forge_reuse_score"] = reuse_score(stats.forge_files_with_import, stats.py_files,
                                                                       stats.forge_import_count, stats.import_total)
                        row["snapshot_src_reuse_score"] = reuse_score(stats.src_files_with_import, stats.py_files,
                                                                     stats.src_import_count, stats.import_total)
                    except Exception:
                        pass
                cur = nxt
        else:
            # Single snapshot at --until (faster); repeated into each month row for simplicity.
            sha = gl.project_commits_latest_sha_before(project, default_branch, until)
            if sha:
                try:
                    tarbytes = gl.project_repo_archive_tar_gz(pid, sha)
                    py_sources = extract_python_sources_from_archive(tarbytes)
                    stats = analyze_python_snapshot(py_sources, forge_prefixes, src_prefixes)

                    snap = {
                        "snapshot_py_files": stats.py_files,
                        "snapshot_py_nonblank_lines": stats.py_nonblank_lines,
                        "snapshot_comment_ratio": doc_ratio(stats.comment_lines, stats.py_nonblank_lines),
                        "snapshot_docstring_ratio": doc_ratio(stats.docstring_lines, stats.py_nonblank_lines),
                        "snapshot_forge_reuse_score": reuse_score(stats.forge_files_with_import, stats.py_files,
                                                                 stats.forge_import_count, stats.import_total),
                        "snapshot_src_reuse_score": reuse_score(stats.src_files_with_import, stats.py_files,
                                                               stats.src_import_count, stats.import_total),
                    }
                    for m in months:
                        row = proj_rows[(pid, m)]
                        for k, v in snap.items():
                            row[k] = v
                except Exception:
                    pass

    # Write CSVs
    os.makedirs(args.outdir, exist_ok=True)
    df_user = pd.DataFrame(list(user_rows.values())).sort_values(["month", "username"])
    df_proj = pd.DataFrame(list(proj_rows.values())).sort_values(["month", "project"])

    user_path = os.path.join(args.outdir, "user_monthly.csv")
    proj_path = os.path.join(args.outdir, "project_monthly.csv")

    df_user.to_csv(user_path, index=False)
    df_proj.to_csv(proj_path, index=False)

    print("Wrote:")
    print(" -", user_path)
    print(" -", proj_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
