#!/usr/bin/env python3
"""
GitLab MLOps adoption metrics extractor
- Produces:
  1) project_metrics.csv (aggregated per project)
  2) mr_metrics.csv (one row per merged MR)
  3) pipeline_metrics.csv (one row per pipeline on default branch)

Requires:
  pip install python-gitlab pandas python-dateutil

Auth:
  export GITLAB_URL=...
  export GITLAB_TOKEN=...
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import gitlab
from dateutil import parser as dtparser


LOG = logging.getLogger("gitlab-metrics")


def parse_dt(x: Optional[str]) -> Optional[datetime]:
    if not x:
        return None
    dt = dtparser.isoparse(x)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours(delta) -> Optional[float]:
    if delta is None:
        return None
    return delta.total_seconds() / 3600.0


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    return (sum(xs) / len(xs)) if xs else None


def safe_median(values: List[Optional[float]]) -> Optional[float]:
    xs = sorted([v for v in values if v is not None and not math.isnan(v)])
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if (len(xs) % 2 == 1) else (xs[mid - 1] + xs[mid]) / 2


@dataclass
class RepoHygiene:
    has_ci: bool
    has_readme: bool
    has_src: bool
    has_tests: bool
    has_docs: bool
    has_pyproject: bool
    has_requirements: bool
    has_precommit: bool
    has_linter: bool
    has_typecheck: bool
    has_makefile: bool
    has_dockerfile: bool
    notebooks_count: Optional[int] = None
    python_files_count: Optional[int] = None

    @property
    def hygiene_score(self) -> int:
        flags = [
            self.has_ci, self.has_readme, self.has_src, self.has_tests, self.has_docs,
            self.has_pyproject or self.has_requirements,
            self.has_precommit, self.has_linter, self.has_typecheck,
            self.has_makefile, self.has_dockerfile,
        ]
        return int(sum(bool(x) for x in flags))


class GitLabMetricsExtractor:
    def __init__(self, url: str, token: str, per_page: int = 100) -> None:
        self.gl = gitlab.Gitlab(url=url, private_token=token, per_page=per_page)
        self.gl.auth()

    def _file_exists(self, project, file_path: str, ref: str) -> bool:
        try:
            project.files.get(file_path=file_path, ref=ref)
            return True
        except gitlab.exceptions.GitlabGetError:
            return False

    def repo_hygiene(self, project, ref: str, recursive_scan: bool, max_tree: int) -> RepoHygiene:
        # Fast existence checks via Repository Files API 8
        has_ci = self._file_exists(project, ".gitlab-ci.yml", ref)
        has_readme = any(self._file_exists(project, p, ref) for p in ["README.md", "README.rst", "README.txt"])
        has_pyproject = self._file_exists(project, "pyproject.toml", ref)
        has_requirements = any(self._file_exists(project, p, ref) for p in ["requirements.txt", "requirements-dev.txt"])
        has_precommit = self._file_exists(project, ".pre-commit-config.yaml", ref)
        has_linter = any(self._file_exists(project, p, ref) for p in ["ruff.toml", ".ruff.toml", ".flake8"])
        has_typecheck = any(self._file_exists(project, p, ref) for p in ["mypy.ini", "pyrightconfig.json"])
        has_makefile = self._file_exists(project, "Makefile", ref)
        has_dockerfile = self._file_exists(project, "Dockerfile", ref)

        # Path checks via repository tree (non-recursive first)
        # Repository tree API 9
        try:
            root_items = project.repository_tree(ref=ref, path="", per_page=100, get_all=True)
            root_names = {x.get("name") for x in root_items}
        except gitlab.exceptions.GitlabGetError:
            root_names = set()

        has_src = "src" in root_names
        has_tests = ("tests" in root_names) or ("test" in root_names)
        has_docs = "docs" in root_names

        notebooks_count = None
        python_files_count = None
        if recursive_scan:
            try:
                items = project.repository_tree(ref=ref, path="", recursive=True, per_page=100, get_all=True)
                items = items[:max_tree]
                notebooks_count = sum(1 for x in items if x.get("type") == "blob" and str(x.get("path", "")).endswith(".ipynb"))
                python_files_count = sum(1 for x in items if x.get("type") == "blob" and str(x.get("path", "")).endswith(".py"))
            except gitlab.exceptions.GitlabGetError:
                pass

        return RepoHygiene(
            has_ci=has_ci,
            has_readme=has_readme,
            has_src=has_src,
            has_tests=has_tests,
            has_docs=has_docs,
            has_pyproject=has_pyproject,
            has_requirements=has_requirements,
            has_precommit=has_precommit,
            has_linter=has_linter,
            has_typecheck=has_typecheck,
            has_makefile=has_makefile,
            has_dockerfile=has_dockerfile,
            notebooks_count=notebooks_count,
            python_files_count=python_files_count,
        )

    def list_group_projects(self, group_path: str, include_subgroups: bool = True) -> List[Any]:
        group = self.gl.groups.get(group_path)
        projects = group.projects.list(include_subgroups=include_subgroups, all=True)
        # Fetch full project objects (need default_branch etc.)
        full = []
        for p in projects:
            try:
                full.append(self.gl.projects.get(p.id))
            except gitlab.exceptions.GitlabGetError:
                continue
        return full

    def merged_mrs_in_window(self, project, since: datetime, until: datetime) -> List[Any]:
        # Merge Requests API supports filtering by state and updated_after 10
        mrs = project.mergerequests.list(
            state="merged",
            scope="all",
            updated_after=since.isoformat(),
            per_page=100,
            get_all=True,
        )
        out = []
        for mr in mrs:
            merged_at = parse_dt(getattr(mr, "merged_at", None))
            if merged_at and since <= merged_at <= until:
                out.append(mr)
        return out

    def mr_first_reviewer_comment_time(self, mr) -> Optional[datetime]:
        # Notes / discussions via python-gitlab 11
        try:
            notes = mr.notes.list(get_all=True)
        except gitlab.exceptions.GitlabGetError:
            return None

        author_id = getattr(mr, "author", {}).get("id")
        best: Optional[datetime] = None
        for n in notes:
            # ignore system notes where possible
            if getattr(n, "system", False):
                continue
            n_author = getattr(n, "author", {}).get("id")
            if author_id is not None and n_author == author_id:
                continue
            t = parse_dt(getattr(n, "created_at", None))
            if t and (best is None or t < best):
                best = t
        return best

    def mr_review_comment_stats(self, mr) -> Tuple[int, int]:
        """Returns (review_comment_count, unique_reviewer_count)."""
        try:
            notes = mr.notes.list(get_all=True)
        except gitlab.exceptions.GitlabGetError:
            return (0, 0)

        author_id = getattr(mr, "author", {}).get("id")
        count = 0
        reviewers = set()
        for n in notes:
            if getattr(n, "system", False):
                continue
            n_author = getattr(n, "author", {}).get("id")
            if author_id is not None and n_author == author_id:
                continue
            count += 1
            if n_author is not None:
                reviewers.add(n_author)
        return count, len(reviewers)

    def mr_approvals_summary(self, project_id: int, mr_iid: int) -> Dict[str, Any]:
        # Approvals API exists but is tier-dependent 12
        # Use raw GET to avoid object mismatches across versions.
        try:
            data = self.gl.http_get(f"/projects/{project_id}/merge_requests/{mr_iid}/approvals")
            approved_by = data.get("approved_by", []) or []
            return {
                "approvals_required": data.get("approvals_required"),
                "approvals_left": data.get("approvals_left"),
                "approvals_given": len(approved_by),
            }
        except Exception:
            return {
                "approvals_required": None,
                "approvals_left": None,
                "approvals_given": None,
            }

    def pipelines_in_window(self, project, ref: str, since: datetime, until: datetime) -> List[Any]:
        # Pipelines API 13
        pipes = project.pipelines.list(
            ref=ref,
            updated_after=since.isoformat(),
            per_page=100,
            get_all=True,
        )
        out = []
        for p in pipes:
            created_at = parse_dt(getattr(p, "created_at", None))
            if created_at and since <= created_at <= until:
                out.append(p)
        return out

    def pipeline_details(self, project, pipeline_id: int) -> Dict[str, Any]:
        try:
            p = project.pipelines.get(pipeline_id)
            return {
                "id": pipeline_id,
                "status": getattr(p, "status", None),
                "created_at": getattr(p, "created_at", None),
                "updated_at": getattr(p, "updated_at", None),
                "duration": getattr(p, "duration", None),
                # Coverage may be available if configured in CI 14
                "coverage": getattr(p, "coverage", None),
                "ref": getattr(p, "ref", None),
                "sha": getattr(p, "sha", None),
                "web_url": getattr(p, "web_url", None),
            }
        except gitlab.exceptions.GitlabGetError:
            return {"id": pipeline_id}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="GitLab group path, e.g. 'gcqa/mlops'")
    ap.add_argument("--since", required=True, help="ISO date/time, e.g. 2025-10-01 or 2025-10-01T00:00:00Z")
    ap.add_argument("--until", required=True, help="ISO date/time, e.g. 2026-01-01")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--include-subgroups", action="store_true", default=True)
    ap.add_argument("--recursive-repo-scan", action="store_true", help="Counts *.ipynb, *.py via recursive tree (slower)")
    ap.add_argument("--max-tree", type=int, default=5000, help="Max items inspected in recursive tree")
    args = ap.parse_args()

    url = os.getenv("GITLAB_URL")
    token = os.getenv("GITLAB_TOKEN")
    if not url or not token:
        LOG.error("Missing GITLAB_URL / GITLAB_TOKEN environment variables.")
        return 2

    since = parse_dt(args.since)
    until = parse_dt(args.until)
    if not since or not until:
        LOG.error("Could not parse --since/--until")
        return 2
    if until < since:
        LOG.error("--until must be after --since")
        return 2

    ex = GitLabMetricsExtractor(url=url, token=token)

    projects = ex.list_group_projects(args.group, include_subgroups=args.include_subgroups)
    LOG.info("Found %d projects in group '%s'", len(projects), args.group)

    mr_rows: List[Dict[str, Any]] = []
    pipe_rows: List[Dict[str, Any]] = []
    proj_rows: List[Dict[str, Any]] = []

    for project in projects:
        project_id = project.id
        project_path = getattr(project, "path_with_namespace", None)
        default_branch = getattr(project, "default_branch", None) or "main"

        LOG.info("Project: %s (id=%s) default_branch=%s", project_path, project_id, default_branch)

        hygiene = ex.repo_hygiene(
            project,
            ref=default_branch,
            recursive_scan=args.recursive_repo_scan,
            max_tree=args.max_tree,
        )

        # --- Merge Requests ---
        mrs = ex.merged_mrs_in_window(project, since, until)
        lead_times_h = []
        first_review_h = []
        comment_counts = []
        reviewers_counts = []
        mr_sizes = []

        merges_to_default = 0
        merges_to_main_or_master = 0

        for mr in mrs:
            mr_iid = mr.iid
            created_at = parse_dt(getattr(mr, "created_at", None))
            merged_at = parse_dt(getattr(mr, "merged_at", None))
            updated_at = parse_dt(getattr(mr, "updated_at", None))
            target_branch = getattr(mr, "target_branch", None)

            if target_branch == default_branch:
                merges_to_default += 1
            if target_branch in ("main", "master"):
                merges_to_main_or_master += 1

            first_review = ex.mr_first_reviewer_comment_time(mr)
            review_comment_count, unique_reviewer_count = ex.mr_review_comment_stats(mr)
            approvals = ex.mr_approvals_summary(project_id, mr_iid)

            lt_h = hours(merged_at - created_at) if (created_at and merged_at) else None
            fr_h = hours(first_review - created_at) if (created_at and first_review) else None
            post_update_to_merge_h = hours(merged_at - updated_at) if (updated_at and merged_at) else None

            lead_times_h.append(lt_h)
            first_review_h.append(fr_h)
            comment_counts.append(review_comment_count)
            reviewers_counts.append(unique_reviewer_count)

            # size fields may be present depending on GitLab version/settings
            additions = getattr(mr, "additions", None)
            deletions = getattr(mr, "deletions", None)
            changes_count = getattr(mr, "changes_count", None)
            try:
                if changes_count is not None:
                    changes_count = int(changes_count)
            except Exception:
                pass

            size_changed_lines = None
            try:
                if additions is not None and deletions is not None:
                    size_changed_lines = int(additions) + int(deletions)
            except Exception:
                pass
            mr_sizes.append(size_changed_lines)

            comments_per_100_lines = None
            if size_changed_lines and size_changed_lines > 0:
                comments_per_100_lines = review_comment_count / (size_changed_lines / 100.0)

            mr_rows.append({
                "project_id": project_id,
                "project": project_path,
                "default_branch": default_branch,
                "mr_iid": mr_iid,
                "mr_title": getattr(mr, "title", None),
                "mr_web_url": getattr(mr, "web_url", None),
                "author": (getattr(mr, "author", {}) or {}).get("username"),
                "created_at": getattr(mr, "created_at", None),
                "updated_at": getattr(mr, "updated_at", None),
                "merged_at": getattr(mr, "merged_at", None),
                "target_branch": target_branch,
                "lead_time_hours": lt_h,
                "time_to_first_review_comment_hours": fr_h,
                "time_from_last_update_to_merge_hours": post_update_to_merge_h,
                "review_comment_count": review_comment_count,
                "unique_reviewer_count": unique_reviewer_count,
                "comments_per_100_changed_lines": comments_per_100_lines,
                "approvals_required": approvals.get("approvals_required"),
                "approvals_left": approvals.get("approvals_left"),
                "approvals_given": approvals.get("approvals_given"),
                "additions": additions,
                "deletions": deletions,
                "changes_count": changes_count,
                "changed_lines": size_changed_lines,
                "commits_count": getattr(mr, "commits_count", None),
            })

        # --- Pipelines ---
        pipes = ex.pipelines_in_window(project, ref=default_branch, since=since, until=until)
        pipe_durations = []
        coverages = []
        success = 0
        failed = 0

        for p in pipes:
            pid = p.id
            details = ex.pipeline_details(project, pid)
            status = details.get("status")
            duration = details.get("duration")
            coverage = details.get("coverage")

            if isinstance(duration, (int, float)):
                pipe_durations.append(float(duration))
            try:
                if coverage is not None:
                    coverages.append(float(coverage))
            except Exception:
                pass

            if status == "success":
                success += 1
            elif status in ("failed", "canceled"):
                failed += 1

            pipe_rows.append({
                "project_id": project_id,
                "project": project_path,
                "ref": default_branch,
                **details,
            })

        total_pipes = len(pipes)
        success_rate = (success / total_pipes) if total_pipes else None

        proj_rows.append({
            "project_id": project_id,
            "project": project_path,
            "default_branch": default_branch,

            # throughput
            "mrs_merged_in_window": len(mrs),
            "merges_to_default_branch": merges_to_default,
            "merges_to_main_or_master": merges_to_main_or_master,

            # review efficiency
            "lead_time_hours_mean": safe_mean(lead_times_h),
            "lead_time_hours_median": safe_median(lead_times_h),
            "time_to_first_review_comment_hours_mean": safe_mean(first_review_h),
            "time_to_first_review_comment_hours_median": safe_median(first_review_h),
            "review_comment_count_mean": safe_mean([float(x) for x in comment_counts]) if comment_counts else None,
            "unique_reviewer_count_mean": safe_mean([float(x) for x in reviewers_counts]) if reviewers_counts else None,
            "comments_per_100_lines_mean": safe_mean([
                (c / (s / 100.0)) for c, s in zip(comment_counts, mr_sizes) if s and s > 0
            ]) if mr_sizes else None,

            # CI/CD
            "pipelines_on_default_branch": total_pipes,
            "pipeline_success_rate": success_rate,
            "pipeline_duration_seconds_mean": safe_mean(pipe_durations),
            "pipeline_duration_seconds_median": safe_median(pipe_durations),
            "coverage_mean": safe_mean(coverages),
            "coverage_median": safe_median(coverages),

            # repo hygiene / reusability proxies
            "repo_hygiene_score": hygiene.hygiene_score,
            "has_ci": hygiene.has_ci,
            "has_src": hygiene.has_src,
            "has_tests": hygiene.has_tests,
            "has_docs": hygiene.has_docs,
            "has_readme": hygiene.has_readme,
            "has_pyproject": hygiene.has_pyproject,
            "has_requirements": hygiene.has_requirements,
            "has_precommit": hygiene.has_precommit,
            "has_linter": hygiene.has_linter,
            "has_typecheck": hygiene.has_typecheck,
            "has_makefile": hygiene.has_makefile,
            "has_dockerfile": hygiene.has_dockerfile,
            "notebooks_count": hygiene.notebooks_count,
            "python_files_count": hygiene.python_files_count,
        })

    # Write outputs
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    df_proj = pd.DataFrame(proj_rows).sort_values(["project"], na_position="last")
    df_mr = pd.DataFrame(mr_rows).sort_values(["project", "merged_at"], na_position="last")
    df_pipe = pd.DataFrame(pipe_rows).sort_values(["project", "created_at"], na_position="last")

    proj_path = os.path.join(outdir, "project_metrics.csv")
    mr_path = os.path.join(outdir, "mr_metrics.csv")
    pipe_path = os.path.join(outdir, "pipeline_metrics.csv")

    df_proj.to_csv(proj_path, index=False)
    df_mr.to_csv(mr_path, index=False)
    df_pipe.to_csv(pipe_path, index=False)

    LOG.info("Wrote: %s", proj_path)
    LOG.info("Wrote: %s", mr_path)
    LOG.info("Wrote: %s", pipe_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
