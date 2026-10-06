"""Architecture invariants: one definition per responsibility, one entry point.

These tests are cheap structural checks that fail loudly if a duplicate creeps
back in during future work.
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "src" / "image_story"


def top_level_defs(path: Path) -> set[str]:
    """Names defined at module level (classes, functions, assignments)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def class_defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {n.name for n in tree.body if isinstance(n, ast.ClassDef)}


def package_modules() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py"))


class TestNoDuplicateDefinitions:
    """A domain contract is defined exactly once across the package."""

    #: Contracts that must have a single definition site.
    SINGLE_DEFINITION = (
        "EvidenceRecord",
        "VisualObservations",
        "WorldState",
        "WorldEntity",
        "SceneSummary",
        "CollectionMemory",
        "EntityMemory",
        "StateTransition",
        "NarrativeElement",
        "CreativePlan",
        "StoryPlan",
        "StoryBeat",
        "StoryDraft",
        "StoryClaim",
        "VerificationResult",
        "EvaluationResult",
        "PipelineConfig",
        "ProcessingConfig",
        "CollectionPipelineConfig",
        "ImageRecord",
        "ImageCollection",
        "ProcessingJob",
        "StorySession",
        "CacheEntry",
        "PipelineOrchestrator",
    )

    @pytest.mark.parametrize("name", SINGLE_DEFINITION)
    def test_defined_once(self, name):
        sites = [p.relative_to(PACKAGE).as_posix() for p in package_modules() if name in class_defs(p)]
        assert len(sites) == 1, f"{name} defined in {sites}; expected exactly one definition site"

    def test_no_enum_defined_twice_in_one_module(self):
        """The enums module previously defined the collection enums five times."""
        for path in package_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            seen: set[str] = set()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    assert node.name not in seen, f"{path.name}: {node.name} defined twice"
                    seen.add(node.name)

    def test_evidence_text_projection_is_single(self):
        """`_evidence_to_text` was duplicated four times; only the shared helper remains."""
        offenders = [
            p.relative_to(PACKAGE).as_posix()
            for p in package_modules()
            if re.search(r"def _evidence_to_text", p.read_text(encoding="utf-8"))
        ]
        assert not offenders, f"duplicate evidence->text in {offenders}"


class TestLayering:
    def test_domain_does_not_import_upstream_layers(self):
        """`domain` must not depend on pipeline stages."""
        offenders = []
        for path in (PACKAGE / "domain").rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.lstrip()
                if stripped.startswith(("import ", "from ")) and "image_story" in stripped:
                    if any(
                        layer in stripped
                        for layer in ("vision", "memory", "retrieval", "pipeline", "narrative", "evaluation")
                    ):
                        offenders.append(f"{path.name}: {stripped}")
        assert not offenders, offenders

    def test_only_orchestrator_and_cli_import_every_stage(self):
        """Broad imports are the orchestrator's job; check nothing else grew them."""
        allowed = {"orchestrator.py", "cli.py", "runners.py", "benchmarks.py", "__init__.py"}
        for path in package_modules():
            text = path.read_text(encoding="utf-8")
            layers = sum(
                1
                for layer in ("vision", "memory", "retrieval", "narrative", "evaluation", "world")
                if f"from ..{layer}" in text
            )
            if layers >= 4:
                assert path.name in allowed, f"{path.relative_to(PACKAGE)} imports {layers} layers"

    def test_every_package_has_a_docstring(self):
        for path in package_modules():
            if path.name == "__init__.py" or path.parent == PACKAGE:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            assert ast.get_docstring(tree), f"{path.relative_to(PACKAGE)} has no module docstring"


class TestEntryPoints:
    def test_module_entry_point_exists(self):
        assert (PACKAGE / "__main__.py").is_file()
        assert (PACKAGE / "cli.py").is_file()

    def test_no_stray_main_scripts_at_repo_root(self):
        strays = [
            p.name
            for p in REPO_ROOT.glob("*.py")
            if p.name not in ("conftest.py",)
        ]
        assert not strays, f"competing entry points at repo root: {strays}"

    def test_challenge_has_its_own_entry_point(self):
        assert (REPO_ROOT / "challenges" / "image-story-v1" / "main.py").is_file()

    def test_cli_exposes_one_run_path(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PACKAGE.parent)
        result = subprocess.run(
            [sys.executable, "-m", "image_story", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "--collection" in result.stdout
        assert "--multi" in result.stdout


class TestV3NotImplemented:
    """V3 subsystems are roadmap items, not code. They must not appear yet."""

    V3_MODULES = ("causal",)

    @pytest.mark.parametrize("name", V3_MODULES)
    def test_no_v3_package_exists(self, name):
        assert not (PACKAGE / name).exists(), (
            f"{name}/ exists; V3 is not implemented (see docs/architecture/roadmap.md)"
        )
