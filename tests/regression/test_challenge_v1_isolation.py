"""Challenge V1 stays runnable from its own directory.

The V1 suite is self-contained: it imports its own modules by bare name and
runs with its own dependencies. These tests only assert that this separation
still holds, without importing V1 code into the production package.
"""
import ast
import re
from pathlib import Path

import pytest

CHALLENGE_DIR = Path(__file__).resolve().parents[2] / "challenges" / "image-story-v1"

V1_MODULES = ("main.py", "baseline.py", "seeing.py", "context_builder.py", "metric.py")

PRODUCTION_DIR_MARKERS = (
    "from image_story",
    "import image_story",
)


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.skipif(not CHALLENGE_DIR.is_dir(), reason="challenge directory not present")
class TestChallengeIsolation:
    def test_all_v1_modules_present(self):
        for module in V1_MODULES:
            assert (CHALLENGE_DIR / module).is_file(), f"missing {module}"

    def test_v1_never_imports_the_production_package(self):
        for module in V1_MODULES:
            tree = parse_module(CHALLENGE_DIR / module)
            assert not (imported_names(tree) & {"image_story"}), (
                f"{module} imports image_story; V1 must stay self-contained"
            )

    def test_production_package_never_imports_the_challenge(self):
        repo_root = CHALLENGE_DIR.parents[1]
        offenders = []
        for path in (repo_root / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bchallenges?\b", text) and "import" in text:
                for line in text.splitlines():
                    if line.lstrip().startswith(("import ", "from ")) and "challenge" in line:
                        offenders.append(f"{path}: {line.strip()}")
        assert not offenders, "production code imports the challenge: " + "; ".join(offenders)

    def test_v1_has_its_own_requirements(self):
        assert (CHALLENGE_DIR / "requirements.txt").is_file()

    def test_v1_has_its_own_readme_and_tests(self):
        assert (CHALLENGE_DIR / "README.md").is_file()
        assert (CHALLENGE_DIR / "test_pipeline.py").is_file()

    def test_v1_result_artifacts_are_preserved(self):
        """The historical result files are the baseline evidence; keep them."""
        expected = {
            "results_final.csv",
            "results_final_fixlen.csv",
            "vision_final.json",
            "vision_final_fixlen.json",
        }
        present = {p.name for p in CHALLENGE_DIR.iterdir() if p.is_file()}
        assert expected <= present, f"missing artifacts: {sorted(expected - present)}"

    def test_v1_entry_point_is_not_a_package_module(self):
        """V1 keeps its own script; production has a single entry point."""
        assert (CHALLENGE_DIR / "main.py").is_file()
        repo_root = CHALLENGE_DIR.parents[1]
        assert not (repo_root / "main.py").exists()
        assert not (repo_root / "main_v2.py").exists()
