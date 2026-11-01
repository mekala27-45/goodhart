"""Every gate gets three tests: a clean pass, a deliberate violation, and a refusal on nothing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import check_no_em_dash
import check_published_numbers
import check_statement
import check_tokens
import check_vocabulary
import pytest
import scan_for_planted_identifiers
from goodhart_core.manifest import ManifestBuilder, write
from goodhart_core.statements import STATEMENT
from goodhart_render.render import ClaimGateError, Renderer, hand_typed_numbers
from goodhart_render.targets import discover, template_roots

ROOT = Path(__file__).resolve().parents[2]


def _file(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# em dash


def test_dash_gate_passes_clean_text(tmp_path: Path) -> None:
    f = _file(tmp_path, "a.md", "Ranges run from 3 to 5, asides go in parentheses.\n")
    assert check_no_em_dash.check(tmp_path, [f]) == 1


def test_dash_gate_catches_an_em_dash(tmp_path: Path) -> None:
    f = _file(tmp_path, "a.md", "a pause" + chr(0x2014) + "like this\n")
    with pytest.raises(ValueError, match="em dash"):
        check_no_em_dash.check(tmp_path, [f])


def test_dash_gate_refuses_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        check_no_em_dash.check(tmp_path, [])


# vocabulary


def test_vocabulary_gate_passes_plain_prose(tmp_path: Path) -> None:
    f = _file(tmp_path, "a.md", "The reward paid for a query that returned the whole table.\n")
    assert check_vocabulary.check(tmp_path, [f]) == 1


def test_vocabulary_gate_catches_a_banned_phrase(tmp_path: Path) -> None:
    f = _file(tmp_path, "a.md", "We " + "lever" + "aged the twins.\n")
    with pytest.raises(ValueError, match="vocabulary"):
        check_vocabulary.check(tmp_path, [f])


def test_vocabulary_gate_refuses_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        check_vocabulary.check(tmp_path, [])


# statement


def test_statement_gate_passes_a_surface_with_the_statement(tmp_path: Path) -> None:
    f = _file(tmp_path, "README.md", f"# goodhart\n\n> {STATEMENT}\n")
    assert check_statement.check(tmp_path, [f]) == 1


def test_statement_gate_finds_it_in_html_between_tags(tmp_path: Path) -> None:
    words = STATEMENT.split(" ")
    half = len(words) // 2
    page = f"<p>{' '.join(words[:half])} <span>{' '.join(words[half:])}</span></p>"
    f = _file(tmp_path, "index.html", page)
    assert check_statement.check(tmp_path, [f]) == 1


def test_statement_gate_catches_a_missing_statement(tmp_path: Path) -> None:
    f = _file(tmp_path, "README.md", "# goodhart\n\nNo statement here.\n")
    with pytest.raises(ValueError, match="missing the statement"):
        check_statement.check(tmp_path, [f])


def test_statement_gate_refuses_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no surfaces"):
        check_statement.check(tmp_path, [])


# tokens


def _git(tmp_path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "goodhart-gpu@users.noreply.github.com")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    return tmp_path


def test_token_scan_passes_clean_files(tmp_path: Path) -> None:
    f = _file(tmp_path, "config.yaml", "database: postgresql://user:password@host/db\n")
    n_files, _ = check_tokens.check(tmp_path, files=[f])
    assert n_files == 1


def test_token_scan_catches_a_kaggle_token_in_a_file(tmp_path: Path) -> None:
    f = _file(tmp_path, "notes.md", "token: " + "KG" + "AT_" + "a1b2c3d4e5f6g7h8i9j0k1l2\n")
    with pytest.raises(ValueError, match="Kaggle API token"):
        check_tokens.check(tmp_path, files=[f])


def test_token_scan_reads_commit_diffs_not_only_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    leaked = "url: postgresql://owner:" + "s3cretpass" + "@ep-x.neon.tech/db\n"
    _file(repo, "a.txt", leaked)
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "add")
    _file(repo, "a.txt", "url: removed\n")
    _git(repo, "commit", "-q", "-am", "remove")
    clean = _file(repo, "b.txt", "nothing\n")
    with pytest.raises(ValueError, match="commit"):
        check_tokens.check(repo, files=[clean])


def test_token_scan_refuses_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        check_tokens.check(tmp_path, files=[])


# identifiers


def test_identifier_scan_passes_clean_files(tmp_path: Path) -> None:
    f = _file(tmp_path, "a.md", "Contact noreply@github.com, listen on 127.0.0.1 and 0.0.0.0.\n")
    n_files, _ = scan_for_planted_identifiers.check(tmp_path, files=[f])
    assert n_files == 1


def test_identifier_scan_catches_the_planted_fixture() -> None:
    fixture = ROOT / "tests" / "fixtures" / "planted_identifier.txt"
    with pytest.raises(ValueError, match="identifier scan failed"):
        scan_for_planted_identifiers.check(ROOT, files=[fixture])


def test_identifier_scan_refuses_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        scan_for_planted_identifiers.check(tmp_path, files=[])


# the claim gate


def _claim_repo(tmp_path: Path, template: str) -> Path:
    b = ManifestBuilder("2026-10-31")
    b.value("champion_twin", 0.512, "pct1", origin="test", population="test", source="build")
    write(b.build(), tmp_path / "results" / "manifest.json")
    _file(tmp_path, "docs/templates/README.md.j2", template)
    return tmp_path


def test_claim_gate_renders_and_then_matches(tmp_path: Path) -> None:
    repo = _claim_repo(tmp_path, "Twin confirmed accuracy is {{ v('champion_twin') }}.\n")
    renderer = Renderer(
        repo,
        check_published_numbers.load_manifest(repo / "results" / "manifest.json"),
        discover(repo),
        template_roots(repo),
    )
    renderer.write_all()
    assert (repo / "README.md").read_text() == "Twin confirmed accuracy is 51.2%.\n"
    assert renderer.check_all() == []


def test_claim_gate_catches_a_hand_edit(tmp_path: Path) -> None:
    repo = _claim_repo(tmp_path, "Twin confirmed accuracy is {{ v('champion_twin') }}.\n")
    renderer = Renderer(
        repo,
        check_published_numbers.load_manifest(repo / "results" / "manifest.json"),
        discover(repo),
        template_roots(repo),
    )
    renderer.write_all()
    (repo / "README.md").write_text("Twin confirmed accuracy is 61.2%.\n")
    drifts = renderer.check_all()
    assert len(drifts) == 1 and "61.2%" in drifts[0].diff


def test_claim_gate_catches_a_number_typed_into_a_template() -> None:
    assert hand_typed_numbers("The champion reached 51.2 percent.", []) == ["51.2"]
    assert hand_typed_numbers("The champion reached {{ v('x') }}.", []) == []


def test_claim_gate_refuses_nothing(tmp_path: Path) -> None:
    b = ManifestBuilder("2026-10-31")
    with pytest.raises(ClaimGateError, match="no documents"):
        Renderer(tmp_path, b.build(), [], [])


def test_claim_gate_refuses_an_empty_manifest(tmp_path: Path) -> None:
    repo = _claim_repo(tmp_path, "x\n")
    with pytest.raises(ClaimGateError, match="empty manifest"):
        Renderer(repo, ManifestBuilder("2026-10-31").build(), discover(repo), template_roots(repo))


# the palette validator


def _palette(tmp_path: Path, slots: list[str]) -> Path:
    config = json.loads((ROOT / "web" / "src" / "theme" / "palette.json").read_text())
    config["light"]["categorical"] = slots
    p = tmp_path / "palette.json"
    p.write_text(json.dumps(config))
    return p


def _validate(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(ROOT / "scripts" / "validate_palette.js"), "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_palette_validator_passes_the_committed_palette() -> None:
    done = _validate(ROOT / "web" / "src" / "theme" / "palette.json")
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout)["failures"] == 0


def test_palette_validator_catches_a_red_green_pair(tmp_path: Path) -> None:
    done = _validate(_palette(tmp_path, ["#C0263F", "#2E8B2E", "#0A7F72"]))
    assert done.returncode != 0


def test_palette_validator_refuses_an_empty_palette(tmp_path: Path) -> None:
    done = _validate(_palette(tmp_path, []))
    assert done.returncode != 0


# registration


def _registered_root(tmp_path: Path) -> Path:
    import shutil

    import check_registration

    for rel in check_registration.REGISTERED.values():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, target)
    check_registration.register(tmp_path)
    return tmp_path


def test_registration_gate_passes_what_was_registered(tmp_path: Path) -> None:
    import check_registration

    root = _registered_root(tmp_path)
    assert check_registration.check(root) == []


def test_registration_gate_catches_a_change_without_a_dated_entry(tmp_path: Path) -> None:
    import check_registration

    root = _registered_root(tmp_path)
    plan = root / "evaluation" / "plan.yaml"
    plan.write_text(plan.read_text(encoding="utf-8") + "\n# a quiet change\n", encoding="utf-8")
    new = check_registration.current(root)["plan_hash"][:12]
    problems = check_registration.check(root)
    assert len(problems) == 1 and problems[0].startswith("plan_hash")
    (root / "DECISIONS.md").write_text(f"## 2001-01-01 an old entry\n\nNames {new}.\n", encoding="utf-8")
    assert check_registration.check(root), "an entry dated before the registration does not count"
    today = check_registration.dt.date.today().isoformat()
    (root / "DECISIONS.md").write_text(
        f"## {today} the plan changed\n\nThe new plan hash is {new}.\n", encoding="utf-8"
    )
    assert check_registration.check(root) == []


def test_registration_gate_refuses_when_nothing_is_registered(tmp_path: Path) -> None:
    import check_registration

    assert check_registration.check(tmp_path) == ["nothing is registered: evaluation/registration.json does not exist"]
    root = _registered_root(tmp_path)
    with pytest.raises(SystemExit):
        check_registration.register(root)


# skills


def test_skills_gate_passes_the_committed_skills() -> None:
    import check_skills

    assert check_skills.check() == []


def test_skills_gate_catches_a_skill_without_a_description(tmp_path: Path) -> None:
    import check_skills

    for name in check_skills.REQUIRED:
        folder = tmp_path / ".claude" / "skills" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Does one thing.\n---\n\nBody.\n")
    bad = tmp_path / ".claude" / "skills" / "add-a-reward" / "SKILL.md"
    bad.write_text("---\nname: add-a-reward\n---\n\nBody.\n")
    problems = check_skills.check(tmp_path)
    assert problems and all(p.startswith("add-a-reward") for p in problems)


def test_skills_gate_refuses_when_there_are_no_skills(tmp_path: Path) -> None:
    import check_skills

    problems = check_skills.check(tmp_path)
    assert "no skill folder under .claude/skills" in problems and len(problems) == 4
