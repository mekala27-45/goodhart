"""Every promotion gate passes what it should and fails, on purpose, what it should not; a gate
without its numbers does not pass; the champion is the best checkpoint that cleared them all."""

from __future__ import annotations

from typing import Any

from goodhart_evaluation import gates
from goodhart_evaluation.metrics import ItemScore


def _scores(right: int, items: int = 40, refused: int = 0, refuse_items: int = 0) -> list[ItemScore]:
    """items answerable items in ten clusters, the first `right` confirmed, the first `refused` refused."""
    out = [
        ItemScore(
            f"q{i}", f"c{i % 10}", "lookup", "doorstep", False, n=1, confirmed=int(i < right), refused=int(i < refused)
        )
        for i in range(items)
    ]
    out += [ItemScore(f"r{i}", "refusal", "refusal", "ledger", True, n=1, refused=1) for i in range(refuse_items)]
    return out


def test_readback_test_needs_a_gain_whose_interval_clears_zero() -> None:
    assert gates.readback_test(_scores(30), _scores(10)).passed
    tied = gates.readback_test(_scores(10), _scores(10))
    assert not tied.passed and tied.measured and "not clear of zero" in tied.detail
    assert not gates.readback_test(None, _scores(10)).measured


def test_over_refusal_allows_two_points_and_not_three() -> None:
    assert gates.over_refusal(_scores(10, items=100, refused=2), _scores(10, items=100)).passed
    failed = gates.over_refusal(_scores(10, items=100, refused=3), _scores(10, items=100))
    assert not failed.passed and failed.value is not None and abs(failed.value - 0.03) < 1e-9
    # Refusing a refusal item is right, and never counts as over refusal.
    assert gates.over_refusal(_scores(10, refuse_items=20), _scores(10)).value == 0.0


def test_ifeval_allows_two_points_below_the_base_and_not_more() -> None:
    base = {"prompt_strict": 0.40}
    assert gates.ifeval({"prompt_strict": 0.385}, base).passed
    assert not gates.ifeval({"prompt_strict": 0.37}, base).passed
    assert not gates.ifeval(None, base).measured


def test_a_hack_detector_alert_in_the_last_fifty_steps_holds_the_promotion() -> None:
    late = [{"stat": "share_literal_only", "step": 180}]
    assert not gates.detectors(late, last_step=200).passed
    assert gates.detectors([{"stat": "share_literal_only", "step": 120}], last_step=200).passed
    # A training health alert is watched, not a hack.
    assert gates.detectors([{"stat": "frac_reward_zero_std", "step": 199}], last_step=200).passed
    assert gates.detectors(None, None).passed, "SFT and DPO have no steps to watch"


def test_the_seed_spread_is_bounded_where_there_are_seeds() -> None:
    assert not gates.seed_spread({0: 0.40, 1: 0.47}).passed
    assert gates.seed_spread({0: 0.40, 1: 0.43, 2: 0.42}).passed
    assert gates.seed_spread({0: 0.40}).passed
    assert not gates.seed_spread({}).measured


def test_parity_and_the_license() -> None:
    assert gates.parity({"compared": 84, "mismatches": 0}).passed
    assert not gates.parity({"compared": 84, "mismatches": 1}).passed
    assert not gates.parity(None).measured
    assert gates.license_gate("apache-2.0").passed and not gates.license_gate("llama3.2").passed


def _inputs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "readback_test": _scores(30),
        "base_readback_test": _scores(10),
        "test": _scores(20, items=100),
        "base_test": _scores(10, items=100),
        "ifeval": {"prompt_strict": 0.41},
        "base_ifeval": {"prompt_strict": 0.40},
        "alerts": [],
        "last_step": 200,
        "by_seed": {0: 0.5},
        "parity": {"compared": 84, "mismatches": 0},
        "license": "apache-2.0",
    }
    return {**base, **over}


def test_the_champion_is_the_best_checkpoint_that_cleared_every_gate() -> None:
    results = {
        "M1": gates.evaluate(_inputs()),
        "M3": gates.evaluate(_inputs(alerts=[{"stat": "share_unfiltered", "step": 190}])),
        "M4": gates.evaluate(_inputs(parity=None)),
    }
    assert (
        results["M1"]["passed"] and results["M3"]["failed"] == ["detectors"] and results["M4"]["failed"] == ["parity"]
    )
    assert gates.champion(results, {"M1": 0.5, "M3": 0.6, "M4": 0.7}) == "M1"
    assert gates.champion({"M3": results["M3"]}, {"M3": 0.6}) is None
    assert [g["gate"] for g in results["M1"]["gates"]] == list(gates.GATES)
