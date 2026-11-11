"""Checks that run without pytest: `python -m gvai.selftest`.

These exist because the interesting failures in this pipeline are quiet ones.
A NaN policy loss still lets training run and the value head still improves, so
nothing announces that the policy objective has stopped working.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

from .data import SelfPlayData, collate, normalized_adjacency, read_game
from .model import GraphNet, ModelConfig, policy_value_loss, top1_agreement


def _fake_example(num_nodes: int, num_features: int, seed: int):
    from .data import Example

    rng = np.random.default_rng(seed)
    # A path graph, so adjacency is non-trivial but easy to reason about.
    edges = np.array(
        [[i, i + 1] for i in range(num_nodes - 1)]
        + [[i + 1, i] for i in range(num_nodes - 1)],
        dtype=np.uint32,
    )
    policy = rng.random(num_nodes + 1).astype(np.float32)
    policy /= policy.sum()
    return Example(
        node_features=rng.random((num_nodes, num_features)).astype(np.float32),
        adjacency=normalized_adjacency(num_nodes, edges),
        policy_nodes=policy[:-1],
        policy_pass=float(policy[-1]),
        value=float(rng.uniform(-1, 1)),
    )


def test_mixed_board_sizes_give_a_finite_loss() -> None:
    """Regression: padded nodes once produced a NaN policy loss.

    Target is 0 and log_prob was -inf for padding, and 0 * -inf is NaN. It
    only appears when boards of different sizes share a batch, so
    single-topology training looked healthy while mixed training was broken.
    """
    batch = collate([_fake_example(n, 4, seed=n) for n in (9, 25, 49)])
    assert batch["x"].shape[1] == 49, "should pad to the largest board"
    assert batch["mask"][0].sum().item() == 9

    net = GraphNet(ModelConfig(num_features=4, hidden=16, layers=2))
    node_logits, pass_logit, value = net(batch["x"], batch["a"], batch["mask"])
    loss, policy_loss, value_loss = policy_value_loss(
        node_logits, pass_logit, value, batch
    )
    for name, tensor in [
        ("loss", loss),
        ("policy", policy_loss),
        ("value", value_loss),
    ]:
        assert torch.isfinite(tensor), f"{name} is not finite: {tensor.item()}"

    loss.backward()
    for parameter_name, parameter in net.named_parameters():
        assert parameter.grad is not None, f"{parameter_name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{parameter_name} grad not finite"


def test_padding_does_not_change_predictions() -> None:
    """A board must predict the same alone as it does padded in a batch.

    This is what makes mixed-topology batching legitimate; if the mask were
    applied incorrectly anywhere, padding would leak into the result.
    """
    torch.manual_seed(0)
    net = GraphNet(ModelConfig(num_features=4, hidden=16, layers=3)).eval()
    small = _fake_example(9, 4, seed=1)

    alone = collate([small])
    padded = collate([small, _fake_example(40, 4, seed=2)])

    with torch.no_grad():
        logits_alone, pass_alone, value_alone = net(
            alone["x"], alone["a"], alone["mask"]
        )
        logits_padded, pass_padded, value_padded = net(
            padded["x"], padded["a"], padded["mask"]
        )

    assert torch.allclose(logits_alone[0], logits_padded[0, :9], atol=1e-5), (
        "node logits changed when the board was padded"
    )
    assert torch.allclose(pass_alone[0], pass_padded[0], atol=1e-5), (
        "pass logit changed when the board was padded"
    )
    assert torch.allclose(value_alone[0], value_padded[0], atol=1e-5), (
        "value changed when the board was padded"
    )


def test_normalized_adjacency_matches_gcn_definition() -> None:
    """D^-1/2 (A + I) D^-1/2, symmetric, with self-loops."""
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.uint32)
    a = normalized_adjacency(3, edges)
    assert np.allclose(a, a.T), "must be symmetric"
    assert a[0, 0] > 0 and a[1, 1] > 0, "self-loops must be present"
    # Node 1 has degree 3 with its self-loop, node 0 has 2.
    assert np.isclose(a[0, 1], 1.0 / np.sqrt(2 * 3)), a[0, 1]
    assert np.isclose(a[0, 2], 0.0), "non-adjacent nodes stay zero"


def test_reads_real_self_play_data() -> None:
    root = pathlib.Path(__file__).resolve().parents[2] / "data"
    files = sorted(root.rglob("*.bin")) if root.exists() else []
    if not files:
        print("  (skipped: no self-play data under ./data)")
        return
    game = read_game(files[0])
    assert len(game) > 0
    assert game.node_features.shape == (
        len(game),
        game.num_nodes,
        game.num_features,
    )
    sums = game.policies.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), f"policies should be distributions: {sums[:3]}"
    assert game.values.min() >= -1.0 and game.values.max() <= 1.0


def test_untrained_policy_loss_is_about_uniform() -> None:
    """A fresh net should score roughly ln(actions), which catches gross
    wiring errors in the loss."""
    batch = collate([_fake_example(16, 4, seed=7) for _ in range(4)])
    net = GraphNet(ModelConfig(num_features=4, hidden=16, layers=2))
    node_logits, pass_logit, value = net(batch["x"], batch["a"], batch["mask"])
    _, policy_loss, _ = policy_value_loss(node_logits, pass_logit, value, batch)
    expected = np.log(17)
    assert abs(policy_loss.item() - expected) < 0.6, (
        f"policy loss {policy_loss.item():.3f} is far from ln(17)={expected:.3f}"
    )


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as error:
            print(f"FAIL {test.__name__}: {error}")
            failures += 1
        except Exception as error:  # noqa: BLE001
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")
            failures += 1
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
