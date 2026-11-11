"""A graph network with policy and value heads.

Deliberately *not* PyTorch Geometric. One message-passing layer here is

    h' = relu(h W_self + (A_norm h) W_neigh + b)

which is a dense matmul against the normalized adjacency. At board sizes up to
a few hundred nodes a dense matrix is tiny (361x361 floats is ~130 KB) and a
matmul beats sparse scatter, while exporting to ONNX cleanly enough to run
under onnxruntime-web (issue #6). PyG's gather/scatter kernels are the usual
place that export falls over.

`A` is an *input*, not a buried constant, so one set of weights serves every
board topology and size: grids, tori, cube surfaces, circular boards. That is
the entire reason for using a graph net rather than a CNN.

The policy is split into per-node logits and a single pass logit instead of a
flat ``num_nodes + 1`` vector. Mixed board sizes in one batch mean the pass
slot would otherwise land at a different index per example.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


# Logit assigned to padded nodes before the softmax. Finite on purpose; see
# policy_value_loss.
MASKED_LOGIT = -1.0e9


@dataclass
class ModelConfig:
    num_features: int = 4
    hidden: int = 64
    layers: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


class MessagePassing(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_transform = nn.Linear(hidden, hidden)
        self.neighbor_transform = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # a: [B, N, N] normalized adjacency with self-loops.
        gathered = torch.matmul(a, h)
        return torch.relu(self.self_transform(h) + self.neighbor_transform(gathered))


class GraphNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        hidden = self.config.hidden

        self.embed = nn.Linear(self.config.num_features, hidden)
        self.blocks = nn.ModuleList(
            [MessagePassing(hidden) for _ in range(self.config.layers)]
        )
        self.policy_head = nn.Linear(hidden, 1)
        self.pass_head = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, x: torch.Tensor, a: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B, N, F]; a: [B, N, N]; mask: [B, N] with 1 for real nodes.

        Returns per-node policy logits [B, N], a pass logit [B], and a value
        in [-1, 1] [B], all from the perspective of the player to move.

        Node logits are returned unmasked: at play time there is no padding,
        and during training the caller masks before the softmax. That keeps
        the exported graph free of large-negative constants.
        """
        mask_c = mask.unsqueeze(-1)

        h = torch.relu(self.embed(x)) * mask_c
        for block in self.blocks:
            h = block(h, a) * mask_c

        node_logits = self.policy_head(h).squeeze(-1)

        # Masked mean over real nodes, so a padded board pools the same as it
        # would alone in a batch.
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask_c).sum(dim=1) / counts

        pass_logit = self.pass_head(pooled).squeeze(-1)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)
        return node_logits, pass_logit, value


def policy_value_loss(
    node_logits: torch.Tensor,
    pass_logit: torch.Tensor,
    value: torch.Tensor,
    batch: dict,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AlphaZero's loss: cross-entropy to the MCTS visit distribution, plus
    squared error to the game outcome.

    The target policy is already a distribution over legal actions, so this is
    a soft-target cross-entropy rather than a hard-label one.
    """
    mask = batch["mask"]
    # Mask with a large *finite* number, not -inf. Padded nodes have target 0
    # and would give log_prob -inf, and 0 * -inf is NaN. That only shows up
    # once boards of different sizes share a batch -- which is exactly the
    # mixed-topology training this model exists for -- so single-board runs
    # look fine while mixed runs silently produce a NaN loss.
    masked_logits = node_logits.masked_fill(mask < 0.5, MASKED_LOGIT)
    logits = torch.cat([masked_logits, pass_logit.unsqueeze(1)], dim=1)
    log_probs = torch.log_softmax(logits, dim=1)

    target = torch.cat([batch["policy_nodes"], batch["policy_pass"].unsqueeze(1)], dim=1)
    policy_loss = -(target * log_probs).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(value, batch["value"])
    return policy_loss + value_weight * value_loss, policy_loss, value_loss


def top1_agreement(
    node_logits: torch.Tensor, pass_logit: torch.Tensor, batch: dict
) -> torch.Tensor:
    """Fraction of positions where the net's best action is MCTS's best action."""
    mask = batch["mask"]
    masked_logits = node_logits.masked_fill(mask < 0.5, MASKED_LOGIT)
    logits = torch.cat([masked_logits, pass_logit.unsqueeze(1)], dim=1)
    target = torch.cat([batch["policy_nodes"], batch["policy_pass"].unsqueeze(1)], dim=1)
    return (logits.argmax(dim=1) == target.argmax(dim=1)).float().mean()
