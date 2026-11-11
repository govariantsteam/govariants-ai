"""Export a checkpoint to ONNX, and verify the export actually matches.

    python -m gvai.export --checkpoint ../models/gvai_mixed.pt --out ../models/gvai.onnx

The board size is a dynamic axis, so one file serves every topology. An export
that silently disagrees with the PyTorch model would be worse than no export
at all, so this always runs a parity check against onnxruntime at two
different board sizes before declaring success.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from .model import GraphNet, ModelConfig


def load_checkpoint(path: str | pathlib.Path) -> tuple[GraphNet, ModelConfig]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig(**blob["config"])
    net = GraphNet(config)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, config


def random_inputs(num_nodes: int, num_features: int, batch: int = 1, seed: int = 0):
    """Inputs shaped like the real thing.

    The adjacency must be an actual normalized adjacency, not a dense random
    matrix. A dense one has row sums in the tens, which drives activations
    through four layers into magnitudes where float32 summation order matters,
    and the resulting "parity failure" says nothing about the export. Real
    boards are sparse with row sums near 1.
    """
    from .data import normalized_adjacency

    rng = np.random.default_rng(seed)
    x = rng.random((batch, num_nodes, num_features)).astype(np.float32)

    # A square-ish grid graph of num_nodes nodes, as a stand-in for a board.
    width = int(np.ceil(np.sqrt(num_nodes)))
    edges = []
    for node in range(num_nodes):
        row, col = divmod(node, width)
        right = node + 1
        if col + 1 < width and right < num_nodes:
            edges += [[node, right], [right, node]]
        down = node + width
        if down < num_nodes:
            edges += [[node, down], [down, node]]
    edge_array = np.array(edges, dtype=np.uint32) if edges else np.zeros((0, 2), np.uint32)

    adjacency = normalized_adjacency(num_nodes, edge_array)
    a = np.repeat(adjacency[None, :, :], batch, axis=0)
    mask = np.ones((batch, num_nodes), dtype=np.float32)
    return x, a, mask


def export(net: GraphNet, config: ModelConfig, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    x, a, mask = random_inputs(25, config.num_features)
    args = (torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(mask))

    torch.onnx.export(
        net,
        args,
        str(out),
        input_names=["x", "a", "mask"],
        output_names=["node_logits", "pass_logit", "value"],
        dynamic_axes={
            "x": {0: "batch", 1: "nodes"},
            "a": {0: "batch", 1: "nodes", 2: "nodes2"},
            "mask": {0: "batch", 1: "nodes"},
            "node_logits": {0: "batch", 1: "nodes"},
            "pass_logit": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def check_parity(
    net: GraphNet, config: ModelConfig, onnx_path: pathlib.Path, num_nodes: int
) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x, a, mask = random_inputs(num_nodes, config.num_features, seed=num_nodes)

    with torch.no_grad():
        torch_out = net(torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(mask))

    onnx_out = session.run(None, {"x": x, "a": a, "mask": mask})

    worst = 0.0
    for name, torch_tensor, onnx_array in zip(
        ("node_logits", "pass_logit", "value"), torch_out, onnx_out
    ):
        diff = float(np.abs(torch_tensor.numpy() - onnx_array).max())
        print(f"    {name:<12} max abs diff {diff:.3e}")
        worst = max(worst, diff)
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="../models/gvai.onnx")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    net, config = load_checkpoint(args.checkpoint)
    out = pathlib.Path(args.out)
    export(net, config, out)
    size_kb = out.stat().st_size / 1024
    print(f"exported {out} ({size_kb:.0f} KB), config {config.to_dict()}")

    worst = 0.0
    # Two different board sizes, neither of them the size used for tracing,
    # which is what proves the dynamic axis actually works.
    for num_nodes in (49, 81):
        print(f"  parity at {num_nodes} nodes:")
        worst = max(worst, check_parity(net, config, out, num_nodes))

    if worst > args.tolerance:
        print(f"FAILED: worst difference {worst:.3e} exceeds {args.tolerance:.0e}")
        return 1
    print(f"parity OK: worst difference {worst:.3e} <= {args.tolerance:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
