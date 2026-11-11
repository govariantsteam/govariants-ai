"""Training half of govariants-ai: reads self-play data written by the Rust
engine, trains a graph network, and exports ONNX for the engine to play with."""

__all__ = ["data", "model", "train", "export"]
