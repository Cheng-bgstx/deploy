# Tracker weights are rebuilt from Scalecar's ONNX, not exported from a checkpoint

The Scalecar d1146 Tracker checkpoints (`model_*.ema.pt`) exist only on the training server; ModelScope ships just `actor.onnx` plus `deployment.json`. We therefore rebuild the PyTorch actor from the ONNX initializers (all 87 map back to parameters, max error ≤ 2.1e-6 against onnxruntime) and re-export it through a wrapper that matches sim2real's conventions: unbatched flat inputs taken from existing sim2real observation helpers, Reference Tokens computed inside the graph, and a single `action` output. sim2real gets no new inference code; every export is checked numerically against the original ONNX. If the `.pt` files ever turn up, the same wrapper can export from them instead.

## Considered Options

- Keep the original ONNX and adapt sim2real (sidecar `out_keys`, batch-dim handling, new 3-D observation classes): most sim2real code, rejected because the user wants sim2real changed as little as possible.
- Graph surgery with `onnx.compose` in front of the untouched graph: keeps the original nodes byte-for-byte but leaves a dynamic-batch subgraph inside and needs a new tool anyway.

## Consequences

- The export script is ours to maintain; a new Tracker checkpoint means re-running it.
- The re-exported graph's sha256 no longer matches `deployment.json.actor.sha256`, so provenance is the export script plus its numeric comparison, not the hash.
