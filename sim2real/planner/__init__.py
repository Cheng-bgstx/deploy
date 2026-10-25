"""The Scalecar Planner: depth + recent history -> Chunks of whole-body Reference (GLOSSARY: Planner, Chunk).

- ``scalecar_dp/``: Scalecar_v2's Planner core, copied verbatim (``SOURCES.json``).
- ``checkpoint``: the pinned d1146 checkpoint and its ``weights_only=True`` loader.
- ``worker``: the CUDA Planner subprocess (spawn) and its latest-value mailboxes.
- ``provider``: the 50 Hz side (Chunk acceptance, resampling, late blend, coverage, Go, history).
- ``reference``: the ``planner`` motion backend of ``tracking.py``.
- ``sim``: vehicle sim2sim pieces (depth rendering, simulation-time clock with injected latency).
"""
