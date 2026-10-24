# Tracker and Planner run on an external x86 Host; the G1 only captures depth

The G1's Onboard Computer is an Orin NX. There is no evidence the Planner (ResNet-18 + DiT, 10 flow steps, bf16 on CUDA) meets its latency budget there, while on an x86 RTX-class GPU a plan takes about 40 ms. We therefore run sim2real's Tracker loop, the Bridge (`--robot-io zmq` + `scripts/g1/real_bridge.py`, since sim2real's inline backend is only declared for aarch64) and the Planner subprocess on an external Host tethered by Ethernet, and leave only the RealSense capture program on the Onboard Computer. This mirrors Scalecar_v2's own topology.

## Consequences

- All Planner, Tracker and depth timestamps share the Host's monotonic clock; depth frames are mapped onto it by a round-trip clock sync over the SSH tunnel.
- Moving inference onboard later is a separate decision that has to re-measure Planner and Tracker latency on the Orin NX.
