# The Scalecar deployment uses sim2real's state machine, not Scalecar's safety layer

Scalecar_v2 deploys behind a C++ 200 Hz controller that adds a state machine (Start/A/A/Select), command and state watchdogs, a latched damping fault, gated shaping, delayed-auto-recovery (`command_loss_hold`), shadow runs and commissioning records. For this phase we deliberately drop all of it and run the Tracker under sim2real's existing modes: A = Shaping (`init`), R1 = Tracker, B = Go, R2 = E-stop (`zero`, joint targets follow measured positions). The goal is to keep sim2real's runtime model unchanged; the cost is that there is no watchdog independent of the policy process and nothing latches. This is not an oversight — do not "restore" Scalecar's layer piecemeal. If it is ported later, it becomes a Bridge, not a "controller" (that word means an operator input device here).

## Consequences

- Soft-limit clipping and the 15 rad/s target rate limit are kept, because training applied them; they are part of the Tracker, not of the dropped safety layer.
- Shaping targets the Motion's first-frame pose (within soft limits) rather than the policy default pose, since the Tracker was only trained from that start state.
