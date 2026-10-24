# The vehicle sim2sim uses Scalecar's scene, robot model and physics, cropped to the Task Envelope

The Tracker and Planner results we compare against were produced in Scalecar_v2's MuJoCo setup: its compiled car scene, its own G1 model (torso and hand collision shapes, collision exclusions and torque limits differ materially from sim2real's G1), and 8 × 0.625 ms implicitfast substeps with MuJoCo position actuators plus backward-Euler damping. Vehicle entry is contact-rich, so the vehicle sim2sim reuses all of that unchanged, with MuJoCo pinned to 3.11.0 in the root env, instead of sim2real's 5 ms Python-PD loop and G1 model.

Unchanged, a 20 ms tick costs about 150 ms because 99% of each step goes to 194k static collision hulls. We therefore rebuild the scene locally from the ModelScope sources with an offline script (USD tooling declared inline, never in the root env) and keep only the hulls inside the Task Envelope swept by all 1146 Motions plus 0.3 m, and turn `vis.global.bvactive` off. On the reference episode this is bitwise identical to the delivered `scene.mjb` and about 3.5× faster. The delivered MJB stays as the reference for re-checking identity, and a per-tick guard marks an episode invalid when the robot comes near the envelope boundary, so cropping can never change a result silently.

## Considered Options

- Delivered `scene.mjb` with only `bvactive` off, run in parallel: no rebuild tooling, but about 0.16× real time per episode.
- `fusestatic` plus a permanently sleeping dummy body: about 16× faster, but contact order changes (sub-millimetre drift) and the sleeping body is not a documented use.
- sim2real's own G1 model and physics loop: contact behaviour against 1 mm hull shells at a 5 ms step is untested and the collision shapes differ.

## Consequences

- Any MuJoCo upgrade means rebuilding the scene; the MJB format is version-specific.
- The envelope is a scene parameter: new Motions or start ranges need a new envelope.
- Vehicle sim2sim physics intentionally differs from sim2real's other sim2sim paths.
