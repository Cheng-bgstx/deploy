"""Deterministic bidirectional sampled surface-distance validation, metres."""
from pathlib import Path

import numpy as np


def surface_distance_report(source_vertices: np.ndarray, source_faces: np.ndarray,
                            converted_vertices: np.ndarray, converted_faces: np.ndarray, *,
                            threshold_m: float = .01, samples: int = 4096,
                            seed: int = 101) -> dict:
    """Report sampled max/p95 in each direction and reject excess distances.

    This is a reproducible numerical audit, not a certified Hausdorff bound.
    Exact unchanged arrays receive an analytic zero-distance report.
    ``passed`` must be checked by the converter before scene publication.
    """
    import trimesh

    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError('Surface distance threshold must be finite and positive')
    if type(samples) is not int or samples < 100:
        raise ValueError('Surface distance samples must be an integer >=100')
    arrays = [np.asarray(source_vertices), np.asarray(converted_vertices)]
    face_arrays = [np.asarray(source_faces), np.asarray(converted_faces)]
    for vertices, faces in zip(arrays, face_arrays):
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError('Surface vertices must be finite [N,3]')
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or not len(faces):
            raise ValueError('Surface faces must be nonempty integer [M,3]')
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('Surface face indices outside vertex array')
    same = np.array_equal(arrays[0], arrays[1]) and np.array_equal(face_arrays[0], face_arrays[1])
    report = dict(schema_version=1, method='exact_identity' if same else 'area_weighted_surface_samples',
                  samples_per_direction=0 if same else samples, seed=seed, threshold_m=threshold_m,
                  certified_global_bound=same, units='metres')
    meshes = [trimesh.Trimesh(v, f, process=False) for v, f in zip(arrays, face_arrays)]
    for direction, a, b in [('converted_to_source', 1, 0), ('source_to_converted', 0, 1)]:
        if same:
            distances = np.zeros(1)
        else:
            points, _ = trimesh.sample.sample_surface(meshes[a], samples, seed=seed+a)
            chunks = []
            for start in range(0, len(points), 128):
                chunks.append(trimesh.proximity.closest_point(meshes[b], points[start:start+128])[1])
            distances = np.concatenate(chunks)
        if not np.isfinite(distances).all():
            raise ValueError(f'Nonfinite surface distances in {direction}')
        report[direction] = dict(max_m=float(distances.max()), p95_m=float(np.percentile(distances, 95)))
    report['passed'] = all(report[d]['max_m'] <= threshold_m for d in ['converted_to_source','source_to_converted'])
    return report


def enforce_surface_distance(report: dict, component: str, output: Path) -> None:
    """Retain failure evidence before rejecting an invalid component."""
    import json

    output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise ValueError(f'{component} surface differs from source beyond {report["threshold_m"]}m; '
                         f'see {output}')
