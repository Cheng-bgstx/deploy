"""Surface-preserving spatial hull conversion for static vehicle triangle meshes."""
from collections.abc import Iterator

import numpy as np
from scipy.spatial import ConvexHull


def spatial_triangle_hulls(vertices: np.ndarray, faces: np.ndarray, *,
                           cell_m: float = .04, shell_m: float = .001,
                           local_error_m: float = .006
                           ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield local convex shells without bridging distant vehicle surfaces.

    Input/output coordinates are metres in the same frame. Long triangles are
    subdivided before centroid binning; every nondegenerate source triangle is
    retained. A thin shell makes planar patches valid MuJoCo convex meshes.
    The cell size is a partition parameter, not a certified surface-error bound.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Vehicle vertices must be finite [N,3] metre coordinates')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Vehicle faces must be integer [M,3] triangle indices')
    if not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Vehicle triangle indices are empty or outside the vertex array')
    if not np.isfinite(cell_m) or not .005 <= cell_m <= .10:
        raise ValueError('cell_m must be finite and in [0.005,0.10] metres')
    if not np.isfinite(shell_m) or not 0 < shell_m <= cell_m / 10:
        raise ValueError('shell_m must be positive and at most cell_m/10')
    if not np.isfinite(local_error_m) or local_error_m <= shell_m:
        raise ValueError('local_error_m must be finite and larger than shell_m')
    pending = vertices[faces]
    complete = []
    while len(pending):
        edges = np.stack((pending[:, 1] - pending[:, 0], pending[:, 2] - pending[:, 1],
                          pending[:, 0] - pending[:, 2]), axis=1)
        length2 = np.sum(edges**2, axis=2)
        long = length2.max(axis=1) > cell_m**2
        complete.append(pending[~long])
        if not long.any():
            break
        triangles = pending[long]
        edge = length2[long].argmax(axis=1)
        order = np.stack((edge, (edge + 1) % 3, (edge + 2) % 3), axis=1)
        triangles = np.take_along_axis(triangles, order[:, :, None], axis=1)
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        mid = (a + b) / 2
        pending = np.concatenate((np.stack((a, mid, c), axis=1), np.stack((mid, b, c), axis=1)))
    triangles = np.concatenate(complete)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    magnitude = np.linalg.norm(normal, axis=1)
    triangles, normal = triangles[magnitude > 1e-12], normal[magnitude > 1e-12]
    if not len(triangles):
        raise ValueError('Vehicle mesh has no nondegenerate triangles')
    normal /= magnitude[magnitude > 1e-12, None]
    keys = np.floor(triangles.mean(axis=1) / cell_m).astype(np.int64)
    # Opposite or perpendicular surfaces can share a spatial cell. Keeping
    # their dominant signed normals separate avoids filling thin cavities.
    dominant = np.abs(normal).argmax(axis=1)
    direction = 2 * dominant + (normal[np.arange(len(normal)), dominant] < 0)
    keys = np.column_stack((keys, direction))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    boundaries = np.r_[0, np.cumsum(np.bincount(inverse))]
    def convert(ids):
        # Per-triangle offsets preserve both sides, including opposed normals.
        points = np.concatenate((triangles[ids] + shell_m/2*normal[ids, None],
                                 triangles[ids] - shell_m/2*normal[ids, None])).reshape(-1, 3)
        points = np.unique(points, axis=0)
        hull = ConvexHull(points)
        points = points[hull.vertices]
        hull = ConvexHull(points)
        facets = hull.simplices.copy()
        face_normals = np.cross(points[facets[:, 1]] - points[facets[:, 0]],
                                points[facets[:, 2]] - points[facets[:, 0]])
        reverse = np.sum(face_normals * hull.equations[:, :3], axis=1) < 0
        facets[reverse] = facets[reverse][:, [0, 2, 1]]
        if len(ids) > 1:
            import trimesh
            surface = points[facets]
            probes = np.concatenate((surface.mean(1), (surface[:, 0]+surface[:, 1])/2,
                                     (surface[:, 1]+surface[:, 2])/2, (surface[:, 2]+surface[:, 0])/2))
            source = triangles[ids]
            minimum = np.full(len(probes), np.inf)
            for i in range(0, len(source), 32):
                chunk = source[i:i+32]
                for j in range(0, len(probes), 64):
                    p = probes[j:j+64]
                    repeated = np.repeat(chunk, len(p), axis=0)
                    query = np.tile(p, (len(chunk), 1))
                    nearest = trimesh.triangles.closest_point(repeated, query)
                    distances = np.linalg.norm(nearest-query, axis=1).reshape(len(chunk),len(p)).min(0)
                    minimum[j:j+len(p)] = np.minimum(minimum[j:j+len(p)], distances)
            if minimum.max() > local_error_m:
                centroids = triangles[ids].mean(1)
                axis = np.ptp(centroids, axis=0).argmax()
                sorted_ids = ids[np.argsort(centroids[:,axis], kind='stable')]
                mid = len(sorted_ids)//2
                yield from convert(sorted_ids[:mid])
                yield from convert(sorted_ids[mid:])
                return
        yield points, facets

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        yield from convert(order[start:end])
