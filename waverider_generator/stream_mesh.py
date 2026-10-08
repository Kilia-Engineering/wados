"""
Triangulated half-model mesh from upper/lower surface streams.

Shared by the stream-based generators (VMOF, Hybrid GOC) for STL and
Cart3D TRI export.  Streams follow the OC convention: a list of (n_pts, 3)
arrays, one per spanwise station, index 0 on the symmetry plane (Z = 0),
point 0 of each stream on the leading edge.
"""

import numpy as np


def build_stream_mesh(upper_streams, lower_streams):
    """
    Build a closed triangulated half-model from upper and lower streams.

    Returns
    -------
    vertices : ndarray (n_verts, 3)
    triangles : ndarray (n_tri, 3)  — 0-based vertex indices
    """
    us = upper_streams
    ls = lower_streams

    n_span = len(us)

    # Streams may have different lengths (tip stream has only 2 points in OC).
    # Use the most common stream length for the mesh; degenerate tip stream
    # is handled by clamping to its last valid point.
    n_stream_vals = [s.shape[0] for s in us]
    n_stream = max(n_stream_vals)

    def _pad_stream(s, n_target):
        """Extend a short stream to n_target by repeating its last point."""
        if s.shape[0] >= n_target:
            return s[:n_target]
        pad = np.tile(s[-1:, :], (n_target - s.shape[0], 1))
        return np.vstack([s, pad])

    us_padded = [_pad_stream(s, n_stream) for s in us]
    ls_padded = [_pad_stream(s, n_stream) for s in ls]

    verts_upper = np.vstack(us_padded)   # (n_span * n_stream, 3)
    verts_lower = np.vstack(ls_padded)   # (n_span * n_stream, 3)
    vertices = np.vstack([verts_upper, verts_lower])
    lower_start = n_span * n_stream

    triangles = []

    # Upper surface quads
    for i in range(n_span - 1):
        for j in range(n_stream - 1):
            v00 = i       * n_stream + j
            v01 = i       * n_stream + j + 1
            v10 = (i + 1) * n_stream + j
            v11 = (i + 1) * n_stream + j + 1
            triangles.append([v00, v10, v01])
            triangles.append([v01, v10, v11])

    # Lower surface quads (reversed winding)
    for i in range(n_span - 1):
        for j in range(n_stream - 1):
            v00 = lower_start + i       * n_stream + j
            v01 = lower_start + i       * n_stream + j + 1
            v10 = lower_start + (i + 1) * n_stream + j
            v11 = lower_start + (i + 1) * n_stream + j + 1
            triangles.append([v00, v01, v10])
            triangles.append([v01, v11, v10])

    # Leading edge cap: upper LE row to lower LE row
    for i in range(n_span - 1):
        u0 = i       * n_stream
        u1 = (i + 1) * n_stream
        l0 = lower_start + i       * n_stream
        l1 = lower_start + (i + 1) * n_stream
        triangles.append([u0, l0, u1])
        triangles.append([u1, l0, l1])

    # Trailing edge cap (base plane)
    for i in range(n_span - 1):
        u0 = i       * n_stream + (n_stream - 1)
        u1 = (i + 1) * n_stream + (n_stream - 1)
        l0 = lower_start + i       * n_stream + (n_stream - 1)
        l1 = lower_start + (i + 1) * n_stream + (n_stream - 1)
        triangles.append([u0, u1, l0])
        triangles.append([u1, l1, l0])

    # Symmetry-plane cap (i = 0, Z = 0)
    for j in range(n_stream - 1):
        u0 = j
        u1 = j + 1
        l0 = lower_start + j
        l1 = lower_start + j + 1
        triangles.append([u0, u1, l0])
        triangles.append([u1, l1, l0])

    # Wingtip cap (i = n_span - 1)
    i_tip = n_span - 1
    for j in range(n_stream - 1):
        u0 = i_tip * n_stream + j
        u1 = i_tip * n_stream + j + 1
        l0 = lower_start + i_tip * n_stream + j
        l1 = lower_start + i_tip * n_stream + j + 1
        triangles.append([u0, l0, u1])
        triangles.append([u1, l0, l1])

    return vertices, np.array(triangles, dtype=int)


def write_ascii_stl(vertices, triangles, filename, solid_name="waverider"):
    """Write a triangle mesh as an ASCII STL file."""
    with open(filename, 'w') as fh:
        fh.write(f"solid {solid_name}\n")
        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            nrm = np.linalg.norm(normal)
            if nrm > 1e-14:
                normal /= nrm
            else:
                normal = np.array([0.0, 0.0, 1.0])
            fh.write(
                f"  facet normal "
                f"{normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
            )
            fh.write("    outer loop\n")
            fh.write(
                f"      vertex "
                f"{v0[0]:.8f} {v0[1]:.8f} {v0[2]:.8f}\n"
            )
            fh.write(
                f"      vertex "
                f"{v1[0]:.8f} {v1[1]:.8f} {v1[2]:.8f}\n"
            )
            fh.write(
                f"      vertex "
                f"{v2[0]:.8f} {v2[1]:.8f} {v2[2]:.8f}\n"
            )
            fh.write("    endloop\n")
            fh.write("  endfacet\n")
        fh.write(f"endsolid {solid_name}\n")


def write_tri(vertices, triangles, filename):
    """Write a triangle mesh in NASA Cart3D TRI format (1-based indices)."""
    n_v = len(vertices)
    n_t = len(triangles)

    with open(filename, 'w') as fh:
        fh.write(f"{n_v}\n")
        fh.write(f"{n_t}\n")
        for v in vertices:
            fh.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in triangles:
            # TRI format: 1-based vertex indices
            fh.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
