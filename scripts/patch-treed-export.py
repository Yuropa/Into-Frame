#!/usr/bin/env python3
"""
Patch Tree-D Fusion's vendored Magic123 so the exported mesh carries colour.

lib/TreeDFusion/Magic123/nerf/renderer.py (git-ignored, installed by hand) writes
the marching-cubes geometry as a bare OBJ: an `mtllib mesh.mtl` line that refers
to a file it never writes, vertices and faces, nothing else. The NeRF it was
trained from has an albedo field, and this patch queries it at every vertex and
writes `v x y z r g b` lines, which trimesh reads as vertex colours and which
pipeline/tree_generation/model_generation_treed_imp.py then exports into the GLB.

Idempotent: run it after (re)installing Tree-D Fusion.

    python scripts/patch-treed-export.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "lib" / "TreeDFusion" / "Magic123" / "nerf" / "renderer.py"
MARK = "# [into-frame] vertex colours from the albedo field"

OLD = """        obj_file = os.path.join(path, f'mesh.obj')
        # clean

        
        v_np = vertices  # [N, 3]
        f_np = triangles  # [M, 3]
"""
NEW = """        obj_file = os.path.join(path, f'mesh.obj')
        # clean

        
        v_np = vertices  # [N, 3]
        f_np = triangles  # [M, 3]

        %s
        # The stock export writes geometry only; the trained field also predicts an
        # albedo per point, so sample it at each vertex and write it as an OBJ
        # vertex colour.
        c_np = None
        try:
            with torch.no_grad():
                v_t = torch.from_numpy(np.asarray(v_np, dtype=np.float32)).to(self.aabb_train.device)
                chunks = []
                for chunk in v_t.split(65536):
                    chunks.append(self.density(chunk)['albedo'].float().clamp(0, 1).cpu().numpy())
                c_np = np.concatenate(chunks, axis=0)
                if c_np.shape[0] != len(v_np) or c_np.shape[-1] != 3:
                    c_np = None
        except Exception as e:
            logger.info(f'[INFO] vertex colour export skipped: {e}')
            c_np = None
""" % MARK

OLD_WRITE = """            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \\n')
"""
NEW_WRITE = """            for i, v in enumerate(v_np):
                if c_np is not None:
                    c = c_np[i]
                    fp.write(f'v {v[0]} {v[1]} {v[2]} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} \\n')
                else:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \\n')
"""


def main() -> int:
    if not TARGET.exists():
        print(f"{TARGET} not found; is Tree-D Fusion installed under lib/?")
        return 1
    text = TARGET.read_text()
    if MARK in text:
        print("already patched")
        return 0
    if text.count(OLD) != 1 or text.count(OLD_WRITE) != 1:
        print("renderer.py does not match the expected export_mesh; patch by hand")
        return 1
    TARGET.write_text(text.replace(OLD, NEW).replace(OLD_WRITE, NEW_WRITE))
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
