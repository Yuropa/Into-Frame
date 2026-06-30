import sys
import subprocess
from pathlib import Path

# Bootstrap sys.path before any server imports.
def _bootstrap():
    for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        if p.name == "server":
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            pkgs = p.parent / "lib" / "packages"
            if str(pkgs) not in sys.path:
                sys.path.append(str(pkgs))
            return p
    raise RuntimeError("Could not locate 'server' directory")

_SERVER = _bootstrap()
_LIB = _SERVER.parent / "lib"
_CHECKPOINTS = _SERVER.parent / "checkpoints"

import trimesh
from remote_connection.remote_server import RemoteServer


class TreeDGenerator(RemoteServer):
    def setup(self):
        pass  # Magic123 loads lazily per invocation

    def perform(self, action: str, temp_path: Path, input) -> str:
        if action != "meshify":
            raise ValueError(f"Unknown action: {action}")

        image = input["image"] if isinstance(input, dict) else input
        species = input.get("species", "") if isinstance(input, dict) else ""

        rgba_path = (temp_path / "rgba.png").resolve()
        image.save(str(rgba_path))

        ckpt_dir = _CHECKPOINTS / "treed"
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise RuntimeError(f"No .ckpt checkpoint found in {ckpt_dir}")
        ckpt_path = ckpt_files[0]

        prompt = f"_{species}_tree_" if species else "_tree_"
        treed_dir = _LIB / "TreeDFusion"
        workspace = temp_path / "output"

        cmd = [
            sys.executable, "Magic123/main.py", "-O",
            "--text", prompt,
            "--sd_version", "1.5",
            "--image", str(rgba_path),
            "--workspace", str(workspace),
            "--optim", "adam",
            "--iters", "2000",
            "--h", "64",
            "--w", "64",
            "--mcubes_resolution", "128",
            "--guidance", "SD", "zero123",
            "--lambda_guidance", "1.0", "8.0",
            "--guidance_scale", "20.0", "5.0",
            "--latent_iter_ratio", "0",
            "--t_range", "0.2", "0.6",
            "--lambda_depth", "0",
            "--lambda_depth_mse", "0",
            "--lambda_normal", "0",
            "--lambda_normal_smooth", "0",
            "--lambda_normal_smooth2d", "0",
            "--lambda_mask", "20.0",
            "--lambda_rgb", "5.0",
            "--lambda_opacity", "0.01",
            "--bg_radius", "-1",
            "--blob_density", "8", "--blob_radius", "0.35",
            "--save_mesh",
            "--zero123_config", "zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml",
            "--zero123_ckpt", str(ckpt_path),
        ]
        log_path = temp_path / "magic123.log"
        print(f"Magic123 log: {log_path}", flush=True)
        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=str(treed_dir), stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"Magic123 failed (exit {result.returncode}):\n{tail}")

        obj_path = workspace / "mesh" / "mesh.obj"
        if not obj_path.exists():
            raise RuntimeError(f"TreeD did not produce a mesh at {obj_path}")

        mesh = trimesh.load(str(obj_path))
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)

        mesh_path = temp_path / "mesh.glb"
        mesh.export(str(mesh_path), include_normals=True)
        return str(mesh_path)


if __name__ == "__main__":
    TreeDGenerator.run()
