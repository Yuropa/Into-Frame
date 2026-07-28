"""
Offline harness for TerrainNoiseRefinementStage.run — uses REAL landlab
(`pip install landlab` into the venv; it installs cleanly with modern numpy/scipy).
Stubs only torch + the pipeline framework, NOT landlab.

Reproduces the saved final terrain's roughness profile closely; absolute mean|Δ| is
~0.34 m (a floor set by landlab-version RNG differences in fBm + hydro erosion, not by
the code under test), so use it for *statistics / visuals*, not bit-exactness.

Paths via env: IF_SERVER, IF_CONTEXT (see heightmap_harness.py).
"""
import os, sys, types, json, numpy as np

SERVER = os.path.abspath(os.environ.get("IF_SERVER", os.path.join(os.path.dirname(__file__), "..", "..", "server")))
D      = os.environ.get("IF_CONTEXT", "/Users/Josh/Desktop/Mount.debug 2/context/0bc34849-eb4d-e372-bc30-97d7d11064f0")

_t = types.ModuleType("torch"); _t.device = str; _t.dtype = object
_t.Tensor = type("Tensor", (), {})   # scipy array-api-compat probes torch.Tensor
sys.modules["torch"] = _t

ps = types.ModuleType("pipeline.pipeline_stage")
class PipelineStageConfiguration:
    def __init__(self, name=None, device=None, torch_dtype=None, log=None, keys=None, seed=0, **kw):
        self.name=name; self.seed=seed; self.keys=keys
class PipelineStage:
    def __init__(self, config): self.config=config; self.temp=None
    def create_progress(self,*a,**k): return None
    def advance_progress(self,*a,**k): pass
    def finish_progress(self,*a,**k): pass
    def log_info(self,m): print("  [noise]",m)
    def log_warning(self,m): print("  [noise][WARN]",m)
ps.PipelineStageConfiguration=PipelineStageConfiguration; ps.PipelineStage=PipelineStage
sys.modules["pipeline.pipeline_stage"]=ps
pc = types.ModuleType("pipeline.pipeline_context")
_KEYS=["HEIGHT_MAP","HEIGHT_MAP_OBSERVED_MASK","HEIGHT_MAP_PARAMS","CLIFF_MASK",
       "HEIGHT_MAP_CERTAINTY","ROAD_SKELETON","MOUNTAIN_RIDGE_CHAINS"]
class ContextKey: pass
for k in _KEYS: setattr(ContextKey,k,k)
class PipelineContext: pass
pc.ContextKey=ContextKey; pc.PipelineContext=PipelineContext
sys.modules["pipeline.pipeline_context"]=pc

sys.path.insert(0, SERVER)
from util.depth_utils import Depth
import importlib.util
spec=importlib.util.spec_from_file_location(
    "pipeline.terrain.terrain_noise_refinement", f"{SERVER}/pipeline/terrain/terrain_noise_refinement.py")
nr=importlib.util.module_from_spec(spec); sys.modules["pipeline.terrain.terrain_noise_refinement"]=nr
spec.loader.exec_module(nr)

DEPTHS={
  "HEIGHT_MAP": f"{D}/Terrain Reconstruction/height_map.npy",     # reconstruction output
  "HEIGHT_MAP_OBSERVED_MASK": f"{D}/Height Map/height_map_observed_mask.npy",
  "CLIFF_MASK": f"{D}/Terrain Reconstruction/cliff_mask.npy",
  "HEIGHT_MAP_CERTAINTY": f"{D}/Height Map/height_map_certainty.npy",
  "ROAD_SKELETON": f"{D}/Region Map/road_skeleton.npy",
}
OBJS={
  "HEIGHT_MAP_PARAMS": json.load(open(f"{D}/Height Map/height_map_params.json")),
  "MOUNTAIN_RIDGE_CHAINS": json.load(open(f"{D}/Region Map/mountain_ridge_chains.json")),
}
class FakeCtx:
    def __init__(self, hm_override=None): self._hm=hm_override; self.outputs={}
    def input_depth(self,key):
        if key=="HEIGHT_MAP" and self._hm is not None: return Depth(self._hm)
        p=DEPTHS.get(key); return Depth(np.load(p)) if p else None
    def input_object(self,key): return OBJS.get(key)
    def add_depth(self,key,depth): self.outputs[key]=depth.depth
    def add_object(self,key,obj): self.outputs[key]=obj

# Production config.yaml values that differ from the code defaults.
PROD=dict(road_blur_sigma=4.0, road_terrain_smooth_sigma=6.0, noise_scale=80.0,
          linear_diffusivity=1.0e-3, diffusion_dt=2250.0)

def run(hm_override=None, prod=True, **cfg_over):
    cfg=dict(PROD) if prod else {}; cfg.update(cfg_over)
    config=nr.TerrainNoiseRefinementConfiguration("Terrain Noise Refinement","cpu",None,None,seed=0,**cfg)
    stage=nr.TerrainNoiseRefinementStage(config); stage.temp=None
    ctx=FakeCtx(hm_override=hm_override); stage.run(ctx); return ctx.outputs

if __name__=="__main__":
    import time; t=time.time(); hm=run()["HEIGHT_MAP"]
    print(f"ran in {time.time()-t:.0f}s  Y[{hm.min():.1f},{hm.max():.1f}]")
