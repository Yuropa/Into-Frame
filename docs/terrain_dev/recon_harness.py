"""
Offline harness for TerrainReconstructionStage.run — runs the REAL reconstruction
(cliff mask, ridge anchors, slope/shelf envelope, harmonic solve, restore) on a
saved debug context. Stubs ONLY the framework plumbing (torch type hints, the
PipelineStage/PipelineContext base classes, and landlab's raster grid); every line
of reconstruction math is the real code, so there's no drift.

Reproduces the saved reconstruction to HEIGHT_MAP max|Δ|=0.03 m (0.05% of range),
CLIFF_MASK exact, on the reference capture.

Usage:
    from recon_harness import run, radial, D, SCR, tr
    out = run()                          # -> {"HEIGHT_MAP":..., "CLIFF_MASK":..., ...}
    out = run(hm_override=<4096x4096>)   # feed a modified height map
    out = run(envelope_smooth_m=8.0)     # override any TerrainReconstruction cfg

Paths via env: IF_SERVER, IF_CONTEXT, IF_SCRATCH (see heightmap_harness.py).
"""
import os, sys, types, json, numpy as np
from pathlib import Path

SERVER = os.environ.get("IF_SERVER",  os.path.join(os.path.dirname(__file__), "..", "..", "server"))
D      = os.environ.get("IF_CONTEXT", "/Users/Josh/Desktop/Mount.debug 2/context/0bc34849-eb4d-e372-bc30-97d7d11064f0")
SCR    = os.environ.get("IF_SCRATCH", "/tmp/if_scratch")
SERVER = os.path.abspath(SERVER); os.makedirs(SCR, exist_ok=True)

# ── stub torch (only referenced in type hints) ──
_t = types.ModuleType("torch"); _t.device = str; _t.dtype = object; sys.modules["torch"] = _t

# ── stub framework plumbing ──
ps = types.ModuleType("pipeline.pipeline_stage")
class PipelineStageConfiguration:
    def __init__(self, name=None, device=None, torch_dtype=None, log=None, keys=None, seed=0, **kw):
        self.name=name; self.seed=seed; self.keys=keys
class PipelineStage:
    def __init__(self, config): self.config=config; self.temp=None
    def create_progress(self,*a,**k): return None
    def advance_progress(self,*a,**k): pass
    def finish_progress(self,*a,**k): pass
    def log_info(self,m): print("  [recon]",m)
    def log_warning(self,m): print("  [recon][WARN]",m)
ps.PipelineStageConfiguration=PipelineStageConfiguration; ps.PipelineStage=PipelineStage
sys.modules["pipeline.pipeline_stage"]=ps

pc = types.ModuleType("pipeline.pipeline_context")
_KEYS=["HEIGHT_MAP","HEIGHT_MAP_CELL_RELIEF","HEIGHT_MAP_CELL_SLOPE","HEIGHT_MAP_OBSERVED_MASK",
       "HEIGHT_MAP_REAL_SAMPLE_MASK","HEIGHT_MAP_CERTAINTY","HEIGHT_MAP_PARAMS","MOUNTAIN_RIDGE_CHAINS",
       "CLIFF_MASK","LINEAR_GRAPH","WATER_CHAINS","HEIGHT_MAP_PANO_UV_TRUST_MASK"]
class ContextKey: pass
for k in _KEYS: setattr(ContextKey,k,k)
class PipelineContext: pass
pc.ContextKey=ContextKey; pc.PipelineContext=PipelineContext
sys.modules["pipeline.pipeline_context"]=pc

# ── stub landlab.RasterModelGrid: node k = row*W+col (C-order), 4-neighbours,
#    -1 at grid edges (Neumann). This is all the harmonic solve uses landlab for. ──
ll = types.ModuleType("landlab")
class RasterModelGrid:
    BC_NODE_IS_CORE = 0; BC_NODE_IS_FIXED_VALUE = 1
    def __init__(self, shape, xy_spacing=1.0):
        self.H,self.W = shape; self.number_of_nodes = self.H*self.W
        self.status_at_node = np.zeros(self.number_of_nodes, dtype=np.int64); self._fields={}
        r = np.arange(self.number_of_nodes)//self.W; c = np.arange(self.number_of_nodes)%self.W
        nb = np.full((self.number_of_nodes,4), -1, dtype=np.int64); k = np.arange(self.number_of_nodes)
        nb[c+1<self.W,0]=k[c+1<self.W]+1; nb[r+1<self.H,1]=k[r+1<self.H]+self.W
        nb[c-1>=0,2]=k[c-1>=0]-1;         nb[r-1>=0,3]=k[r-1>=0]-self.W
        self.adjacent_nodes_at_node = nb
    def add_field(self, name, vals, at="node"): self._fields[name]=np.asarray(vals).copy(); return self._fields[name]
    @property
    def core_nodes(self): return np.where(self.status_at_node==self.BC_NODE_IS_CORE)[0]
ll.RasterModelGrid=RasterModelGrid; sys.modules["landlab"]=ll

sys.path.insert(0, SERVER)
from util.depth_utils import Depth
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pipeline.terrain.terrain_reconstruction", f"{SERVER}/pipeline/terrain/terrain_reconstruction.py")
tr = importlib.util.module_from_spec(spec); sys.modules["pipeline.terrain.terrain_reconstruction"]=tr
spec.loader.exec_module(tr)

HM=f"{D}/Height Map"
DEPTHS={
    "HEIGHT_MAP": f"{HM}/height_map.npy",
    "HEIGHT_MAP_CELL_RELIEF": f"{HM}/height_map_cell_relief.npy",
    "HEIGHT_MAP_CELL_SLOPE": f"{HM}/height_map_cell_slope.npy",
    "HEIGHT_MAP_OBSERVED_MASK": f"{HM}/height_map_observed_mask.npy",
    "HEIGHT_MAP_REAL_SAMPLE_MASK": f"{HM}/height_map_real_sample_mask.npy",
    "HEIGHT_MAP_CERTAINTY": f"{HM}/height_map_certainty.npy",
}
OBJS={
    "HEIGHT_MAP_PARAMS": json.load(open(f"{HM}/height_map_params.json")),
    "MOUNTAIN_RIDGE_CHAINS": json.load(open(f"{D}/Region Map/mountain_ridge_chains.json")),
    "WATER_CHAINS": json.load(open(f"{D}/Region Map/water_chains.json")),
    "LINEAR_GRAPH": None,   # no rivers in this capture
}
class FakeCtx:
    def __init__(self, hm_override=None): self._hm=hm_override; self.outputs={}
    def input_depth(self, key):
        if key=="HEIGHT_MAP" and self._hm is not None: return Depth(self._hm)
        p=DEPTHS.get(key); return Depth(np.load(p)) if p else None
    def input_object(self, key): return OBJS.get(key)
    def add_depth(self, key, depth): self.outputs[key]=depth.depth
    def add_object(self, key, obj): self.outputs[key]=obj

# config.yaml "Terrain Reconstruction" values for the reference capture.
CFG=dict(solve_resolution=512, ridge_min_anchor_distance=0.2, ridge_max_slope_angle_deg=38.0,
    ridge_override_min_distance_m=30.0, ridge_override_feather_m=15.0, river_valley_depth=0.5,
    river_drop_per_segment=0.05, lake_y_range_threshold=0.3, cliff_slope_angle_low_deg=50.0,
    cliff_slope_angle_high_deg=75.0, cliff_max_slope_angle_deg=82.0, cliff_shelf_angle_deg=8.0,
    cliff_shelf_crest_percentile=85.0, cliff_shelf_min_blob_cells=20, envelope_smooth_m=3.0,
    envelope_min_radius_m=15.0, envelope_max_reach_m=25.0, envelope_reach_feather_m=10.0)

def run(hm_override=None, temp=None, **cfg_over):
    cfg=dict(CFG); cfg.update(cfg_over)
    config=tr.TerrainReconstructionConfiguration("Terrain Reconstruction","cpu",None,None,seed=0,**cfg)
    stage=tr.TerrainReconstructionStage(config); stage.temp=Path(temp) if temp else None
    ctx=FakeCtx(hm_override=hm_override); stage.run(ctx)
    return ctx.outputs

def radial(hm, label=""):
    G=hm.shape[0]; cc=(np.arange(G)-G/2+0.5)*(200.0/G); X,Z=np.meshgrid(cc,cc); r=np.sqrt(X**2+Z**2)
    print(f"  [{label}] p50:", " ".join(f"{a}:{np.median(hm[(r>=a)&(r<b)]):.1f}"
        for a,b in [(0,10),(10,20),(20,30),(30,45),(45,60),(60,80),(80,100)]))

if __name__=="__main__":
    import time; t=time.time(); out=run(); hm=out["HEIGHT_MAP"]
    print(f"ran in {time.time()-t:.0f}s"); radial(hm,"harness")
    saved=np.load(f"{D}/Terrain Reconstruction/height_map.npy"); radial(saved,"saved  ")
    print(f"HEIGHT_MAP vs saved: max|Δ|={np.abs(hm-saved).max():.3f}")
    print(f"CLIFF_MASK vs saved: max|Δ|={np.abs(out['CLIFF_MASK']-np.load(f'{D}/Terrain Reconstruction/cliff_mask.npy')).max():.3f}")
    np.save(os.path.join(SCR,"recon_base.npy"), hm)
