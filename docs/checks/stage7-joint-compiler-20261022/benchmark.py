import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(1)
import pytensor
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash
from cyber_threat_modelling.data.store import write_json

parser=argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
folder=Path('artifacts/stage7/joint-level-v1/primary/representative-7101')
run=json.loads((folder/'run.json').read_text())
with np.load(folder/'data.npz') as f: data=dict(f)
with np.load(folder/'truth.npz') as f: truth=dict(f)
start=time.perf_counter()
model=build_model(data,run['options'],truth)
graph=time.perf_counter()-start
start=time.perf_counter()
fused=model.compile_fn([model.logp(),model.dlogp()],inputs=model.value_vars,mode='NUMBA')
compile_time=time.perf_counter()-start
point=model.initial_point()
start=time.perf_counter();initial=fused(point);first=time.perf_counter()-start
start=time.perf_counter();native=model.compile_fn([model.logp(),model.dlogp()],inputs=model.value_vars)
native_compile=time.perf_counter()-start
points=[point]
rng=np.random.default_rng(251022)
points.append({k:v+rng.normal(0,.05,np.shape(v)) for k,v in point.items()})
near={k:np.array(v,copy=True) for k,v in point.items()}
near['f_phi_interval__']=np.log((1+np.array([.35,.9999]))/(1-np.array([.35,.9999])))
points.append(near)
comparisons=[]
for p in points:
 a,b=fused(p),native(p)
 np.testing.assert_allclose(a[0],b[0],rtol=1e-8,atol=1e-6)
 np.testing.assert_allclose(a[1],b[1],rtol=1e-6,atol=1e-6)
 assert np.isfinite(a[0]) and np.isfinite(a[1]).all()
 comparisons.append({'logp_abs_error':float(abs(a[0]-b[0])), 'gradient_max_abs_error':float(np.max(abs(a[1]-b[1])))})
repeats=[]
for i in range(100):
 start=time.perf_counter();fused(point);repeats.append(time.perf_counter()-start)
write_json(args.output/'audit.json',dict(passed=True,source_sha256=file_hash(Path(__file__)),
 source_files={p.as_posix():file_hash(p) for p in Path('src').rglob('*.py')},
 run_sha256=file_hash(folder/'run.json'), data_sha256=file_hash(folder/'data.npz'),
 graph_seconds=graph,compile_seconds=compile_time,first_call_seconds=first,
 native_compile_seconds=native_compile, comparisons=comparisons,
 median_fused_seconds=float(np.median(repeats)),repeated_seconds=repeats,
 numba_opt=os.environ.get('NUMBA_OPT'),numba_fastmath=bool(pytensor.config.numba__fastmath),
 numba_cache=bool(pytensor.config.numba__cache), real_data_read=False,
 interpretation='Full representative density and gradient; deterministic points; concurrent workload. Not a sampling speedup or controlled cross-profile timing experiment.'))
print(json.dumps({'passed':True,'median_fused_seconds':float(np.median(repeats)),'compile_seconds':compile_time,'first_call_seconds':first}),flush=True)
