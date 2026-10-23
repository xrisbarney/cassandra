"""Save complete retained chains without consuming sampler RNG or changing draws."""

from pathlib import Path

import pymc as pm
from pymc.backends.base import MultiTrace

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash


def save_complete_chain(trace, model, output, chain, warmup, draws):
    output = Path(output)
    destination = output/f'chain_{chain}.nc'
    marker = output/f'chain_{chain}.json'
    temporary = output/f'chain_{chain}.writing.nc'
    if any(p.exists() for p in (destination, marker, temporary)):
        raise FileExistsError('Preserve the existing completed-chain checkpoint.')
    if trace.chain != chain or len(trace) != warmup+draws:
        raise ValueError('Only the exact complete chain can be checkpointed.')
    retained = MultiTrace([trace])[warmup:]
    if len(retained) != draws:
        raise ValueError('Checkpoint retained length differs from the budget.')
    saved = pm.to_inference_data(trace=retained, model=model, save_warmup=False,
                                 log_likelihood=False)
    try:
        saved.posterior = saved.posterior.assign_coords(chain=[chain])
        saved.sample_stats = saved.sample_stats.assign_coords(chain=[chain])
        if saved.posterior.sizes['chain'] != 1 or saved.posterior.sizes['draw'] != draws:
            raise ValueError('Checkpoint posterior dimensions differ.')
        saved.to_netcdf(temporary)
        temporary.rename(destination)
        write_json(marker, {'chain': chain, 'warmup_excluded': warmup, 'retained_draws': draws,
                            'posterior_sha256': file_hash(destination),
                            'resumable_sampler_state': False})
    finally:
        saved.close()
    return {'chain': chain, 'path': destination.name, 'sha256': file_hash(destination),
            'marker_sha256': file_hash(marker)}
