"""Plot retained chains from the first, unadmitted rank-two fallback attempt."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cyber_threat_modelling.provenance import file_hash

folder=Path('artifacts/stage7/rank-two-joint-v1/primary/fallback_zero-7101')
output=Path(__file__).parent
with xr.open_dataset(folder/'posterior_0.nc',group='posterior') as post:
    phi=post['f_phi'].values[:,:,1]
    factor_mean=post['f_states'].values[:,:,1:,1].mean(axis=2)
    levels=post['m'].values[:,:,1]
    scale=post['m_sd'].values
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for chain in range(4):
    color=f'C{chain}'
    for ax,values in zip(axes.flat[:3],[phi,factor_mean,scale]):
        ax.plot(np.arange(phi.shape[1]),values[chain],lw=.65,alpha=.7,
                color=color,label=f'Chain {chain+1}')
    axes[1,1].scatter(factor_mean[chain,::3],levels[chain,::3],s=5,alpha=.35,
                       color=color,label=f'Chain {chain+1}')
for ax,title in zip(axes.flat[:3],['Second-factor persistence',
                                    'Second-factor mean over observed months',
                                    'Known-category level scale']):
    ax.set(title=title,xlabel='Retained draw')
axes[0,0].set_ylabel('AR coefficient')
axes[0,1].set_ylabel('Factor mean')
axes[1,0].set_ylabel('Level standard deviation')
axes[1,1].set(title='Level / factor-mean dependence',
               xlabel='Second-factor time mean',ylabel='Second category level')
axes[0,0].legend(ncol=2,fontsize=8)
fig.suptitle('Unadmitted fallback attempt: chain-mixing diagnosis only',fontsize=13)
for suffix in ['png','pdf']:
    fig.savefig(output/f'chain_mixing.{suffix}',dpi=180)
plt.close(fig)
(output/'figure_audit.json').write_text(json.dumps({
    'source_sha256':file_hash(Path(__file__)),
    'posterior_sha256':file_hash(folder/'posterior_0.nc'),
    'attempt_sha256':file_hash(folder/'attempt_0.json'),
    'chains':4,'retained_draws_per_chain':int(phi.shape[1]),
    'recovery_evidence':False,
    'artifacts':{f'chain_mixing.{s}':file_hash(output/f'chain_mixing.{s}') for s in ['png','pdf']},
    'interpretation':'Failed numerical gates. Plots diagnose persistence, level and state-mean mixing; they are not an admitted posterior or a truth-recovery result.'
},indent=2),encoding='utf-8')
