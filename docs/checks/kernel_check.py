"""Historical exploratory kernel check; not the current specification.

The canonical fixed-centre check is kernel_check2.py. See README.md here.
This file preserves the earlier learned-centre construction and +/-200 reference.
"""

import numpy as np
def sig(x): return 1/(1+np.exp(-x))
b=12.0
# manuscript: two-sided, d in [-200,200]
d=np.arange(-200,201); w=sig((b-np.abs(d))/2); w/=np.sqrt((w**2).sum())
def acov(w,lag): return float((w[:-lag]*w[lag:]).sum()) if lag>0 else float((w**2).sum())
# causal truncated: d in [0,48]
d2=np.arange(0,49); w2=sig((b-np.abs(d2-b))/2); w2/=np.sqrt((w2**2).sum())
print("two-sided lag12:",acov(w,12)); print("causal lag12:",acov(w2,12))
# Causal representation with the same finite truncation, integer shift 200.
d3=np.arange(0,401); w3=sig((b-np.abs(d3-200))/2); w3/=np.sqrt((w3**2).sum())
print("causal wide-truncation lag12 (integer shift 200):",acov(w3,12))
# Retained edge weight at d=48 for the learned-centre example with b=18.
print("sigmoid at edge b=18,d=48:", sig((18-abs(48-18))/2))
