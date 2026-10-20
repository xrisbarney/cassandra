# Data storage

Stage 4 has populated raw and interim directories and audited source manifests.
Processed modelling panels belong to Stage 5. Successful raw source responses are
immutable and identified by retrieval timestamps and hashes.

The three data directories and generated artifacts are ignored by version control.

## Stage 4 snapshot

`raw/stage4-20261020` holds pinned HTTP responses; `interim/stage4-20261020`
holds normalized records and audit outputs. Both remain local and ignored by Git.
Published provenance lives under `docs/audits/stage4-20261020`.
BEA raw responses include echoed API credentials: do not publish them.

