# Best-of-Breed Results

This directory contains the best-of-breed pipeline outputs copied from
`/Users/aaronsteiner/Documents/GitHub/PyDI/pipelines`.

Layout:

```text
results/best of breeds/<domain>/<variant>/
```

Domains: `companies`, `games`, `music`, `papers`, `products`.
Variants: `baseline`, `easy`, `medium`, `hard`.

Each run directory preserves the PyDI result artifacts for that domain and
variant, including the fused output table (`fused.csv`), predicted
correspondences (`correspondences.csv`), per-stage scores
(`per_stage_summary.csv`), stage-selection JSON files, the run `summary.md`,
and available end-to-end panel outputs.

Source run mapping:

| Domain | Variant | PyDI source run |
|---|---|---|
| companies | baseline | `pipelines/companies/run_slurm_baseline_255975` |
| companies | easy | `pipelines/companies/run_slurm_easy_255976` |
| companies | medium | `pipelines/companies/run_slurm_medium_255977` |
| companies | hard | `pipelines/companies/run_slurm_hard_255978` |
| games | baseline | `pipelines/games/run_slurm_baseline_255979` |
| games | easy | `pipelines/games/run_slurm_easy_255980` |
| games | medium | `pipelines/games/run_slurm_medium_255981` |
| games | hard | `pipelines/games/run_slurm_hard_255982` |
| music | baseline | `pipelines/music/run_slurm_baseline_255983` |
| music | easy | `pipelines/music/run_slurm_easy_255984` |
| music | medium | `pipelines/music/run_slurm_medium_255985` |
| music | hard | `pipelines/music/run_slurm_hard_255986` |
| papers | baseline | `pipelines/papers/run_slurm_baseline_256772` |
| papers | easy | `pipelines/papers/run_slurm_easy_256553` |
| papers | medium | `pipelines/papers/run_slurm_medium_256543` |
| papers | hard | `pipelines/papers/run_slurm_hard_256557` |
| products | baseline | `pipelines/products/run_slurm_baseline_255987` |
| products | easy | `pipelines/products/run_slurm_easy_255988` |
| products | medium | `pipelines/products/run_slurm_medium_255989` |
| products | hard | `pipelines/products/run_slurm_hard_255990` |
