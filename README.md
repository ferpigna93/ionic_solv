# ionic_solv

A Python toolkit for computing **absolute solvation free energies of ions** using
Gaussian 16 and *ab initio* / DFT methods.

## Overview

Two thermodynamic approaches are implemented:

### 1 — Direct SMD/PCM
```
Ion(g) ──────────────────────────────→ Ion(aq)
         ΔG°_solv = G°(solv) − G°(gas) + ΔG°_ss
```
*Fast, widely used. Recommended default.*

### 2 — Cluster-Continuum (CC)
```
Ion(g) + n·H₂O(g) → [Ion·(H₂O)_n](g)   ΔG°_bind  (DFT/gas-phase)
[Ion·(H₂O)_n](g)  → [Ion·(H₂O)_n](aq)  ΔG°_cont  (DFT/SMD)
n·H₂O(l)          → n·H₂O(g)            n·ΔG°_vap (+6.32 kcal/mol each)

ΔG°_solv(ion) = ΔG°_bind + ΔG°_cont − n·ΔG°_vap + ΔG°_ss
```
*More accurate for highly charged or strongly coordinating ions.*

---

## Repository layout

```
ionic_solv/
├── config/
│   └── calc_config.yaml          # all default parameters in one place
├── input_prep/
│   ├── generate_inputs.py        # generate Gaussian .com files
│   └── cluster_builder.py        # build [ion·(H₂O)_n] clusters
├── job_submission/
│   ├── submit_slurm.py           # write + submit SLURM scripts
│   └── templates/
│       └── slurm_g16.sh          # SLURM template (edit modules as needed)
├── post_processing/
│   ├── parse_gaussian.py         # parse .log files with cclib
│   └── solvation_energy.py       # assemble ΔG°_solv from parsed data
├── utils/
│   └── xyz_tools.py              # XYZ I/O, geometry extraction, RMSD
├── examples/
│   └── ions/                     # sample XYZ files (Na, Cl, K, Li, F, water)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/ferpigna93/ionic_solv.git
cd ionic_solv
pip install -r requirements.txt
```

Gaussian 16 must be installed and accessible as `g16` on your HPC cluster.

---

## Quick start — Direct SMD (recommended for most ions)

### Step 1 — Prepare inputs

```bash
# Single ion (Na+), B3LYP/6-311+G(d,p), SMD/Water
python input_prep/generate_inputs.py \
    -f examples/ions/Na.xyz \
    -c 1 -m 1 \
    -o jobs/Na/

# Multiple ions in one call
python input_prep/generate_inputs.py \
    -f examples/ions/*.xyz \
    -c -1 -m 1 \
    --functional M06-2X --basis "6-311+G(d,p)" \
    -o jobs/anions/
```

Two `.com` files are generated per ion:
- `Na_gas.com` — gas-phase optimisation + frequency
- `Na_solv.com` — solution-phase optimisation + frequency (SMD)

### Step 2 — Submit to SLURM

```bash
# Write SLURM scripts (no submission yet)
python job_submission/submit_slurm.py \
    -d jobs/Na/ --nproc 8 --mem 16GB

# Write AND submit
python job_submission/submit_slurm.py \
    -d jobs/Na/ --config config/calc_config.yaml --submit
```

### Step 3 — Post-processing

```bash
# Parse output files
python post_processing/parse_gaussian.py \
    -f jobs/Na/Na/Na_gas.log jobs/Na/Na/Na_solv.log \
    --csv Na_parsed.csv

# Compute delta-G solvation
python post_processing/solvation_energy.py direct \
    -g jobs/Na/Na/Na_gas.log \
    -s jobs/Na/Na/Na_solv.log \
    --ion-label Na+
```

Example output:
```
Ion               : Na
G(gas)            : -162.17654321 Ha
G(solv)           : -162.31572816 Ha
dG_solv (raw)     : -87.38 kcal/mol
dG_ss correction  : +1.89 kcal/mol
dG_solv (total)   : -85.49 kcal/mol  |  -357.68 kJ/mol

dG_solv(calc)  = -85.5 kcal/mol
dG_solv(exp.)  = -87.2 kcal/mol  (ref)
Error          = +1.7 kcal/mol
```

---

## Cluster-Continuum workflow

### Step 1 — Build clusters

```bash
python input_prep/cluster_builder.py \
    -f examples/ions/Na.xyz \
    -c 1 -m 1 \
    -n 1 2 3 4 \
    --nconformers 5 \
    -o cluster_jobs/Na/
```

This creates 5 random conformers for each n = 1, 2, 3, 4 and generates
`_gas.com` + `_solv.com` inputs for each.

Also run the bare ion and water monomer gas-phase jobs:
```bash
python input_prep/generate_inputs.py -f examples/ions/Na.xyz    -c 1 -m 1 -o ref_jobs/
python input_prep/generate_inputs.py -f examples/ions/water.xyz -c 0 -m 1 -o ref_jobs/
```

### Step 2 — Submit and run

```bash
python job_submission/submit_slurm.py -d cluster_jobs/Na/ --submit
python job_submission/submit_slurm.py -d ref_jobs/        --submit
```

### Step 3 — Compute cluster-continuum result

```bash
python post_processing/solvation_energy.py cluster \
    --ion-gas   ref_jobs/Na/Na_gas.log \
    --water-gas ref_jobs/water/water_gas.log \
    --cluster-dir cluster_jobs/Na/ \
    --n-water 1 2 3 4 \
    --ion-label Na+
```

---

## Configuration

Edit `config/calc_config.yaml` for project-wide defaults. Any parameter can be
overridden on the CLI.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `functional` | `B3LYP` | M06-2X or wB97X-D recommended |
| `basis_set` | `6-311+G(d,p)` | aug-cc-pVTZ for high accuracy |
| `scrf_model` | `SMD` | SMD > IEFPCM for absolute solvation energies |
| `solvent` | `Water` | Any Gaussian solvent keyword |
| `apply_standard_state_correction` | `true` | +1.89 kcal/mol (1 atm to 1 mol/L) |
| `cluster_continuum.n_water` | `[1,2,3,4]` | n values to scan |

---

## Python API

```python
from input_prep.generate_inputs import GaussianInputGenerator, CalcConfig
from post_processing.parse_gaussian import parse_log
from post_processing.solvation_energy import calc_direct, compare_to_experiment

# Generate inputs
cfg = CalcConfig.from_yaml("config/calc_config.yaml")
gen = GaussianInputGenerator(cfg)
gen.write_all("examples/ions/Na.xyz", charge=1, mult=1, outdir="jobs/Na")

# Parse outputs (after Gaussian runs)
gas  = parse_log("jobs/Na/Na_gas.log")
solv = parse_log("jobs/Na/Na_solv.log")

# Compute solvation free energy
result = calc_direct(gas, solv)
print(result.summary())
print(compare_to_experiment("Na+", result.dG_solv_kcal))
```

---

## Recommended DFT methods

| Method | Notes |
|--------|-------|
| B3LYP/6-311+G(d,p) | Fast, widely benchmarked baseline |
| M06-2X/6-311+G(d,p) | Better non-covalent interactions |
| wB97X-D/aug-cc-pVTZ | Dispersion + diffuse functions; best accuracy |
| MP2/aug-cc-pVTZ // DFT geometry | High-level single-point correction |

Diffuse functions (+ or aug-) are critical for anions.

---

## Key references

- Marenich, Cramer, Truhlar. *J. Phys. Chem. B* 2009, 113, 6378 — SMD model
- Pliego & Riveros. *J. Phys. Chem. A* 2001, 105, 7241 — cluster-continuum method
- Kelly, Cramer, Truhlar. *J. Phys. Chem. B* 2006, 110, 16066 — standard-state correction
- Tissandier et al. *J. Phys. Chem. A* 1998, 102, 7787 — proton hydration reference
- Marcus, Y. *Ion Solvation* 1985 — experimental reference data

---

## Public tools used

- [cclib](https://cclib.github.io/) — parsing Gaussian output files
- [GoodVibes](https://github.com/patonlab/GoodVibes) — quasi-RRHO thermochemistry
- [ccinput](https://github.com/cyllab/ccinput) — computational chemistry input generator
- [iGen](https://github.com/grahamhaug/iGen) — interactive Gaussian input generator
- [gaussianutility](https://github.com/Sungil-Hong/gaussianutility) — Gaussian utilities
