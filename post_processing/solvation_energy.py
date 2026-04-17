#!/usr/bin/env python3
"""solvation_energy.py — Assemble the thermodynamic cycle and compute ΔG°_solv.

Two supported approaches:

1. DIRECT SMD (or PCM)
   ─────────────────────
   Ion(g) → Ion(aq)

   ΔG°_solv = G°(ion, solv) − G°(ion, gas)
            + ΔG°_ss          [standard-state correction, +1.89 kcal/mol]

   Requires:
     • <stem>_gas.log   — gas-phase opt+freq
     • <stem>_solv.log  — solution-phase opt+freq (SMD)


2. CLUSTER-CONTINUUM (n = 1, 2, 3, … explicit waters)
   ────────────────────────────────────────────────────
   Ion(g) + n·H₂O(g) → [Ion·(H₂O)_n](g)      ΔG°_bind
   [Ion·(H₂O)_n](g)  → [Ion·(H₂O)_n](aq)     ΔG°_cont
   n·H₂O(l)          → n·H₂O(g)               n·ΔG°_vap

   ΔG°_solv(ion) = ΔG°_bind + ΔG°_cont − n·ΔG°_vap + ΔG°_ss

   where:
     ΔG°_bind = G°(cluster, gas) − G°(ion, gas) − n·G°(H₂O, gas)
     ΔG°_cont = G°(cluster, solv) − G°(cluster, gas)
     ΔG°_vap  = +6.32 kcal/mol (H₂O vaporisation, 298 K)
     ΔG°_ss   = +1.89 kcal/mol (1 atm → 1 mol/L standard-state)

   Requires per n:
     • <stem>_n<N>_confK_gas.log   — cluster gas-phase opt+freq
     • <stem>_n<N>_confK_solv.log  — cluster solution-phase opt+freq
     • <stem>_gas.log              — bare ion gas-phase
     • water_gas.log               — single water gas-phase

Usage (CLI):
  # Direct:
  python solvation_energy.py direct -g Na_gas.log -s Na_solv.log

  # Cluster-continuum (Boltzmann-averaged over conformers):
  python solvation_energy.py cluster \\
      --ion-gas Na_gas.log \\
      --water-gas water_gas.log \\
      --cluster-jobs jobs/Na/ \\
      --n-water 1 2 3 4

Usage (Python API):
  from post_processing.solvation_energy import calc_direct, calc_cluster_continuum
  dG = calc_direct(gas_result, solv_result)
  dG = calc_cluster_continuum(ion_gas, water_gas, cluster_gas_list, cluster_solv_list, n=4)
"""

from __future__ import annotations

import argparse
import logging
import math
import textwrap
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from post_processing.parse_gaussian import GaussianResult, parse_log, parse_directory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

HARTREE_TO_KCAL = 627.5094740631
RT_298 = 0.5922  # RT at 298.15 K in kcal/mol  (R = 1.987 cal/mol·K)

# Standard-state correction: gas (1 atm, 24.5 L/mol) → solution (1 mol/L)
# ΔG°_ss = RT·ln(RT/P°V°) = RT·ln(24.5) ≈ +1.89 kcal/mol
DELTA_G_SS_KCAL = 1.89

# Free energy of water vaporisation: H₂O(l) → H₂O(g), 298.15 K, 1 atm
# Experimental: ΔG°_vap = +6.32 kcal/mol
DELTA_G_VAP_WATER_KCAL = 6.32

# Boltzmann constant in kcal/mol
KB_KCAL = 0.0019872041  # kcal mol⁻¹ K⁻¹


# ---------------------------------------------------------------------------
# Direct SMD approach
# ---------------------------------------------------------------------------

@dataclass
class DirectSolvationResult:
    ion_name: str
    G_gas_hartree: float
    G_solv_hartree: float
    dG_solv_hartree: float
    dG_solv_kcal: float
    dG_solv_kJ: float
    ss_correction_kcal: float
    G_gas_log: str
    G_solv_log: str

    def summary(self) -> str:
        lines = [
            f"Ion               : {self.ion_name}",
            f"G°(gas)           : {self.G_gas_hartree:.8f} Ha",
            f"G°(solv)          : {self.G_solv_hartree:.8f} Ha",
            f"ΔG°_solv (raw)    : {(self.G_solv_hartree - self.G_gas_hartree)*HARTREE_TO_KCAL:.2f} kcal/mol",
            f"ΔG°_ss correction : {self.ss_correction_kcal:+.2f} kcal/mol",
            f"ΔG°_solv (total)  : {self.dG_solv_kcal:.2f} kcal/mol  |  {self.dG_solv_kJ:.2f} kJ/mol",
        ]
        return "\n".join(lines)


def calc_direct(
    gas: GaussianResult,
    solv: GaussianResult,
    apply_ss_correction: bool = True,
    delta_G_ss: float = DELTA_G_SS_KCAL,
) -> DirectSolvationResult:
    """Compute ΔG°_solv from the direct SMD/PCM thermodynamic cycle.

    Requires both gas and solv results to have thermochemistry (G_corr).
    If freq data is missing, falls back to E_elec only (less accurate).
    """
    _check_normal(gas)
    _check_normal(solv)

    G_gas = gas.G_total_hartree
    G_solv = solv.G_total_hartree

    if G_gas is None or G_solv is None:
        logger.warning(
            "Thermal correction data missing — using E_elec only (less accurate)"
        )
        G_gas = gas.scf_energy_hartree
        G_solv = solv.scf_energy_hartree

    dG_raw_kcal = (G_solv - G_gas) * HARTREE_TO_KCAL
    ss = delta_G_ss if apply_ss_correction else 0.0
    dG_total_kcal = dG_raw_kcal + ss
    dG_total_kJ = dG_total_kcal * 4.184

    return DirectSolvationResult(
        ion_name=gas.stem.replace("_gas", ""),
        G_gas_hartree=G_gas,
        G_solv_hartree=G_solv,
        dG_solv_hartree=G_solv - G_gas,
        dG_solv_kcal=dG_total_kcal,
        dG_solv_kJ=dG_total_kJ,
        ss_correction_kcal=ss,
        G_gas_log=gas.path,
        G_solv_log=solv.path,
    )


# ---------------------------------------------------------------------------
# Cluster-continuum approach
# ---------------------------------------------------------------------------

@dataclass
class ClusterSolvationResult:
    ion_name: str
    n_water: int
    n_conformers: int
    dG_bind_kcal: float         # binding of n H₂O to ion in gas phase
    dG_cont_kcal: float         # continuum solvation of cluster
    dG_vap_correction_kcal: float  # −n·ΔG°_vap  (negative because H₂O(l)→(g))
    ss_correction_kcal: float
    dG_solv_kcal: float         # total
    dG_solv_kJ: float
    boltzmann_weights: List[float]

    def summary(self) -> str:
        lines = [
            f"Ion               : {self.ion_name}",
            f"n(H₂O)            : {self.n_water}  ({self.n_conformers} conformers)",
            f"ΔG°_bind          : {self.dG_bind_kcal:.2f} kcal/mol",
            f"ΔG°_cont          : {self.dG_cont_kcal:.2f} kcal/mol",
            f"−n·ΔG°_vap        : {self.dG_vap_correction_kcal:.2f} kcal/mol",
            f"ΔG°_ss            : {self.ss_correction_kcal:+.2f} kcal/mol",
            f"ΔG°_solv (total)  : {self.dG_solv_kcal:.2f} kcal/mol  |  {self.dG_solv_kJ:.2f} kJ/mol",
            f"Boltzmann weights : {[f'{w:.3f}' for w in self.boltzmann_weights]}",
        ]
        return "\n".join(lines)


def _boltzmann_average(
    energies_kcal: List[float],
    T: float = 298.15,
) -> Tuple[float, List[float]]:
    """Return Boltzmann-averaged energy and individual weights."""
    betas = [-e / (KB_KCAL * T) for e in energies_kcal]
    max_b = max(betas)
    exp_vals = [math.exp(b - max_b) for b in betas]
    Z = sum(exp_vals)
    weights = [e / Z for e in exp_vals]
    avg = sum(w * e for w, e in zip(weights, energies_kcal))
    return avg, weights


def calc_cluster_continuum(
    ion_gas: GaussianResult,
    water_gas: GaussianResult,
    cluster_gas_list: List[GaussianResult],
    cluster_solv_list: List[GaussianResult],
    n_water: int,
    T: float = 298.15,
    apply_ss_correction: bool = True,
    delta_G_ss: float = DELTA_G_SS_KCAL,
    delta_G_vap: float = DELTA_G_VAP_WATER_KCAL,
) -> ClusterSolvationResult:
    """Compute ΔG°_solv using the cluster-continuum thermodynamic cycle.

    Parameters
    ----------
    ion_gas          : parsed gas-phase result for bare ion
    water_gas        : parsed gas-phase result for a single water molecule
    cluster_gas_list : parsed gas-phase results for each [ion·(H₂O)_n] conformer
    cluster_solv_list: parsed SMD results for the same conformers (same order)
    n_water          : number of explicit water molecules in the cluster
    """
    _check_normal(ion_gas)
    _check_normal(water_gas)
    for r in cluster_gas_list + cluster_solv_list:
        _check_normal(r)

    def G(r: GaussianResult) -> float:
        g = r.G_total_hartree
        if g is None:
            logger.warning(f"{r.stem}: no thermal data, using E_elec")
            g = r.scf_energy_hartree
        return g

    G_ion = G(ion_gas)
    G_water = G(water_gas)

    # ΔG°_bind per conformer (gas phase)
    dG_bind_per_conf = [
        (G(cg) - G_ion - n_water * G_water) * HARTREE_TO_KCAL
        for cg in cluster_gas_list
    ]

    # ΔG°_cont per conformer (continuum solvation of cluster)
    dG_cont_per_conf = [
        (G(cs) - G(cg)) * HARTREE_TO_KCAL
        for cg, cs in zip(cluster_gas_list, cluster_solv_list)
    ]

    # Total ΔG°_solv per conformer (before Boltzmann averaging)
    n_conf = len(cluster_gas_list)
    vap_term = -n_water * delta_G_vap   # we transfer n H₂O from liquid to gas
    ss = delta_G_ss if apply_ss_correction else 0.0

    dG_total_per_conf = [
        dG_bind_per_conf[i] + dG_cont_per_conf[i] + vap_term + ss
        for i in range(n_conf)
    ]

    # Boltzmann average using gas-phase cluster energies to weight
    gas_G_vals = [G(cg) * HARTREE_TO_KCAL for cg in cluster_gas_list]
    _, weights = _boltzmann_average(gas_G_vals, T=T)

    dG_solv_avg = sum(w * dG for w, dG in zip(weights, dG_total_per_conf))
    dG_bind_avg = sum(w * dG for w, dG in zip(weights, dG_bind_per_conf))
    dG_cont_avg = sum(w * dG for w, dG in zip(weights, dG_cont_per_conf))

    return ClusterSolvationResult(
        ion_name=ion_gas.stem.replace("_gas", ""),
        n_water=n_water,
        n_conformers=n_conf,
        dG_bind_kcal=dG_bind_avg,
        dG_cont_kcal=dG_cont_avg,
        dG_vap_correction_kcal=vap_term,
        ss_correction_kcal=ss,
        dG_solv_kcal=dG_solv_avg,
        dG_solv_kJ=dG_solv_avg * 4.184,
        boltzmann_weights=weights,
    )


# ---------------------------------------------------------------------------
# Comparison with experimental data
# ---------------------------------------------------------------------------

EXPERIMENTAL_REF_KCAL: Dict[str, float] = {
    "Li+":  -113.5, "Na+":   -87.2, "K+":    -70.5,
    "Rb+":   -65.7, "Cs+":   -59.8,
    "Mg2+": -455.5, "Ca2+": -380.8,
    "F-":   -111.1, "Cl-":   -74.9, "Br-":   -67.8,
    "I-":    -57.8, "OH-":  -105.0,
}


def compare_to_experiment(
    ion_label: str,
    dG_calc_kcal: float,
    ref: Optional[Dict[str, float]] = None,
) -> str:
    if ref is None:
        ref = EXPERIMENTAL_REF_KCAL
    exp = ref.get(ion_label)
    if exp is None:
        return f"No experimental reference for '{ion_label}'"
    error = dG_calc_kcal - exp
    return (
        f"ΔG°_solv(calc)  = {dG_calc_kcal:.1f} kcal/mol\n"
        f"ΔG°_solv(exp.)  = {exp:.1f} kcal/mol  (ref)\n"
        f"Error           = {error:+.1f} kcal/mol"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_normal(r: GaussianResult) -> None:
    if not r.normal_termination:
        raise RuntimeError(
            f"Gaussian job did not terminate normally: {r.path}\n"
            "Check the output file for errors before post-processing."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute ΔG°_solv from Gaussian 16 output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Direct SMD:
              python solvation_energy.py direct -g Na_gas.log -s Na_solv.log

              # Cluster-continuum:
              python solvation_energy.py cluster \\
                  --ion-gas Na_gas.log --water-gas water_gas.log \\
                  --cluster-dir jobs/Na/ --n-water 1 2 3 4
        """),
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- direct ----
    dp = sub.add_parser("direct", help="Direct SMD/PCM approach")
    dp.add_argument("-g", "--gas", required=True, metavar="LOG", help="Gas-phase log")
    dp.add_argument("-s", "--solv", required=True, metavar="LOG", help="Solution-phase log")
    dp.add_argument("--no-ss", action="store_true", help="Skip standard-state correction")
    dp.add_argument("--ion-label", default=None, help="Label for experimental comparison (e.g. Na+)")

    # ---- cluster ----
    cp = sub.add_parser("cluster", help="Cluster-continuum approach")
    cp.add_argument("--ion-gas", required=True, metavar="LOG", help="Bare ion gas-phase log")
    cp.add_argument("--water-gas", required=True, metavar="LOG", help="Single water gas-phase log")
    cp.add_argument("--cluster-dir", required=True, metavar="DIR", help="Directory with cluster logs")
    cp.add_argument("--n-water", nargs="+", type=int, default=[1, 2, 3, 4], metavar="N")
    cp.add_argument("--no-ss", action="store_true")
    cp.add_argument("--ion-label", default=None)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.mode == "direct":
        gas = parse_log(args.gas)
        solv = parse_log(args.solv)
        result = calc_direct(gas, solv, apply_ss_correction=not args.no_ss)
        print(result.summary())
        if args.ion_label:
            print()
            print(compare_to_experiment(args.ion_label, result.dG_solv_kcal))

    elif args.mode == "cluster":
        ion_gas = parse_log(args.ion_gas)
        water_gas = parse_log(args.water_gas)
        cluster_dir = Path(args.cluster_dir)

        for n in args.n_water:
            # Expect files named *_n{n}_conf*_gas.log / *_n{n}_conf*_solv.log
            gas_logs = sorted(cluster_dir.rglob(f"*_n{n}_conf*_gas.log"))
            solv_logs = sorted(cluster_dir.rglob(f"*_n{n}_conf*_solv.log"))

            if not gas_logs:
                logger.warning(f"No cluster gas logs found for n={n}")
                continue

            cg_list = [parse_log(f) for f in gas_logs]
            cs_list = [parse_log(f) for f in solv_logs]

            result = calc_cluster_continuum(
                ion_gas, water_gas, cg_list, cs_list,
                n_water=n,
                apply_ss_correction=not args.no_ss,
            )
            print(f"\n{'='*55}")
            print(result.summary())
            if args.ion_label:
                print()
                print(compare_to_experiment(args.ion_label, result.dG_solv_kcal))


if __name__ == "__main__":
    main()
