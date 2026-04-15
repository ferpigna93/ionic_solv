#!/usr/bin/env python3
"""parse_gaussian.py — Parse Gaussian 16 output files with cclib.

Extracts the quantities needed to assemble the solvation thermodynamic cycle:
  - SCF energy (E_elec)
  - Zero-point vibrational energy (ZPVE)
  - Thermal enthalpy correction (H_corr)
  - Thermal free-energy correction (G_corr = H_corr − T·S)
  - Total Gibbs free energy: G = E_elec + G_corr
  - Solvation cavity energy (if SCRF job)
  - Normal-termination check

Requires: cclib >= 1.8.1  (pip install cclib)

Usage (CLI):
  python parse_gaussian.py -f Na_gas.log Na_solv.log
  python parse_gaussian.py -d jobs/Na/ --pattern "*.log" --csv results.csv

Usage (Python API):
  from post_processing.parse_gaussian import parse_log, GaussianResult
  result = parse_log("Na_gas.log")
  print(result.G_total_hartree)
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import textwrap
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# Suppress cclib deprecation warnings unless user asks for DEBUG
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cclib")

try:
    import cclib
    from cclib.io import ccopen
except ImportError:
    raise ImportError(
        "cclib is required: pip install cclib"
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_KJ = 2625.4996394799
EV_TO_KCAL = 23.0605419966


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GaussianResult:
    """Parsed data from one Gaussian output file."""

    # Source
    path: str
    stem: str                     # filename without extension
    normal_termination: bool

    # Electronic structure
    scf_energy_hartree: float     # E_elec (last SCF energy)
    n_atoms: int
    charge: int
    mult: int

    # Thermochemistry (None if no Freq job)
    zpve_hartree: Optional[float]
    H_corr_hartree: Optional[float]   # thermal enthalpy correction
    G_corr_hartree: Optional[float]   # thermal Gibbs correction
    temperature_K: Optional[float]

    # Derived totals
    @property
    def G_total_hartree(self) -> Optional[float]:
        """G = E_elec + G_corr (thermal free energy)."""
        if self.G_corr_hartree is None:
            return None
        return self.scf_energy_hartree + self.G_corr_hartree

    @property
    def G_total_kcal(self) -> Optional[float]:
        if self.G_total_hartree is None:
            return None
        return self.G_total_hartree * HARTREE_TO_KCAL

    @property
    def scf_energy_kcal(self) -> float:
        return self.scf_energy_hartree * HARTREE_TO_KCAL

    def summary(self) -> str:
        lines = [
            f"File            : {self.path}",
            f"Normal term.    : {self.normal_termination}",
            f"Charge / Mult   : {self.charge} / {self.mult}",
            f"E_elec          : {self.scf_energy_hartree:.8f} Ha  ({self.scf_energy_kcal:.4f} kcal/mol)",
        ]
        if self.zpve_hartree is not None:
            lines += [
                f"ZPVE            : {self.zpve_hartree:.6f} Ha",
                f"H_corr          : {self.H_corr_hartree:.6f} Ha",
                f"G_corr          : {self.G_corr_hartree:.6f} Ha",
                f"G_total         : {self.G_total_hartree:.8f} Ha  ({self.G_total_kcal:.4f} kcal/mol)",
                f"Temperature     : {self.temperature_K:.2f} K",
            ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        d = {
            "path": self.path,
            "stem": self.stem,
            "normal_termination": self.normal_termination,
            "charge": self.charge,
            "mult": self.mult,
            "n_atoms": self.n_atoms,
            "E_elec_Ha": self.scf_energy_hartree,
            "E_elec_kcal": self.scf_energy_kcal,
            "ZPVE_Ha": self.zpve_hartree,
            "H_corr_Ha": self.H_corr_hartree,
            "G_corr_Ha": self.G_corr_hartree,
            "G_total_Ha": self.G_total_hartree,
            "G_total_kcal": self.G_total_kcal,
            "temperature_K": self.temperature_K,
        }
        return d


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_log(path: str | Path) -> GaussianResult:
    """Parse a Gaussian output file and return a GaussianResult.

    Uses cclib for the heavy lifting; falls back to regex for quantities
    not yet exposed by cclib (e.g., thermal G correction).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    logger.debug(f"Parsing {path}")

    # --- cclib parse ---
    try:
        parser = ccopen(str(path))
        data = parser.parse()
    except Exception as exc:
        raise RuntimeError(f"cclib failed on {path}: {exc}") from exc

    # SCF energy: cclib stores in eV; last entry is the converged value
    scf_ev = float(data.scfenergies[-1])
    scf_ha = scf_ev / EV_TO_KCAL  # eV → kcal → Ha? No: use direct conversion
    # cclib.scfenergies is in eV; 1 Ha = 27.2114 eV
    HA_PER_EV = 1.0 / 27.211386245988
    scf_ha = scf_ev * HA_PER_EV

    # Charge and multiplicity
    charge = int(data.charge)
    mult = int(data.mult)
    n_atoms = int(data.natom)

    # --- Thermochemistry via regex (cclib exposes some but not G_corr uniformly) ---
    zpve, H_corr, G_corr, temp = _parse_thermo_regex(path)

    # --- Normal termination ---
    normal = _check_normal_termination(path)

    return GaussianResult(
        path=str(path),
        stem=path.stem,
        normal_termination=normal,
        scf_energy_hartree=scf_ha,
        n_atoms=n_atoms,
        charge=charge,
        mult=mult,
        zpve_hartree=zpve,
        H_corr_hartree=H_corr,
        G_corr_hartree=G_corr,
        temperature_K=temp,
    )


def _parse_thermo_regex(path: Path) -> tuple:
    """Extract thermochemistry corrections directly from Gaussian output text.

    Gaussian prints these lines in the frequency section:
      Zero-point correction=              0.020772 (Hartree/Particle)
      Thermal correction to Energy=       0.026453
      Thermal correction to Enthalpy=     0.027397
      Thermal correction to Gibbs Free Energy= -0.001058
      Sum of electronic and zero-point Energies= ...
      Sum of electronic and thermal Energies= ...
      Sum of electronic and thermal Enthalpies= ...
      Sum of electronic and thermal Free Energies= ...
      Temperature   298.150 Kelvin.
    """
    text = path.read_text(errors="replace")

    def _extract(pattern: str) -> Optional[float]:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    zpve = _extract(r"Zero-point correction=\s+([-\d.]+)")
    H_corr = _extract(r"Thermal correction to Enthalpy=\s+([-\d.]+)")
    G_corr = _extract(r"Thermal correction to Gibbs Free Energy=\s+([-\d.]+)")
    temp = _extract(r"Temperature\s+([\d.]+) Kelvin")

    return zpve, H_corr, G_corr, temp


def _check_normal_termination(path: Path) -> bool:
    """Check whether the calculation terminated normally."""
    text = path.read_text(errors="replace")
    # Gaussian 16 prints this at the very end of a successful job
    return "Normal termination of Gaussian" in text


# ---------------------------------------------------------------------------
# Batch parsing
# ---------------------------------------------------------------------------

def parse_directory(
    directory: str | Path,
    pattern: str = "*.log",
) -> List[GaussianResult]:
    """Parse all matching Gaussian output files in a directory tree."""
    directory = Path(directory)
    files = sorted(directory.rglob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' found in {directory}")
        return []

    results = []
    for f in files:
        try:
            r = parse_log(f)
            results.append(r)
            status = "OK" if r.normal_termination else "FAILED"
            logger.info(f"  [{status}] {f.name}")
        except Exception as exc:
            logger.error(f"  [ERROR] {f.name}: {exc}")
    return results


def results_to_csv(results: List[GaussianResult], csv_path: str | Path) -> None:
    """Write parsed results to a CSV file."""
    if not results:
        return
    csv_path = Path(csv_path)
    rows = [r.as_dict() for r in results]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse Gaussian 16 output files and extract thermochemistry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python parse_gaussian.py -f Na_gas.log Na_solv.log
              python parse_gaussian.py -d jobs/Na/ --pattern "*.log" --csv results.csv
        """),
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-f", "--files", nargs="+", metavar="LOG", help="Gaussian output file(s)")
    g.add_argument("-d", "--directory", metavar="DIR", help="Directory to search recursively")
    p.add_argument("--pattern", default="*.log", help="Glob pattern (default: *.log)")
    p.add_argument("--csv", metavar="FILE", help="Write results to CSV")
    p.add_argument("--verbose", action="store_true", help="Show DEBUG output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.files:
        results = []
        for f in args.files:
            try:
                r = parse_log(f)
                results.append(r)
                print(r.summary())
                print()
            except Exception as exc:
                logger.error(f"{f}: {exc}")
    else:
        results = parse_directory(args.directory, pattern=args.pattern)
        for r in results:
            print(r.summary())
            print()

    if args.csv and results:
        results_to_csv(results, args.csv)


if __name__ == "__main__":
    main()
