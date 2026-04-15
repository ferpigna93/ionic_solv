#!/usr/bin/env python3
"""generate_inputs.py — Gaussian 16 input file generator for ionic solvation.

Generates the full set of .com files needed to compute absolute solvation free
energies of ions via the direct SMD approach or the cluster-continuum (CC)
thermodynamic cycle.

Calculation workflow (direct SMD):
  1. Gas-phase optimization + frequency  →  G°_gas
  2. Solution-phase opt + freq (SMD)     →  G°_solv
  ΔG°_solv = G°_solv − G°_gas  (+ optional standard-state correction)

Usage (CLI):
  python generate_inputs.py -f ion.xyz -c -1 -m 1 --solvent Water --nproc 8 --mem 16GB
  python generate_inputs.py -f ion.xyz -c 1  -m 1 --method M06-2X --basis 6-311+G(d,p)
  python generate_inputs.py --config ../config/calc_config.yaml -f *.xyz -c -1

Usage (Python API):
  from input_prep.generate_inputs import GaussianInputGenerator, CalcConfig
  cfg = CalcConfig.from_yaml("config/calc_config.yaml")
  gen = GaussianInputGenerator(cfg)
  gen.write_all("Na.xyz", charge=1, mult=1, outdir="jobs/Na")
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARTREE_TO_KCAL = 627.5094740631  # 1 Eh = 627.509 kcal/mol

SUPPORTED_SOLVENTS = {
    # Gaussian keyword → human label
    "water": "Water",
    "methanol": "Methanol",
    "ethanol": "Ethanol",
    "acetonitrile": "Acetonitrile",
    "dmso": "DMSO",
    "thf": "THF",
    "dichloromethane": "Dichloromethane",
    "chloroform": "Chloroform",
    "acetone": "Acetone",
    "toluene": "Toluene",
    "hexane": "Hexane",
}

SUPPORTED_MODELS = {"SMD", "IEFPCM", "CPCM"}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalcConfig:
    functional: str = "B3LYP"
    basis_set: str = "6-311+G(d,p)"
    nproc: int = 8
    mem: str = "16GB"
    solvent: str = "Water"
    scrf_model: str = "SMD"
    temperature: float = 298.15
    pressure: float = 1.0
    extra_keywords: str = ""
    opt_keyword: str = "Opt"
    freq_keyword: str = "Freq=NoRaman"
    apply_ss_correction: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CalcConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        m = data.get("method", {})
        r = data.get("resources", {})
        s = data.get("solvation", {})
        o = data.get("optimization", {})
        f = data.get("frequency", {})
        return cls(
            functional=m.get("functional", cls.functional),
            basis_set=m.get("basis_set", cls.basis_set),
            nproc=r.get("nproc", cls.nproc),
            mem=r.get("mem", cls.mem),
            solvent=s.get("solvent", cls.solvent),
            scrf_model=s.get("model", cls.scrf_model),
            temperature=f.get("temperature", cls.temperature),
            pressure=f.get("pressure", cls.pressure),
            extra_keywords=m.get("extra_keywords", cls.extra_keywords),
            opt_keyword=o.get("keyword", cls.opt_keyword),
            freq_keyword=f.get("keyword", cls.freq_keyword),
            apply_ss_correction=s.get("apply_standard_state_correction", cls.apply_ss_correction),
        )


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class GaussianInputGenerator:
    """Generate Gaussian 16 .com files for the ionic solvation workflow."""

    def __init__(self, config: CalcConfig):
        self.cfg = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write_all(
        self,
        xyz_path: str | Path,
        charge: int,
        mult: int,
        outdir: str | Path = ".",
        name: Optional[str] = None,
    ) -> List[Path]:
        """Generate the complete set of .com files for one ion.

        Returns a list of Paths to the generated files.
        """
        xyz_path = Path(xyz_path)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        stem = name or _sanitize(xyz_path.stem)
        coords = _read_xyz(xyz_path)

        files = []
        files.append(
            self._write_gas_opt_freq(coords, charge, mult, stem, outdir)
        )
        files.append(
            self._write_solv_opt_freq(coords, charge, mult, stem, outdir)
        )
        return files

    def write_gas_opt_freq(
        self,
        xyz_path: str | Path,
        charge: int,
        mult: int,
        outdir: str | Path = ".",
        name: Optional[str] = None,
    ) -> Path:
        """Gas-phase optimization + frequency — standalone helper."""
        xyz_path = Path(xyz_path)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = name or _sanitize(xyz_path.stem)
        coords = _read_xyz(xyz_path)
        return self._write_gas_opt_freq(coords, charge, mult, stem, outdir)

    def write_solv_opt_freq(
        self,
        xyz_path: str | Path,
        charge: int,
        mult: int,
        outdir: str | Path = ".",
        name: Optional[str] = None,
    ) -> Path:
        """Solution-phase optimization + frequency — standalone helper."""
        xyz_path = Path(xyz_path)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = name or _sanitize(xyz_path.stem)
        coords = _read_xyz(xyz_path)
        return self._write_solv_opt_freq(coords, charge, mult, stem, outdir)

    def write_sp_correction(
        self,
        xyz_path: str | Path,
        charge: int,
        mult: int,
        sp_functional: str,
        sp_basis: str,
        phase: str = "solv",
        outdir: str | Path = ".",
        name: Optional[str] = None,
    ) -> Path:
        """High-level single-point energy correction (e.g. MP2/aug-cc-pVTZ).

        Reads geometry from the previous checkpoint file.
        phase: "gas" or "solv"
        """
        xyz_path = Path(xyz_path)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = name or _sanitize(xyz_path.stem)

        label = f"{stem}_{phase}_sp"
        ref_chk = f"{stem}_{phase}"

        route = self._build_route_sp(sp_functional, sp_basis, phase)
        header = self._link0(label, ref_chk)
        body = f"Charge {charge}, mult {mult} — single-point {sp_functional}/{sp_basis}"

        content = self._assemble(header, route, body, charge, mult, coords=None)
        out = outdir / f"{label}.com"
        out.write_text(content)
        print(f"  Written: {out}")
        return out

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _write_gas_opt_freq(
        self,
        coords: List[Tuple[str, float, float, float]],
        charge: int,
        mult: int,
        stem: str,
        outdir: Path,
    ) -> Path:
        label = f"{stem}_gas"
        route = self._build_route_opt_freq(phase="gas")
        header = self._link0(label)
        title = f"{stem} — gas phase opt+freq  {self.cfg.functional}/{self.cfg.basis_set}"
        content = self._assemble(header, route, title, charge, mult, coords)
        out = outdir / f"{label}.com"
        out.write_text(content)
        print(f"  Written: {out}")
        return out

    def _write_solv_opt_freq(
        self,
        coords: List[Tuple[str, float, float, float]],
        charge: int,
        mult: int,
        stem: str,
        outdir: Path,
    ) -> Path:
        label = f"{stem}_solv"
        route = self._build_route_opt_freq(phase="solv")
        header = self._link0(label)
        title = (
            f"{stem} — {self.cfg.scrf_model}/{self.cfg.solvent} opt+freq  "
            f"{self.cfg.functional}/{self.cfg.basis_set}"
        )
        content = self._assemble(header, route, title, charge, mult, coords)
        out = outdir / f"{label}.com"
        out.write_text(content)
        print(f"  Written: {out}")
        return out

    def _link0(self, label: str, oldchk: Optional[str] = None) -> str:
        lines = []
        if oldchk:
            lines.append(f"%OldChk={oldchk}.chk")
        lines.append(f"%Chk={label}.chk")
        lines.append(f"%NProcShared={self.cfg.nproc}")
        lines.append(f"%Mem={self.cfg.mem}")
        return "\n".join(lines)

    def _build_route_opt_freq(self, phase: str) -> str:
        method = f"{self.cfg.functional}/{self.cfg.basis_set}"
        keywords = ["#p", method, self.cfg.opt_keyword, self.cfg.freq_keyword]

        temp_kw = f"Temperature={self.cfg.temperature:.2f}"
        keywords.append(temp_kw)

        if phase == "solv":
            scrf = f"SCRF=({self.cfg.scrf_model},Solvent={self.cfg.solvent})"
            keywords.append(scrf)

        if self.cfg.extra_keywords:
            keywords.append(self.cfg.extra_keywords)

        return " ".join(keywords)

    def _build_route_sp(
        self, functional: str, basis: str, phase: str
    ) -> str:
        method = f"{functional}/{basis}"
        keywords = ["#p", method, "SP", "Geom=AllCheck", "Guess=Read"]
        if phase == "solv":
            scrf = f"SCRF=({self.cfg.scrf_model},Solvent={self.cfg.solvent})"
            keywords.append(scrf)
        return " ".join(keywords)

    def _assemble(
        self,
        header: str,
        route: str,
        title: str,
        charge: int,
        mult: int,
        coords: Optional[List[Tuple[str, float, float, float]]],
    ) -> str:
        parts = [header, "", route, "", title, "", f"{charge} {mult}"]
        if coords:
            for sym, x, y, z in coords:
                parts.append(f" {sym:<3s}  {x:>14.8f}  {y:>14.8f}  {z:>14.8f}")
        parts.extend(["", ""])   # trailing blank lines required by Gaussian
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# XYZ helpers
# ---------------------------------------------------------------------------

def _read_xyz(path: Path) -> List[Tuple[str, float, float, float]]:
    """Parse a standard XYZ file; returns list of (symbol, x, y, z)."""
    lines = path.read_text().splitlines()
    try:
        n_atoms = int(lines[0].strip())
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse atom count from first line of {path}")
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        coords.append((sym, x, y, z))
    if not coords:
        raise ValueError(f"No coordinates found in {path}")
    return coords


def _sanitize(name: str) -> str:
    """Remove characters that cause issues in filenames and Gaussian titles."""
    return re.sub(r"[^\w\-]", "_", name)


# ---------------------------------------------------------------------------
# Batch generation from a directory of XYZ files
# ---------------------------------------------------------------------------

def batch_generate(
    xyz_files: List[Path],
    charge: int,
    mult: int,
    config: CalcConfig,
    outdir: Path,
) -> None:
    """Generate inputs for a list of XYZ files with shared charge/mult."""
    gen = GaussianInputGenerator(config)
    for xyz in xyz_files:
        ion_dir = outdir / _sanitize(xyz.stem)
        print(f"\n[{xyz.name}] → {ion_dir}/")
        gen.write_all(xyz, charge=charge, mult=mult, outdir=ion_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Gaussian 16 .com files for ionic solvation calculations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Single ion, default SMD/Water, B3LYP/6-311+G(d,p)
              python generate_inputs.py -f Cl.xyz -c -1

              # Batch: all XYZ in a folder, custom method
              python generate_inputs.py -f ions/*.xyz -c 1 -m 1 \\
                  --functional M06-2X --basis 6-311+G(d,p) --nproc 16

              # Load defaults from YAML, override solvent
              python generate_inputs.py --config ../config/calc_config.yaml \\
                  -f Na.xyz -c 1 --solvent Methanol
        """),
    )
    p.add_argument(
        "-f", "--files", nargs="+", required=True, metavar="XYZ",
        help="Input XYZ file(s). Glob patterns are supported.",
    )
    p.add_argument("-c", "--charge", type=int, required=True, help="Molecular charge")
    p.add_argument(
        "-m", "--mult", type=int, default=1, help="Spin multiplicity (default: 1)"
    )
    p.add_argument("--config", metavar="YAML", help="Path to calc_config.yaml")
    p.add_argument("--functional", default=None, help="DFT functional")
    p.add_argument("--basis", default=None, help="Basis set")
    p.add_argument("--nproc", type=int, default=None, help="Number of processors")
    p.add_argument("--mem", default=None, help="Memory (e.g. 16GB)")
    p.add_argument("--solvent", default=None, help="Gaussian solvent keyword")
    p.add_argument(
        "--model", default=None, choices=list(SUPPORTED_MODELS),
        help="Solvation model (default: SMD)",
    )
    p.add_argument(
        "-o", "--outdir", default="gaussian_inputs",
        help="Output directory (default: gaussian_inputs/)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.config:
        cfg = CalcConfig.from_yaml(args.config)
    else:
        cfg = CalcConfig()

    if args.functional:
        cfg.functional = args.functional
    if args.basis:
        cfg.basis_set = args.basis
    if args.nproc:
        cfg.nproc = args.nproc
    if args.mem:
        cfg.mem = args.mem
    if args.solvent:
        cfg.solvent = args.solvent
    if args.model:
        cfg.scrf_model = args.model

    xyz_files = [Path(f) for f in args.files]
    missing = [f for f in xyz_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"XYZ files not found: {missing}")

    outdir = Path(args.outdir)
    gen = GaussianInputGenerator(cfg)
    for xyz in xyz_files:
        ion_dir = outdir / _sanitize(xyz.stem)
        print(f"\n[{xyz.name}] → {ion_dir}/")
        gen.write_all(xyz, charge=args.charge, mult=args.mult, outdir=ion_dir)

    print(f"\nDone. {len(xyz_files)} ion(s) processed → {outdir}/")


if __name__ == "__main__":
    main()
