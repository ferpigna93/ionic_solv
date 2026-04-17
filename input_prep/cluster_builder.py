#!/usr/bin/env python3
"""cluster_builder.py — Build explicit ion–water clusters for the cluster-continuum approach.

The cluster-continuum thermodynamic cycle (Pliego & Riveros 2001, Kelly et al. 2006)
improves absolute solvation free energies for ions by wrapping them with n explicit
water molecules before applying a continuum solvation model:

  Ion(g) + n·H₂O(g) → [Ion·(H₂O)_n](g)          ΔG°_bind   (DFT/gas)
  [Ion·(H₂O)_n](g)  → [Ion·(H₂O)_n](aq)          ΔG°_cont   (DFT/SMD)
  n·H₂O(l)          → n·H₂O(g)                    n·ΔG°_vap  (+6.32 kcal/mol each)

  ΔG°_solv(ion) = ΔG°_bind + ΔG°_cont − n·ΔG°_vap + ΔG°_ss

This module:
  1. Reads an ion XYZ file.
  2. Distributes n water molecules on a sphere around it (random or uniform).
  3. Writes cluster XYZ files + corresponding Gaussian .com files.

Usage:
  python cluster_builder.py -f Na.xyz -c 1 -n 1 2 3 4 --nconformers 5
  python cluster_builder.py -f Cl.xyz -c -1 -n 4 6 --min_dist 2.0 --max_dist 3.5
"""

from __future__ import annotations

import argparse
import math
import random
import textwrap
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from input_prep.generate_inputs import CalcConfig, GaussianInputGenerator, _sanitize

# ---------------------------------------------------------------------------
# Water molecule geometry (experimental O–H bond = 0.9572 Å, HOH = 104.52°)
# ---------------------------------------------------------------------------

_WATER_O = np.array([0.0, 0.0, 0.0])
_WATER_H1 = np.array([0.9572 * math.sin(math.radians(52.26)), 0.0,  0.9572 * math.cos(math.radians(52.26))])
_WATER_H2 = np.array([-0.9572 * math.sin(math.radians(52.26)), 0.0, 0.9572 * math.cos(math.radians(52.26))])

WATER_ATOMS: List[Tuple[str, np.ndarray]] = [
    ("O", _WATER_O),
    ("H", _WATER_H1),
    ("H", _WATER_H2),
]

# Default ion–O distances (Å) for initial placement
# These are rough first-shell radii; Gaussian will optimize the geometry.
DEFAULT_ION_O_DIST: dict[str, float] = {
    "Li": 1.95, "Na": 2.35, "K": 2.80, "Rb": 2.95, "Cs": 3.15,
    "Mg": 2.05, "Ca": 2.45, "Sr": 2.65, "Ba": 2.85,
    "F":  2.65, "Cl": 3.10, "Br": 3.30, "I":  3.55,
    "default": 2.50,
}


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _random_unit_vectors(n: int, seed: int | None = None) -> np.ndarray:
    """Return n random unit vectors uniformly distributed on the unit sphere."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _uniform_unit_vectors(n: int) -> np.ndarray:
    """Approximately uniform placement on the unit sphere via Fibonacci lattice."""
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    golden = (1 + math.sqrt(5)) / 2
    indices = np.arange(n)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n)
    phi = 2 * math.pi * indices / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _random_rotation() -> np.ndarray:
    """Random 3×3 rotation matrix (uniform over SO(3))."""
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


def _place_water(
    ion_pos: np.ndarray,
    direction: np.ndarray,
    ion_o_dist: float,
) -> List[Tuple[str, float, float, float]]:
    """Place one water molecule with O pointing toward the ion.

    The O is at distance `ion_o_dist` from the ion along `direction`.
    The molecule is randomly rotated around the O–ion axis.
    """
    o_pos = ion_pos + direction * ion_o_dist

    # Orient water so the dipole points toward the ion (O lone pairs toward ion)
    rot = _random_rotation()
    water_coords = []
    for sym, local in WATER_ATOMS:
        world = o_pos + rot @ local
        water_coords.append((sym, float(world[0]), float(world[1]), float(world[2])))
    return water_coords


def build_cluster(
    ion_coords: List[Tuple[str, float, float, float]],
    n_water: int,
    ion_o_dist: float | None = None,
    uniform: bool = False,
    seed: int | None = None,
) -> List[Tuple[str, float, float, float]]:
    """Build a cluster of the ion + n water molecules.

    Parameters
    ----------
    ion_coords : coordinates of the bare ion (list of (sym, x, y, z))
    n_water    : number of water molecules to place
    ion_o_dist : ion–O distance in Å (auto-detected from ion symbol if None)
    uniform    : use Fibonacci-lattice placement instead of random
    seed       : random seed for reproducibility

    Returns
    -------
    Full cluster coordinates as list of (symbol, x, y, z).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Use centroid of ion as the reference point
    ion_center = np.mean(
        [[x, y, z] for _, x, y, z in ion_coords], axis=0
    )

    # Auto-detect ion–O distance from first atom symbol
    if ion_o_dist is None:
        first_sym = ion_coords[0][0]
        ion_o_dist = DEFAULT_ION_O_DIST.get(first_sym, DEFAULT_ION_O_DIST["default"])

    # Place water molecules
    if uniform:
        directions = _uniform_unit_vectors(n_water)
    else:
        directions = _random_unit_vectors(n_water, seed=seed)

    cluster = list(ion_coords)
    for direction in directions:
        water = _place_water(ion_center, direction, ion_o_dist)
        cluster.extend(water)

    return cluster


# ---------------------------------------------------------------------------
# XYZ writer
# ---------------------------------------------------------------------------

def write_xyz(
    coords: List[Tuple[str, float, float, float]],
    path: Path,
    comment: str = "",
) -> None:
    lines = [str(len(coords)), comment]
    for sym, x, y, z in coords:
        lines.append(f"{sym:<3s}  {x:>14.8f}  {y:>14.8f}  {z:>14.8f}")
    path.write_text("\n".join(lines) + "\n")


def read_xyz(path: Path) -> List[Tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        coords.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return coords


# ---------------------------------------------------------------------------
# Batch cluster generation
# ---------------------------------------------------------------------------

def generate_clusters(
    ion_xyz: Path,
    charge: int,
    mult: int,
    n_water_list: List[int],
    n_conformers: int,
    config: CalcConfig,
    outdir: Path,
    uniform: bool = False,
    ion_o_dist: float | None = None,
) -> None:
    """Generate XYZ + Gaussian .com files for each (n_water, conformer) pair.

    Directory layout:
      outdir/
        ion_stem_n1_conf0/
          ion_stem_n1_conf0_gas.com
          ion_stem_n1_conf0_solv.com
          ion_stem_n1_conf0_cluster.xyz
        ion_stem_n1_conf1/
          ...
    """
    ion_coords = read_xyz(ion_xyz)
    stem = _sanitize(ion_xyz.stem)
    gen = GaussianInputGenerator(config)

    for n in n_water_list:
        for conf in range(n_conformers):
            label = f"{stem}_n{n}_conf{conf}"
            job_dir = outdir / label
            job_dir.mkdir(parents=True, exist_ok=True)

            cluster = build_cluster(
                ion_coords,
                n_water=n,
                ion_o_dist=ion_o_dist,
                uniform=uniform and conf == 0,   # first conformer is uniform
                seed=conf,
            )

            # Write cluster XYZ
            xyz_out = job_dir / f"{label}_cluster.xyz"
            write_xyz(
                cluster,
                xyz_out,
                comment=f"{stem} + {n} H2O — conformer {conf}",
            )

            # Generate Gaussian inputs
            gen.write_all(
                xyz_path=xyz_out,
                charge=charge,
                mult=mult,
                outdir=job_dir,
                name=label,
            )

            print(f"  Cluster n={n}, conf={conf}: {job_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ion–water clusters for the cluster-continuum approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python cluster_builder.py -f Na.xyz -c 1 -n 1 2 3 4
              python cluster_builder.py -f Cl.xyz -c -1 -n 4 6 --nconformers 10 --uniform
        """),
    )
    p.add_argument("-f", "--file", required=True, metavar="XYZ", help="Bare ion XYZ file")
    p.add_argument("-c", "--charge", type=int, required=True, help="Ion charge")
    p.add_argument("-m", "--mult", type=int, default=1, help="Spin multiplicity")
    p.add_argument(
        "-n", "--n-water", nargs="+", type=int, default=[1, 2, 3, 4],
        metavar="N", help="Number(s) of water molecules",
    )
    p.add_argument(
        "--nconformers", type=int, default=5, metavar="K",
        help="Random conformers per n (default: 5)",
    )
    p.add_argument("--uniform", action="store_true", help="Use uniform (Fibonacci) placement for first conformer")
    p.add_argument("--ion-o-dist", type=float, default=None, metavar="Å", help="Ion–O distance override (Å)")
    p.add_argument("--config", metavar="YAML", help="Path to calc_config.yaml")
    p.add_argument("--functional", default=None)
    p.add_argument("--basis", default=None)
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--mem", default=None)
    p.add_argument("--solvent", default=None)
    p.add_argument("-o", "--outdir", default="cluster_inputs", help="Output directory")
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

    ion_xyz = Path(args.file)
    if not ion_xyz.exists():
        raise FileNotFoundError(ion_xyz)

    outdir = Path(args.outdir)
    print(f"Generating clusters for {ion_xyz.name}: n_water={args.n_water}, conformers={args.nconformers}")
    generate_clusters(
        ion_xyz=ion_xyz,
        charge=args.charge,
        mult=args.mult,
        n_water_list=args.n_water,
        n_conformers=args.nconformers,
        config=cfg,
        outdir=outdir,
        uniform=args.uniform,
        ion_o_dist=args.ion_o_dist,
    )
    print(f"\nDone → {outdir}/")


if __name__ == "__main__":
    main()
