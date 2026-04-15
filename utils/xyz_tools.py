#!/usr/bin/env python3
"""xyz_tools.py — Utilities for reading, writing and manipulating XYZ files.

Provides:
  - read_xyz / write_xyz
  - centroid, translate, rotate
  - create_monatomic_xyz  (for bare ions)
  - extract_geometry_from_gaussian  (pull optimised coords from a .log file)
  - rmsd  (geometry comparison)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# (symbol, x, y, z) in Å
Coords = List[Tuple[str, float, float, float]]


# ---------------------------------------------------------------------------
# Read / Write
# ---------------------------------------------------------------------------

def read_xyz(path: str | Path) -> Tuple[Coords, str]:
    """Read a standard XYZ file.

    Returns
    -------
    coords  : list of (symbol, x, y, z)
    comment : second line of the file
    """
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    coords: Coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) >= 4:
            coords.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    if len(coords) != n:
        raise ValueError(f"Expected {n} atoms, got {len(coords)} in {path}")
    return coords, comment


def write_xyz(
    coords: Coords,
    path: str | Path,
    comment: str = "",
) -> None:
    """Write coordinates to a standard XYZ file."""
    lines = [str(len(coords)), comment]
    for sym, x, y, z in coords:
        lines.append(f"{sym:<3s}  {x:>14.8f}  {y:>14.8f}  {z:>14.8f}")
    Path(path).write_text("\n".join(lines) + "\n")


def create_monatomic_xyz(
    symbol: str,
    path: str | Path,
    comment: str = "",
) -> Path:
    """Create a single-atom XYZ file (useful for bare ions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz([(symbol, 0.0, 0.0, 0.0)], path, comment=comment or f"{symbol} ion")
    return path


# ---------------------------------------------------------------------------
# Geometry operations
# ---------------------------------------------------------------------------

def centroid(coords: Coords) -> np.ndarray:
    """Return the geometric centroid (unweighted) of the coordinates."""
    pts = np.array([[x, y, z] for _, x, y, z in coords])
    return pts.mean(axis=0)


def translate(coords: Coords, vector: np.ndarray) -> Coords:
    """Translate all atoms by a displacement vector."""
    return [(s, x + vector[0], y + vector[1], z + vector[2]) for s, x, y, z in coords]


def center(coords: Coords) -> Coords:
    """Move the centroid of the molecule to the origin."""
    c = centroid(coords)
    return translate(coords, -c)


def rotate(coords: Coords, R: np.ndarray) -> Coords:
    """Apply a 3×3 rotation matrix R to all coordinates."""
    result = []
    for sym, x, y, z in coords:
        v = R @ np.array([x, y, z])
        result.append((sym, float(v[0]), float(v[1]), float(v[2])))
    return result


def rmsd(coords1: Coords, coords2: Coords) -> float:
    """Root-mean-square deviation between two geometries (Å).

    Assumes the atoms are already aligned (same ordering, same origin).
    """
    if len(coords1) != len(coords2):
        raise ValueError("RMSD requires equal numbers of atoms")
    pts1 = np.array([[x, y, z] for _, x, y, z in coords1])
    pts2 = np.array([[x, y, z] for _, x, y, z in coords2])
    return float(np.sqrt(np.mean(np.sum((pts1 - pts2) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Extract optimised geometry from Gaussian output
# ---------------------------------------------------------------------------

_ORIENTATION_RE = re.compile(
    r"Standard orientation:.*?-{5,}\n(.*?)-{5,}", re.DOTALL
)
_ATOM_LINE_RE = re.compile(
    r"^\s+\d+\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)

# Atomic number → symbol mapping (first 36 elements cover most chemistry)
_ATOMIC_NUM = {
    1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",
    8: "O",  9: "F",  10: "Ne",11: "Na",12: "Mg",13: "Al",14: "Si",
    15: "P", 16: "S", 17: "Cl",18: "Ar",19: "K", 20: "Ca",
    26: "Fe",27: "Co",28: "Ni",29: "Cu",30: "Zn",
    35: "Br",36: "Kr",53: "I", 56: "Ba",
}


def extract_geometry_from_gaussian(log_path: str | Path) -> Optional[Coords]:
    """Extract the last optimised geometry from a Gaussian output file.

    Reads the final "Standard orientation" block printed after convergence.
    Returns None if no geometry block is found (e.g. SP calculation).
    """
    text = Path(log_path).read_text(errors="replace")
    matches = list(_ORIENTATION_RE.finditer(text))
    if not matches:
        return None

    # Last block = final optimised geometry
    block = matches[-1].group(1)
    coords: Coords = []
    for line in block.splitlines():
        m = _ATOM_LINE_RE.match(line)
        if m:
            atomic_num = int(m.group(1))
            x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
            sym = _ATOMIC_NUM.get(atomic_num, f"X{atomic_num}")
            coords.append((sym, x, y, z))
    return coords or None


def gaussian_to_xyz(log_path: str | Path, out_path: str | Path) -> Optional[Path]:
    """Pull optimised geometry from a Gaussian .log and save as XYZ.

    Returns the path to the created XYZ file, or None on failure.
    """
    log_path = Path(log_path)
    coords = extract_geometry_from_gaussian(log_path)
    if coords is None:
        return None
    out_path = Path(out_path)
    write_xyz(coords, out_path, comment=f"Optimised geometry from {log_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Convenience: batch extraction from a directory of log files
# ---------------------------------------------------------------------------

def batch_extract_geometries(
    log_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "*.log",
) -> List[Path]:
    """Extract final geometries from all Gaussian logs in a directory."""
    log_dir = Path(log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for log in sorted(log_dir.rglob(pattern)):
        out = out_dir / (log.stem + ".xyz")
        result = gaussian_to_xyz(log, out)
        if result:
            written.append(result)
            print(f"  {log.name} → {out.name}")
        else:
            print(f"  {log.name}: no geometry found (skipped)")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="XYZ tools for ionic_solv")
    sub = p.add_subparsers(dest="cmd", required=True)

    # create
    c = sub.add_parser("create", help="Create a monatomic ion XYZ file")
    c.add_argument("symbol", help="Atomic symbol, e.g. Na")
    c.add_argument("-o", "--output", required=True, help="Output .xyz path")

    # extract
    e = sub.add_parser("extract", help="Extract optimised geometry from Gaussian log")
    e.add_argument("-f", "--file", required=True, help="Gaussian .log file")
    e.add_argument("-o", "--output", required=True, help="Output .xyz path")

    # batch-extract
    b = sub.add_parser("batch-extract", help="Extract geometries from all logs in a directory")
    b.add_argument("-d", "--directory", required=True)
    b.add_argument("-o", "--outdir", required=True)

    args = p.parse_args()

    if args.cmd == "create":
        out = create_monatomic_xyz(args.symbol, args.output)
        print(f"Created: {out}")

    elif args.cmd == "extract":
        result = gaussian_to_xyz(args.file, args.output)
        if result:
            print(f"Written: {result}")
        else:
            print("No optimised geometry found in the log file.")

    elif args.cmd == "batch-extract":
        paths = batch_extract_geometries(args.directory, args.outdir)
        print(f"\n{len(paths)} geometries extracted → {args.outdir}/")
