#!/usr/bin/env python3
"""visualize.py — Optional interactive visualization of molecular structures.

Supports two modes, auto-detected:
  - Jupyter notebook  : nglview interactive 3D widget (via RDKit or direct XYZ)
  - CLI / script      : self-contained HTML saved to disk (3Dmol.js via CDN)

All visualization dependencies (rdkit, nglview) are optional.
The module raises clear ImportError messages with install instructions
if a required package is missing.

Jupyter usage:
  from utils.visualize import show_xyz, show_cluster_grid
  show_xyz("Na_n4_conf0_cluster.xyz", charge=1)
  show_cluster_grid(["n1.xyz", "n2.xyz", "n3.xyz", "n4.xyz"])

CLI usage (saves HTML):
  python utils/visualize.py -f Na_n4_conf0_cluster.xyz --charge 1
  python utils/visualize.py -d test_clusters/Na_n4_conf0/ --open
"""

from __future__ import annotations

import argparse
import textwrap
import webbrowser
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _is_jupyter() -> bool:
    """Return True if running inside a Jupyter kernel."""
    try:
        shell = get_ipython().__class__.__name__   # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "google.colab._shell")
    except NameError:
        return False


# ---------------------------------------------------------------------------
# RDKit loader
# ---------------------------------------------------------------------------

def _load_rdkit_mol(xyz_path: str | Path, charge: int = 0):
    """Load an XYZ file as an RDKit Mol with bonds determined.

    Requires rdkit >= 2022.09 for MolFromXYZFile support.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        raise ImportError(
            "RDKit is required for this function.\n"
            "Install with:  conda install -c conda-forge rdkit\n"
            "           or: pip install rdkit"
        )

    mol = Chem.MolFromXYZFile(str(xyz_path))
    if mol is None:
        raise ValueError(f"RDKit could not parse {xyz_path}")

    # DetermineBonds adds connectivity based on interatomic distances.
    # This is needed for show_rdkit() to draw bonds correctly.
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception:
        # Clusters with unusual geometries may fail bond determination;
        # the viewer will still show atoms without bonds.
        pass

    return mol


# ---------------------------------------------------------------------------
# nglview widget  (Jupyter only)
# ---------------------------------------------------------------------------

def show_xyz(
    xyz_path: str | Path,
    charge: int = 0,
    use_rdkit: bool = True,
    representation: str = "ball+stick",
    sphere_scale: float = 0.3,
    html_output: Optional[str | Path] = None,
) -> object:
    """Visualize a single XYZ file.

    In Jupyter  : returns an nglview widget (display it with the last line of a cell).
    Outside     : saves an HTML file (html_output path, or <xyz_stem>.html by default)
                  and returns the HTML path.

    Parameters
    ----------
    xyz_path       : path to the XYZ file
    charge         : total molecular charge (used for RDKit bond determination)
    use_rdkit      : if True, load via RDKit for richer bond display; if False,
                     pass the XYZ directly to nglview (simpler, no bond info)
    representation : nglview representation name (Jupyter mode only)
    sphere_scale   : atom sphere scale for ball+stick (Jupyter mode only)
    html_output    : override path for HTML output (non-Jupyter mode)
    """
    xyz_path = Path(xyz_path)

    if _is_jupyter():
        return _show_nglview(xyz_path, charge, use_rdkit, representation, sphere_scale)
    else:
        out = Path(html_output) if html_output else xyz_path.with_suffix(".html")
        _save_html_3dmol([xyz_path], out, title=xyz_path.stem)
        print(f"Visualization saved → {out}")
        return out


def show_cluster_grid(
    xyz_paths: List[str | Path],
    charge: int = 0,
    labels: Optional[List[str]] = None,
    html_output: Optional[str | Path] = None,
) -> object:
    """Display multiple XYZ structures side by side.

    In Jupyter  : returns an ipywidgets HBox of nglview widgets.
    Outside     : saves a single HTML file with all structures in a grid.

    Parameters
    ----------
    xyz_paths  : list of XYZ file paths to display
    charge     : shared charge for RDKit bond determination
    labels     : optional titles shown above each viewer
    html_output: override output HTML path (non-Jupyter mode)
    """
    paths = [Path(p) for p in xyz_paths]
    lbls = labels or [p.stem for p in paths]

    if _is_jupyter():
        return _show_nglview_grid(paths, charge, lbls)
    else:
        out = (Path(html_output) if html_output
               else paths[0].parent / "cluster_grid.html")
        _save_html_3dmol(paths, out, title="Cluster grid", labels=lbls)
        print(f"Grid visualization saved → {out}")
        return out


# ---------------------------------------------------------------------------
# nglview backend  (Jupyter)
# ---------------------------------------------------------------------------

def _show_nglview(
    xyz_path: Path,
    charge: int,
    use_rdkit: bool,
    representation: str,
    sphere_scale: float,
):
    try:
        import nglview as nv
    except ImportError:
        raise ImportError(
            "nglview is required for Jupyter visualization.\n"
            "Install with:  pip install nglview\n"
            "               jupyter nbextension enable --py --sys-prefix nglview"
        )

    if use_rdkit:
        mol = _load_rdkit_mol(xyz_path, charge)
        view = nv.show_rdkit(mol)
    else:
        view = nv.show_file(str(xyz_path))

    view.clear_representations()
    if representation == "ball+stick":
        view.add_ball_and_stick(sphere_scale=sphere_scale)
    else:
        view.add_representation(representation)

    view.center()
    return view


def _show_nglview_grid(paths: List[Path], charge: int, labels: List[str]):
    try:
        import nglview as nv
        import ipywidgets as widgets
    except ImportError:
        raise ImportError(
            "nglview and ipywidgets are required for grid display.\n"
            "Install with:  pip install nglview ipywidgets"
        )

    views = []
    for path, lbl in zip(paths, labels):
        mol = _load_rdkit_mol(path, charge)
        v = nv.show_rdkit(mol)
        v.clear_representations()
        v.add_ball_and_stick(sphere_scale=0.3)
        v.center()
        views.append(widgets.VBox([widgets.Label(lbl), v]))

    return widgets.HBox(views)


# ---------------------------------------------------------------------------
# 3Dmol.js HTML backend  (CLI / browser)
# ---------------------------------------------------------------------------

_3DMOL_CDN = "https://3Dmol.org/build/3Dmol-min.js"

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <script src="{cdn}"></script>
  <style>
    body  {{ font-family: sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 10px; }}
    h2    {{ text-align: center; margin-bottom: 8px; }}
    .grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; }}
    .cell {{ text-align: center; }}
    .lbl  {{ font-size: 13px; margin-bottom: 4px; color: #a0c4ff; }}
    .view {{ width: {width}px; height: {height}px; position: relative; border-radius: 6px;
             border: 1px solid #444; background: #111; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="grid">
{cells}
  </div>
  <script>
{scripts}
  </script>
</body>
</html>
"""

_CELL_TEMPLATE = """\
    <div class="cell">
      <div class="lbl">{label}</div>
      <div class="view" id="{vid}"></div>
    </div>"""

_SCRIPT_TEMPLATE = """\
    (function() {{
      let v = $3Dmol.createViewer("{vid}", {{backgroundColor: "0x111111"}});
      v.addModel(`{xyz}`, "xyz");
      v.setStyle({{}}, {{stick: {{radius: 0.12}}, sphere: {{scale: 0.28}}}});
      v.zoomTo();
      v.render();
    }})();"""


def _save_html_3dmol(
    xyz_paths: List[Path],
    output: Path,
    title: str = "Molecular Viewer",
    labels: Optional[List[str]] = None,
    cell_width: int = 480,
    cell_height: int = 420,
) -> None:
    """Generate a self-contained HTML file with 3Dmol.js viewers."""
    labels = labels or [p.stem for p in xyz_paths]

    cells, scripts = [], []
    for i, (path, lbl) in enumerate(zip(xyz_paths, labels)):
        vid = f"v{i}"
        xyz_content = path.read_text().replace("`", "'")   # escape backticks
        cells.append(_CELL_TEMPLATE.format(label=lbl, vid=vid))
        scripts.append(_SCRIPT_TEMPLATE.format(vid=vid, xyz=xyz_content))

    html = _HTML_TEMPLATE.format(
        title=title,
        cdn=_3DMOL_CDN,
        width=cell_width,
        height=cell_height,
        cells="\n".join(cells),
        scripts="\n".join(scripts),
    )
    output.write_text(html)


# ---------------------------------------------------------------------------
# Convenience: visualize all conformers of a given (ion, n) in one HTML
# ---------------------------------------------------------------------------

def visualize_conformers(
    cluster_dir: str | Path,
    pattern: str = "*_cluster.xyz",
    charge: int = 0,
    output: Optional[str | Path] = None,
    open_browser: bool = False,
) -> Path:
    """Find all cluster XYZ files in a directory and render them to HTML.

    Parameters
    ----------
    cluster_dir  : directory to search (recursively)
    pattern      : glob pattern for XYZ files
    charge       : total charge for RDKit bond determination
    output       : output HTML path (default: cluster_dir/viewer.html)
    open_browser : if True, open the HTML in the default web browser
    """
    cluster_dir = Path(cluster_dir)
    xyz_files = sorted(cluster_dir.rglob(pattern))

    if not xyz_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {cluster_dir}"
        )

    out = Path(output) if output else cluster_dir / "viewer.html"

    if _is_jupyter():
        return show_cluster_grid(xyz_files, charge=charge)

    title = cluster_dir.name
    labels = [p.parent.name for p in xyz_files]
    _save_html_3dmol(xyz_files, out, title=title, labels=labels)

    print(f"Viewer saved → {out}  ({len(xyz_files)} structures)")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize XYZ molecular structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Single file → saves Na_n4_conf0_cluster.html
              python utils/visualize.py -f test_clusters/Na_n4_conf0/Na_n4_conf0_cluster.xyz

              # All conformers in a directory → saves viewer.html
              python utils/visualize.py -d test_clusters/ --open

              # Grid of specific files
              python utils/visualize.py -f n1.xyz n2.xyz n3.xyz n4.xyz --output grid.html
        """),
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-f", "--files", nargs="+", metavar="XYZ")
    g.add_argument("-d", "--directory", metavar="DIR")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--output", default=None, metavar="HTML")
    p.add_argument("--open", action="store_true", help="Open in default browser after saving")
    p.add_argument("--pattern", default="*_cluster.xyz",
                   help="Glob pattern when using -d (default: *_cluster.xyz)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.directory:
        visualize_conformers(
            args.directory,
            pattern=args.pattern,
            charge=args.charge,
            output=args.output,
            open_browser=args.open,
        )
    else:
        paths = [Path(f) for f in args.files]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Files not found: {missing}")

        if len(paths) == 1:
            out = show_xyz(paths[0], charge=args.charge,
                           html_output=args.output)
        else:
            out = show_cluster_grid(paths, charge=args.charge,
                                    html_output=args.output)

        if args.open and isinstance(out, Path):
            webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
