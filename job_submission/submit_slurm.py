#!/usr/bin/env python3
"""submit_slurm.py — Generate and optionally submit SLURM scripts for Gaussian 16 jobs.

Discovers all .com files in a directory tree, renders a SLURM script for each,
and (optionally) submits them to the queue with `sbatch`.

Usage:
  # Generate scripts only (dry run):
  python submit_slurm.py -d gaussian_inputs/ --nproc 16 --mem 32GB

  # Generate and submit immediately:
  python submit_slurm.py -d gaussian_inputs/ --nproc 8 --mem 16GB --submit

  # Submit a specific file:
  python submit_slurm.py -f jobs/Na/Na_gas.com --submit

  # Load resources from config:
  python submit_slurm.py -d gaussian_inputs/ --config config/calc_config.yaml --submit
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import yaml

TEMPLATE_PATH = Path(__file__).parent / "templates" / "slurm_g16.sh"


# ---------------------------------------------------------------------------
# SLURM script renderer
# ---------------------------------------------------------------------------

def render_slurm_script(
    com_path: Path,
    nproc: int = 8,
    mem: str = "16GB",
    time: str = "24:00:00",
    partition: str = "normal",
    account: str = "",
) -> str:
    """Fill in the SLURM template for a given .com file."""
    template = TEMPLATE_PATH.read_text()

    job_name = com_path.stem          # e.g. Na_gas
    replacements = {
        "__JOBNAME__": job_name,
        "__NPROC__": str(nproc),
        "__MEM__": mem,
        "__TIME__": time,
        "__PARTITION__": partition,
        "__ACCOUNT__": account,
        "__INPUT__": com_path.name,   # relative; script runs in the same dir
    }
    for k, v in replacements.items():
        template = template.replace(k, v)

    return template


def write_slurm_script(
    com_path: Path,
    nproc: int,
    mem: str,
    time: str,
    partition: str,
    account: str,
) -> Path:
    """Write the SLURM .sh file next to the .com file and return its path."""
    content = render_slurm_script(
        com_path, nproc=nproc, mem=mem, time=time,
        partition=partition, account=account,
    )
    sh_path = com_path.with_suffix(".sh")
    sh_path.write_text(content)
    sh_path.chmod(sh_path.stat().st_mode | 0o111)   # make executable
    return sh_path


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_job(sh_path: Path, dry_run: bool = False) -> Optional[str]:
    """Run `sbatch <sh_path>` from the directory containing the script.

    Returns the SLURM job-ID string on success, or None on dry-run.
    """
    if dry_run:
        print(f"  [dry-run] sbatch {sh_path}")
        return None

    result = subprocess.run(
        ["sbatch", sh_path.name],
        cwd=sh_path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] sbatch failed for {sh_path.name}:\n{result.stderr.strip()}", file=sys.stderr)
        return None

    # sbatch output: "Submitted batch job 123456"
    job_id = result.stdout.strip().split()[-1]
    print(f"  [submitted] {sh_path.name} → job {job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_directory(
    directory: Path,
    nproc: int,
    mem: str,
    time: str,
    partition: str,
    account: str,
    submit: bool,
) -> List[str]:
    """Find all .com files recursively and generate/submit SLURM scripts."""
    com_files = sorted(directory.rglob("*.com"))
    if not com_files:
        print(f"No .com files found in {directory}")
        return []

    job_ids = []
    for com in com_files:
        sh = write_slurm_script(com, nproc, mem, time, partition, account)
        print(f"Script: {sh}")
        if submit:
            jid = submit_job(sh, dry_run=False)
            if jid:
                job_ids.append(jid)

    print(f"\nTotal: {len(com_files)} scripts generated, {len(job_ids)} submitted.")
    return job_ids


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config_resources(yaml_path: Path) -> dict:
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    r = data.get("resources", {})
    return {
        "nproc": r.get("nproc", 8),
        "mem": r.get("mem", "16GB"),
        "time": r.get("time", "24:00:00"),
        "partition": r.get("partition", "normal"),
        "account": r.get("account", ""),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SLURM submission scripts for Gaussian 16 .com files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python submit_slurm.py -d gaussian_inputs/ --nproc 16 --mem 32GB
              python submit_slurm.py -d gaussian_inputs/ --config config/calc_config.yaml --submit
              python submit_slurm.py -f jobs/Na/Na_gas.com --submit
        """),
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("-d", "--directory", metavar="DIR", help="Directory containing .com files")
    target.add_argument("-f", "--file", metavar="COM", help="Single .com file")

    p.add_argument("--config", metavar="YAML", help="Load resources from calc_config.yaml")
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--mem", default=None)
    p.add_argument("--time", default=None, metavar="HH:MM:SS")
    p.add_argument("--partition", default=None)
    p.add_argument("--account", default=None)
    p.add_argument("--submit", action="store_true", help="Actually run sbatch (default: write scripts only)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Defaults
    resources = {"nproc": 8, "mem": "16GB", "time": "24:00:00", "partition": "normal", "account": ""}

    if args.config:
        resources.update(load_config_resources(Path(args.config)))

    # CLI overrides
    for k in ("nproc", "mem", "time", "partition", "account"):
        v = getattr(args, k, None)
        if v is not None:
            resources[k] = v

    if args.directory:
        process_directory(
            Path(args.directory),
            submit=args.submit,
            **resources,
        )
    else:
        com = Path(args.file)
        sh = write_slurm_script(com, **resources)
        print(f"Script: {sh}")
        if args.submit:
            submit_job(sh)


if __name__ == "__main__":
    main()
