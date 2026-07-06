#!/usr/bin/env python3
"""Overlay pT78 (phosphorylated) vs WT (reference) stage4 CSVs.

Reads phospho/output/stage4/*.csv and wt_reference/output_reference/stage4/*.csv,
writes overlay PNGs and a summary_stats.json into comparison/.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
P_DIR = HERE / "phospho" / "output" / "stage4"
WT_DIR = HERE / "wt_reference" / "output_reference" / "stage4"
OUT_DIR = HERE / "comparison"
OUT_DIR.mkdir(exist_ok=True)


def plot_timeseries(fname, ycol, ylabel, title, out_name):
    p = pd.read_csv(P_DIR / fname)
    wt = pd.read_csv(WT_DIR / fname)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(p["time_ps"] / 1000, p[ycol], label="pT78 (phosphorylated)")
    ax.plot(wt["time_ps"] / 1000, wt[ycol], label="WT (reference)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)
    return p, wt


def plot_rmsf():
    p = pd.read_csv(P_DIR / "rmsf.csv")
    wt = pd.read_csv(WT_DIR / "rmsf.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p["resid"], p["rmsf_angstrom"], label="pT78 (phosphorylated)")
    ax.plot(wt["resid"], wt["rmsf_angstrom"], label="WT (reference)")
    ax.axvline(78, color="gray", linestyle="--", linewidth=1, label="resid 78")
    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (Angstrom)")
    ax.set_title("Per-residue RMSF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmsf.png", dpi=150)
    plt.close(fig)
    return p, wt


def tail_stats(df, col, frac=0.25):
    n = max(1, int(len(df) * frac))
    tail = df[col].iloc[-n:]
    return {"mean": float(tail.mean()), "std": float(tail.std())}


summary = {}

rmsd_p, rmsd_wt = plot_timeseries(
    "rmsd.csv", "rmsd_angstrom", "RMSD (Angstrom)", "C-alpha RMSD", "rmsd.png"
)
summary["rmsd_angstrom_last_quarter"] = {
    "pT78": tail_stats(rmsd_p, "rmsd_angstrom"),
    "WT": tail_stats(rmsd_wt, "rmsd_angstrom"),
}

rg_p, rg_wt = plot_timeseries(
    "radius_of_gyration.csv", "rg_angstrom", "Rg (Angstrom)", "Radius of Gyration", "rg.png"
)
summary["rg_angstrom_last_quarter"] = {
    "pT78": tail_stats(rg_p, "rg_angstrom"),
    "WT": tail_stats(rg_wt, "rg_angstrom"),
}

sasa_p, sasa_wt = plot_timeseries(
    "sasa.csv", "sasa_angstrom2", "SASA (Angstrom^2)", "Phosphosite (resid 78) SASA", "sasa_resid78.png"
)
summary["sasa_resid78_angstrom2_last_quarter"] = {
    "pT78": tail_stats(sasa_p, "sasa_angstrom2"),
    "WT": tail_stats(sasa_wt, "sasa_angstrom2"),
}

contacts_p, contacts_wt = plot_timeseries(
    "contacts.csv", "n_contacts", "# contacts", "Contact Count", "contacts.png"
)
summary["n_contacts_last_quarter"] = {
    "pT78": tail_stats(contacts_p, "n_contacts"),
    "WT": tail_stats(contacts_wt, "n_contacts"),
}

plot_rmsf()

(OUT_DIR / "summary_stats.json").write_text(json.dumps(summary, indent=2))
print(f"Wrote comparison plots and summary_stats.json to {OUT_DIR}")
for metric, vals in summary.items():
    print(f"  {metric}: pT78={vals['pT78']}  WT={vals['WT']}")
