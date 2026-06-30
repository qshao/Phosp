from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD

out = Path(__file__).parent / "fixtures"
out.mkdir(exist_ok=True)

u = mda.Universe(PSF, DCD)
with mda.Writer(str(out / "mini_traj.xtc"), n_atoms=u.atoms.n_atoms) as W:
    for ts in u.trajectory[:5]:
        W.write(u.atoms)
u.atoms.write(str(out / "mini_traj.pdb"))
