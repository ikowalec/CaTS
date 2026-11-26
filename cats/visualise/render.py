def render_povray(atoms, filename):

    from ase.io.pov import write_pov, get_bondpairs
    from ase.data import covalent_radii
    import numpy as np

    # Set atomic radii for rendering
    covalent_radii = covalent_radii
    radii_scale = 0.65  # scaling factor for ball and stick model
    plotting_var_settings = {}
    plotting_var_settings["radii"] = (
        np.array([covalent_radii[z] for z in atoms.numbers]) * radii_scale
    )

    # Set custom bonds based on modified radius and other settings
    bond_radius_scaling_factor = 0.9
    povray_settings = {}
    povray_settings["textures"] = ["intermediate"] * len(atoms)
    povray_settings["bondatoms"] = get_bondpairs(
        atoms, radius=bond_radius_scaling_factor
    )
    povray_settings["canvas_width"] = 1080
    povray_settings["display"] = False

    renderer = write_pov(
        filename=f"{filename}.pov",
        atoms=atoms,
        povray_settings=povray_settings,
        **plotting_var_settings,
    )
    # Point to your executable here
    renderer.render(povray_executable=r"C:\Program Files\POV-Ray\v3.7\bin\pvengine.exe")


import os

os.chdir(r"<desktop-data-root>\PdZn_formates\110")
from ase.io import read

filename = "HCOOm_H"
atoms = read(f"{filename}_PdZn.traj")

for atom in atoms:
    atom.x -= 5
    atom.y += 1

atoms.wrap()
import numpy as np

minx = np.amin([atom.x for atom in atoms if atom.symbol == "Pd"]).copy()
miny = np.amin([atom.y for atom in atoms if atom.symbol == "Pd"]).copy()

for atom in atoms:
    atom.x -= minx
    atom.y -= miny

atoms_list = [atoms.copy()]

atoms.rotate((1, 0, 0), -90, rotate_cell=True)

atoms_list.append(atoms.copy())

from ase.visualize import view

# view(atoms_list)


for i, atoms in enumerate(atoms_list):
    render_povray(atoms, f"{filename}_{i}")

toluene = Atoms()

[
    Atom("C", position=(1.18536, -0.18846, 0.07428)),
    Atom("C", position=(0.96657, -1.56679, 0.06122)),
    Atom("C", position=(-0.33392, -2.07005, -0.00928)),
    Atom("C", position=(-1.41067, -1.18290, -0.06403)),
    Atom("C", position=(-1.18546, 0.19439, -0.05069)),
    Atom("C", position=(0.11464, 0.71428, 0.01523)),
    Atom("C", position=(0.35558, 2.20037, -0.00365)),
    Atom("H", position=(2.20612, 0.19715, 0.13400)),
    Atom("H", position=(1.81553, -2.25105, 0.11067)),
    Atom("H", position=(-0.50749, -3.14719, -0.01627)),
    Atom("H", position=(-2.43199, -1.56490, -0.11286)),
    Atom("H", position=(-2.03397, 0.88197, -0.08931)),
    Atom("H", position=(0.47043, 2.56802, -1.03615)),
    Atom("H", position=(1.27246, 2.46591, 0.54053)),
    Atom("H", position=(-0.48322, 2.74924, 0.44627)),
]
