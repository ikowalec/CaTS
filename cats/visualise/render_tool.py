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
    povray_settings["textures"] = ["ase2"] * len(atoms)
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

from ase.io import read
folder = r"<research-data-root>\2025\11.November\O2_chemisorbed_dissociation"

atoms = read(rf"{folder}\initial.traj")
render_povray(atoms, rf"{folder}\initial_rendered")