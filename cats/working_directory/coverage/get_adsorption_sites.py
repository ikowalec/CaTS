from acat.adsorption_sites import ClusterAdsorptionSites
import pandas as pd
from ase.io import read

atoms = read(r"<repo-root>\.data\Au63.xyz")
atoms.center()

# Identify all adsorption sites on the nanoalloy
cas = ClusterAdsorptionSites(atoms, composition_effect=True)

# Print information of each site
sites = cas.get_sites()
df = pd.DataFrame(sites)

print(df)