def get_example_slab():
    # generate example structure
    from ase.build import slab
    atoms = slab("ZnO", (1, 1, 0), vacuum=10.0)
    
    return atoms


def cluster_mlopt(fragment, fmax, optimiser, opt_kwargs):
    '''
    Parameters:
        fragment: Fragment object 
            (py-ChemShell) object containing structural information
        fmax: float
            Convergence criterion in eV/A for max force on each atom
            Note: This will need interconverting between CS and ASE
        optimiser: Optimiser Class from ASE
        opt_kwargs: dict
            Keyword arguments to be passed to the ASE optimiser
    
    returns bool -> True if successful, False if not.


    Working theory:
    
    - Set up a Fragment object
    - Extract QM region as an Atoms object
    - CatLearn requests energy and forces
    - calculator is a wrapper which must return energy and forces obtained from the ChemShell calculation
    - wrapper calls the ChemShell/NWChem calculation including gradients
    - results including forces and the final energy must be provided
    - when catlearn makes a prediction the QM region (Atoms) is updated
    - Original Fragment is updated with new coordinates shift from Atoms
    '''



    return None


from ase.calculators.calculator import Calculator

class PyChemShellWrapperCalculator(Calculator):
    # Premise - work on QM region only

    def __init__(self, theory, fragment):

        self.theory = theory
        self.fragment = fragment

        return super().__init_subclass__()


    @property
    def theory_settings(self):
        '''The theory objects settings'''

        return self.theory.kwargs # figure out how to extract proeprties from Theory



    from chemsh.utils import atoms_from_frag, atoms_to_frag
    from chemsh.tasks.sp import SP
    
    # TODO: turn into setter function
    '''Select the QM region'''
    atoms = atoms_from_frag(self.fragment.SelectByRegion(1))
    
    
    job = SP(theory=self.theory)
    job.run(gradients=True)
    job.results
    
    pass
