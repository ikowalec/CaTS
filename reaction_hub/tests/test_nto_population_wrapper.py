import inspect
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from jobs.submission import _build_wrapper_script
from jobs.tasks import _validate_analysis_request


class PopulationOnlyWrapperTests(SimpleTestCase):
    def test_reuses_nto_and_exits_before_electronic_and_density_work(self):
        clean = _validate_analysis_request('density_analysis', {
            'source_run_id':3440, 'ct_run_id':7636, 'excitation_numbers':[10],
            'nto_spin':'beta', 'nto_population_only':True,
        })
        self.assertTrue(clean['nto_population_only'])
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'/'density_analysis'
            source.mkdir(parents=True)
            (source/'state_010_nto.mwfn').write_text('retained NTO',encoding='utf-8')
            work = root/'work'
            work.mkdir()
            params = {name:(None if 'settings' in name or name in ('solvent','spin_multiplicity','max_al_steps_per_iter')
                           else False if name=='use_tddft_optimisation'
                           else 1 if name in ('n_iter','nodes','tasks_per_node','cpus_per_task','al_steps_per_iter')
                           else 0.1 if name=='fmax' else 'orca')
                      for name, param in inspect.signature(_build_wrapper_script).parameters.items()
                      if param.default is inspect.Parameter.empty}
            params.update(run_mode='charge_transfer',charge_transfer_config={
                'source_run_id':3440,'source_working_directory':str(source.parent),
                'density_analysis_only':True,'density_analysis':{
                    'nto_population_only':True,'excitation_numbers':[10],'hole_selector':'hb'},
            })
            script = _build_wrapper_script(**params)
            try:
                os.chdir(work)
                with patch('cats.multiwfn.orbital_composition.mulliken_orbital_composition',return_value={'method':'Mulliken'}) as analysis, \
                     patch('cats.multiwfn.density_analysis.generate_excitation_density_analysis',side_effect=AssertionError('Redundant density analysis')), \
                     patch('cats.multiwfn.tools.mulliken_population_routine',side_effect=AssertionError('Ground charges must not be changed')):
                    with self.assertRaises(SystemExit) as stopped:
                        exec(compile(script,'native_wrapper','exec'),{})
                    self.assertEqual(stopped.exception.code,0)
                self.assertEqual([c.kwargs['role'] for c in analysis.call_args_list],['hole','electron'])
                self.assertTrue(all(c.kwargs['spin']=='beta' for c in analysis.call_args_list))
                summary=json.loads(Path('reaction_hub_summary.json').read_text())
                self.assertEqual(summary['charge_transfer']['analysis_scope'],'nto_orbital_mulliken')
                self.assertFalse(Path('orca.molden.input').exists())
            finally:
                os.chdir(previous)
