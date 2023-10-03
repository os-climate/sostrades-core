'''
Copyright 2022 Airbus SAS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate a doe scenario
"""
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_processes.test.test_sellar_grid_search',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        # add disciplines Sellar
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.sellar_status.'
        mods_dict = {'Sellar_Problem': disc_dir + 'SellarProblem',
                     'Sellar_2': disc_dir + 'Sellar2',
                     'Sellar_1': disc_dir + 'Sellar1'}
        builder_list = self.create_builder_list(mods_dict,
                                                ns_dict={'ns_OptimSellar': self.ee.study_name,
                                                         'ns_sampling': f'{self.ee.study_name}.Eval',
                                                         'ns_eval': f'{self.ee.study_name}.Eval'}
                                                )
        # evaluator builder
        eval_builder = self.ee.factory.create_mono_instance_driver(
            'Eval', builder_list)

        # sample generator builder
        mod_sg = 'sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper.SampleGeneratorWrapper'
        sg_builder = self.ee.factory.get_builder_from_module(
            'SampleGenerator', mod_sg)
        return eval_builder + [sg_builder]
