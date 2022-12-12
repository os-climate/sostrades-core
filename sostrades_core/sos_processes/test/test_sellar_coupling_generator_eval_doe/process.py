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
        'label': 'Core Test Sellar Coupling Sample Generator doe',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        # simple 2-disc process NOT USING nested scatters

        # add disciplines Sellar
        if 1 == 1:
            repo = 'sostrades_core.sos_processes.test'
            sub_proc = 'test_sellar_coupling'
            coupling_builder = self.ee.factory.get_builder_from_process(
                repo=repo, mod_id=sub_proc)
        else:
            disc_dir = 'sostrades_core.sos_wrapping.test_discs.sellar.'
            mods_dict = {
                'Sellar_Problem': disc_dir + 'SellarProblem',
                'Sellar_2': disc_dir + 'Sellar2',
                'Sellar_1': disc_dir + 'Sellar1',
            }
            builder_list_sellar = self.create_builder_list(
                mods_dict,
                # ns_dict={'ns_OptimSellar': self.ee.study_name}
            )

            coupling_builder = self.ee.factory.create_builder_coupling(
                "SellarCoupling")
            coupling_builder.set_builder_info(
                'cls_builder', builder_list_sellar)

        # shared namespace
        # self.ee.ns_manager.add_ns(
        #    'ns_scatter_scenario', f'{self.ee.study_name}')
        # self.ee.ns_manager.add_ns(
        #    'ns_OptimSellar', f'{self.ee.study_name}.SellarCoupling')
        #self.ee.ns_manager.add_ns('ns_sampling', f'{self.ee.study_name}')
        #self.ee.ns_manager.add_ns('ns_eval', f'{self.ee.study_name}')

        # if Mono_instance
        self.ee.ns_manager.add_ns(
            'ns_OptimSellar', f'{self.ee.study_name}.Eval.SellarCoupling')
        # if Multi_instance
        # self.ee.ns_manager.add_ns(
        #    'ns_OptimSellar', f'{self.ee.study_name}.SellarCoupling')

        self.ee.ns_manager.add_ns('ns_sampling', f'{self.ee.study_name}.Eval')
        self.ee.ns_manager.add_ns('ns_eval', f'{self.ee.study_name}.Eval')

        # multi scenario driver builder
        multi_scenarios = self.ee.factory.create_driver(
            'Eval', coupling_builder, flatten_subprocess=True)

        # sample generator builder
        mod_cp = 'sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper.SampleGeneratorWrapper'
        cp_builder = self.ee.factory.get_builder_from_module(
            'SampleGenerator', mod_cp)

        return multi_scenarios + [cp_builder]
