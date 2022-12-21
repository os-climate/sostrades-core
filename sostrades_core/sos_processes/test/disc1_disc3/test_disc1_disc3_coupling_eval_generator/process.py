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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
#-- Generate test 1 process
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Test Disc1 Disc3 Coupling Eval Generator',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        # Select the nested subprocess
        repo = 'sostrades_core.sos_processes.test.disc1_disc3'
        sub_proc = 'test_disc1_disc3_coupling'
        coupling_builder = self.ee.factory.get_builder_from_process(
            repo=repo, mod_id=sub_proc)

        # driver builder
        eval_driver = self.ee.factory.create_driver(
            'Eval', coupling_builder
        )
        #======================================================================
        # # shift nested subprocess namespaces
        # coupling_name = 'D1_D3_Coupling'
        # self.ee.ns_manager.add_ns(
        #     'ns_disc3', f'{self.ee.study_name}.Eval.{coupling_name}')
        # self.ee.ns_manager.add_ns(
        #     'ns_out_disc3', f'{self.ee.study_name}.Eval.{coupling_name}')
        # self.ee.ns_manager.add_ns(
        #     'ns_ac', f'{self.ee.study_name}.Eval.{coupling_name}')
        # self.ee.ns_manager.add_ns(
        #     'ns_data_ac', f'{self.ee.study_name}.Eval.{coupling_name}')
        #======================================================================

        # driver namespaces
        self.ee.ns_manager.add_ns('ns_sampling', f'{self.ee.study_name}.Eval')
        self.ee.ns_manager.add_ns('ns_eval', f'{self.ee.study_name}.Eval')
        # sample generator builder
        mod_generator = 'sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper.SampleGeneratorWrapper'
        generator_builder = self.ee.factory.get_builder_from_module(
            'Sample_Generator', mod_generator)

        return eval_driver + [generator_builder]
