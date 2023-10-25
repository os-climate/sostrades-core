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
# -- Generate test 1 process
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_processes.test.test_grid_search_mult',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        mod1_path = 'sostrades_core.sos_wrapping.test_discs.disc1_grid.Disc1'
        grid_search = 'GridSearch'

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', mod1_path)

        self.ee.ns_manager.add_ns(SampleGeneratorWrapper.NS_SAMPLING, f'{self.ee.study_name}.Eval')
        self.ee.ns_manager.add_ns(SampleGeneratorWrapper.NS_DRIVER, f'{self.ee.study_name}.Eval')
        self.ee.ns_manager.add_ns('ns_test', f'{self.ee.study_name}.Eval.Disc1')

        # sample generator builder
        mod_sg = 'sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper.SampleGeneratorWrapper'
        sg_builder = self.ee.factory.get_builder_from_module(
            'SampleGenerator', mod_sg)

        # multipliers builder
        mod_mp = 'sostrades_core.execution_engine.disciplines_wrappers.multipliers_wrapper.MultipliersWrapper'
        mp_builder = self.ee.factory.get_builder_from_module(
            'Multipliers', mod_mp)

        # evaluator builder
        eval_builder = self.ee.factory.create_mono_instance_driver(
            'Eval', [mp_builder, disc1_builder])

        return eval_builder + [sg_builder]