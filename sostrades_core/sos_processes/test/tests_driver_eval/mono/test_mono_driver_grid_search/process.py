'''
Copyright 2024 Capgemini

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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test Mono Driver Eval Grid Search',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        mod1_path = 'sostrades_core.sos_wrapping.test_discs.disc1_grid.Disc1'
        grid_search = 'GridSearch'

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', mod1_path)

        self.ee.ns_manager.add_ns('ns_test', f'{self.ee.study_name}.Eval.Disc1')

        # evaluator builder
        eval_builder = self.ee.factory.create_mono_instance_driver(
            'Eval', [disc1_builder])

        return eval_builder
