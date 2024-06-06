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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test Sellar subprocess Mono Instance Eval with Sample Generator',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        # Select the nested subprocess
        repo = 'sostrades_core.sos_processes.test.sellar'
        sub_proc = 'test_sellar_list'
        coupling_builder = self.ee.factory.get_builder_from_process(
            repo=repo, mod_id=sub_proc)

        # driver builder
        eval_driver = self.ee.factory.create_mono_instance_driver(
            'Eval', coupling_builder)

        # shift nested subprocess namespaces
        coupling_name = 'subprocess'
        self.ee.ns_manager.add_ns(
            'ns_OptimSellar', f'{self.ee.study_name}.{coupling_name}')

        return eval_driver
