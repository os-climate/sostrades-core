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
# -- Generate test 1 process
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Test Disc1 Disc3 Coupling Mono Instance Eval with Sample Generator',
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
        eval_driver = self.ee.factory.create_mono_instance_driver(
            'Eval', coupling_builder
        )

        return eval_driver
