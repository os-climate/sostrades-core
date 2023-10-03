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
        'label': 'Core Test Sellar Coupling Eval Generator',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''

        # Select the nested subprocess
        cls_list = self.ee.factory.get_builder_from_process(repo='sostrades_core.sos_processes.test',
                                                            mod_id='test_disc1_disc2_coupling')

        uq_name = 'UncertaintyQuantification'

        # driver namespaces
        self.ee.ns_manager.add_ns('ns_eval', f'{self.ee.study_name}.Eval')

        eval_driver = self.ee.factory.create_mono_instance_driver(
            'Eval', cls_list, with_sample_generator=True)

        uq_builder = self.ee.factory.add_uq_builder(uq_name)

        return eval_driver + [uq_builder]
