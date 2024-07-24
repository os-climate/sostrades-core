'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/13-2024/05/16 Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.proxy_sample_generator import ProxySampleGenerator
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate a doe scenario
"""


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test Sample Generator',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''

        builder = self.ee.factory.create_sample_generator('SampleGenerator')

        self.ee.ns_manager.add_ns_def(
            {ProxySampleGenerator.NS_SAMPLING: f'{self.ee.study_name}.SampleGenerator'},
        )

        return builder
