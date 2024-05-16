'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2023 Capgemini

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
# -- Generate test 2 process

import pandas as pd

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test Architecture Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Remy', 'Tomato_producer'],
             'Current': ['Remy', 'Tomato_producer', 'Opex', 'CAPEX', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline',
                      'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'product_list', 'ValueBlockDiscipline'),
                        ('scatter', 'product_list', 'ValueBlockDiscipline'),
                        ('scatter', 'product_list', 'ValueBlockDiscipline')],
             'Activation': [True, True, False, False, False], })

        builder = self.ee.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.ee.ns_manager.add_ns_def({'ns_vbdict': self.ee.study_name,
                                       'ns_public': self.ee.study_name,
                                       'ns_business': f'{self.ee.study_name}.Business',
                                       'ns_Remy': f'{self.ee.study_name}.Business.Remy',
                                       'ns_Tomato': f'{self.ee.study_name}.Business.Tomato_producer'})
        return builder
