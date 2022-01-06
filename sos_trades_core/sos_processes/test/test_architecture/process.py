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
#-- Generate test 2 process

import pandas as pd
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    def get_builders(self):

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_business',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.ee.smaps_manager.add_build_map('AC_list', mydict)

        vb_type_list = ['SumValueBlockDiscipline',
                        'ValueBlockDiscipline',
                        'ValueBlockDiscipline',
                        'ValueBlockDiscipline',
                        'ValueBlockDiscipline',
                        'ValueBlockDiscipline']
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Airbus', 'Boeing', 'Services'],
             'Current': ['Airbus', 'Boeing', 'AC_Sales', 'Services', 'AC_Sales', 'FHS'],
             'Type': vb_type_list,
             'Action': [('standard'), ('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline'), ('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline'), ('scatter', 'AC_list', 'ValueBlockDiscipline')],
             'Activation': [True, True, False, False, False, False]})

        builder = self.ee.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.ee.ns_manager.add_ns_def({'ns_vbdict': self.ee.study_name,
                                       'ns_public': self.ee.study_name,
                                       'ns_segment_services': self.ee.study_name,
                                       'ns_services': self.ee.study_name,
                                       'ns_services_ac': self.ee.study_name,
                                       'ns_seg': self.ee.study_name,
                                       'ns_ac': self.ee.study_name,
                                       'ns_coc': self.ee.study_name,
                                       'ns_data_ac': self.ee.study_name,
                                       'ns_business_ac': self.ee.study_name,
                                       'ns_rc': self.ee.study_name,
                                       'ns_business': f'{self.ee.study_name}.Business',
                                       'ns_Airbus': f'{self.ee.study_name}.Business.Airbus',
                                       'ns_Boeing': f'{self.ee.study_name}.Business.Boeing'})
        return builder
