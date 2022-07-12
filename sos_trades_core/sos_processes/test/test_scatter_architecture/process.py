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

    # ontology information
    _ontology_data = {
        'label': 'Core Test Scatter Architecture',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):

        # add namespaces definition
        self.ee.ns_manager.add_ns_def({'ns_public': f'{self.ee.study_name}.Business',
                                       'ns_subsystem': f'{self.ee.study_name}.Business',
                                       'ns_ac': f'{self.ee.study_name}.Business',
                                       'ns_actors': self.ee.study_name})

        # actor, subsystem and AC_list scatter maps dict
        subsystem_map = {'input_name': 'subsystem_list',

                         'input_ns': 'ns_subsystem',
                         'output_name': 'subsystem',
                         'scatter_ns': 'ns_subsystem_scatter'}

        ac_list_map = {'input_name': 'AC_list',

                       'input_ns': 'ns_ac',
                       'output_name': 'AC_name',
                       'scatter_ns': 'ns_ac_scatter'}

        actors_list_map = {'input_name': 'actors_list',

                           'input_ns': 'ns_public',
                           'output_name': 'actor_name',
                           'scatter_ns': 'ns_actors',
                           'ns_to_update': ['ns_subsystem', 'ns_ac']}

        # add actor, subsystem and AC_list maps
        self.ee.smaps_manager.add_build_map(
            'subsystem_list_map', subsystem_map)
        self.ee.smaps_manager.add_build_map(
            'AC_list_map', ac_list_map)
        self.ee.smaps_manager.add_build_map(
            'actors_list', actors_list_map)

        architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Services', 'Services', None],
             'Current': ['OSS', 'FHS', 'Pool', 'TSP', 'Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline',
                      'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline'))],
             'Activation': [False, False, False, False, False]})

        builder_architecture = self.ee.factory.create_architecture_builder(
            'Business', architecture_df)

        scatter_builder = self.ee.factory.create_scatter_builder('Business',
                                                                 'actors_list', builder_architecture)

        return scatter_builder
