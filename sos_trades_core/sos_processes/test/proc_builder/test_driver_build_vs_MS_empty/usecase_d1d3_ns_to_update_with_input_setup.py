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
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd


class Study(StudyManager):
    '''This is an example of usecase study for
     the test_disc1_disc3_vs_MS_from_proc process.
    This process instantiates the multiscenario of (disc1_scenario, disc3_scenario).
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        # provide a process (with disciplines) to the driver
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = mod_id
        sub_process_inputs_dict['usecase_name'] = 'Empty'
        sub_process_inputs_dict['usecase_data'] = {}

        # provide an associated scenario_map to the driver
        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = ['ns_data_ac', 'ns_ac', 'ns_disc3', 'ns_out_disc3']
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        ######### Numerical values   ####
        x1 = 2.
        x2 = 4.
        a1 = 3.
        b1 = 4.
        a2 = 6.
        b2 = 2.
        scenario_list = ['scenario_1', 'scenario_2']
        ######### Fill the dictionary for dm   ####
        dict_values = {}

        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
        dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list

        dict_values[f'{self.study_name}.vs_MS.scenario_1.a'] = a1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.x'] = x1

        dict_values[f'{self.study_name}.vs_MS.scenario_2.a'] = a2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.x'] = x2

        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.constant'] = 3
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.power'] = 1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.z'] = 1.2

        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.constant'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.z'] = 1.2

        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.ee.display_treeview_nodes()
    uc_cls.ee.display_treeview_nodes(True)
    uc_cls.run(for_test=True)
