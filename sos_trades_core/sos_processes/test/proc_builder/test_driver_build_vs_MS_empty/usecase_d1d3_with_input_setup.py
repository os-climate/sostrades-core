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
from sos_trades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
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

        process_builder_parameter_type = ProcessBuilderParameterType(mod_id, repo, 'Empty')

        # provide an associated scenario_map to the driver
        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = ['ns_ac', 'ns_disc3', 'ns_out_disc3']
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        ######### Numerical values   ####
        x = 2.0
        a = 3.0
        b1 = 4.0
        b2 = 2.0
        scenario_list = ['scenario_1', 'scenario_2']
        ######### Fill the dictionary for dm   ####
        dict_values = {}

        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list

        dict_values[f'{self.study_name}.vs_MS.a'] = a
        dict_values[f'{self.study_name}.vs_MS.x'] = x

        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.constant'] = 3.0
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.z'] = 1.2

        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.constant'] = 3.0
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.z'] = 1.5

        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.ee.display_treeview_nodes()
    uc_cls.ee.display_treeview_nodes(True)
    uc_cls.run(for_test=True)
