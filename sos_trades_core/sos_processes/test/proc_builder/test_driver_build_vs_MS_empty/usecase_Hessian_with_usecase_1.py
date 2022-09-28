'''
Copyright 2022 Airbus SA

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
     the test_disc_hessian_vs_MS_from_proc process.
    This process instantiates the multiscenario of a Hessian Discipline.
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        # provide a process (with disciplines) to the driver
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'

        usecase_name = 'usecase1'

        anonymize_input_dict = {}
        anonymize_input_dict['<study_ph>.Hessian.ax2'] = 4.0
        anonymize_input_dict['<study_ph>.Hessian.by2'] = 5.0
        anonymize_input_dict['<study_ph>.Hessian.cx'] = 6.0
        anonymize_input_dict['<study_ph>.Hessian.dy'] = 7.0
        anonymize_input_dict['<study_ph>.Hessian.exy'] = 12.0
        anonymize_input_dict['<study_ph>.Hessian.x'] = 2.0
        anonymize_input_dict['<study_ph>.Hessian.y'] = 3.0

        process_builder_parameter_type = ProcessBuilderParameterType(mod_id, repo, usecase_name)
        process_builder_parameter_type.usecase_data = anonymize_input_dict

        # provide an associated scenario_map to the driver
        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = []
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        ######### Numerical values   ####
        x = 2.0
        y = 3.0

        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0
        #scenario_list = ['scenario_1', 'scenario_2']
        scenario_list = ['scenario_1', 'scenario_2', 'reference']
        ######### Fill the dictionary for dm   ####
        dict_values = {}

        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list

        scenario = scenario_list[0]
        my_root = f'{self.study_name}' + '.vs_MS.' + scenario
        dict_values[f'{my_root}' + '.Hessian.x'] = x
        dict_values[f'{my_root}' + '.Hessian.y'] = y
        dict_values[f'{my_root}' + '.Hessian.ax2'] = ax2
        dict_values[f'{my_root}' + '.Hessian.by2'] = by2
        dict_values[f'{my_root}' + '.Hessian.cx'] = cx
        dict_values[f'{my_root}' + '.Hessian.dy'] = dy
        dict_values[f'{my_root}' + '.Hessian.exy'] = exy

        scenario = scenario_list[1]
        my_root = f'{self.study_name}' + '.vs_MS.' + scenario
        dict_values[f'{my_root}' + '.Hessian.x'] = x + 10.
        dict_values[f'{my_root}' + '.Hessian.y'] = y + 10.
        dict_values[f'{my_root}' + '.Hessian.ax2'] = ax2 + 10.
        dict_values[f'{my_root}' + '.Hessian.by2'] = by2 + 10.
        dict_values[f'{my_root}' + '.Hessian.cx'] = cx + 10.
        dict_values[f'{my_root}' + '.Hessian.dy'] = dy + 10.
        dict_values[f'{my_root}' + '.Hessian.exy'] = exy + 10.

        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
