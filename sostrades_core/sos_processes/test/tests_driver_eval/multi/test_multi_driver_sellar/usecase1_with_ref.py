'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2024/05/16 Copyright 2023 Capgemini

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
import pandas as pd
from numpy import array

from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.proc_builder.process_builder_parameter_type import (
    ProcessBuilderParameterType,
)


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """Usecase for Eval simple on  Sellar Problem"""
        ns = f'{self.study_name}'

        disc_dict = {}
        # build the scenarios
        scenario_df = pd.DataFrame({'selected_scenario': [True, False, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_W',
                                                      'scenario_2']})
        disc_dict[f'{self.study_name}.Eval.samples_df'] = scenario_df
        disc_dict[f'{ns}.Eval.instance_reference'] = True
        disc_dict[f'{ns}.Eval.reference_mode'] = 'copy_mode'

        with_modal = True
        anonymize_input_dict_from_usecase = {}
        if with_modal:
            repo = 'sostrades_core.sos_processes.test.sellar'
            mod_id = 'test_sellar_list'
            my_usecase = 'Empty'
            process_builder_parameter_type = ProcessBuilderParameterType(
                mod_id, repo, my_usecase)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            disc_dict[f'{ns}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        # Sellar reference inputs
        # ======================================================================
        local_dv = 10.
        disc_dict[f'{ns}.Eval.ReferenceScenario.x'] = array([2.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.y_1'] = array([1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.y_2'] = array([1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.z'] = array([1., 1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.Sellar_Problem.local_dv'] = local_dv
        # ======================================================================

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    uc_cls.run()
