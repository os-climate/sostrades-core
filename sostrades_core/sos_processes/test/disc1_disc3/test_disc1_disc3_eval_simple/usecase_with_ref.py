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
import pandas as pd
from numpy import array

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase from test_01_multi_instance_configuration_from_df_without_reference_scenario
        """

        # reference var values
        self.x = 2.
        self.a = 3
        self.constant = 3
        self.power = 2
        self.b = 8
        self.z = 12
        self.b1 = 4
        self.b2 = 2
        self.z1 = 1.2
        self.z2 = 1.5

        disc_dict = {}
        # build the scenarios
        scenario_df = pd.DataFrame({'selected_scenario': [True, False, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_W',
                                                      'scenario_2']})
        disc_dict[f'{self.study_name}.Eval.scenario_df'] = scenario_df
        disc_dict[f'{self.study_name}.Eval.builder_mode'] = 'multi_instance'
        disc_dict[f'{self.study_name}.Eval.instance_reference'] = True
        disc_dict[f'{self.study_name}.Eval.reference_mode'] = 'linked_mode'

        # configure the Reference scenario
        # Non-trade variables (to propagate)
        disc_dict[f'{self.study_name}.Eval.ReferenceScenario.a'] = self.a
        disc_dict[f'{self.study_name}.Eval.ReferenceScenario.x'] = self.x
        disc_dict[self.study_name +
                  '.Eval.ReferenceScenario.Disc3.constant'] = self.constant
        disc_dict[self.study_name +
                  '.Eval.ReferenceScenario.Disc3.power'] = self.power
        # Trade variables reference (not to propagate)
        disc_dict[self.study_name + '.Eval.ReferenceScenario.Disc1.b'] = self.b
        disc_dict[self.study_name + '.Eval.ReferenceScenario.z'] = self.z

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    uc_cls.run()
