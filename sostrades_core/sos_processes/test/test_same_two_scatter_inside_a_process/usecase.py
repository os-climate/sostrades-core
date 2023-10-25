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
from sostrades_core.study_manager.study_manager import StudyManager
from numpy import array


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        """

        """
        ns = f'{self.study_name}'

        private_val_list = [{f'{ns}.multi_scenarios.scenario_1.Disc1.a': 3,
                             f'{ns}.multi_scenarios.scenario_1.Disc1.b': 4,
                             f'{ns}.multi_scenarios.scenario_1.Disc2.constant': 10,
                             f'{ns}.multi_scenarios.scenario_1.Disc2.power': 2,
                             f'{ns}.multi_scenarios.scenario_1.x': 2,
                             f'{ns}.multi_scenarios.scenario_2.Disc1.a': 6,
                             f'{ns}.multi_scenarios.scenario_2.Disc1.b': 2,
                             f'{ns}.multi_scenarios.scenario_2.Disc2.constant': 20,
                             f'{ns}.multi_scenarios.scenario_2.Disc2.power': 3,
                             f'{ns}.multi_scenarios.scenario_2.x': 4}]

        private_val = {}
        for dic in private_val_list:
            private_val.update(dic)

        private_val_scatter1 = {name.replace(
            f'{ns}.', f'{ns}.Scatter1.'): value for name, value in private_val.items()}
        private_val_scatter2 = {name.replace(
            f'{ns}.', f'{ns}.Scatter2.'): value for name, value in private_val.items()}
        private_val_scatter1.update(private_val_scatter2)

        return private_val_scatter1


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
#     uc_cls.execution_engine.root_process.coupling_structure.graph.write_full_graph("here.pdf")
