'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        disc_dict = {}
        disc_dict[f'{ns}.proc1.Disc2.constant'] = constant1
        disc_dict[f'{ns}.proc1.Disc2.power'] = power1
        disc_dict[f'{ns}.proc2.Disc2.constant'] = constant2
        disc_dict[f'{ns}.proc2.Disc2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        disc_dict[f'{ns}.proc1.x'] = x1
        disc_dict[f'{ns}.proc2.x'] = x2
        disc_dict[f'{ns}.proc1.Disc1.a'] = a1
        disc_dict[f'{ns}.proc2.Disc1.a'] = a2
        disc_dict[f'{ns}.proc1.Disc1.b'] = b1
        disc_dict[f'{ns}.proc2.Disc1.b'] = b2

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
#     uc_cls.execution_engine.root_process.coupling_structure.graph.write_full_graph("here.pdf")
