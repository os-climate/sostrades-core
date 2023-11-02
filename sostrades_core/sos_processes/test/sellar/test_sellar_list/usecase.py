'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/02 Copyright 2023 Capgemini

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
        Usecase for a list of Sellar Problem
        """
        ns = f'{self.study_name}'

        disc_dict = {}
        # Sellar inputs
        # ======================================================================
        # disc_dict[f'{ns}.x'] = array([1.])
        # disc_dict[f'{ns}.y_1'] = array([1.])
        # disc_dict[f'{ns}.y_2'] = array([1.])
        # disc_dict[f'{ns}.z'] = array([1., 1.])
        # ======================================================================

        disc_dict[f'{ns}.x'] = array([1.])
        disc_dict[f'{ns}.y_1'] = array([1.])
        disc_dict[f'{ns}.y_2'] = array([1.])
        disc_dict[f'{ns}.z'] = array([1., 1.])

        disc_dict[f'{ns}.Sellar_Problem.local_dv'] = 10.

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
