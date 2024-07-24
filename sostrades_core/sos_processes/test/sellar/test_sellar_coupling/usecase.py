'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/05/16 Copyright 2023 Capgemini

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
from os.path import dirname, join

from numpy import array

from sostrades_core.study_manager.run_usecase import run_usecase
from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase for a coupling of Sellar Problem
        """
        ns = f'{self.study_name}'
        coupling_name = "SellarCoupling"

        disc_dict = {}
        # Sellar inputs
        # ======================================================================
        disc_dict[f'{ns}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{ns}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{ns}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{ns}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{ns}.{coupling_name}.Sellar_Problem.local_dv'] = 10.
        # ======================================================================

        return [disc_dict]


if '__main__' == __name__:
    # uc_cls = Study()
    # uc_cls.load_data()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    # uc_cls.run()
    #  uc_cls.execution_engine.root_process.coupling_structure.graph.write_full_graph("here.pdf")
    proc_name = "sostrades_core.sos_processes.test.sellar.test_sellar_coupling"
    run_usecase(proc_name, join(dirname(__file__), 'usecase_dataset_sellar_coupling.json'))
