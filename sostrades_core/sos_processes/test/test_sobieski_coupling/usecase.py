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
from sos_trades_core.study_manager.study_manager import StudyManager
from numpy import array
import pandas as pd


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        sc_name = "SobieskyCoupling"
        disc_dict = {
                      '{ns}.z': [0.05,45000,1.6,5.5,55.,1000],
                      '{ns}.y_14': [50606.9,7306.20],
                      '{ns}.y_24':  [4.15], 
                      '{ns}.y_34': [1.10], 
                      '{ns}.x_1': [0.25,1.0],
                      '{ns}.y_21':  [50606.9], 
                      '{ns}.y_31': [6354.32], 
                      '{ns}.x_2': [1.0],
                      '{ns}.y_12':  [50606.9,0.95], 
                      '{ns}.y_32': [12194.2], 
                      '{ns}.x_3': [0.5],
                      '{ns}.y_23':  [12194.2], 
                      }

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
