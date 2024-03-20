'''
Copyright 2022 Airbus SAS
Modifications on 29/01/2024 Copyright 2024 Capgemini

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
from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import DatasetsConnectorManager
from sostrades_core.study_manager.run_usecase import run_usecase
from sostrades_core.study_manager.study_manager import StudyManager
from os.path import join, dirname

class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        dict_values = {
            f'{self.study_name}.Disc1.x': 3,
            f'{self.study_name}.Disc1.a': 1,
            f'{self.study_name}.Disc1.b': 5,
            f'{self.study_name}.Disc1.name': 'A1'
            }
        return dict_values


if '__main__' == __name__:
    #uc_cls = Study()
    #uc_cls.load_data()

    # copy json datasets into arango dataset
    arangoCnx = DatasetsConnectorManager.get_connector("Arango_connector")
    jsonCnx = DatasetsConnectorManager.get_connector("MVP0_datasets_connector")
    
    arangoCnx.copy_dataset_from(jsonCnx, 'dataset_all_types', {'x': 'float', 'a': 'int', 'b': 'int', 'name': 'string', 
                                                               'x_dict': 'dict', 'y_array': 'array', 'z_list': 'list', 'b_bool': 'bool',
                                                                 'd': 'dataframe', 'linearization_mode': 'string', 'cache_type': 'string', 
                                                                 'cache_file_path': 'string', 'debug_mode': 'string'},
                                                                 create_if_not_exists=True, override=True)

    # execute the usecase
    proc_name = "sostrades_core.sos_processes.test.test_disc1_all_types"
    run_usecase(proc_name, join(dirname(__file__),'usecase_dataset.json'))

    
