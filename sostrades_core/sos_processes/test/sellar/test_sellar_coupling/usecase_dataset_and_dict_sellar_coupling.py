'''
Copyright 2024 Capgemini

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
from os.path import dirname, join, realpath
from pathlib import Path

from numpy import array

from sostrades_core.datasets.dataset_manager import DatasetsConnectorManager
from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import DatasetConnectorType
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
        disc_dict[f'{ns}.{coupling_name}.x'] = array([21.])
        disc_dict[f'{ns}.{coupling_name}.y_1'] = [21.]
        disc_dict[f'{ns}.{coupling_name}.y_2'] = [21.]
        disc_dict[f'{ns}.{coupling_name}.z'] = array([21., 21.])
        disc_dict[f'{ns}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        return [disc_dict]

    def get_dataset_mapping(self):
        # create connector

        connector_args = {
             "file_path": join(Path(__file__).parents[4],"tests", "data","test_92_datasets_db.json")
        }

        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_datasets_connector",
                                                    connector_type=DatasetConnectorType.get_enum_value("JSON"),
                                                    **connector_args)
        # Get dataset file NOTE it is not the same as the usecase name because it uses same dataset as other use case
        datasets_file = join(dirname(realpath(__file__)), "usecase_dataset_sellar_coupling.json")
        # Deserialize it
        return DatasetsMapping.from_json_file(datasets_file)

