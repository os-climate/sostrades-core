'''
Copyright 2023 Capgemini

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
import os

from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import DatasetConnectorType
from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import DatasetsConnectorManager
from sostrades_core.study_manager.study_manager import StudyManager


if '__main__' == __name__:
    # Create a connector for demonstration
    test_data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "tests", "data")

    DatasetsConnectorManager.register_connector(
        connector_identifier="JSON_datasets",
        connector_type=DatasetConnectorType.JSON,
        file_path=os.path.join(test_data_folder, "test_92_datasets_db.json"),
    )

    # Instanciate study without the study class
    repo = "sostrades_core.sos_processes.test"
    study_name = "dataset_test"
    proc_name = "test_disc1_disc2_dataset"

    # test study with only one dataset
    json_study_file_path = os.path.join(os.path.dirname(__file__), 'usecase_dataset.json')
    uc_cls = StudyManager(repo, proc_name, study_name)
    uc_cls.load_study(json_study_file_path)
    uc_cls.run()

    # test with 2 datasets
    json_study_file_path = os.path.join(os.path.dirname(__file__), 'usecase_2datasets.json')
    uc_cls2 = StudyManager(repo, proc_name, study_name)
    uc_cls2.load_study(json_study_file_path)
    uc_cls2.run()
