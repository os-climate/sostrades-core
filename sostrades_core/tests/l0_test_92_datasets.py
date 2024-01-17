"""
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
"""
from pathlib import Path
import unittest
import os

from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import DatasetConnectorType
from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import DatasetsConnectorManager
from sostrades_core.study_manager.study_manager import StudyManager


class TestDatasets(unittest.TestCase):
    """
    Discipline to test datasets
    """

    def setUp(self):
        # register connector for tests
        self.test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        DatasetsConnectorManager.register_connector(
            connector_identifier="JSON_datasets",
            connector_type=DatasetConnectorType.JSON,
            file_path=os.path.join(self.test_data_folder, "test_92_datasets_db.json"),
        )

        self.repo = "sostrades_core.sos_processes.test"
        self.study_name = "dataset_test"
        self.proc_name = "test_disc1_disc2_dataset"
        self.process_path = os.path.join(Path(__file__).parents[1], "sos_processes", "test", self.proc_name)
        self.study = StudyManager(self.repo, self.proc_name, self.study_name)

    def test_01_usecase1(self):
        dm = self.study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("dataset_test.a"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc1VirtualNode.x"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc2VirtualNode.x"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc1.b"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc2.b"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc1.c"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc2.c"), None)

        self.study.load_study(os.path.join(self.process_path, "usecase_dataset.json"))

        self.assertEqual(dm.get_value("dataset_test.a"), 1)
        self.assertEqual(dm.get_value("dataset_test.Disc1VirtualNode.x"), 4)
        self.assertEqual(dm.get_value("dataset_test.Disc2VirtualNode.x"), 4)
        self.assertEqual(dm.get_value("dataset_test.Disc1.b"), "string_2")
        self.assertEqual(dm.get_value("dataset_test.Disc2.b"), "string_2")
        self.assertEqual(dm.get_value("dataset_test.Disc1.c"), "string_3")
        self.assertEqual(dm.get_value("dataset_test.Disc2.c"), "string_3")

    def test_02_usecase2(self):
        dm = self.study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("dataset_test.a"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc1VirtualNode.x"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc2VirtualNode.x"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc1.b"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc2.b"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc1.c"), None)
        self.assertEqual(dm.get_value("dataset_test.Disc2.c"), None)

        self.study.load_study(os.path.join(self.process_path, "usecase_2datasets.json"))

        self.assertEqual(dm.get_value("dataset_test.a"), 10)
        self.assertEqual(dm.get_value("dataset_test.Disc1VirtualNode.x"), 20)
        self.assertEqual(dm.get_value("dataset_test.Disc2VirtualNode.x"), 20)
        self.assertEqual(dm.get_value("dataset_test.Disc1.b"), "string_1")
        self.assertEqual(dm.get_value("dataset_test.Disc2.b"), "string_1")
        self.assertEqual(dm.get_value("dataset_test.Disc1.c"), "string_2")
        self.assertEqual(dm.get_value("dataset_test.Disc2.c"), "string_2")

    def test_03_mapping(self):
        """
        Some example to work with dataset mapping
        """
        json_file_path = os.path.join(self.test_data_folder, "test_92_example_mapping.json")

        dataset_mapping = DatasetsMapping.from_json_file(file_path=json_file_path)
        self.assertEqual(dataset_mapping.datasets_infos["Dataset1"].connector_id, "<1connector_id>")
        self.assertEqual(dataset_mapping.datasets_infos["Dataset1"].dataset_id, "<1dataset_id>")
        self.assertEqual(dataset_mapping.datasets_infos["Dataset2"].connector_id, "<2connector_id>")
        self.assertEqual(dataset_mapping.datasets_infos["Dataset2"].dataset_id, "<2dataset_id>")

        self.assertEqual(
            dataset_mapping.namespace_datasets_mapping["namespace1"], [dataset_mapping.datasets_infos["Dataset1"]]
        )
        self.assertEqual(
            set(dataset_mapping.namespace_datasets_mapping["namespace2"]),
            set([dataset_mapping.datasets_infos["Dataset1"], dataset_mapping.datasets_infos["Dataset2"]]),
        )
