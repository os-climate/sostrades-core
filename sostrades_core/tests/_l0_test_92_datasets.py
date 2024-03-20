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
import logging
from pathlib import Path
import unittest
import os

import numpy as np
import pandas as pd

from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.study_manager.study_manager import StudyManager


class TestDatasets(unittest.TestCase):
    """
    Discipline to test datasets
    """

    def setUp(self):
        # Set logging level to debug for datasets
        logging.getLogger("sostrades_core.datasets").setLevel(logging.DEBUG)

        self.process_module_name = "sostrades_core.sos_processes.test.test_disc1_disc2_dataset"
        self.study_name = "usecase_dataset"
        self.process_path = os.path.join(Path(__file__).parents[1], "sos_processes", "test", "test_disc1_disc2_dataset")
        self.study = StudyManager(self.process_module_name, self.study_name)

    def test_01_usecase1(self):
        dm = self.study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("usecase_dataset.a"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.c"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.c"), None)

        #check numerical parameters
        self.assertEqual(dm.get_value("usecase_dataset.linearization_mode"), None)
        self.assertEqual(dm.get_value("usecase_dataset.debug_mode"), None)
        self.assertEqual(dm.get_value("usecase_dataset.cache_type"), None)
        self.assertEqual(dm.get_value("usecase_dataset.cache_file_path"), None)
        self.assertEqual(dm.get_value("usecase_dataset.sub_mda_class"), None)
        self.assertEqual(dm.get_value("usecase_dataset.max_mda_iter"), None)
        self.assertEqual(dm.get_value("usecase_dataset.n_processes"), None)
        self.assertEqual(dm.get_value("usecase_dataset.chain_linearize"), None)
        self.assertEqual(dm.get_value("usecase_dataset.tolerance"), None)
        self.assertEqual(dm.get_value("usecase_dataset.use_lu_fact"), None)
        self.assertEqual(dm.get_value("usecase_dataset.warm_start"), None)
        self.assertEqual(dm.get_value("usecase_dataset.acceleration"), None)
        self.assertEqual(dm.get_value("usecase_dataset.warm_start_threshold"), None)
        self.assertEqual(dm.get_value("usecase_dataset.n_subcouplings_parallel"), None)
        self.assertEqual(dm.get_value("usecase_dataset.tolerance_gs"), None)
        self.assertEqual(dm.get_value("usecase_dataset.relax_factor"), None)
        self.assertEqual(dm.get_value("usecase_dataset.epsilon0"), None)
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDO"), None)
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDO_preconditioner"), None)
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDO_options"), None)
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDA"), None)
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDA_preconditioner"), None)
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDA_options"), None)
        self.assertEqual(dm.get_value("usecase_dataset.group_mda_disciplines"), None)
        self.assertEqual(dm.get_value("usecase_dataset.propagate_cache_to_children"), None)

        self.study.load_study(os.path.join(self.process_path, "usecase_dataset.json"))

        self.assertEqual(dm.get_value("usecase_dataset.a"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1VirtualNode.x"), 4)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2VirtualNode.x"), 4)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), "string_2")
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.b"), "string_2")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.c"), "string_3")
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.c"), "string_3")

        #check numerical parameters
        self.assertEqual(dm.get_value("usecase_dataset.linearization_mode"), "auto")
        self.assertEqual(dm.get_value("usecase_dataset.debug_mode"), "")
        self.assertEqual(dm.get_value("usecase_dataset.cache_type"), "None")
        self.assertEqual(dm.get_value("usecase_dataset.cache_file_path"), "")
        self.assertEqual(dm.get_value("usecase_dataset.sub_mda_class"), "MDAJacobi")
        self.assertEqual(dm.get_value("usecase_dataset.max_mda_iter"), 30)
        self.assertEqual(dm.get_value("usecase_dataset.n_processes"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.chain_linearize"), False)
        self.assertEqual(dm.get_value("usecase_dataset.tolerance"), 1.0e-6)
        self.assertEqual(dm.get_value("usecase_dataset.use_lu_fact"), False)
        self.assertEqual(dm.get_value("usecase_dataset.warm_start"), False)
        self.assertEqual(dm.get_value("usecase_dataset.acceleration"), "m2d")
        self.assertEqual(dm.get_value("usecase_dataset.warm_start_threshold"), -1)
        self.assertEqual(dm.get_value("usecase_dataset.n_subcouplings_parallel"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.tolerance_gs"), 10.0)
        self.assertEqual(dm.get_value("usecase_dataset.relax_factor"), 0.99)
        self.assertEqual(dm.get_value("usecase_dataset.epsilon0"), 1.0e-6)
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDO"), "GMRES")
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDO_preconditioner"), "None")
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDO_options"), {
            "max_iter": 1000,
            "tol": 1.0e-8})
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDA"), "GMRES")
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDA_preconditioner"), "None")
        self.assertEqual(dm.get_value("usecase_dataset.linear_solver_MDA_options"), {
            "max_iter": 1000,
            "tol": 1.0e-8})
        self.assertEqual(dm.get_value("usecase_dataset.group_mda_disciplines"), False)
        self.assertEqual(dm.get_value("usecase_dataset.propagate_cache_to_children"), False)


    def test_02_usecase2(self):
        dm = self.study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("usecase_dataset.a"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.c"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.c"), None)

        self.study.load_study(os.path.join(self.process_path, "usecase_2datasets.json"))

        self.assertEqual(dm.get_value("usecase_dataset.a"), 10)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1VirtualNode.x"), 20)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2VirtualNode.x"), 20)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), "string_1")
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.b"), "string_1")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.c"), "string_2")
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.c"), "string_2")

    def test_03_mapping(self):
        """
        Some example to work with dataset mapping
        """
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        json_file_path = os.path.join(test_data_folder, "test_92_example_mapping.json")

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
    

    def test_04_datasets_types(self):
        process_module_name = "sostrades_core.sos_processes.test.test_disc1_all_types"
        study_name = "usecase_dataset"
        process_path = os.path.join(Path(__file__).parents[1], "sos_processes", "test", "test_disc1_all_types")
        study = StudyManager(process_module_name, study_name)

        dm = study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {})
        self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"),np.array([])))
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [])
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), True)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.d"), None)

        study.load_study(os.path.join(process_path, "usecase_dataset.json"))

        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), 2)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), "A1")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1":1,"test2":2})
        self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0,2.0,3.0])))
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [1.0,2.0,3.0])
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), False)
        self.assertTrue((dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame({"years":[2023,2024],"x":[1.0,10.0]})).all().all())
    
    def test_05_nested_process_level0(self):
        process_module_name = "sostrades_core.sos_processes.test.sellar.test_sellar_coupling"
        study_name = "usecase_dataset_sellar_coupling"
        process_path = os.path.join(Path(__file__).parents[1], "sos_processes", "test", "sellar", "test_sellar_coupling")
        study = StudyManager(process_module_name, study_name)

        dm = study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.x"), None)
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_1"), None)
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_2"), None)
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.z"), None)
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.Sellar_Problem.local_dv"), None)

        study.load_study(os.path.join(process_path, "usecase_dataset_sellar_coupling.json"))

        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.x"), [1.0])
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_1"), [2.0])
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_2"), [3.0])
        self.assertTrue((dm.get_value(f"{study_name}.SellarCoupling.z")== [4.0,5.0]).all())
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.Sellar_Problem.local_dv"), 10.0)
