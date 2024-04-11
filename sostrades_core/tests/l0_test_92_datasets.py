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
import unittest
import os

import numpy as np
import pandas as pd

from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.study_manager.study_manager import StudyManager
import sostrades_core.sos_processes.test.test_disc1_disc2_dataset.usecase_dataset
import sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset
import sostrades_core.sos_processes.test.sellar.test_sellar_coupling.usecase_dataset_sellar_coupling
import sostrades_core.sos_processes.test.sellar.test_sellar_coupling.usecase_dataset_and_dict_sellar_coupling as uc_dataset_dict


class TestDatasets(unittest.TestCase):
    """
    Discipline to test datasets
    """

    def setUp(self):
        # Set logging level to debug for datasets
        logging.getLogger("sostrades_core.datasets").setLevel(logging.DEBUG)

    def test_01_usecase1(self):
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_disc2_dataset.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        dm = study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("usecase_dataset.a"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.c"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.c"), None)

        study.load_study(os.path.join(process_path, "usecase_dataset.json"))

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
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_disc2_dataset.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        dm = study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("usecase_dataset.a"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2VirtualNode.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.c"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.c"), None)

        study.load_study(os.path.join(process_path, "usecase_2datasets.json"))

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
        self.assertEqual(dataset_mapping.datasets_infos["<1connector_id>|<1dataset_id>|*"].connector_id, "<1connector_id>")
        self.assertEqual(dataset_mapping.datasets_infos["<1connector_id>|<1dataset_id>|*"].dataset_id, "<1dataset_id>")
        self.assertEqual(dataset_mapping.datasets_infos["<2connector_id>|<2dataset_id>|*"].connector_id, "<2connector_id>")
        self.assertEqual(dataset_mapping.datasets_infos["<2connector_id>|<2dataset_id>|*"].dataset_id, "<2dataset_id>")

        self.assertEqual(
            dataset_mapping.namespace_datasets_mapping["namespace1"], [dataset_mapping.datasets_infos["<1connector_id>|<1dataset_id>|*"]]
        )
        self.assertEqual(
            set(dataset_mapping.namespace_datasets_mapping["namespace2"]),
            set([dataset_mapping.datasets_infos["<1connector_id>|<1dataset_id>|*"], dataset_mapping.datasets_infos["<2connector_id>|<2dataset_id>|*"]]),
        )
    

    def test_04_datasets_types(self):
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)

        dm = study.execution_engine.dm

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
        usecase_file_path = sostrades_core.sos_processes.test.sellar.test_sellar_coupling.usecase_dataset_sellar_coupling.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        study_name = "usecase_dataset_sellar_coupling"

        dm = study.execution_engine.dm

        study.load_study(os.path.join(process_path, "usecase_dataset_sellar_coupling.json"))

        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.x"), [1.0])
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_1"), [2.0])
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_2"), [3.0])
        self.assertTrue((dm.get_value(f"{study_name}.SellarCoupling.z")== [4.0,5.0]).all())
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.Sellar_Problem.local_dv"), 10.0)


    def test_06_parameter_change_returned_in_load_data_using_both_dict_and_datasets(self):
        usecase_file_path = uc_dataset_dict.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        uc = uc_dataset_dict.Study()
        param_changes = study.load_data(from_input_dict=uc.setup_usecase())
        param_changes.extend(study.load_study(os.path.join(process_path, "usecase_dataset_sellar_coupling.json")))
        if len(param_changes) != 9:
            msg = ["WRONG ParameterChanges : "] + [p.__str__() for p in param_changes]
            raise ValueError("\n".join(msg))
        self.assertEqual(len([p for p in param_changes if p.connector_id is None and p.dataset_id is None]), 5)
        self.assertEqual(len([p for p in param_changes if p.connector_id is not None and p.dataset_id is not None]), 4) # there is one variable in common