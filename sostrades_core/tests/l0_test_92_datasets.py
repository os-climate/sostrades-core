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
import os
import unittest

import numpy as np
import pandas as pd

import sostrades_core.sos_processes.test.sellar.test_sellar_coupling.usecase_dataset_and_dict_sellar_coupling as uc_dataset_dict
import sostrades_core.sos_processes.test.sellar.test_sellar_coupling.usecase_dataset_sellar_coupling
import sostrades_core.sos_processes.test.test_disc1_disc2_dataset.usecase_dataset
import sostrades_core.sos_processes.test.test_disc1_nested_types.usecase_local_dataset
from sostrades_core.datasets.dataset_info.dataset_info_v0 import DatasetInfoV0
from sostrades_core.datasets.dataset_info.dataset_info_v1 import DatasetInfoV1
from sostrades_core.datasets.dataset_mapping import (
    DatasetsMapping,
    DatasetsMappingException,
)
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    DatasetGenericException,
)
from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import DatasetConnectorType
from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import DatasetsConnectorManager
from sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset import Study
from sostrades_core.sos_processes.test.test_disc1_disc2_coupling.usecase_coupling_2_disc_test import (
    Study as StudyDisc1Disc2,
)
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.compare_data_manager_tooling import dict_are_equal


class TestDatasets(unittest.TestCase):
    """
    Discipline to test datasets
    """

    def setUp(self):
        # Set logging level to debug for datasets
        logging.getLogger("sostrades_core.datasets").setLevel(logging.DEBUG)

        # create connector for test
        connector_args = {
             "file_path": "./sostrades_core/tests/data/test_92_datasets_db.json"
        }

        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_datasets_connector",
                                                    connector_type=DatasetConnectorType.get_enum_value("JSON"),
                                                    **connector_args)
        
        # create local connector
        connector_args = {
             "root_directory_path": "./sostrades_core/tests/data/local_datasets_db/",
             "create_if_not_exists": True
        }

        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_local_datasets_connector",
                                                    connector_type=DatasetConnectorType.get_enum_value("Local"),
                                                    **connector_args)
        
        
        # nested types reference values to be completed with more nested types
        df1 = pd.DataFrame({'years': [2020, 2021, 2022],
                            'type': ['alpha', 'beta', 'gamma']})
        df2 = pd.DataFrame({'years': [2020, 2021, 2022],
                            'price': [20.33, 60.55, 72.67]})
        dict_df = {'df1': df1.copy(), 'df2': df2.copy()}
        dict_dict_df = {'dict': {'df1': df1.copy(), 'df2': df2.copy()}}
        dict_dict_float = {'dict': {'f1': 0.033, 'f2': 333.66}}
        array_string = np.array(['s1', 's2'])
        array_df = np.array([df1.copy(), df2.copy()])
        dspace_dict_lists = {'variable': ['x', 'z', 'y_1', 'y_2'],
                             'value': [[1.], [5., 2.], [1.], [1.]],
                             'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                             'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                             'enable_variable': [True, True, True, True],
                             'activated_elem': [[True], [True, True], [True], [True]]}
        dspace_dict_array = {'variable': ['x', 'z', 'y_1', 'y_2'],
                             'value': [np.array([1.]), np.array([5., 2.]), np.array([1.]), np.array([1.])],
                             'lower_bnd': [np.array([0.]), np.array([-10., 0.]), np.array([-100.]), np.array([-100.])],
                             'upper_bnd': [np.array([10.]), np.array([10., 10.]), np.array([100.]), np.array([100.])],
                             'enable_variable': [True, True, True, True],
                             'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_lists = pd.DataFrame(dspace_dict_lists)
        dspace_array = pd.DataFrame(dspace_dict_array)

        self.nested_types_reference_dict = {'X_dict_df': dict_df,
                                            'X_dict_dict_df': dict_dict_df,
                                            'X_dict_dict_float': dict_dict_float,
                                            'X_array_string': array_string,
                                            'X_array_df': array_df,
                                            'X_dspace_lists': dspace_lists,
                                            'X_dspace_array': dspace_array,
                                            }

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

        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_dataset.json")))

        self.assertEqual(dm.get_value("usecase_dataset.a"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1VirtualNode.x"), 4)
        self.assertEqual(dm.get_value("usecase_dataset.Disc2VirtualNode.x"), 4)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), "string_2")
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.b"), "string_2")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.c"), "string_3")
        self.assertEqual(dm.get_value("usecase_dataset.Disc2.c"), "string_3")

        # check numerical parameters
        self.assertEqual(dm.get_value("usecase_dataset.linearization_mode"), "auto")
        self.assertEqual(dm.get_value("usecase_dataset.debug_mode"), "")
        self.assertEqual(dm.get_value("usecase_dataset.cache_type"), "")
        self.assertEqual(dm.get_value("usecase_dataset.cache_file_path"), "")
        self.assertEqual(dm.get_value("usecase_dataset.inner_mda_name"), "MDAJacobi")
        self.assertEqual(dm.get_value("usecase_dataset.max_mda_iter"), 30)
        self.assertEqual(dm.get_value("usecase_dataset.n_processes"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.chain_linearize"), False)
        self.assertEqual(dm.get_value("usecase_dataset.tolerance"), 1.0e-6)
        self.assertEqual(dm.get_value("usecase_dataset.use_lu_fact"), False)
        self.assertEqual(dm.get_value("usecase_dataset.warm_start"), False)
        self.assertEqual(dm.get_value("usecase_dataset.acceleration_method"), "Alternate2Delta")
        self.assertEqual(dm.get_value("usecase_dataset.n_subcouplings_parallel"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.tolerance_gs"), 10.0)
        self.assertEqual(dm.get_value("usecase_dataset.over_relaxation_factor"), 0.99)
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

        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_2datasets.json")))

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

        self.assertEqual(dataset_mapping.datasets_infos["V0|<1connector_id>|<1dataset_id>"].connector_id,
                         "<1connector_id>")
        self.assertEqual(dataset_mapping.datasets_infos["V0|<1connector_id>|<1dataset_id>"].dataset_id, "<1dataset_id>")
        self.assertEqual(dataset_mapping.datasets_infos["V0|<2connector_id>|<2dataset_id>"].connector_id,
                         "<2connector_id>")
        self.assertEqual(dataset_mapping.datasets_infos["V0|<2connector_id>|<2dataset_id>"].dataset_id, "<2dataset_id>")

        self.assertEqual(
            dataset_mapping.namespace_datasets_mapping["namespace1"], ["V0|<1connector_id>|<1dataset_id>"]
        )
        self.assertEqual(
            set(dataset_mapping.namespace_datasets_mapping["namespace2"]),
            set(["V0|<1connector_id>|<1dataset_id>", "V0|<2connector_id>|<2dataset_id>"]),
        )

    def test_04_datasets_types(self):
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)

        dm = study.execution_engine.dm

        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_dataset.json")))

        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), 2)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), "A1")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1": 1, "test2": 2})
        self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0, 2.0, 3.0])))
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [1.0, 2.0, 3.0])
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), False)
        self.assertTrue((dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame(
            {"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

    def test_05_nested_process_level0(self):
        usecase_file_path = sostrades_core.sos_processes.test.sellar.test_sellar_coupling.usecase_dataset_sellar_coupling.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        study_name = "usecase_dataset_sellar_coupling"

        dm = study.execution_engine.dm

        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_dataset_sellar_coupling.json")))

        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.x"), [1.0])
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_1"), [2.0])
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.y_2"), [3.0])
        self.assertTrue((dm.get_value(f"{study_name}.SellarCoupling.z") == [4.0, 5.0]).all())
        self.assertEqual(dm.get_value(f"{study_name}.SellarCoupling.Sellar_Problem.local_dv"), 10.0)

    def test_06_parameter_change_returned_in_load_data_using_both_dict_and_datasets(self):
        
        
        
        usecase_file_path = uc_dataset_dict.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        uc = uc_dataset_dict.Study()

        param_changes = study.load_data(from_input_dict=uc.setup_usecase())
        param_changes.extend(study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_dataset_sellar_coupling.json"))))
        x_parameterchanges = [_pc for _pc in param_changes if
                              _pc.parameter_id == 'usecase_dataset_and_dict_sellar_coupling.SellarCoupling.x']
        z_parameterchanges = [_pc for _pc in param_changes if
                              _pc.parameter_id == 'usecase_dataset_and_dict_sellar_coupling.SellarCoupling.z']

        self.assertEqual(x_parameterchanges[0].variable_type, 'array')
        self.assertEqual(x_parameterchanges[0].old_value, None)
        self.assertTrue(np.all(x_parameterchanges[0].new_value == [21.]))
        self.assertEqual(z_parameterchanges[0].variable_type, 'array')
        self.assertEqual(z_parameterchanges[0].old_value, None)
        self.assertTrue(np.all(z_parameterchanges[0].new_value == [21., 21.]))
        self.assertEqual(x_parameterchanges[0].dataset_id, None)
        self.assertEqual(x_parameterchanges[0].connector_id, None)
        self.assertEqual(z_parameterchanges[0].dataset_id, None)
        self.assertEqual(z_parameterchanges[0].connector_id, None)

        self.assertEqual(x_parameterchanges[1].variable_type, 'array')
        self.assertTrue(np.all(x_parameterchanges[1].old_value == [21.]))
        self.assertTrue(np.all(x_parameterchanges[1].new_value == [1.]))
        self.assertEqual(z_parameterchanges[1].variable_type, 'array')
        self.assertTrue(np.all(z_parameterchanges[1].old_value == [21., 21.]))
        self.assertTrue(np.all(z_parameterchanges[1].new_value == [4., 5.]))
        self.assertEqual(x_parameterchanges[1].dataset_id, 'V0|MVP0_datasets_connector|dataset_sellar')
        self.assertEqual(x_parameterchanges[1].connector_id, 'MVP0_datasets_connector')
        self.assertEqual(z_parameterchanges[1].dataset_id, 'V0|MVP0_datasets_connector|dataset_sellar')
        self.assertEqual(z_parameterchanges[1].connector_id, 'MVP0_datasets_connector')

    def test_07_datasets_local_connector_with_all_non_nested_types(self):
        """
        Check correctness of loaded values after loading a handcrafted local directories' dataset,  testing usage of
        LocalDatasetsConnector and FileSystemDatasetsSerializer.
        """
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)

        dm = study.execution_engine.dm

        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset.json")))

        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), 2)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), "A1")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1": 1, "test2": 2})
        self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0, 2.0, 3.0])))
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [1.0, 2.0, 3.0])
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), False)
        self.assertTrue((dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame(
            {"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

    def test_07b_datasets_local_connector_with_several_namespace(self):
        """
        Check correctness of loaded values after loading a handcrafted local directories' dataset,  testing usage of
        LocalDatasetsConnector and FileSystemDatasetsSerializer. and more than one namespace
        """
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_disc2_coupling.usecase_coupling_2_disc_test.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyDisc1Disc2()
        study.load_data()
        study.run()
        dm = study.execution_engine.dm

        data_types_dict = {'a': 'float',
            'x': 'float',
            'b': 'float',
            'y': 'float',
            'z': 'float',
            'constant': 'float',
            'power': 'int',
            'indicator': 'float'
            }

        # export study in another folder
        # create connector test for export
        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_datasets_db_export_test/",
            "create_if_not_exists": True
        }

        export_connector = DatasetsConnectorManager.register_connector(connector_identifier="MVP0_local_datasets_connector_export_test",
                                                    connector_type=DatasetConnectorType.get_enum_value("Local"),
                                                    **connector_args)
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        export_mapping_repo_file_path = os.path.join(test_data_folder, "test_92_export_mapping_disc1_disc2.json")

        # test export
        mapping = DatasetsMapping.from_json_file(export_mapping_repo_file_path)
        study.export_data_from_dataset_mapping(mapping)
        exported_data = export_connector.get_values_all(DatasetInfoV0("MVP0_local_datasets_connector_export_test",
                                                                      "test_dataset_disc1_disc2"), data_types_dict)
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.Disc1.a"), exported_data.get("a"))
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.x"), exported_data.get("x"))
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.Disc1.b"), exported_data.get("b"))
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.Disc1.indicator"), exported_data.get("indicator"))
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.y"), exported_data.get("y"))
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.Disc2.constant"), exported_data.get("constant"))
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.Disc2.power"), exported_data.get("power"))
        self.assertEqual(dm.get_value("usecase_coupling_2_disc_test.z"), exported_data.get("z"))

        # test import
        study2 = StudyManager(file_path=usecase_file_path)
        study2.update_data_from_dataset_mapping(mapping)
        dm2 = study2.execution_engine.dm
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.Disc1.a"), exported_data.get("a"))
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.x"), exported_data.get("x"))
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.Disc1.b"), exported_data.get("b"))
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.Disc1.indicator"), exported_data.get("indicator"))
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.y"), exported_data.get("y"))
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.Disc2.constant"), exported_data.get("constant"))
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.Disc2.power"), exported_data.get("power"))
        self.assertEqual(dm2.get_value("usecase_coupling_2_disc_test.z"), exported_data.get("z"))

        export_connector.clear(remove_root_directory=True)

    def test_08_json_to_local_connector_conversion_and_loading(self):
        """
        Use a local connector to copy values from a JSON connector then load them in the study and check correctness,
        thus testing ability of LocalConnector to both write and load values.
        """
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )
        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_datasets_db_copy_test/",
            "create_if_not_exists": True
        }
        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_local_datasets_connector_copy_test",
                                                    connector_type=DatasetConnectorType.get_enum_value("Local"),
                                                    **connector_args)
        
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)

        dm = study.execution_engine.dm
        connector_to = DatasetsConnectorManager.get_connector('MVP0_local_datasets_connector_copy_test')
        connector_json = DatasetsConnectorManager.get_connector('MVP0_datasets_connector')

        dataset_vars = ["a",
                        "x",
                        "b",
                        "name",
                        "x_dict",
                        "y_array",
                        "z_list",
                        "b_bool",
                        "d"]

        data_types_dict = {_k: dm.get_data(f"usecase_dataset.Disc1.{_k}", "type") for _k in dataset_vars}

        try:
            connector_to.copy_dataset_from(connector_from=connector_json,
                                           dataset_identifier=DatasetInfoV0(connector_json, "dataset_all_types"),
                                           data_types_dict=data_types_dict,
                                           create_if_not_exists=True)

            study.update_data_from_dataset_mapping(
                DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset_copy_test.json")))
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 1)
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), 2)
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), "A1")
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1": 1, "test2": 2})
            self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0, 2.0, 3.0])))
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [1.0, 2.0, 3.0])
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), False)
            self.assertTrue((dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame(
                {"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())
            connector_to.clear(remove_root_directory=True)
        except Exception as cm:
            connector_to.clear(remove_root_directory=True)
            raise cm

    def test_09_dataset_error(self):
        """
        Some example to check datasets error
        """
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")

        # check mapping file error
        mapping_error_json_file_path = os.path.join(test_data_folder, "test_92_example_mapping_error_format.json")
        with self.assertRaises(DatasetsMappingException):
            DatasetsMapping.from_json_file(mapping_error_json_file_path)

        # check dataset reading error
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        mapping = DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset_error.json"))
        with self.assertRaises(DatasetGenericException):
            study.update_data_from_dataset_mapping(mapping)

    def test_10_repository_dataset_connector(self):
        """
        Some example to check repository datasets connector
        """
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")

        mapping_repo_file_path = os.path.join(test_data_folder, "test_92_mapping_repository.json")

        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        dm = study.execution_engine.dm
        study.update_data_from_dataset_mapping(DatasetsMapping.from_json_file(mapping_repo_file_path))

        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), 2)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), "A1")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1": 1, "test2": 2})
        self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0, 2.0, 3.0])))
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [1.0, 2.0, 3.0])
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), False)
        self.assertTrue((dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame(
            {"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

    def test_11_datasets_local_connector_nested_types(self):
        """
        Check correctness of loaded values after loading a handcrafted local directories' dataset, testing usage of
        LocalDatasetsConnector and FileSystemDatasetsSerializer pickle-based loading for the following nested types:
            - dict[str: DataFrame]
            - dict[str: dict[str: DataFrame]]
            - dict[str: dict[str: float]]  (THIS IS JSONIFIABLE)
            - array[str]
            - array[DataFrame]
            - design space with lists (DataFrame with string, list, and bool columns)
            - design space with arrays (DataFrame with string, array, list and bool columns)
        """

        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_nested_types.usecase_local_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        dm = study.execution_engine.dm
        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset.json")))
        dm_dict = {var: dm.get_value(f"usecase_local_dataset.Disc1.{var}") for var in self.nested_types_reference_dict}
        self.assertTrue(dict_are_equal(self.nested_types_reference_dict, dm_dict))

    def test_12_local_to_local_connector_conversion_and_loading_for_nested_types(self):
        """
        Use a local connector to copy values from a validated local connector then load them in the study and check
        correctness, thus testing ability of LocalConnector to both write and load values in the scope of the nested
        types listed in test_11 above, some of which need to be stored in a separate pickle (1 pickle per dataset).

        Note the following differences between connector dump to pickle and the hand-crafted dataset used for test_11:
        - since dict[str: dict[str: float]] is jsonifiable it will be saved in the descriptor.json, and not pickled
        - since dataframe dumping is based on GUI method, it can dump design-space-like dataframes to csv, not pickled
        """
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )
        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_datasets_db_copy_test_nested/",
            "create_if_not_exists": True
        }
        DatasetsConnectorManager.register_connector(
            connector_identifier="MVP0_local_datasets_connector_copy_test_nested",
            connector_type=DatasetConnectorType.get_enum_value("Local"),
            **connector_args)
        
        
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_nested_types.usecase_local_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)

        dm = study.execution_engine.dm
        connector_to = DatasetsConnectorManager.get_connector('MVP0_local_datasets_connector_copy_test_nested')
        connector_local = DatasetsConnectorManager.get_connector('MVP0_local_datasets_connector')

        data_types_dict = {_k: dm.get_data(f"usecase_local_dataset.Disc1.{_k}", "type") for _k in
                           self.nested_types_reference_dict}

        try:
            connector_to.copy_dataset_from(connector_from=connector_local,
                                           dataset_identifier=DatasetInfoV0(connector_local, "dataset_nested_types"),
                                           data_types_dict=data_types_dict,
                                           create_if_not_exists=True)

            study.update_data_from_dataset_mapping(
                DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset_copy_test.json")))
            dm_dict = {var: dm.get_value(f"usecase_local_dataset.Disc1.{var}") for var in
                       self.nested_types_reference_dict}

            self.assertTrue(dict_are_equal(self.nested_types_reference_dict, dm_dict))
            connector_to.clear(remove_root_directory=True)
        except Exception as cm:
            connector_to.clear(remove_root_directory=True)
            raise cm

    def test_13_export_with_repository_dataset_connector(self):
        """
        Some example to check repository datasets connector export
        """
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )

        # create usecase with data
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        mapping_repo_file_path = os.path.join(test_data_folder, "test_92_mapping_repository.json")
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        study.update_data_from_dataset_mapping(DatasetsMapping.from_json_file(mapping_repo_file_path))

        # export study in another folder to compare the datasets
        # create connector test for export
        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_datasets_db_export_test/",
            "create_if_not_exists": True
        }

        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_local_datasets_connector_export_test",
                                                    connector_type=DatasetConnectorType.get_enum_value("Local"),
                                                    **connector_args)
        export_mapping_repo_file_path = os.path.join(test_data_folder, "test_92_export_mapping_repository.json")

        study.export_data_from_dataset_mapping(DatasetsMapping.from_json_file(export_mapping_repo_file_path))

        dm = study.execution_engine.dm
        connector_export = DatasetsConnectorManager.get_connector('MVP0_local_datasets_connector_export_test')

        dataset_vars = ["a",
                        "x",
                        "b",
                        "name",
                        "x_dict",
                        "y_array",
                        "z_list",
                        "b_bool",
                        "d"]

        data_types_dict = {_k: dm.get_data(f"usecase_dataset.Disc1.{_k}", "type") for _k in dataset_vars}

        try:
            values = connector_export.get_values_all(
                dataset_identifier=DatasetInfoV0('MVP0_local_datasets_connector_export_test', "dataset_disc1"),
                data_types_dict=data_types_dict)
            self.assertEqual(values["a"], 1)
            self.assertEqual(values["x"], 4.0)
            self.assertEqual(values["b"], 2)
            self.assertEqual(values["name"], "A1")
            self.assertEqual(values["x_dict"], {"test1": 1, "test2": 2})
            self.assertTrue(np.array_equal(values["y_array"], np.array([1.0, 2.0, 3.0])))
            self.assertEqual(values["z_list"], [1.0, 2.0, 3.0])
            self.assertEqual(values["b_bool"], False)
            self.assertTrue((values["d"] == pd.DataFrame({"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())
            connector_export.clear(remove_root_directory=True)
        except Exception as cm:
            connector_export.clear(remove_root_directory=True)
            raise

    def test_14_test_import_parameter_level(self):

        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        # this
        mapping_repo_file_path = os.path.join(test_data_folder, "test_92_mapping_parameters_level_repository.json")
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        dm = study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {})
        self.assertEqual(len(dm.get_value("usecase_dataset.Disc1.d")), 0)
        self.assertEqual(dm.get_value("usecase_dataset.linearization_mode"), "auto")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.linearization_mode"), "auto")

        study.update_data_from_dataset_mapping(DatasetsMapping.from_json_file(mapping_repo_file_path))

        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 2)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), None)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {})
        self.assertEqual(len(dm.get_value("usecase_dataset.Disc1.d")), 2)

        # check numerical parameters
        self.assertEqual(dm.get_value("usecase_dataset.linearization_mode"), "auto")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.linearization_mode"), "auto")

    def test_15_test_export_parameter_level(self):
        """
        Some example to check repository datasets connector export
        """
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )

        # create usecase with data
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        mapping_repo_file_path = os.path.join(test_data_folder, "test_92_mapping_repository.json")
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        study.update_data_from_dataset_mapping(DatasetsMapping.from_json_file(mapping_repo_file_path))

        # export study in another folder to compare the datasets
        # create connector test for export
        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_test_export_param/",
            "create_if_not_exists": True
        }

        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_local_export_test_param",
                                                    connector_type=DatasetConnectorType.get_enum_value("Local"),
                                                    **connector_args)
        export_mapping_repo_file_path = os.path.join(test_data_folder, "test_92_export_mapping_param_level.json")
        try:
            study.export_data_from_dataset_mapping(DatasetsMapping.from_json_file(export_mapping_repo_file_path))

            dm = study.execution_engine.dm
            connector_export = DatasetsConnectorManager.get_connector('MVP0_local_export_test_param')

            dataset_vars = ["a",
                            "x",
                            "b",
                            "name",
                            "x_dict",
                            "y_array",
                            "z_list",
                            "b_bool",
                            "d"]

            data_types_dict = {_k: dm.get_data(f"usecase_dataset.Disc1.{_k}", "type") for _k in dataset_vars}
            values = connector_export.get_values_all(
                dataset_identifier=DatasetInfoV0('MVP0_local_export_test_param', "dataset_all_types"),
                data_types_dict=data_types_dict)

            self.assertEqual(values["x"], 4.0)
            self.assertEqual(values["b"], 1)
            self.assertTrue((values["d"] == pd.DataFrame({"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())
            connector_export.clear(remove_root_directory=True)
        except Exception as cm:
            connector_export.clear(remove_root_directory=True)
            raise

    def _test_16_bigquery_plain_types_export_import(self):
        """
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )
        connector_args = {
            "project_id": "gcp-businessplanet"
        }
        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_bigquery_connector_copy_test",
                                                    connector_type=DatasetConnectorType.get_enum_value("Bigquery"),
                                                    **connector_args)
        
        
        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_all_types.usecase_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)

        dm = study.execution_engine.dm
        connector_to = DatasetsConnectorManager.get_connector('MVP0_bigquery_connector_copy_test')
        connector_json = DatasetsConnectorManager.get_connector('MVP0_datasets_connector')

        dataset_vars = ["a",
                        "x",
                        "b",
                        "name",
                        "x_dict",
                        "y_array",
                        "z_list",
                        "b_bool",
                        "d"]

        data_types_dict = {_k: dm.get_data(f"usecase_dataset.Disc1.{_k}", "type") for _k in dataset_vars}

        connector_to.copy_dataset_from(connector_from=connector_json,
                                       dataset_identifier=DatasetInfoV0(connector_json, "dataset_all_types"),
                                       data_types_dict=data_types_dict,
                                       create_if_not_exists=True)

        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_bigquery_dataset_copy_test.json")))
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 1)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), 2)
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), "A1")
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1": 1, "test2": 2})
        self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0, 2.0, 3.0])))
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [1.0, 2.0, 3.0])
        self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), False)
        self.assertTrue((dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame(
            {"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

    def _test_17_bigquery_colum_name_characters_compatibility_on_dataframe_and_dict_tables(self):
        """
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )
        connector_args = {
            "project_id": "gcp-businessplanet"
        }
        DatasetsConnectorManager.register_connector(connector_identifier="MVP0_bigquery_connector_copy_test",
                                                    connector_type=DatasetConnectorType.get_enum_value("Bigquery"),
                                                    **connector_args)
        
        
        connector_to = DatasetsConnectorManager.get_connector('MVP0_bigquery_connector_copy_test')
        connector_from = DatasetsConnectorManager.get_connector('MVP0_local_datasets_connector')

        data_types_dict = {"WITNESS_gdp": "dataframe",
                           "dict_strange_keys": "dict"}

        connector_to.copy_dataset_from(connector_from=connector_from,
                                       dataset_identifier=DatasetInfoV0(connector_from, "dataset_df_bq"),
                                       data_types_dict=data_types_dict,
                                       create_if_not_exists=True)

        data_values = connector_to.get_values(DatasetInfoV0('MVP0_bigquery_connector_copy_test', "dataset_df_bq"),
                                              data_to_get=data_types_dict)

        data_name = "WITNESS_gdp"
        ref_df = pd.read_csv(os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                           "data", "local_datasets_db", "dataset_df_bq",
                                                           data_name + ".csv")))

        self.assertTrue((ref_df == data_values[data_name]).all().all())

        dict_strange_keys = data_values["dict_strange_keys"]
        dict_strange_keys_ref = {
            "key1 (?)": "whatever",
            "key2 #^/": "whatever"
        }
        self.assertEqual(dict_strange_keys_ref, dict_strange_keys)

    def test_19_json_V1_import(self):
        """
        Use a local connector to copy values from a JSON connector then load them in the study and check correctness,
        thus testing ability of LocalConnector to both write and load values.
        """
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        # this
        json_db_path = os.path.join(test_data_folder, "test_92_datasets_db_v1.json")
        connector_args = {
            "file_path": json_db_path

        }
        DatasetsConnectorManager.register_connector(connector_identifier="json_v1_connector",
                                                    connector_type=DatasetConnectorType.JSON_V1,
                                                    **connector_args)
        study = Study()
        dm = study.execution_engine.dm

        import_mapping_repo_file_path = os.path.join(test_data_folder, "test_92_datasets_mapping_v1.json")
        try:
            study.load_data(from_datasets_mapping=DatasetsMapping.from_json_file(import_mapping_repo_file_path))

            self.assertEqual(dm.get_value("usecase_dataset.Disc1.a"), 1)
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.x"), 4.0)
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.b"), 2)
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.name"), "A1")
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1": 1, "test2": 2})
            self.assertTrue(np.array_equal(dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0, 2.0, 3.0])))
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.z_list"), [1.0, 2.0, 3.0])
            self.assertEqual(dm.get_value("usecase_dataset.Disc1.b_bool"), False)
            self.assertTrue((dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame(
                {"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

        except Exception as cm:
            raise cm

    def test_19_json_V1_export(self):
        """
        Use a local connector to copy values from a JSON connector then load them in the study and check correctness,
        thus testing ability of LocalConnector to both write and load values.
        """
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        json_db_path = os.path.join(test_data_folder, "test_92_datasets_db_v1_test.json")
        connector_args = {
            "file_path": json_db_path
        }
        DatasetsConnectorManager.register_connector(connector_identifier="json_v1_connector",
                                                    connector_type=DatasetConnectorType.JSON_V1,
                                                    **connector_args)

        connector_export = DatasetsConnectorManager.get_connector("json_v1_connector")
        study = Study()
        study.load_data()  # load from mapping V0
        dm = study.execution_engine.dm

        dataset_vars_disc1 = [
            "a",
            "x",
            "b",
            "name",
            "x_dict",
            "y_array",
            "z_list",
            "b_bool",
            "d"]

        data_types_dict_disc1 = {_k: dm.get_data(f"usecase_dataset.Disc1.{_k}", "type") for _k in dataset_vars_disc1}

        dataset_vars = ["inner_mda_name", "max_mda_iter", "n_processes", "linear_solver_MDA_preconditioner"]
        data_types_dict = {_k: dm.get_data(f"usecase_dataset.{_k}", "type") for _k in dataset_vars}
        export_mapping_repo_file_path = os.path.join(test_data_folder, "test_92_datasets_mapping_v1.json")
        try:

            study.export_data_from_dataset_mapping(
                from_datasets_mapping=DatasetsMapping.from_json_file(export_mapping_repo_file_path))
            values = connector_export.get_values_all(
                dataset_identifier=DatasetInfoV1('connector_export', "dataset_all_types", "<study_ph>"),
                data_types_dict=data_types_dict)
            values.update(connector_export.get_values_all(
                dataset_identifier=DatasetInfoV1('connector_export', "dataset_all_types", "<study_ph>.Disc1"),
                data_types_dict=data_types_dict_disc1))
            self.assertEqual(values["a"], 1)
            self.assertEqual(values["x"], 4.0)
            self.assertEqual(values["b"], 2)
            self.assertEqual(values["name"], "A1")
            self.assertEqual(values["x_dict"], {"test1": 1, "test2": 2})
            self.assertTrue(np.array_equal(values["y_array"], np.array([1.0, 2.0, 3.0])))
            self.assertEqual(values["z_list"], [1.0, 2.0, 3.0])
            self.assertEqual(values["b_bool"], False)
            self.assertTrue((values["d"] == pd.DataFrame({"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

        except Exception as cm:
            raise cm
        connector_export.clear_dataset("dataset_all_types")

    def test_20_file_system_V1_export_import(self):
        """
        Use a local connector to copy values from a JSON connector then load them in the study and check correctness,
        thus testing ability of LocalConnector to both write and load values.
        """
        from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import (
            DatasetConnectorType,
        )
        from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import (
            DatasetsConnectorManager,
        )
        test_data_folder = os.path.join(os.path.dirname(__file__), "data")
        
        connector_local = DatasetsConnectorManager.get_connector('MVP0_local_datasets_connector')

        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_test_20_import_V1/",
            "create_if_not_exists": True
        }
        DatasetsConnectorManager.register_connector(connector_identifier="json_v1_connector",
                                                    connector_type=DatasetConnectorType.Local_V1,
                                                    **connector_args)

        connector_export = DatasetsConnectorManager.get_connector("json_v1_connector")
        study = Study()
        study.load_data()  # load from mapping V0
        dm = study.execution_engine.dm

        # prepare data_type dict
        dataset_vars_disc1 = [
            "a",
            "x",
            "b",
            "name",
            "x_dict",
            "y_array",
            "z_list",
            "b_bool",
            "d"]

        data_types_dict_disc1 = {_k: dm.get_data(f"usecase_dataset.Disc1.{_k}", "type") for _k in dataset_vars_disc1}

        dataset_vars = ["inner_mda_name", "max_mda_iter", "n_processes", "linear_solver_MDA_preconditioner"]
        data_types_dict = {_k: dm.get_data(f"usecase_dataset.{_k}", "type") for _k in dataset_vars}

        # export data with the file system V1 connector
        export_mapping_repo_file_path = os.path.join(test_data_folder, "test_92_datasets_mapping_v1.json")
        try:
            study.export_data_from_dataset_mapping(
                from_datasets_mapping=DatasetsMapping.from_json_file(export_mapping_repo_file_path))
            values = connector_export.get_values_all(
                dataset_identifier=DatasetInfoV1('connector_export', "dataset_all_types", "<study_ph>"),
                data_types_dict=data_types_dict)
            values.update(connector_export.get_values_all(
                dataset_identifier=DatasetInfoV1('connector_export', "dataset_all_types", "<study_ph>.Disc1"),
                data_types_dict=data_types_dict_disc1))
            self.assertEqual(values["a"], 1)
            self.assertEqual(values["x"], 4.0)
            self.assertEqual(values["b"], 2)
            self.assertEqual(values["name"], "A1")
            self.assertEqual(values["x_dict"], {"test1": 1, "test2": 2})
            self.assertTrue(np.array_equal(values["y_array"], np.array([1.0, 2.0, 3.0])))
            self.assertEqual(values["z_list"], [1.0, 2.0, 3.0])
            self.assertEqual(values["b_bool"], False)
            self.assertTrue((values["d"] == pd.DataFrame({"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

        except Exception as cm:
            connector_export.clear_dataset("dataset_all_types")
            raise cm

        # then import in an empty study
        # create empty study:
        empty_study = Study()
        empty_stdy_dm = empty_study.execution_engine.dm

        import_mapping_repo_file_path = os.path.join(test_data_folder, "test_92_datasets_mapping_v1.json")
        try:
            empty_study.load_data(from_datasets_mapping=DatasetsMapping.from_json_file(import_mapping_repo_file_path))

            self.assertEqual(empty_stdy_dm.get_value("usecase_dataset.Disc1.a"), 1)
            self.assertEqual(empty_stdy_dm.get_value("usecase_dataset.Disc1.x"), 4.0)
            self.assertEqual(empty_stdy_dm.get_value("usecase_dataset.Disc1.b"), 2)
            self.assertEqual(empty_stdy_dm.get_value("usecase_dataset.Disc1.name"), "A1")
            self.assertEqual(empty_stdy_dm.get_value("usecase_dataset.Disc1.x_dict"), {"test1": 1, "test2": 2})
            self.assertTrue(
                np.array_equal(empty_stdy_dm.get_value("usecase_dataset.Disc1.y_array"), np.array([1.0, 2.0, 3.0])))
            self.assertEqual(empty_stdy_dm.get_value("usecase_dataset.Disc1.z_list"), [1.0, 2.0, 3.0])
            self.assertEqual(empty_stdy_dm.get_value("usecase_dataset.Disc1.b_bool"), False)
            self.assertTrue((empty_stdy_dm.get_value("usecase_dataset.Disc1.d") == pd.DataFrame(
                {"years": [2023, 2024], "x": [1.0, 10.0]})).all().all())

        except Exception as cm:
            connector_export.clear_dataset("dataset_all_types")
            raise cm

        connector_export.clear(True)

    def test_21_datasets_local_connector_nested_types_V1(self):
        """
        for dataset V1
        Check correctness of loaded values after loading a handcrafted local directories' dataset, testing usage of
        LocalDatasetsConnector and FileSystemDatasetsSerializer pickle-based loading for the following nested types:
            - dict[str: DataFrame]
            - dict[str: dict[str: DataFrame]]
            - dict[str: dict[str: float]]  (THIS IS JSONIFIABLE)
            - array[str]
            - array[DataFrame]
            - design space with lists (DataFrame with string, list, and bool columns)
            - design space with arrays (DataFrame with string, array, list and bool columns)
        """

        usecase_file_path = sostrades_core.sos_processes.test.test_disc1_nested_types.usecase_local_dataset.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)
        dm = study.execution_engine.dm

        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_test_21_import_V1/",
            "create_if_not_exists": True
        }
        DatasetsConnectorManager.register_connector(connector_identifier="local_datasets_V1",
                                                    connector_type=DatasetConnectorType.Local_V1,
                                                    **connector_args)
        local_connector_v1 = DatasetsConnectorManager.get_connector("local_datasets_V1")

        study.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset.json")))
        dm_dict = {var: dm.get_value(f"usecase_local_dataset.Disc1.{var}") for var in self.nested_types_reference_dict}
        self.assertTrue(dict_are_equal(self.nested_types_reference_dict, dm_dict))

        # export in V1
        study.export_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset_V1.json")))
        # import again in empty study
        study_empty = StudyManager(file_path=usecase_file_path)
        dm_empty = study_empty.execution_engine.dm
        study_empty.update_data_from_dataset_mapping(
            DatasetsMapping.from_json_file(os.path.join(process_path, "usecase_local_dataset_V1.json")))
        dm_dict = {var: dm_empty.get_value(f"usecase_local_dataset.Disc1.{var}") for var in
                   self.nested_types_reference_dict}
        self.assertTrue(dict_are_equal(self.nested_types_reference_dict, dm_dict))

        local_connector_v1.clear(True)

    def test_22_compatibility_V0_V1(self):
        """
        check that there is an error when we try to copy a dataset from connector V0 to connector V1
        """

        connector_args = {
            "root_directory_path": "./sostrades_core/tests/data/local_test_22_import_V1/",
            "create_if_not_exists": True
        }
        DatasetsConnectorManager.register_connector(connector_identifier="local_datasets_V1",
                                                    connector_type=DatasetConnectorType.Local_V1,
                                                    **connector_args)
        
        
        connector_to = DatasetsConnectorManager.get_connector('local_datasets_V1')
        connector_from = DatasetsConnectorManager.get_connector('MVP0_local_datasets_connector')
        with self.assertRaises(DatasetGenericException):
            connector_to.copy_dataset_from(connector_from=connector_from,
                                           dataset_identifier=DatasetInfoV1(connector_from, "dataset_all_types",
                                                                            "test"),
                                           data_types_dict={},
                                           create_if_not_exists=True)
        connector_to.clear(True)


if __name__ == "__main__":
    cls = TestDatasets()
    cls.setUp()
    cls.test_07b_datasets_local_connector_with_several_namespace()
