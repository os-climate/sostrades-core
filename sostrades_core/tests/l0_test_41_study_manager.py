'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16-2024/06/10 Copyright 2024 Capgemini

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

from __future__ import annotations

import math
import unittest
from importlib.resources import files
from pathlib import Path
from tempfile import gettempdir

import pytest

from sostrades_core.execution_engine.data_manager import DataManager
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tests import data
from sostrades_core.tools.folder_operations import rmtree_safe
from sostrades_core.tools.rw.load_dump_dm_data import CryptedLoadDump


class TestStudyManager(unittest.TestCase):
    """BaseStudyManager test class"""

    def setUp(self):  # noqa: D102
        unittest.TestCase.setUp(self)
        self.__repository = "sostrades_core.sos_processes.test"
        self.__process = "test_disc1_disc2_coupling"
        self.__study_name = "TestStudyManager"
        self.__temp_dir = Path(gettempdir())
        self.__dump_dir = self.__temp_dir / self.__study_name

        self.__study_data_values = {
            f"{self.__study_name}.x": 10.0,
            f"{self.__study_name}.Disc1.a": 5.0,
            f"{self.__study_name}.Disc1.b": 25431.0,
            f"{self.__study_name}.y": 4.0,
            f"{self.__study_name}.Disc2.constant": math.pi,
            f"{self.__study_name}.Disc2.power": 2,
        }

        data_folder = files(data)
        self.__rsa_private_key_file = data_folder / "private_key.pem"
        self.__rsa_public_key_file = data_folder / "public_key.pem"

    def tearDown(self):  # noqa: D102
        if Path(self.__dump_dir).is_dir():
            rmtree_safe(self.__dump_dir)

    def test_01_create_study(self):
        """Check only basique execution engine setp from study manager."""
        study = BaseStudyManager(self.__repository, self.__process, self.__study_name)

        assert study is not None, "Study has not been initialized"

        assert len(study.execution_engine.dm.data_dict) > 0, "Missing data into the data manager"

        assert len(study.execution_engine.dm.disciplines_dict) > 0, "Missing disciplines into the data manager"

    def test_02_load_data_into_study(self):
        """Check that study manager correctly load data."""
        study = BaseStudyManager(self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)

        for key, value in self.__study_data_values.items():
            var_id = study.execution_engine.dm.get_data_id(key)
            assert study.execution_engine.dm.data_dict[var_id][DataManager.VALUE] == value

    def test_03_dump_and_load_into_study_with_same_name(self):
        """
        Check that load and dump on file function are working when source
        and destination study have the same process and the same name
        """
        study = BaseStudyManager(self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)
        study.dump_data(self.__dump_dir)

        study_bis = BaseStudyManager(self.__repository, self.__process, self.__study_name)

        study_bis.load_data(self.__dump_dir)

        for key in self.__study_data_values:
            var_id = study.execution_engine.dm.get_data_id(key)
            var_id_bis = study_bis.execution_engine.dm.get_data_id(key)
            assert (
                study.execution_engine.dm.data_dict[var_id][DataManager.VALUE]
                == study_bis.execution_engine.dm.data_dict[var_id_bis][DataManager.VALUE]
            )

    def test_04_dump_and_load_into_study_with_different_name(self):
        """
        Check that load and dump on file function are working when source
        and destination study have the same process but not the same name
        (verify that pickle dump are correctly save without study information)
        """
        study = BaseStudyManager(self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)
        study.dump_data(self.__dump_dir)

        study_bis_name = f"{self.__study_name}_bis"
        study_bis = BaseStudyManager(self.__repository, self.__process, study_bis_name)

        study_bis.load_data(self.__dump_dir)

        for key in self.__study_data_values:
            var_id = study.execution_engine.dm.get_data_id(key)
            var_id_bis = study_bis.execution_engine.dm.get_data_id(key.replace(self.__study_name, study_bis_name))
            assert (
                study.execution_engine.dm.data_dict[var_id][DataManager.VALUE]
                == study_bis.execution_engine.dm.data_dict[var_id_bis][DataManager.VALUE]
            )

    def test_05_dump_and_load_into_study_with_encryption(self):
        """
        All the previous test check that basic writing strategy is working.
        This test check that encryption strategy is working.
        """
        study = BaseStudyManager(self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)

        # Change strategy from Direct(default) to encrypted strategy
        rw_strategy = CryptedLoadDump(
            private_key_file=self.__rsa_private_key_file, public_key_file=self.__rsa_public_key_file
        )

        study.rw_strategy = rw_strategy

        study.dump_data(self.__dump_dir)

        study_bis_name = f"{self.__study_name}_bis"
        study_bis = BaseStudyManager(self.__repository, self.__process, study_bis_name)

        # -- disable this test part because returned exception change depending of the platform running the test

        # The study_bis is initialized with default strategy (Direct),
        # make sure there no way to load previously saved data
        # with self.assertRaises(LoadDumpException):
        # study_bis.load_data(self.__dump_dir)

        # Update the strategy and try to laod data and then make comparison
        # with the source one
        study_bis.rw_strategy = rw_strategy
        study_bis.load_data(self.__dump_dir)

        for key in self.__study_data_values:
            var_id = study.execution_engine.dm.get_data_id(key)
            var_id_bis = study_bis.execution_engine.dm.get_data_id(key.replace(self.__study_name, study_bis_name))
            assert (
                study.execution_engine.dm.data_dict[var_id][DataManager.VALUE]
                == study_bis.execution_engine.dm.data_dict[var_id_bis][DataManager.VALUE]
            ), f"error for parameter {key}"

    def test_06_merge_design_spaces(self):
        """Check the merging of design spaces."""
        study = StudyManager("", run_usecase=False)
        var1 = {
            "value": [0],
            "lower_bnd": [-5],
            "upper_bnd": [5],
            "activated_elem": [True],
            "enable_variable": True,
        }
        study.update_dspace_dict_with(name="var1", **var1)
        ds2 = {
            "var2": {
                "value": [0, 1],
                "lower_bnd": [-5, -10],
                "upper_bnd": [5, 10],
                "activated_elem": [True, True],
                "enable_variable": True,
            },
            "dspace_size": 2,
        }
        study.merge_design_spaces([ds2])
        assert study.dspace == {"var1": var1, "var2": ds2["var2"], "dspace_size": 3}

    def test_07_test_merge_design_spaces_fail(self):
        """Check that merging two design spaces with the same variable correctly raises an exception."""
        study = StudyManager("", run_usecase=False)
        var1 = {
            "value": [0],
            "lower_bnd": [-5],
            "upper_bnd": [5],
            "activated_elem": [True],
            "enable_variable": True,
        }
        study.update_dspace_dict_with(name="var1", **var1)
        ds2 = {
            "var1": {
                "value": [0, 1],
                "lower_bnd": [-5, -10],
                "upper_bnd": [5, 10],
                "activated_elem": [True, True],
                "enable_variable": True,
            },
            "dspace_size": 2,
        }
        with pytest.raises(expected_exception=ValueError, match=r"Failed to merge the design spaces;.* var1"):
            study.merge_design_spaces([ds2])
