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
import unittest
from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
from tempfile import gettempdir
from sos_trades_core.execution_engine.data_manager import DataManager
from os.path import join, dirname
from sos_trades_core.tools.rw.load_dump_dm_data import CryptedLoadDump
from sos_trades_core.tests.data import __file__ as data_folder
from pathlib import Path
from time import sleep
from shutil import rmtree


class TestStudyManager(unittest.TestCase):
    """
    BaseStudyManager test class
    """

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.__repository = 'sos_trades_core.sos_processes.test'
        self.__process = 'test_disc1_disc2_coupling'
        self.__study_name = 'TestStudyManager'
        self.__temp_dir = gettempdir()
        self.__dump_dir = join(self.__temp_dir, self.__study_name)

        self.__study_data_values = {
            f'{self.__study_name}.x': 10.,
            f'{self.__study_name}.Disc1.a': 5.,
            f'{self.__study_name}.Disc1.b': 25431.,
            f'{self.__study_name}.y': 4.,
            f'{self.__study_name}.Disc2.constant': 3.1416,
            f'{self.__study_name}.Disc2.power': 2}

        self.__rsa_private_key_file = join(
            dirname(data_folder), 'private_key.pem')
        self.__rsa_public_key_file = join(
            dirname(data_folder), 'public_key.pem')

    def tearDown(self):

        if Path(self.__dump_dir).is_dir():
            rmtree(self.__dump_dir)
        sleep(0.5)

    def test_01_Create_Study(self):
        """ Check only basique execution engine setp from study manager
        """

        study = BaseStudyManager(
            self.__repository, self.__process, self.__study_name)

        self.assertIsNotNone(study, 'Study has not been initialized')

        self.assertTrue(len(study.execution_engine.dm.data_dict)
                        > 0, 'Missing data into the data manager')

        self.assertTrue(len(study.execution_engine.dm.disciplines_dict)
                        > 0, 'Missing disciplines into the data manager')

    def test_02_Load_Data_Into_Study(self):
        """ Check that study manager correctly load data
        """
        study = BaseStudyManager(
            self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)

        for key, value in self.__study_data_values.items():
            var_id = study.execution_engine.dm.get_data_id(key)
            self.assertEqual(
                study.execution_engine.dm.data_dict[var_id][DataManager.VALUE], value)

    def test_03_Dump_And_Load_Into_Study_With_Same_Name(self):
        """ Check that load and dump on file function are working when source 
        and destination study have the same process and the same name
        """
        study = BaseStudyManager(
            self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)
        study.dump_data(self.__dump_dir)

        study_bis = BaseStudyManager(
            self.__repository, self.__process, self.__study_name)

        study_bis.load_data(self.__dump_dir)

        for key in self.__study_data_values.keys():
            var_id = study.execution_engine.dm.get_data_id(key)
            var_id_bis = study_bis.execution_engine.dm.get_data_id(key)
            self.assertEqual(
                study.execution_engine.dm.data_dict[var_id][DataManager.VALUE], study_bis.execution_engine.dm.data_dict[var_id_bis][DataManager.VALUE])

    def test_04_Dump_And_Load_Into_Study_With_Different_Name(self):
        """ Check that load and dump on file function are working when source 
        and destination study have the same process but not the same name
        (verify that pickle dump are correctly save without study information)
        """
        study = BaseStudyManager(
            self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)
        study.dump_data(self.__dump_dir)

        study_bis_name = f'{self.__study_name}_bis'
        study_bis = BaseStudyManager(
            self.__repository, self.__process, study_bis_name)

        study_bis.load_data(self.__dump_dir)

        for key in self.__study_data_values.keys():
            var_id = study.execution_engine.dm.get_data_id(key)
            var_id_bis = study_bis.execution_engine.dm.get_data_id(
                key.replace(self.__study_name, study_bis_name))
            self.assertEqual(
                study.execution_engine.dm.data_dict[var_id][DataManager.VALUE], study_bis.execution_engine.dm.data_dict[var_id_bis][DataManager.VALUE])

    def test_05_Dump_And_Load_Into_Study_With_Encryption(self):
        """ All the previous test check that basic writing strategy is working
        This test check that encryption strategy is working
        """
        study = BaseStudyManager(
            self.__repository, self.__process, self.__study_name)

        study.load_data(from_input_dict=self.__study_data_values)

        # Change strategy from Direct(default) to encrypted strategy
        rw_strategy = CryptedLoadDump(private_key_file=self.__rsa_private_key_file,
                                      public_key_file=self.__rsa_public_key_file)

        study.rw_strategy = rw_strategy

        study.dump_data(self.__dump_dir)

        study_bis_name = f'{self.__study_name}_bis'
        study_bis = BaseStudyManager(
            self.__repository, self.__process, study_bis_name)

        # -- disable this test part because returned exception change depending of the platform running the test

        # The study_bis is initialized with default strategy (Direct),
        # make sure there no way to load previously saved data
        # with self.assertRaises(LoadDumpException):
        # study_bis.load_data(self.__dump_dir)

        # Update the strategy and try to laod data and then make comparison
        # with the source one
        study_bis.rw_strategy = rw_strategy
        study_bis.load_data(self.__dump_dir)

        for key in self.__study_data_values.keys():
            var_id = study.execution_engine.dm.get_data_id(key)
            var_id_bis = study_bis.execution_engine.dm.get_data_id(
                key.replace(self.__study_name, study_bis_name))
            self.assertEqual(study.execution_engine.dm.data_dict[var_id][DataManager.VALUE],
                             study_bis.execution_engine.dm.data_dict[var_id_bis][DataManager.VALUE])
