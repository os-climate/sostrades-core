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
import logging
from pathlib import Path
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.study_manager.study_manager import StudyManager
from os.path import join

class UnitTestHandler(logging.Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        super().__init__()
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestDatasets(unittest.TestCase):
    """
    Discipline to test datasets
    """

    def setUp(self):
        self.repo = 'sostrades_core.sos_processes.test'
        self.study_name = 'dataset_test'
        self.proc_name = "test_disc1_disc2_dataset"
        self.process_path = join(Path(__file__).parents[1], "sos_processes","test", self.proc_name)
        self.study = StudyManager(self.repo, self.proc_name, self.study_name)
        

    def test_01_usecase1(self):
        dm = self.study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value('dataset_test.a'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc1VirtualNode.x'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc2VirtualNode.x'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc1.b'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc2.b'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc1.c'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc2.c'), None)

        self.study.load_study(join(self.process_path,"usecase_dataset.json"))

        
        self.assertEqual(dm.get_value('dataset_test.a'), 1)
        self.assertEqual(dm.get_value('dataset_test.Disc1VirtualNode.x'), 4)
        self.assertEqual(dm.get_value('dataset_test.Disc2VirtualNode.x'), 4)
        self.assertEqual(dm.get_value('dataset_test.Disc1.b'), "2")
        self.assertEqual(dm.get_value('dataset_test.Disc2.b'), "2")
        self.assertEqual(dm.get_value('dataset_test.Disc1.c'), "3")
        self.assertEqual(dm.get_value('dataset_test.Disc2.c'), "3")

    def test_02_usecase2(self):
        dm = self.study.execution_engine.dm
        # assert data are empty
        self.assertEqual(dm.get_value('dataset_test.a'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc1VirtualNode.x'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc2VirtualNode.x'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc1.b'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc2.b'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc1.c'), None)
        self.assertEqual(dm.get_value('dataset_test.Disc2.c'), None)

        self.study.load_study(join(self.process_path,"usecase_2datasets.json"))

        
        self.assertEqual(dm.get_value('dataset_test.a'), 10)
        self.assertEqual(dm.get_value('dataset_test.Disc1VirtualNode.x'), 20)
        self.assertEqual(dm.get_value('dataset_test.Disc2VirtualNode.x'), 20)
        self.assertEqual(dm.get_value('dataset_test.Disc1.b'), "1")
        self.assertEqual(dm.get_value('dataset_test.Disc2.b'), "1")
        self.assertEqual(dm.get_value('dataset_test.Disc1.c'), "2")
        self.assertEqual(dm.get_value('dataset_test.Disc2.c'), "2")