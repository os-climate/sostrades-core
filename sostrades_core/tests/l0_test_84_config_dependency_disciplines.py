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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from logging import Handler

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestConfigDependencyDiscs(unittest.TestCase):
    """
    Tool building test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.study_name = 'MyCase'
        self.exec_eng = ExecutionEngine(self.study_name)
        self.factory = self.exec_eng.factory

        self.repo = 'sostrades_core.sos_processes.test'

    def test_01_add_disc_to_config_dependency_disciplines(self):
        self.repo = 'sostrades_core.sos_processes.test'

        self.exec_eng.select_root_process(self.repo,
                                          'test_disc1_disc2_coupling')

        disc1 = self.exec_eng.root_process.proxy_disciplines[0]
        disc2 = self.exec_eng.root_process.proxy_disciplines[1]

        disc1.add_disc_to_config_dependency_disciplines(disc2)

        self.assertEqual(disc1.config_dependency_disciplines, [disc2])
        self.assertEqual(disc2.config_dependent_disciplines, [disc1])

        with self.assertRaises(Exception) as cm:
            disc2.add_disc_to_config_dependency_disciplines(disc1)

        error_message = f'The discipline {disc1.get_disc_full_name()} has already {disc2.get_disc_full_name()} in its config_dependency_list, it is not possible to add the discipline in config_dependency_list of myself'

        self.assertEqual(str(cm.exception), error_message)

        with self.assertRaises(Exception) as cm:
            disc2.add_disc_to_config_dependency_disciplines(disc2)

        error_message = f'Not possible to add self in the config_dependency_list for disc : {disc2.get_disc_full_name()}'

        self.assertEqual(str(cm.exception), error_message)
        values_dict = {}
        values_dict[f'{self.study_name}.Disc1.a'] = 10.
        values_dict[f'{self.study_name}.Disc1.b'] = 20.
        values_dict[f'{self.study_name}.Disc2.power'] = 2
        values_dict[f'{self.study_name}.Disc2.constant'] = -10.
        values_dict[f'{self.study_name}.x'] = 3.
        self.exec_eng.load_study_from_input_dict(values_dict)

    def test_02_clean_disc_to_config_dependency_disciplines(self):
        self.repo = 'sostrades_core.sos_processes.test'

        self.exec_eng.select_root_process(self.repo,
                                          'test_disc1_disc2_coupling')
        disc1 = self.exec_eng.root_process.proxy_disciplines[0]
        disc2 = self.exec_eng.root_process.proxy_disciplines[1]

        disc1.add_disc_to_config_dependency_disciplines(disc2)

        self.exec_eng.root_process.clean_children([disc2])

        self.assertEqual(disc1.config_dependency_disciplines, [])


if '__main__' == __name__:
    cls = TestConfigDependencyDiscs()
    cls.setUp()
    cls.test_02_clean_disc_to_config_dependency_disciplines()
    cls.tearDown()
