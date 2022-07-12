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

from sostrades_core.sos_processes.processes_factory import SoSProcessFactory


class TestSoSProcessFactory(unittest.TestCase):
    """
    SoSProcessFactory test class
    """

    def setUp(self):
        '''
        set up tests
        '''
        # All this test are based on sostrades_core.sos_processes.test
        self.name = 'TestSoSFactory'
        self.repository_to_check = 'sostrades_core.sos_processes.test'
        self.raw_repository = 'sostrades_core.sos_processes'
        self.SoSPF = SoSProcessFactory([self.raw_repository], False)

    def test_01_instantiate_sosprocessfactory(self):
        '''
        Default initialisation test
        '''
        self.assertIsInstance(self.SoSPF, SoSProcessFactory,
                              "'{}' is not a SoSProcessFactory".format(self.SoSPF))

    def test_02_list_repos(self):
        '''
        List repos test
        '''
        self.assertIn(self.repository_to_check, self.SoSPF.get_repo_list())
        self.assertNotIn(self.raw_repository, self.SoSPF.get_repo_list())

    def test_03_list_test_modules(self):
        '''
        Check if process factory list available tests processes
        '''
        SoSPF_process_list = self.SoSPF.get_processes_id_list(
            self.repository_to_check)

        target_list = ['test_disc1', 'test_disc1_all_types', 'test_disc1_disc2_coupling',
                       'test_disc1_disc2_couplingdefault']

        for target in target_list:
            self.assertIn(target, SoSPF_process_list)
