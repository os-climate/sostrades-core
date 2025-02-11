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

import sostrades_core.sos_processes.test.test_residual_variables.usecase_test_residual_variables
from sostrades_core.study_manager.study_manager import StudyManager


class TestResidualVariables(unittest.TestCase):
    """Discipline to test the new function run_solves_residuals and residuals variable of Gemseo"""

    def setUp(self):
        # Set logging level to debug for residual_vars
        logging.getLogger("sostrades_core.residual_vars").setLevel(logging.DEBUG)

    def _test_01_residual_variable_config(self):
        usecase_file_path = sostrades_core.sos_processes.test.test_residual_variables.usecase_test_residual_variables.__file__
        process_path = os.path.dirname(usecase_file_path)
        study = StudyManager(file_path=usecase_file_path)

        study.setup_usecase()
        study.run()

        study.ee.dm.get_value()
        for proxy in study.ee.root_process.proxy_disciplines:
            self.assertTrue(proxy.discipline_wrapp.discipline.run_solves_residuals)
