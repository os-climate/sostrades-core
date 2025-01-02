'''
Copyright 2025 Capgemini

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
from logging import Handler

from gemseo.mda.base_mda import BaseMDA
from gemseo.scenarios.base_scenario import BaseScenario

from sostrades_core.sos_processes.test.test_script_gemseo.usecase import Study as StudyGEMSEOMDO
from sostrades_core.sos_processes.test.test_script_gemseo_mda.usecase import Study as StudyGEMSEOMDA


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)

class TestResidualVariables(unittest.TestCase):
    """
    Discipline to test the new function run_solves_residuals and residuals variable of Gemseo
    """

    def setUp(self):
        # Set logging level to debug for residual_vars
        self.my_handler = UnitTestHandler()

    def test_01_gemseo_mdoscenario_in_sostrades(self):

        study = StudyGEMSEOMDO()

        study.load_data()
        #study.ee.logger.setLevel(10)
        base_scenario_logger = logging.getLogger('gemseo.scenarios.base_scenario')
        base_scenario_logger.addHandler(self.my_handler)
        scenario=study.ee.root_process.cls_builder
        self.assertTrue(isinstance(scenario,BaseScenario))
        self.assertTrue(study.ee.wrapping_mode == 'GEMSEO')
        study.run(for_test=True)

        self.assertIn('*** Start %s execution ***', self.my_handler.msg_list)
        self.assertIn('*** End %s execution (time: %s) ***', self.my_handler.msg_list)

    def test_02_gemseo_mda_in_sostrades(self):

        study = StudyGEMSEOMDA()

        study.load_data()
        mda = study.ee.root_process.cls_builder
        self.assertTrue(isinstance(mda,BaseMDA))
        self.assertTrue(study.ee.wrapping_mode == 'GEMSEO')
        study.run(for_test=True)

    def test_03_dump_gemseo_script_study(self):
        study = StudyGEMSEOMDA()

        study.load_data()
        mda = study.ee.root_process.cls_builder
        self.assertTrue(isinstance(mda, BaseMDA))
        self.assertTrue(study.ee.wrapping_mode == 'GEMSEO')
        study.run(dump_study=True)
