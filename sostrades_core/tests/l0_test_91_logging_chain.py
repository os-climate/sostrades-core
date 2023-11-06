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
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class UnitTestHandler(logging.Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        super().__init__()
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestLoggingDiscipline(SoSWrapp):
    """
    Discipline to test logging
    """
    DISC_LOGGING_MESSAGE = "Testing logger Discipline"

    def run(self):
        self.logger.info(TestLoggingDiscipline.DISC_LOGGING_MESSAGE)


class TestLoggers(unittest.TestCase):
    """
    Test de chaine de logs
    """

    def setUp(self):
        self.name = 'EETestLoggers'
        self.model_name = 'test_'
        self.ee = ExecutionEngine(self.name, logger=logging.getLogger("TestLoggerName"))
        self.my_handler = UnitTestHandler()
        self.ee.logger.addHandler(self.my_handler)
        self.ee.logger.setLevel(logging.INFO)

    def test_01_loggers_chain_sos_wrap(self):
        """
        Tests if a log made in self.logger of a SoSWrap is correctly linked to execution engine
        """
        self.ee.ns_manager.add_ns_def({})
        mod_path = 'sostrades_core.tests.l0_test_91_logging_chain.TestLoggingDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)
        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.load_study_from_input_dict({})
        self.ee.display_treeview_nodes()
        self.ee.execute()
        self.assertIn(TestLoggingDiscipline.DISC_LOGGING_MESSAGE, self.my_handler.msg_list,
                      "Discipline logging message was not found at execution engine level logger. Logging link is broken")
