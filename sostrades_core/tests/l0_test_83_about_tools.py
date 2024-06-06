'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

from sostrades_core.execution_engine.builder_tools.scatter_tool import ScatterTool
from sostrades_core.execution_engine.builder_tools.tool_builder import ToolBuilder
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestToolBuild(unittest.TestCase):
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

    def test_01_get_tool_from_toolfactory(self):
        scatter_name = 'scatter_name'
        scatter_tool_builder = self.exec_eng.tool_factory.create_tool_builder(
            scatter_name, 'ScatterTool')

        self.assertTrue(isinstance(scatter_tool_builder, ToolBuilder))
        self.assertEqual(
            scatter_tool_builder.sos_name, scatter_name)
        self.assertEqual(scatter_tool_builder.cls, ScatterTool)

    # def test_02_create_driver_with_tool(self):
    #     scatter_name = 'scatter_name'
    #     scatter_tool_builder = self.exec_eng.tool_factory.create_tool_builder(
    #         scatter_name, 'ScatterTool')
    #
    #     cls_builder = self.factory.get_builder_from_process(repo=self.repo,
    #                                                         mod_id='test_disc1_scenario')
    #
    #     multi_scenarios = self.factory.create_driver_with_tool(
    #         'driver', cls_builder, scatter_tool_builder)
    #
    #     self.assertEqual(multi_scenarios[0].cls, ProxyDriverEvaluator)
    #     self.assertEqual(
    #         multi_scenarios[0].args['builder_tool'], scatter_tool_builder)


if '__main__' == __name__:
    cls = TestToolBuild()
    cls.setUp()
    cls.test_01_get_tool_from_toolfactory()
    cls.tearDown()
