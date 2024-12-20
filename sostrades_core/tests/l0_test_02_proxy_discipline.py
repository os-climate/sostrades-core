'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/05/16 Copyright 2023 Capgemini

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

import logging
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.sos_wrapping.test_discs.disc1_all_types import Disc1
from sostrades_core.tools.compare_data_manager_tooling import dict_are_equal


class TestProxyDiscipline(unittest.TestCase):
    """ProxyDiscipline test class"""

    def setUp(self):
        """Initialize third data needed for testing"""
        self.name = 'SoSDisc'
        self.ee = ExecutionEngine('Test')
        self.ns_test = 'Test'
        base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1_all_types.Disc1'
        self.mod1_ns_path = f'{base_path}.disc1.Disc1'
        self.mod1_initexec_path = f'{base_path}.disc1_init_execution.Disc1InitExec'
        self.mod2_path = f'{base_path}.disc2.Disc2'
        self.mod8_path = f'{base_path}.disc8.Disc8'

    def test_01_instantiate_sos_wrapp(self):
        """Default initialisation test"""
        sosdisc_instance = Disc1(self.name, logger=logging.getLogger(__name__))
        assert isinstance(sosdisc_instance, SoSWrapp), f"'{sosdisc_instance}' is not a SoSWrapp"

    def test_02_check_io_data(self):
        """Check selection of coupling variables"""
        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)
        self.ee.configure()

        disc1 = self.ee.root_process.proxy_disciplines[0]
        data_names_in = disc1.get_input_data_names()
        data_names_out = disc1.get_output_data_names()
        for full_input_name in [f'{self.ee.study_name}.Disc1.{input_name}' for input_name in Disc1.DESC_IN]:
            assert full_input_name in data_names_in
        for full_output_name in [f'{self.ee.study_name}.Disc1.{output_name}' for output_name in Disc1.DESC_OUT]:
            assert full_output_name in data_names_out

    def test_03_load_input_data_values(self):
        """Check loading of local data and configure"""
        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)
        self.ee.configure()

        dict_values = {'Test.Disc1.x': 1, 'Test.Disc1.a': 2, 'Test.Disc1.b': 5, 'Test.Disc1.name': 'A1'}
        self.ee.load_study_from_input_dict(dict_values)

        self.ee.display_treeview_nodes()
        assert len(self.ee.root_process.proxy_disciplines) == 1
        assert self.ee.root_process.is_configured()
        assert self.ee.root_process.status == 'CONFIGURE'
        assert self.ee.root_process.proxy_disciplines[0].status == 'CONFIGURE'

    def test_04_execution_success(self):
        """Check discipline execution"""
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x, self.name + '.Disc1.a': a, self.name + '.Disc1.b': b}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()
        self.ee.execute()

        # check status DONE after execution
        for disc_id in self.ee.dm.disciplines_dict:
            assert self.ee.dm.get_discipline(disc_id).status == 'DONE'

        self.ee.execute()

        # check status DONE after execution
        for disc_id in self.ee.dm.disciplines_dict:
            assert self.ee.dm.get_discipline(disc_id).status == 'DONE'

        y = self.ee.dm.get_value(self.name + '.y')
        assert y == a * x + b

    def test_05_execution_failure(self):
        """Check discipline execution failure when no coupling values"""
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        values_dict = {self.ns_test + '.Disc1.a': 1.0, self.ns_test + '.Disc1.b': 3.0}

        self.ee.load_study_from_input_dict(values_dict)

        # exception raised by check_inputs function: Test.x is not set
        self.assertRaises(ValueError, self.ee.execute)

    def test_06_namespace_appliance(self):
        """Check namespace appliance"""
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)
        self.ee.configure()

        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.x': x, self.ns_test + '.Disc1.a': a, self.ns_test + '.Disc1.b': b}

        self.ee.load_study_from_input_dict(values_dict)

        # check namespaced input/output
        disc1 = self.ee.root_process.proxy_disciplines[0]
        assert disc1.get_var_full_name('x', disc1.get_data_in()) == 'Test.x'
        assert disc1.get_var_full_name('y', disc1.get_data_out()) == 'Test.y'

        # check values in dm
        assert self.ee.dm.get_value('Test.Disc1.a') == a
        assert self.ee.dm.get_value('Test.Disc1.b') == b
        assert self.ee.dm.get_value('Test.x') == x

        self.ee.execute()

        assert self.ee.dm.get_value(self.ns_test + '.x') == values_dict[self.ns_test + '.x']

    def test_07_get_sos_io_asdict(self):
        """Check discipline namespace update"""
        ee = ExecutionEngine('Test')

        ns_dict = {'ns_ac': self.ns_test}
        ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        ee.factory.set_builders_to_coupling_builder(disc1_builder)
        ee.configure()

        values_dict = {}
        values_dict[self.ns_test + '.Disc1.a'] = 10.0
        values_dict[self.ns_test + '.Disc1.b'] = 20.0
        values_dict[self.ns_test + '.x'] = 10.0

        ee.load_study_from_input_dict(values_dict)

        # get inputs and compare to reference
        disc1 = ee.root_process.proxy_disciplines[0]
        inp_dict = disc1.get_sosdisc_inputs(['a', 'b'], in_dict=True)
        ref_inp = {'a': 10.0, 'b': 20.0}
        self.assertDictEqual(ref_inp, inp_dict, 'error in input dict')
        ee.execute()

        # get outputs and compare to reference
        out_dict = disc1.get_sosdisc_outputs(['indicator', 'y'], in_dict=True)
        ref_out = {'indicator': 200.0, 'y': 120.0}
        self.assertDictEqual(ref_out, out_dict, 'error in input dict')

    def test_08_get_sos_io_no_inputs(self):
        """Check discipline namespace update"""
        ee = ExecutionEngine('Test')
        ns_dict = {'ns_ac': self.ns_test}
        ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        ee.factory.set_builders_to_coupling_builder(disc1_builder)

        ee.configure()

        values_dict = {}
        values_dict[self.ns_test + '.Disc1.a'] = 10.0
        values_dict[self.ns_test + '.Disc1.b'] = 20.0
        values_dict[self.ns_test + '.x'] = 10.0

        ee.load_study_from_input_dict(values_dict)

        # get inputs and compare to reference
        disc1 = ee.root_process.proxy_disciplines[0]
        inp_dict = disc1.get_sosdisc_inputs()
        ref_inp = {'x': 10.0, 'a': 10.0, 'b': 20.0}

        for key, value in ref_inp.items():
            assert inp_dict[key] == value, 'error in input dict'
        ee.execute()

        # get outputs and compare to reference
        out_dict = disc1.get_sosdisc_outputs()
        ref_out = {'indicator': 200.0, 'y': 120.0}
        self.assertDictEqual(ref_out, out_dict, 'error in input dict')

    def test_09_check_factory_with_1_added_disc(self):
        """
        check if the root of the factory is the discipline if only 1 disc is added
        and a coupling if two are added
        """
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.configure()
        priv_in_values = {self.ns_test + '.x': 99.0, self.ns_test + '.Disc1.a': 1, self.ns_test + '.Disc1.b': 3}

        self.ee.load_study_from_input_dict(priv_in_values)

        self.ee.execute()

        assert isinstance(
            self.ee.root_process, ProxyDiscipline
        ), 'The root of the factory must be a ProxyDiscipline because only one disc has been added'

        # Now we try with two disciplines
        ee2 = ExecutionEngine('Test2')

        ns_dict = {'ns_ac': 'Test2'}
        ee2.ns_manager.add_ns_def(ns_dict)

        disc1_builder = ee2.factory.get_builder_from_module('Disc1', self.mod1_path)
        disc2_builder = ee2.factory.get_builder_from_module('Disc2', self.mod2_path)
        ee2.factory.set_builders_to_coupling_builder([disc1_builder, disc2_builder])
        ee2.configure()

        priv_in_values = {
            'Test2.Disc1.x': 9.0,
            'Test2.Disc1.a': 1,
            'Test2.Disc1.b': 2,
            'Test2.y': 10.0,
            'Test2.Disc2.constant': 4.0,
            'Test2.Disc2.power': 2,
            'Test2.Disc1.name': 'A1',
        }
        ee2.load_study_from_input_dict(priv_in_values)

        ee2.execute()

        assert isinstance(
            ee2.root_process, ProxyCoupling
        ), 'The root of the factory must be a SoSDiscipline because only one disc has been added'

    def test_10_check_overwrite_of_default_values(self):
        """Check defaults for public"""
        ee = ExecutionEngine('Test')
        ns_dict = {'ns_protected': self.ns_test}
        ee.ns_manager.add_ns_def(ns_dict)

        disc8_builder = ee.factory.get_builder_from_module('Disc8', self.mod8_path)
        ee.factory.set_builders_to_coupling_builder(disc8_builder)

        ee.configure()

        values_dict = {}
        values_dict[self.ns_test + '.Disc8.a'] = 10.0
        # default value for 'b' is 2
        values_dict[self.ns_test + '.Disc8.b'] = 20.0
        values_dict[self.ns_test + '.x'] = 10.0

        ee.load_study_from_input_dict(values_dict)

        # get inputs and compare to reference
        disc8 = ee.root_process.proxy_disciplines[0]

        inp_dict = disc8.get_sosdisc_inputs(in_dict=True)
        ref_inp = {'x': 10.0, 'a': 10.0, 'b': 20.0}

        for key, value in ref_inp.items():
            assert inp_dict[key] == value, 'error in input dict'
        ee.execute()

        # get outputs and compare to reference
        out_dict = disc8.get_sosdisc_outputs()
        ref_out = {'indicator': 200.0, 'y': 120.0}
        self.assertDictEqual(ref_out, out_dict, 'error in input dict')

    def test_11_post_processing(self):
        """Check discipline post-processing"""
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x, self.name + '.Disc1.a': a, self.name + '.Disc1.b': b}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()
        self.ee.execute()

        # get post-processing of disc1
        disc1 = self.ee.dm.get_disciplines_with_name('Test.Disc1')[0]
        filter = disc1.get_chart_filter_list()
        graph_list = disc1.get_post_processing_list(filter)
        # graph_list[0].to_plotly().show()

        # test post-processing worked
        assert len(graph_list[0].series[0].abscissa) == 1
        assert graph_list[0].series[0].abscissa[0] == self.ee.dm.get_value('Test.x')
        assert graph_list[0].series[0].ordinate[0] == self.ee.dm.get_value('Test.y')

    def test_12_execution_success_of_discipline_alone(self):
        """
        check discipline execution only of the SoSDiscipline i.e. without executing the root process
        useful test for devs that need to start at (proxy) discipline level and propagate to all other (proxy) classes
        """
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module('Disc1', self.mod1_ns_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x, self.name + '.Disc1.a': a, self.name + '.Disc1.b': b}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()

        self.ee.prepare_execution()
        self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline.validate_input_data = False
        local_data = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline.execute(values_dict)
        ref_local_data = {
            'Test.x': 1.0,
            'Test.Disc1.a': 1.0,
            'Test.Disc1.b': 2.0,
            'Test.Disc1.indicator': 2.0,
            'Test.y': 3.0,
        }
        assert dict_are_equal(local_data, ref_local_data)

    def test_13_retrieve_inputs_at_init_execution(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module('Disc1InitExec', self.mod1_initexec_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {
            self.name + '.x': x,
            self.name + '.Disc1InitExec.a': a,
            self.name + '.Disc1InitExec.b': b,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()

        self.ee.prepare_execution()

        aplus_b = self.ee.factory.proxy_disciplines[0].discipline_wrapp.discipline.sos_wrapp.aplusb
        assert a + b == aplus_b


if __name__ == '__main__':
    testcls = TestProxyDiscipline()
    testcls.setUp()
    testcls.test_12_execution_success_of_discipline_alone()
