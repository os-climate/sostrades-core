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
from gemseo.core.discipline import MDODiscipline
from sostrades_core.execution_engine.sos_mdo_discipline import SoSMDODiscipline

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for linearization mode propagation
"""

import unittest
from logging import Handler
import pandas as pd
from numpy import array, set_printoptions

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


class TestPropagatesLinearizationMode(unittest.TestCase):
    """
    Test de propagation of linearization mode to children disciplines
    """

    def setUp(self):
        self.study_name = 'linearization_mode_propagation'
        self.ns = f'{self.study_name}'
        self.sc_name = "SellarOptimScenario"
        self.c_name = "SellarCoupling"
        self.my_handler = UnitTestHandler()

        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}

        self.dspace = pd.DataFrame(dspace_dict)
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_discopt'

    def test_01_linearization_mode_children_propagation(self):
        print("\n Test 1 : Propagation of linearization mode to children disciplines")
        exec_eng = ExecutionEngine(self.study_name)

        exec_eng.logger.setLevel(10)
        exec_eng.logger.addHandler(self.my_handler)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.study_name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.study_name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        FINITE_DIFFERENCES = MDODiscipline.FINITE_DIFFERENCES
        LINEARIZATION_MODE = SoSMDODiscipline.LINEARIZATION_MODE

        disc_dict[f'{self.study_name}.{coupling_name}.{LINEARIZATION_MODE}'] = FINITE_DIFFERENCES
        exec_eng.load_study_from_input_dict(disc_dict)

        proxy_discs = exec_eng.root_process.proxy_disciplines[0].proxy_disciplines
        # the dm has the proper values
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_Problem.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_1.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_3.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{LINEARIZATION_MODE}'),
                         'auto')

        # the activation has been properly logged
        self.assertIn(f'Discipline Sellar_1 set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline Sellar_3 set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline {coupling_name} set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline {self.study_name} set to linearization mode auto', self.my_handler.msg_list)

        exec_eng.execute()
        self.assertEqual(
            exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.linearization_mode, "auto")
        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)

    def test_02_linearization_mode_children_propagation_from_root_process(self):
        print("\n Test 2 : Propagation of linearization mode to children disciplines from root process")
        exec_eng = ExecutionEngine(self.study_name)

        exec_eng.logger.setLevel(10)
        exec_eng.logger.addHandler(self.my_handler)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.study_name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.study_name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        FINITE_DIFFERENCES = MDODiscipline.FINITE_DIFFERENCES
        LINEARIZATION_MODE = SoSMDODiscipline.LINEARIZATION_MODE

        disc_dict[f'{self.study_name}.{LINEARIZATION_MODE}'] = FINITE_DIFFERENCES
        exec_eng.load_study_from_input_dict(disc_dict)

        proxy_discs = exec_eng.root_process.proxy_disciplines[0].proxy_disciplines
        # the dm has the proper values
        self.assertEqual(
            exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_Problem.{LINEARIZATION_MODE}'),
            FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_1.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_3.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)

        # the activation has been properly logged
        self.assertIn(f'Discipline Sellar_1 set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline Sellar_3 set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline {coupling_name} set to linearization mode {FINITE_DIFFERENCES}',
                      self.my_handler.msg_list)
        self.assertIn(f'Discipline {self.study_name} set to linearization mode {FINITE_DIFFERENCES}',
                      self.my_handler.msg_list)

        exec_eng.execute()
        self.assertEqual(
            exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.linearization_mode, FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)

    def test_03_linearization_mode_children_propagation_from_children_process(self):
        print("\n Test 3 : Propagation of linearization mode to children disciplines from children process")
        exec_eng = ExecutionEngine(self.study_name)

        exec_eng.logger.setLevel(10)
        exec_eng.logger.addHandler(self.my_handler)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.study_name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.study_name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        FINITE_DIFFERENCES = MDODiscipline.FINITE_DIFFERENCES
        LINEARIZATION_MODE = SoSMDODiscipline.LINEARIZATION_MODE

        disc_dict[f'{self.study_name}.{coupling_name}.Sellar_3.{LINEARIZATION_MODE}'] = FINITE_DIFFERENCES
        exec_eng.load_study_from_input_dict(disc_dict)

        proxy_discs = exec_eng.root_process.proxy_disciplines[0].proxy_disciplines
        # the dm has the proper values
        self.assertEqual(
            exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_Problem.{LINEARIZATION_MODE}'),
            "auto")
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_1.{LINEARIZATION_MODE}'),
                         "auto")
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_3.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{LINEARIZATION_MODE}'),
                         "auto")

        # the activation has been properly logged
        self.assertIn(f'Discipline Sellar_1 set to linearization mode {"auto"}', self.my_handler.msg_list)
        self.assertIn(f'Discipline Sellar_3 set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline {coupling_name} set to linearization mode {"auto"}',
                      self.my_handler.msg_list)
        self.assertIn(f'Discipline {self.study_name} set to linearization mode {"auto"}',
                      self.my_handler.msg_list)

        exec_eng.execute()
        self.assertEqual(
            exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.linearization_mode, "auto")
        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            "auto")
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            "auto")
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)

    def test_04_reconfigure_after_run(self):
        print("\n Test 4 : Propagation of linearization mode to children disciplines - revert to auto after run")
        # 1: Set to Finite difference
        exec_eng = ExecutionEngine(self.study_name)

        exec_eng.logger.setLevel(10)
        exec_eng.logger.addHandler(self.my_handler)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.study_name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.study_name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.study_name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        FINITE_DIFFERENCES = MDODiscipline.FINITE_DIFFERENCES
        LINEARIZATION_MODE = SoSMDODiscipline.LINEARIZATION_MODE

        disc_dict[f'{self.study_name}.{LINEARIZATION_MODE}'] = FINITE_DIFFERENCES
        exec_eng.load_study_from_input_dict(disc_dict)

        proxy_discs = exec_eng.root_process.proxy_disciplines[0].proxy_disciplines
        # the dm has the proper values
        self.assertEqual(
            exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_Problem.{LINEARIZATION_MODE}'),
            FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_1.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_3.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{LINEARIZATION_MODE}'),
                         FINITE_DIFFERENCES)

        # the activation has been properly logged
        self.assertIn(f'Discipline Sellar_1 set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline Sellar_3 set to linearization mode {FINITE_DIFFERENCES}', self.my_handler.msg_list)
        self.assertIn(f'Discipline {coupling_name} set to linearization mode {FINITE_DIFFERENCES}',
                      self.my_handler.msg_list)
        self.assertIn(f'Discipline {self.study_name} set to linearization mode {FINITE_DIFFERENCES}',
                      self.my_handler.msg_list)

        exec_eng.execute()
        self.assertEqual(
            exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.linearization_mode, FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            FINITE_DIFFERENCES)

        # 2 : revert to Auto
        # 1: Set to Finite difference


        disc_dict[f'{self.study_name}.{LINEARIZATION_MODE}'] = "auto"
        exec_eng.load_study_from_input_dict(disc_dict)

        proxy_discs = exec_eng.root_process.proxy_disciplines[0].proxy_disciplines
        # the dm has the proper values
        self.assertEqual(
            exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_Problem.{LINEARIZATION_MODE}'),
            "auto")
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_1.{LINEARIZATION_MODE}'),
                         "auto")
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{coupling_name}.Sellar_3.{LINEARIZATION_MODE}'),
                         "auto")
        self.assertEqual(exec_eng.dm.get_value(f'{self.study_name}.{LINEARIZATION_MODE}'),
                         "auto")

        # the activation has been properly logged
        self.assertIn(f'Discipline Sellar_1 set to linearization mode {"auto"}', self.my_handler.msg_list)
        self.assertIn(f'Discipline Sellar_3 set to linearization mode {"auto"}', self.my_handler.msg_list)
        self.assertIn(f'Discipline {coupling_name} set to linearization mode {"auto"}',
                      self.my_handler.msg_list)
        self.assertIn(f'Discipline {self.study_name} set to linearization mode {"auto"}',
                      self.my_handler.msg_list)

        exec_eng.execute()
        self.assertEqual(
            exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.linearization_mode, "auto")
        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            "auto")
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            "auto")
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs(LINEARIZATION_MODE),
            "auto")