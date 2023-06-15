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
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

import numpy as np
from numpy import array

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.sos_processes.test.test_sellar_coupling.usecase import Study as study_sellar_coupling

from logging import Handler, DEBUG

class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)

class TestMDALoop(unittest.TestCase):
    """
    MDA test class
    """

    def setUp(self):
        self.dirs_to_del = []
        self.name = 'EE'
        self.root_dir = gettempdir()
        self.my_handler = UnitTestHandler()

    def tearDown(self):
        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def _test_01_debug_mode_mda_nan(self):
        """
        Checks exception is raised by activating nan debug mode in a bugged discipline.
        """

        exec_eng = ExecutionEngine(self.name)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.error_string'] = 'nan'
        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.debug_mode'] = 'nan'
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_ok = False
        try:
            exec_eng.execute()
            exec_ok = True
        except ValueError as ve:
            self.assertEqual('NaN values found in EE.SellarCoupling.Sellar_3', ve.args[0])
        except:
            raise Exception('Execution failed, and not for the good reason')
        if exec_ok:
            raise Exception('Execution worked, and it should not have')

    def test_02_debug_mode_mda_input_change(self):
        """
        Checks exception is raised by activating input_change debug mode in a bugged discipline.
        """

        exec_eng = ExecutionEngine(self.name)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.error_string'] = 'input_change'
        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.debug_mode'] = 'input_change'
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_ok = False
        try:
            exec_eng.execute()
            exec_ok = True
        except ValueError as ve:
            self.assertIn("Mismatch in .EE.SellarCoupling.y_1.value: 1.0 and 1.5 don't match", ve.args[0])
        except:
            raise Exception('Execution failed, and not for the good reason')
        if exec_ok:
            raise Exception('Execution worked, and it should not have')

    def test_03_debug_mode_mda_linearize_data_change(self):
        """
        Checks exception is raised by activating linearize_data_change debug mode in a bugged discipline.
        """

        exec_eng = ExecutionEngine(self.name)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.error_string'] = 'linearize_data_change'
        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.debug_mode'] = 'linearize_data_change'
        disc_dict[f'{self.name}.{coupling_name}.linearization_mode'] = 'adjoint'
        disc_dict[f'{self.name}.{coupling_name}.sub_mda_class'] = 'MDANewtonRaphson'
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_ok = False
        try:
            exec_eng.execute()
            exec_ok = True
        except ValueError as ve:
            assert "Mismatch in .EE.SellarCoupling.y_1.value: 2.8 and 3.3 don't match" in ve.args[0]
        except:
            raise Exception('Execution failed, and not for the good reason')
        if exec_ok:
            raise Exception('Execution worked, and it should not have')


    def test_04_debug_mode_mda_min_max_coupling(self):
        """
        Checks message is logged after activating min_max_coupling debug mode.
        """

        exec_eng = ExecutionEngine(self.name)
        LOGGER = exec_eng.logger.getChild("ProxyDiscipline.MDODisciplineWrapp.SoSMDODiscipline")
        LOGGER.setLevel(DEBUG)
        LOGGER.addHandler(self.my_handler)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.error_string'] = 'min_max_couplings'
        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.debug_mode'] = 'min_max_couplings'
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.execute()
        self.assertIn('in discipline <EE.SellarCoupling.Sellar_3> : <EE.SellarCoupling.y_1> has the minimum coupling value <1.0>', self.my_handler.msg_list)
        self.assertIn('in discipline <EE.SellarCoupling.Sellar_3> : <EE.SellarCoupling.y_2> has the maximum coupling value <3.515922583453351>', self.my_handler.msg_list)

    def test_05_debug_mode_all(self):
        """
        Check the proper setting of debug mode all on ProxyDiscipline side.
        """
        exec_eng = ExecutionEngine(self.name)

        exec_eng.logger.setLevel(DEBUG)
        exec_eng.logger.addHandler(self.my_handler)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        disc_dict[f'{self.name}.{coupling_name}.Sellar_1.debug_mode'] = 'all'
        exec_eng.load_study_from_input_dict(disc_dict)

        self.assertIn('Discipline Sellar_1 set to debug mode nan', self.my_handler.msg_list)
        self.assertIn('Discipline Sellar_1 set to debug mode input_change', self.my_handler.msg_list)
        self.assertIn('Discipline Sellar_1 set to debug mode min_max_couplings', self.my_handler.msg_list)
        self.assertIn('Discipline Sellar_1 set to debug mode linearize_data_change', self.my_handler.msg_list)
        self.assertIn('Discipline Sellar_1 set to debug mode min_max_grad', self.my_handler.msg_list)

    def test_06_debug_mode_coupling_activation_and_deactivation(self):
        """
        Checks the recursive setting of debug mode flags by the coupling on first configuration and upon activation
        of a higher coupling debug mode and recursive setting off upon deactivation.

        Behaviour: when debug mode changes (structuring), it gets pushed from coupling to children disciplines ignoring
        previous values.
        """
        exec_eng = ExecutionEngine(self.name)

        exec_eng.logger.setLevel(DEBUG)
        exec_eng.logger.addHandler(self.my_handler)

        # add disciplines SellarCoupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling13')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.debug_mode'] = 'all'
        disc_dict[f'{self.name}.{coupling_name}.Sellar_1.debug_mode'] = 'nan'
        disc_dict[f'{self.name}.{coupling_name}.Sellar_3.debug_mode'] = 'min_max_grad'
        disc_dict[f'{self.name}.{coupling_name}.debug_mode'] = 'linearize_data_change'
        disc_dict[f'{self.name}.debug_mode'] = ''


        exec_eng.load_study_from_input_dict(disc_dict)
        proxy_discs = exec_eng.root_process.proxy_disciplines[0].proxy_disciplines
        # the dm has the proper values
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_Problem.debug_mode'),'linearize_data_change')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_1.debug_mode'),'linearize_data_change')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_3.debug_mode'),'linearize_data_change')

        # the activation has been properly logged
        self.assertIn('Discipline Sellar_1 set to debug mode linearize_data_change', self.my_handler.msg_list)
        self.assertIn('Discipline Sellar_3 set to debug mode linearize_data_change', self.my_handler.msg_list)
        self.assertIn(f'Discipline {coupling_name} set to debug mode linearize_data_change', self.my_handler.msg_list)

        # the information has been properly transmitted to the sos_wrapps for execution
        exec_eng.execute()

        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            'linearize_data_change')
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            'linearize_data_change')
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            'linearize_data_change')

        # upon activation of another debug mode for the higher coupling
        disc_dict[f'{self.name}.debug_mode'] = 'input_change'

        exec_eng.load_study_from_input_dict(disc_dict)

        # dm
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.debug_mode'),'input_change')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_Problem.debug_mode'),'input_change')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_1.debug_mode'),'input_change')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_3.debug_mode'),'input_change')

        # log
        self.assertIn('Discipline Sellar_Problem set to debug mode input_change', self.my_handler.msg_list)
        self.assertIn('Discipline Sellar_1 set to debug mode input_change', self.my_handler.msg_list)
        self.assertIn('Discipline Sellar_3 set to debug mode input_change', self.my_handler.msg_list)
        self.assertIn(f'Discipline {coupling_name} set to debug mode input_change', self.my_handler.msg_list)
        self.assertIn(f'Discipline {self.name} set to debug mode input_change', self.my_handler.msg_list)

        # sos_wrapps
        exec_eng.execute()
        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            'input_change')
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            'input_change')
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            'input_change')

        # check that if I deactivate the debug mode of the lower coupling then all the children get deactivated
        disc_dict[f'{self.name}.{coupling_name}.debug_mode'] = ''
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.debug_mode'),'')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_Problem.debug_mode'),'')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_1.debug_mode'),'')
        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.{coupling_name}.Sellar_3.debug_mode'),'')
        exec_eng.execute()
        self.assertEqual(
            proxy_discs[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            '')
        self.assertEqual(
            proxy_discs[1].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            '')
        self.assertEqual(
            proxy_discs[2].mdo_discipline_wrapp.mdo_discipline.sos_wrapp.get_sosdisc_inputs('debug_mode'),
            '')