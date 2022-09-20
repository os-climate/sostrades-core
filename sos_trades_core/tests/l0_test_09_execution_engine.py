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
from logging import getLogger, INFO
from os.path import join, dirname
from pathlib import Path
from shutil import rmtree
from time import sleep
import unittest

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.api import get_sos_logger
from sos_trades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sos_trades_core.study_manager.base_study_manager import BaseStudyManager


LOC_DIRNAME = dirname(__file__)


class TestExecutionEngine(unittest.TestCase):
    """
    Execution engine test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'EETests'
        self.repo = 'sos_trades_core.sos_processes.test'

    def test_01_execution_engine_sosdiscipline(self):
        exec_eng = ExecutionEngine(self.name)

        ns_dict = {'ns_ac': 'EETests'}
        exec_eng.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            disc1_builder)

        exec_eng.configure()

        values_dict = {}
        ns = 'EETests'
        values_dict[ns + '.x'] = 3.
        values_dict[ns + '.Disc1.a'] = 10.
        values_dict[ns + '.Disc1.b'] = 20.
        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.execute()
        print('\ntest_01_execution_engine_sosdiscipline::root_process execution result:')
        res = exec_eng.dm.data_dict

        res_reference = {
            ns + '.x': 3.0,
            ns + '.Disc1.a': 10.0,
            ns + '.Disc1.b': 20.0,
            ns + '.Disc1.indicator': 200.0,
            ns + '.y': 50.0}

        exec_eng.display_treeview_nodes()

        for key in res_reference:
            self.assertEqual(res[exec_eng.dm.data_id_map[key]]
                             ['value'], res_reference[key])

    def test_02_execution_engine_soscoupling(self):
        process = 'test_disc1_disc2_coupling'
        master_logger = get_sos_logger('SoS')
        master_logger.setLevel(INFO)
        master_logger.info(
            f'Master Logger {master_logger} is ready to gather all the loggers of subprocesses')
        exec_eng = ExecutionEngine(self.name)
        ns_dict = {'ns_ac': 'EETests'}
        exec_eng.ns_manager.add_ns_def(ns_dict)
        exec_eng.select_root_process(self.repo, process)
        exec_eng.configure()

        # modify DM ----
        values_dict = {}
        values_dict['EETests.Disc1.a'] = 10.
        values_dict['EETests.Disc1.b'] = 20.
        values_dict['EETests.Disc2.power'] = 2
        values_dict['EETests.Disc2.constant'] = -10.
        values_dict['EETests.x'] = 3.

        exec_eng.dm.set_values_from_dict(values_dict)
        exec_eng.execute()
        print('\ntest_02_execution_engine_soscoupling::root_process execution result:')
        res = exec_eng.dm.data_dict
        print(res)
        print('test_02_execution_engine_soscoupling::exec_engine.dm.disciplines_dict:')
        print(exec_eng.dm.disciplines_dict.keys())
        print('keys into dm')
        for key in exec_eng.dm.data_dict.keys():
            print(' ', key)
        self.assertSetEqual(set(exec_eng.dm.disciplines_id_map.keys()),
                            set(['EETests', 'EETests.Disc2',
                                 'EETests.Disc1']),
                            'bad list of keys stored in exec_engine.dm.disciplines_dict')

        res_target = {
            'EETests.x': 3.0,
            'EETests.Disc1.a': 10.0,
            'EETests.Disc1.b': 20.0,
            'EETests.Disc2.constant': -10.0,
            'EETests.Disc2.power': 2,
            'EETests.Disc1.indicator': 200.0,
            'EETests.y': 50.0,
            'EETests.z': 2490.0}

        for key in res_target:
            self.assertEqual(res[exec_eng.dm.data_id_map[key]]
                             ['value'], res_target[key])

    def test_03_execution_engine_with_serialisation(self):
        root_dir = join(LOC_DIRNAME,
                        'test_03_execution_engine_with_serialisation')
        # Initialize a coupling process with execution engine
        study_name = 'EEtests'
        exec_engine = ExecutionEngine(study_name=study_name,
                                      root_dir=root_dir)
        # Initialize root process with selected process
        exec_engine.select_root_process(self.repo, 'test_disc1_disc2_coupling')

        a_value = 12
        values = {
            study_name +
            '.Disc1.a': a_value}

        exec_engine.dm.set_values_from_dict(values)

        read_value = exec_engine.dm.get_value(
            study_name + '.Disc1.a')

        self.assertEqual(read_value, a_value,
                         f'read_value should be {a_value} not {read_value}')

        # Persist data using the current persistance strategy
        if Path(root_dir).is_dir():
            rmtree(root_dir)
            sleep(0.5)

        dump_dir = join(root_dir, exec_engine.study_name)

        BaseStudyManager.static_dump_data(
            dump_dir, exec_engine, DirectLoadDump())

        # Load previously saved data using the current persistance strategy

        # Initialize a second process coupling process with execution engine
        exec_engine_2 = ExecutionEngine(study_name=study_name,
                                        root_dir=root_dir)

        # Initialize root process with selected process
        exec_engine_2.select_root_process(
            self.repo, 'test_disc1_disc2_coupling')

        BaseStudyManager.static_load_data(
            dump_dir, exec_engine_2, DirectLoadDump())

        read_value = exec_engine_2.dm.get_value(
            study_name + '.Disc1.a')

        self.assertEqual(read_value, a_value,
                         f'read_value should be {a_value} not {read_value}')
        rmtree(root_dir)

    def test_04_execution_engine_with_serialisation_and_defaults(self):
        root_dir = join(
            LOC_DIRNAME, 'test_03_execution_engine_with_serialisation')

        # Initialize a coupling process with execution engine
        study_name = 'EEtests'
        exec_engine = ExecutionEngine(study_name=study_name,
                                      root_dir=root_dir)
        ns_dict = {'ns_ac': 'EETests'}
        exec_engine.ns_manager.add_ns_def(ns_dict)

        # Initialize root process with selected process
        exec_engine.select_root_process(
            self.repo, 'test_disc1_disc2_couplingdefault')

        a_value = 12.0
        values = {
            study_name + '.Disc1.a': a_value}

        exec_engine.dm.set_values_from_dict(values)

        read_value = exec_engine.dm.get_value(
            study_name + '.Disc1.a')

        self.assertEqual(read_value, a_value,
                         f'read_value should be {a_value} not {read_value}')

        # Persist data using the current persistance strategy
        if Path(root_dir).is_dir():
            rmtree(root_dir)
            sleep(0.5)

        dump_dir = join(root_dir, exec_engine.study_name)

        BaseStudyManager.static_dump_data(
            dump_dir, exec_engine, DirectLoadDump())

        # Load previously saved data using the current persistance strategy

        # Initialize a second process coupling process with execution engine
        exec_engine_2 = ExecutionEngine(study_name=study_name,
                                        root_dir=root_dir)

        # Initialize root process with selected process
        exec_engine_2.select_root_process(
            self.repo, 'test_disc1_disc2_couplingdefault')

        BaseStudyManager.static_load_data(
            dump_dir, exec_engine_2, DirectLoadDump())

        read_value = exec_engine_2.dm.get_value(
            study_name + '.Disc1.a')

        self.assertEqual(read_value, a_value,
                         f'read_value should be {a_value} not {read_value}')
        rmtree(root_dir)

    def test_05_exec_engine_logging(self):
        _ee = ExecutionEngine(self.name)
        try:
            issue_using_sos_logging = False
            ee_logger = getLogger('SoSTrades')
            ee_logger.info('test_05_log_execution_engine_sosdiscipline ends')
        except Exception:
            issue_using_sos_logging = True
        assert not issue_using_sos_logging

    def test_06_execution_engine_soscoupling_with_formula(self):
        process = 'test_disc1_disc2_coupling'
        master_logger = get_sos_logger('SoS')
        master_logger.setLevel(INFO)
        master_logger.info(
            f'Master Logger {master_logger} is ready to gather all the loggers of subprocesses')
        exec_eng = ExecutionEngine(self.name)
        ns_dict = {'ns_ac': 'EETests'}
        exec_eng.ns_manager.add_ns_def(ns_dict)
        exec_eng.select_root_process(self.repo, process)
        exec_eng.configure()

        # modify DM ----
        values_dict = {}
        values_dict['EETests.Disc1.a'] = 'EETests.Disc2.power*5'
        values_dict['EETests.Disc1.b'] = 20.
        values_dict['EETests.Disc2.power'] = 2
        values_dict['EETests.Disc2.constant'] = -10.
        values_dict['EETests.x'] = 3.

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.execute()
        print('\ntest_02_execution_engine_soscoupling::root_process execution result:')
        res = exec_eng.dm.data_dict
        print(res)
        print('test_02_execution_engine_soscoupling::exec_engine.dm.disciplines_dict:')
        print(exec_eng.dm.disciplines_dict.keys())
        print('keys into dm')
        for key in exec_eng.dm.data_dict.keys():
            print(' ', key)
        self.assertSetEqual(set(exec_eng.dm.disciplines_id_map.keys()),
                            set(['EETests', 'EETests.Disc2',
                                 'EETests.Disc1']),
                            'bad list of keys stored in exec_engine.dm.disciplines_dict')

        res_target = {
            'EETests.x': 3.0,
            'EETests.Disc1.a': 10.0,
            'EETests.Disc1.b': 20.0,
            'EETests.Disc2.constant': -10.0,
            'EETests.Disc2.power': 2,
            'EETests.Disc1.indicator': 200.0,
            'EETests.y': 50.0,
            'EETests.z': 2490.0}

        for key in res_target:
            self.assertEqual(res[exec_eng.dm.data_id_map[key]]
                             ['value'], res_target[key])
