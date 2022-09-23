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
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal
import pprint
import numpy as np
import pandas as pd
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join
import os

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.execution_engine.proc_builder.build_sos_very_simple_multi_scenario import BuildSoSVerySimpleMultiScenario
from sos_trades_core.execution_engine.scatter_data import SoSScatterData
from tempfile import gettempdir
from sos_trades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class TestBuildVerySimpleMultiScenario(unittest.TestCase):
    """
    BuildSoSVerySimpleMultiScenario test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''

        self.study_name = 'MyStudy'
        self.ns = f'{self.study_name}'
        self.sc_name = "vs_MS"

        self.exec_eng = ExecutionEngine(self.ns)
        self.factory = self.exec_eng.factory

    # Begin : factorized functions to create set of inputs
    def setup_Hessian_usecase_from_direct_input(self, restricted=True):
        """
        Define a set of data inputs with empty usecase and so the subprocess Hessian is filled directly as would be done manually in GUI
        """
        my_usecase = 'Empty'
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'

        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = mod_id
        sub_process_inputs_dict['usecase_name'] = 'Empty'
        sub_process_inputs_dict['usecase_data'] = {}

        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = []
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        ######### Numerical values   ####
        x = 2.0
        y = 3.0

        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0
        scenario_list = ['scenario_1', 'scenario_2']

        ######### Fill the dictionary for dm   ####
        dict_values = {}

        if restricted == False:
            dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
            dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list

        scenario = scenario_list[0]
        my_root = f'{self.study_name}' + '.vs_MS.' + scenario
        dict_values[f'{my_root}' + '.Hessian.x'] = x
        dict_values[f'{my_root}' + '.Hessian.y'] = y
        dict_values[f'{my_root}' + '.Hessian.ax2'] = ax2
        dict_values[f'{my_root}' + '.Hessian.by2'] = by2
        dict_values[f'{my_root}' + '.Hessian.cx'] = cx
        dict_values[f'{my_root}' + '.Hessian.dy'] = dy
        dict_values[f'{my_root}' + '.Hessian.exy'] = exy

        scenario = scenario_list[1]
        my_root = f'{self.study_name}' + '.vs_MS.' + scenario
        dict_values[f'{my_root}' + '.Hessian.x'] = x + 10.0
        dict_values[f'{my_root}' + '.Hessian.y'] = y + 10.0
        dict_values[f'{my_root}' + '.Hessian.ax2'] = ax2 + 10.0
        dict_values[f'{my_root}' + '.Hessian.by2'] = by2 + 10.0
        dict_values[f'{my_root}' + '.Hessian.cx'] = cx + 10.0
        dict_values[f'{my_root}' + '.Hessian.dy'] = dy + 10.0
        dict_values[f'{my_root}' + '.Hessian.exy'] = exy + 10.0

        return [dict_values]

    def setup_Disc1Disc3_usecase_from_direct_input(self, restricted=True):
        """
        Define a set of data inputs with empty usecase and so the subprocess Hessian is filled directly as would be done manually in GUI
        """
        my_usecase = 'Empty'
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'

        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = mod_id
        sub_process_inputs_dict['usecase_name'] = 'Empty'
        sub_process_inputs_dict['usecase_data'] = {}

        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = ['ns_ac', 'ns_disc3', 'ns_out_disc3']
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        ######### Numerical values   ####
        x = 2.0
        a = 3.0
        b1 = 4.0
        b2 = 2.0
        scenario_list = ['scenario_1', 'scenario_2']

        ######### Fill the dictionary for dm   ####
        dict_values = {}

        if restricted == False:
            dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
            dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list

        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.constant'] = 3.0
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.z'] = 1.2

        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.constant'] = 3.0
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.z'] = 1.5
        return [dict_values]

    def setup_Disc1Disc3_ns_all_usecase_from_direct_input(self, restricted=True):
        """
        Define a set of data inputs with empty usecase and so the subprocess Hessian is filled directly as would be done manually in GUI
        """
        my_usecase = 'Empty'
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'

        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = mod_id
        sub_process_inputs_dict['usecase_name'] = 'Empty'
        sub_process_inputs_dict['usecase_data'] = {}

        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = ['ns_ac', 'ns_data_ac', 'ns_disc3', 'ns_out_disc3']
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        ######### Numerical values   ####
        x1 = 2.0
        x2 = 4.0
        a1 = 3.0
        b1 = 4.0
        a2 = 6.0
        b2 = 2.0
        scenario_list = ['scenario_1', 'scenario_2']

        ######### Fill the dictionary for dm   ####
        dict_values = {}

        if restricted == False:
            dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
            dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list

        dict_values[f'{self.study_name}.vs_MS.scenario_1.a'] = a1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.x'] = x1

        dict_values[f'{self.study_name}.vs_MS.scenario_2.a'] = a2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.x'] = x2

        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.constant'] = 3
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.power'] = 1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.z'] = 1.2

        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.constant'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.z'] = 1.2
        return [dict_values]
    #################### End : functions to create set of inputs #############

    #################### Begin : factorized function for test with assert ####
    def check_created_tree_structure(self, target_exp_tv_list):
        exp_tv_str = '\n'.join(target_exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def print_config_state(self):
        # check configuration state
        for disc in self.exec_eng.dm.disciplines_dict.keys():
            my_disc = self.exec_eng.dm.get_discipline(disc)
        # print configuration state
        print('Disciplines configuration status: \n')
        for disc in self.exec_eng.dm.disciplines_dict.keys():
            my_disc = self.exec_eng.dm.get_discipline(disc)
            print(my_disc.get_disc_full_name())
            print('no need to be configured : ' +
                  str(my_disc.is_configured()))
            print('has been configured: ' +
                  str(my_disc.get_configure_status()))
            print('Calculation status: ' + str(my_disc.status))
            print('\n')

    def check_status_state(self, target_status='CONFIGURE'):
        # check configuration state
        for disc in self.exec_eng.dm.disciplines_dict.keys():
            my_disc = self.exec_eng.dm.get_discipline(disc)
            self.assertEqual(my_disc.status, target_status)

    def check_discipline_inputs_list(self, my_disc, target_inputs_list):
        full_inputs_list = my_disc.get_data_io_dict_keys('in')
        for key in target_inputs_list:
            self.assertIn(key, full_inputs_list)

    def check_discipline_outputs_list(self, my_disc, target_outputs_list):
        outputs_list_disc = [
            elem for elem in my_disc.get_data_io_dict_keys('out')]
        self.assertCountEqual(target_outputs_list, outputs_list_disc)

    def check_discipline_value(self, my_disc, my_data_name, target_value, print_flag=True, ioType='in'):
        my_data = my_disc.get_data_io_from_key(
            ioType, my_data_name)
        my_value = my_data['value']
        if isinstance(my_value, pd.DataFrame):
            assert_frame_equal(target_value, my_value)
        else:
            self.assertEqual(target_value, my_value)
        if print_flag:
            print(my_data_name + ': ', my_value)

    def check_discipline_values(self, my_disc, target_values_dict, print_flag=True, ioType='in'):
        if print_flag:
            print(
                f'Check_discipline value for {my_disc.get_disc_full_name()}:')
        for key in target_values_dict.keys():
            self.check_discipline_value(
                my_disc,
                key,
                target_value=target_values_dict[key],
                print_flag=print_flag,
                ioType=ioType,
            )
        if print_flag:
            print('\n')

    def data_value_type_in_gui(self, data):
        if data['editable'] == False or data['io_type'] == 'out':
            value_type = 'READ_ONLY'
        elif not isinstance(data['value'], type(None)):
            value_type = 'USER'
        elif data['default'] != None:
            value_type = 'DEFAULT'
        elif data['optional'] == True:
            value_type = 'OPTIONAL'
        else:
            if data['io_type'] == 'in':
                value_type = 'MISSING'
            else:
                value_type = 'EMPTY'
        return value_type

    def check_discipline_value_type(self, my_disc, my_data_name, target_value, print_flag=True):
        my_data = my_disc.get_data_io_from_key(
            'in', my_data_name)
        my_value_type = self.data_value_type_in_gui(my_data)
        self.assertEqual(target_value, my_value_type)
        if print_flag:
            print(my_data_name + ': ', my_value_type)

    def check_discipline_value_types(self, my_disc, target_values_dict, print_flag=True):
        if print_flag:
            print(
                f'Check_discipline value type for {my_disc.get_disc_full_name()}:')
        for key in target_values_dict.keys():
            self.check_discipline_value_type(
                my_disc,
                key,
                target_value=target_values_dict[key],
                print_flag=print_flag,
            )
        if print_flag:
            print('\n')

    def start_execution_status(self, print_flag=True):
        missing_variables = []
        if print_flag == True:
            print('Start execution status:')
        filter
        for disc in self.exec_eng.dm.disciplines_dict.keys():
            my_disc = self.exec_eng.dm.get_discipline(disc)
            full_inputs_list = my_disc.get_data_io_dict_keys('in')
            for my_data_name in full_inputs_list:
                my_data = my_disc.get_data_io_from_key(
                    'in', my_data_name)
                value_type = self.data_value_type_in_gui(my_data)
                if print_flag == True:
                    print(f'{my_data_name}: {value_type}')
                if value_type == 'MISSING':
                    missing_variables.append(my_data_name)
        if print_flag == True:
            print('\n')
            if missing_variables != []:
                print('Mandatory variables are missing: ')
                print(missing_variables)
            else:
                print('Inputs OK : process ready to be run')
        return missing_variables
    #################### End : factorized function for test with assert ######

    def test_01_build_vs_MS_with_nested_proc_selection_through_process_driver_Hessian_subproc(self):
        '''
        Test the creation of the vs_MS without nested disciplines directly from vs_MS class : 
        through process test_driver_build_vs_MS_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_01_build_vs_MS_with_nested_proc_selection_through_process_driver_Hessian_subproc')
        # Step 0: setup an empty
        print('Step 0: setup an empty driver')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo_proc_builder = 'sos_trades_core.sos_processes.test.proc_builder'
        mod_id_empty_driver = 'test_driver_build_vs_MS_empty'
        self.study_name = 'MyStudy'

        study_dump = BaseStudyManager(
            repo_proc_builder, mod_id_empty_driver, 'MyStudy')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()

        ################ Start checks ##########################
        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_last = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['sub_process_inputs', 'scenario_map']
        inputs_list = inputs_list + [
            'linearization_mode',
            'cache_type',
            'cache_file_path',
            'debug_mode',
        ]
        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = None
        tv_sub_process_inputs_dict['process_name'] = None
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': []}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map
        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)
        ################ End checks ##########################

        # Step 1: Provide subprocess and provide data input
        print('Step 1: provide a process (with disciplines) to the set driver')
        dict_values = self.setup_Hessian_usecase_from_direct_input(restricted=False)[
            0]
        study_dump.load_data(from_input_dict=dict_values)

        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS',
                       f'\t\t|_ scenario_1',
                       f'\t\t\t|_ Hessian',
                       f'\t\t|_ scenario_2',
                       f'\t\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)  # KO if no rebuild done
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = ['scenario_list', 'ns_in_df']
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}

        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = []
        tv_scenario_map = {'input_name': scenario_map_name,
                           #'input_ns': input_ns,
                           #'output_name': output_name,
                           #'scatter_ns': scatter_ns,
                           #'gather_ns': input_ns,
                           'ns_to_update': ns_to_update}

        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map
        target_values_dict['scenario_list'] = ['scenario_1', 'scenario_2']

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline in scenario 1
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_1.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline in scenario 1
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_2.Hessian')[0]
        target_x = 12.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################
        # Step 2: run
        skip_run = False
        if not skip_run:
            print('Step 2: run')
            study_dump.dump_data(dump_dir)
            # print(study_dump.ee.dm.get_data_dict_values())
            study_dump.run()

            # print configuration state:
            if print_flag:
                self.print_config_state()
            # check configuration state
            self.check_status_state(target_status='DONE')

            # Check output
            hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
                f'{self.study_name}.vs_MS.scenario_1.Hessian')[0]
            target_values_dict = {}
            target_values_dict['z'] = 166.0
            self.check_discipline_values(
                hessian_disc, target_values_dict, print_flag=print_flag, ioType='out')

            # self.exec_eng.display_treeview_nodes(True)

            my_data = hessian_disc.get_data_io_from_key(
                'out', 'z')
            my_value = my_data['value']
            tolerance = 1.e-6
            target_x = 166.0
            self.assertAlmostEqual(target_x, my_value, delta=tolerance)

            ########################
            study_load = BaseStudyManager(
                repo_proc_builder, mod_id_empty_driver, 'MyStudy')
            study_load.load_data(from_path=dump_dir)
            # print(study_load.ee.dm.get_data_dict_values())
            study_load.run()
            from shutil import rmtree
            rmtree(dump_dir)

    def test_02_build_vs_MS_test_GUI_sequence_Hessian(self):
        '''
        Test the creation of the driver without nested disciplines directly from vs_MS class : 
        through process test_driver_build_vs_MS_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_02_build_vs_MS_test_GUI_sequence')
        # Step 0: setup an empty driver
        print('Step 0: setup an empty driver')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo_proc_builder = 'sos_trades_core.sos_processes.test.proc_builder'
        mod_id_empty_driver = 'test_driver_build_vs_MS_empty'
        self.study_name = 'MyStudy'

        # create session with empty driver
        print(
            '################################################################################')
        print('STEP_0: create session with empty driver')
        study_dump = BaseStudyManager(
            repo_proc_builder, mod_id_empty_driver, 'MyStudy')
        study_dump.load_data()  # configure

        study_dump.set_dump_directory(dump_dir)
        study_dump.dump_data(dump_dir)

        ################ Start checks ##########################
        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_last = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['sub_process_inputs', 'scenario_map']
        inputs_list = inputs_list
        inputs_list = inputs_list + \
            ['linearization_mode', 'cache_type', 'cache_file_path', 'debug_mode']
        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = None
        tv_sub_process_inputs_dict['process_name'] = None
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': []}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map
        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        # Prepare inputs #########
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        my_usecase = 'usecase1'
        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = None
        sub_process_inputs_dict['usecase_name'] = 'Empty'
        sub_process_inputs_dict['usecase_data'] = {}

        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = []
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        x = 2.0
        y = 3.0

        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0

        scenario_list = ['scenario_1', 'scenario_2']

        ######################## End of prepare inputs ########################

        print(
            '################################################################################')
        print(
            'STEP_1: update with subprocess Hessian selection and filled subprocess data')

        print("\n")
        print("1.1 Provide repo")
        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
        study_dump.load_data(from_input_dict=dict_values)
        # check multi-configure max 100 reached
        #
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new

        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = None
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': []}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map
        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        ################ End checks ##########################
        print("\n")
        print("1.2 Provide process name")
        sub_process_inputs_dict['process_name'] = mod_id
        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
        study_dump.load_data(from_input_dict=dict_values)
        ##
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': []}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        print("\n")
        print("1.3 Provide scenario_map")
        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map
        study_dump.load_data(from_input_dict=dict_values)
        ##
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = ['scenario_list', 'ns_in_df']
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = None

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'MISSING'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['scenario_list']
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################
        print("\n")
        print("1.4 Provide 'scenario_list'")

        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list
        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS',
                       f'\t\t|_ scenario_1',
                       f'\t\t\t|_ Hessian',
                       f'\t\t|_ scenario_2',
                       f'\t\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)  # KO if no rebuild done

        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = ['scenario_1', 'scenario_2']

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_1.Hessian')[0]
        target_x = None
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_2.Hessian')[0]
        target_x = None
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['x', 'y', 'ax2', 'by2', 'cx', 'dy', 'exy',
                                    'x', 'y', 'ax2', 'by2', 'cx', 'dy', 'exy',
                                    'x', 'y', 'ax2', 'by2', 'cx', 'dy', 'exy',
                                    'x', 'y', 'ax2', 'by2', 'cx', 'dy', 'exy']
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################
        print("\n")
        print("1.5 Provide disciplines inputs")

        dict_values = {}
        scenario = scenario_list[0]
        my_root = f'{self.study_name}' + '.vs_MS.' + scenario
        dict_values[f'{my_root}' + '.Hessian.x'] = x
        dict_values[f'{my_root}' + '.Hessian.y'] = y
        dict_values[f'{my_root}' + '.Hessian.ax2'] = ax2
        dict_values[f'{my_root}' + '.Hessian.by2'] = by2
        dict_values[f'{my_root}' + '.Hessian.cx'] = cx
        dict_values[f'{my_root}' + '.Hessian.dy'] = dy
        dict_values[f'{my_root}' + '.Hessian.exy'] = exy

        scenario = scenario_list[1]
        my_root = f'{self.study_name}' + '.vs_MS.' + scenario
        dict_values[f'{my_root}' + '.Hessian.x'] = x + 10.0
        dict_values[f'{my_root}' + '.Hessian.y'] = y + 10.0
        dict_values[f'{my_root}' + '.Hessian.ax2'] = ax2 + 10.0
        dict_values[f'{my_root}' + '.Hessian.by2'] = by2 + 10.0
        dict_values[f'{my_root}' + '.Hessian.cx'] = cx + 10.0
        dict_values[f'{my_root}' + '.Hessian.dy'] = dy + 10.0
        dict_values[f'{my_root}' + '.Hessian.exy'] = exy + 10.0

        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS',
                       f'\t\t|_ scenario_1',
                       f'\t\t\t|_ Hessian',
                       f'\t\t|_ scenario_2',
                       f'\t\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = ['scenario_1', 'scenario_2']

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_1.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_2.Hessian')[0]
        target_x = 12.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################
        print("\n")
        print("1.6 Provide use case name and add scenario_ref")
        sub_process_inputs_dict['usecase_name'] = my_usecase
        if 0 == 0:  # directly provide anonymized dict
            anonymize_input_dict = {}
            anonymize_input_dict['<study_ph>.Hessian.ax2'] = 4.0
            anonymize_input_dict['<study_ph>.Hessian.by2'] = 5.0
            anonymize_input_dict['<study_ph>.Hessian.cx'] = 6.0
            anonymize_input_dict['<study_ph>.Hessian.dy'] = 7.0
            anonymize_input_dict['<study_ph>.Hessian.exy'] = 12.0
            anonymize_input_dict['<study_ph>.Hessian.x'] = 2.0
            anonymize_input_dict['<study_ph>.Hessian.y'] = 3.0
        else:  # get it from usecase name
            sub_process_usecase_full_name = self.get_sub_process_usecase_full_name(
                repo, mod_id, my_usecase)
            anonymize_input_dict = self.import_input_data_from_usecase_of_sub_process(self.exec_eng,
                                                                                      sub_process_usecase_full_name)
        sub_process_inputs_dict['usecase_data'] = anonymize_input_dict

        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        tv_anonymize_input_dict_from_usecase = {}
        tv_anonymize_input_dict_from_usecase['<study_ph>.Hessian.ax2'] = 4.0
        tv_anonymize_input_dict_from_usecase['<study_ph>.Hessian.by2'] = 5.0
        tv_anonymize_input_dict_from_usecase['<study_ph>.Hessian.cx'] = 6.0
        tv_anonymize_input_dict_from_usecase['<study_ph>.Hessian.dy'] = 7.0
        tv_anonymize_input_dict_from_usecase['<study_ph>.Hessian.exy'] = 12.0
        tv_anonymize_input_dict_from_usecase['<study_ph>.Hessian.x'] = 2.0
        tv_anonymize_input_dict_from_usecase['<study_ph>.Hessian.y'] = 3.0
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = my_usecase
        # None because we have empty the anonymized dictionary
        #tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_sub_process_inputs_dict['usecase_data'] = tv_anonymize_input_dict_from_usecase
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = ['scenario_1', 'scenario_2']

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_1.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_2.Hessian')[0]
        target_x = 12.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        # Run
        flag_run = False
        flag_local = True
        if flag_run:
            print(
                '################################################################################')
            print('STEP_2: run')
            if flag_local:
                study_dump.run()
            else:
                study_load = BaseStudyManager(
                    repo, mod_id_empty_driver, 'MyStudy')
                study_load.load_data(from_path=dump_dir)
                print(study_load.ee.dm.get_data_dict_values())
                study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_03_build_vs_MS_test_GUI_sequence_Disc1Disc3(self):
        '''
        Test the creation of the driver without nested disciplines directly from vs_MS class : 
        through process test_driver_build_vs_MS_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        Same as Test 2 but on Disc1Disc3 instead on Hesssian
        '''
        print('test_03_build_vs_MS_test_GUI_sequence_Disc1Disc3')
        # Step 0: setup an empty driver
        print('Step 0: setup an empty driver')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo_proc_builder = 'sos_trades_core.sos_processes.test.proc_builder'
        mod_id_empty_driver = 'test_driver_build_vs_MS_empty'
        self.study_name = 'MyStudy'

        # create session with empty driver
        print(
            '################################################################################')
        print('STEP_0: create session with empty driver')
        study_dump = BaseStudyManager(
            repo_proc_builder, mod_id_empty_driver, 'MyStudy')
        study_dump.load_data()  # configure

        study_dump.set_dump_directory(dump_dir)
        study_dump.dump_data(dump_dir)

        ################ Start checks ##########################
        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_last = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['sub_process_inputs', 'scenario_map']
        inputs_list = inputs_list
        inputs_list = inputs_list + \
            ['linearization_mode', 'cache_type', 'cache_file_path', 'debug_mode']
        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = None
        tv_sub_process_inputs_dict['process_name'] = None
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': []}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map
        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        # Prepare inputs #########
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        my_usecase = 'usecase1'
        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = None
        sub_process_inputs_dict['usecase_name'] = 'Empty'
        sub_process_inputs_dict['usecase_data'] = {}

        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'  # not used
        ns_to_update = ['ns_ac', 'ns_disc3', 'ns_out_disc3']
        #ns_to_update = ['ns_data_ac', 'ns_ac', 'ns_disc3', 'ns_out_disc3']
        scenario_map = {'input_name': scenario_map_name,
                        #'input_ns': input_ns,
                        #'output_name': output_name,
                        #'scatter_ns': scatter_ns,
                        #'gather_ns': input_ns,
                        'ns_to_update': ns_to_update}

        x = 2.0
        a = 3.0
        b1 = 4.0
        b2 = 2.0

        scenario_list = ['scenario_1', 'scenario_2']

        ######################## End of prepare inputs ########################

        print(
            '################################################################################')
        print(
            'STEP_1: update with subprocess Disc1Disc3 selection and filled subprocess data')

        print("\n")
        print("1.1 Provide repo")
        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
        study_dump.load_data(from_input_dict=dict_values)
        # check multi-configure max 100 reached
        #
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new

        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = None
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': []}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map
        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        ################ End checks ##########################
        print("\n")
        print("1.2 Provide process name")
        sub_process_inputs_dict['process_name'] = mod_id
        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
        study_dump.load_data(from_input_dict=dict_values)
        ##
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': ['ns_ac', 'ns_data_ac', 'ns_disc3', 'ns_out_disc3']}
        # updated by complete list of namspace

        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        print("\n")
        print("1.3 Provide scenario_map")
        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.scenario_map'] = scenario_map
        study_dump.load_data(from_input_dict=dict_values)
        ##
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = ['scenario_list', 'ns_in_df']
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = None

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'MISSING'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['scenario_list']
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################
        print("\n")
        print("1.4 Provide 'scenario_list'")

        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list
        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS',
                       f'\t\t|_ scenario_1',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc3',
                       f'\t\t|_ scenario_2',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc3']
        self.check_created_tree_structure(exp_tv_list)  # KO if no rebuild done

        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = ['scenario_1', 'scenario_2']

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        disc1_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_1.Disc1')[0]
        target_b = None
        target_values_dict = {}
        target_values_dict['b'] = target_b
        self.check_discipline_values(
            disc1_disc, target_values_dict, print_flag=print_flag)

        disc1_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_2.Disc1')[0]
        target_b = None
        target_values_dict = {}
        target_values_dict['b'] = target_b
        self.check_discipline_values(
            disc1_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['x', 'a', 'b', 'z', 'constant', 'power',
                                    'x', 'a', 'b', 'z', 'constant', 'power',
                                    'x', 'a', 'b', 'z', 'constant', 'power',
                                    'x', 'a', 'b', 'z', 'constant', 'power']
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################
        print("\n")
        print("1.5 Provide disciplines inputs")

        dict_values = {}
        scenario = scenario_list[0]
        dict_values[f'{self.study_name}.vs_MS.a'] = a
        dict_values[f'{self.study_name}.vs_MS.x'] = x

        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.constant'] = 3.0
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_1.Disc3.z'] = 1.2

        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.constant'] = 3.0
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.power'] = 2
        dict_values[f'{self.study_name}.vs_MS.scenario_2.Disc3.z'] = 1.5

        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS',
                       f'\t\t|_ scenario_1',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc3',
                       f'\t\t|_ scenario_2',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc3']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = ['scenario_1', 'scenario_2']

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        disc1_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_1.Disc1')[0]
        target_b = b1
        target_values_dict = {}
        target_values_dict['b'] = target_b
        self.check_discipline_values(
            disc1_disc, target_values_dict, print_flag=print_flag)

        disc1_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_2.Disc1')[0]
        target_b = b2
        target_values_dict = {}
        target_values_dict['b'] = target_b
        self.check_discipline_values(
            disc1_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################
        print("\n")
        print("1.6 Provide use case name and add scenario_ref")
        sub_process_inputs_dict['usecase_name'] = my_usecase
        if 0 == 0:  # directly provide anonymized dict
            anonymize_input_dict = {}
            anonymize_input_dict['<study_ph>.a'] = a
            anonymize_input_dict['<study_ph>.x'] = x
            anonymize_input_dict['<study_ph>.Disc1.b'] = b1
            anonymize_input_dict['<study_ph>.Disc3.constant'] = 3.0
            anonymize_input_dict['<study_ph>.Disc3.power'] = 2
            anonymize_input_dict['<study_ph>.Disc3.z'] = 1.2
        else:  # get it from usecase name
            sub_process_usecase_full_name = self.get_sub_process_usecase_full_name(
                repo, mod_id, my_usecase)
            anonymize_input_dict = self.import_input_data_from_usecase_of_sub_process(self.exec_eng,
                                                                                      sub_process_usecase_full_name)
        sub_process_inputs_dict['usecase_data'] = anonymize_input_dict

        dict_values = {}
        dict_values[f'{self.study_name}.vs_MS.sub_process_inputs'] = sub_process_inputs_dict
        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_new = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        #print("Added Inputs_list:")
        # print(added_inputs_list)
        #print("Removed Inputs_list:")
        # print(removed_inputs_list)
        target_added_inputs_list = []
        self.assertCountEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertCountEqual(target_removed_inputs_list, removed_inputs_list)

        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc1_disc3_coupling'
        tv_anonymize_input_dict_from_usecase = {}
        tv_anonymize_input_dict_from_usecase['<study_ph>.a'] = a
        tv_anonymize_input_dict_from_usecase['<study_ph>.x'] = x
        tv_anonymize_input_dict_from_usecase['<study_ph>.Disc1.b'] = b1
        tv_anonymize_input_dict_from_usecase['<study_ph>.Disc3.constant'] = 3.0
        tv_anonymize_input_dict_from_usecase['<study_ph>.Disc3.power'] = 2
        tv_anonymize_input_dict_from_usecase['<study_ph>.Disc3.z'] = 1.2
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = repo
        tv_sub_process_inputs_dict['process_name'] = mod_id
        tv_sub_process_inputs_dict['usecase_name'] = my_usecase
        # None because we have empty the anonymized dictionary
        #tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_sub_process_inputs_dict['usecase_data'] = tv_anonymize_input_dict_from_usecase
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = scenario_map
        target_values_dict['scenario_list'] = ['scenario_1', 'scenario_2']

        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        disc1_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_1.Disc1')[0]
        target_b = b1
        target_values_dict = {}
        target_values_dict['b'] = target_b
        self.check_discipline_values(
            disc1_disc, target_values_dict, print_flag=print_flag)

        disc1_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS.scenario_2.Disc1')[0]
        target_b = b2
        target_values_dict = {}
        target_values_dict['b'] = target_b
        self.check_discipline_values(
            disc1_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        target_values_dict['scenario_list'] = 'USER'

        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        # Run
        flag_run = False
        flag_local = True
        if flag_run:
            print(
                '################################################################################')
            print('STEP_2: run')
            if flag_local:
                study_dump.run()
            else:
                study_load = BaseStudyManager(
                    repo, mod_id_empty_driver, 'MyStudy')
                study_load.load_data(from_path=dump_dir)
                print(study_load.ee.dm.get_data_dict_values())
                study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_04_build_driver_with_nested_proc_and_updates(self):
        '''
        Test of changing of nested sub_process
        '''
        print('test_04_build_driver_with_nested_proc_and_updates')
        # Step 0: setup an empty
        print('Step 0: setup an empty driver')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo_proc_builder = 'sos_trades_core.sos_processes.test.proc_builder'
        mod_id_empty_driver = 'test_driver_build_vs_MS_empty'
        self.study_name = 'MyStudy'

        study_dump = BaseStudyManager(
            repo_proc_builder, mod_id_empty_driver, 'MyStudy')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()
        ################ Start checks ##########################
        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select vs_MS disc
        driver_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.vs_MS')[0]
        # check input parameter list and values of vs_MS discipline
        full_inputs_list_last = driver_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['sub_process_inputs', 'scenario_map']
        inputs_list = inputs_list + [
            'linearization_mode',
            'cache_type',
            'cache_file_path',
            'debug_mode',
        ]
        # print(driver_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(driver_disc, inputs_list)
        # check output parameter list  of vs_MS discipline
        # print(driver_disc.get_data_io_dict_keys('out'))
        outputs_list = []
        self.check_discipline_outputs_list(driver_disc, outputs_list)

        # check input values (and print) of vs_MS discipline
        target_values_dict = {}
        tv_sub_process_inputs_dict = {}
        tv_sub_process_inputs_dict['process_repository'] = None
        tv_sub_process_inputs_dict['process_name'] = None
        tv_sub_process_inputs_dict['usecase_name'] = 'Empty'
        tv_sub_process_inputs_dict['usecase_data'] = {}
        tv_scenario_map = {'input_name': None,
                           #'input_ns': '',
                           #'output_name': '',
                           #'scatter_ns': '',
                           #'gather_ns': '',
                           'ns_to_update': []}
        target_values_dict['sub_process_inputs'] = tv_sub_process_inputs_dict
        target_values_dict['scenario_map'] = tv_scenario_map
        self.check_discipline_values(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of vs_MS discipline
        target_values_dict = {}
        target_values_dict['sub_process_inputs'] = 'USER'
        target_values_dict['scenario_map'] = 'USER'
        self.check_discipline_value_types(
            driver_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertCountEqual(target_missing_variables, missing_variables)
        ################ End checks ##########################

        # Step 1: Provide subprocess and provide data input
        print('Step 1: provide a process (with disciplines) to the set driver')
        dict_values = self.setup_Hessian_usecase_from_direct_input(restricted=False)[
            0]
        study_dump.load_data(from_input_dict=dict_values)

        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS',
                       f'\t\t|_ scenario_1',
                       f'\t\t\t|_ Hessian',
                       f'\t\t|_ scenario_2',
                       f'\t\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)  # KO if no rebuild done
        print(
            '################################################################################')
        print(
            'STEP_4: update subprocess selection by changing test_disc_hessian to test_disc1_disc3_coupling')
        #
        dict_values = self.setup_Disc1Disc3_usecase_from_direct_input(restricted=False)[
            0]
        study_dump.load_data(from_input_dict=dict_values)

        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ vs_MS',
                       f'\t\t|_ scenario_1',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc3',
                       f'\t\t|_ scenario_2',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc3']
        self.check_created_tree_structure(
            exp_tv_list)  # KO if no rebuild done


if '__main__' == __name__:
    my_test = TestBuildVerySimpleMultiScenario()
    test_selector = 4
    if test_selector == 1:
        my_test.setUp()
        my_test.test_01_build_vs_MS_with_nested_proc_selection_through_process_driver_Hessian_subproc()
    elif test_selector == 2:
        my_test.setUp()
        my_test.test_02_build_vs_MS_test_GUI_sequence_Hessian()
    elif test_selector == 3:
        my_test.setUp()
        my_test.test_03_build_vs_MS_test_GUI_sequence_Disc1Disc3()
    elif test_selector == 4:
        my_test.setUp()
        my_test.test_04_build_driver_with_nested_proc_and_updates()
