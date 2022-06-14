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

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_wrapping.analysis_discs.build_doe_eval import BuildDoeEval
from copy import deepcopy
from tempfile import gettempdir


class TestMultiScenarioOfDoeEval(unittest.TestCase):
    """
    MultiScenario and doe_eval processes test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''

        self.study_name = 'MyStudy'
        self.ns = f'{self.study_name}'
        self.sc_name = "DoE_Eval"

        self.exec_eng = ExecutionEngine(self.ns)
        self.factory = self.exec_eng.factory

    def setup_Hessian_usecase_from_direct_input(self, restricted=True):
        """
        Define a set of data inputs with empty usecase and so the subprocess Hessian is filled directly as would be done manually in GUI
        """
        my_usecase = 'Empty'
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'

        ######### Numerical values   ####
        x = 2.0
        y = 3.0

        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0

        input_selection_xy = {'selected_input': [True, True, False, False, False, False, False],
                              'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                            'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                            'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
        input_selection_xy = pd.DataFrame(input_selection_xy)

        output_selection_z = {'selected_output': [True],
                              'full_name': ['DoE_Eval.Hessian.z']}
        output_selection_z = pd.DataFrame(output_selection_z)

        dspace_dict_xy = {'variable': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y'],
                          'lower_bnd': [-5., -5.],
                          'upper_bnd': [+5., +5.],
                          #'enable_variable': [True, True],
                          # 'activated_elem': [[True], [True]]
                          }
        my_doe_algo = "lhs"
        n_samples = 4

        dspace_xy = pd.DataFrame(dspace_dict_xy)

        ######### Fill the dictionary for dm   ####
        values_dict = {}
        if restricted == False:
            # Should we use BuildDoeEval.REPO_OF_SUB_PROCESSES?
            values_dict[f'{self.study_name}.DoE_Eval.repo_of_sub_processes'] = repo
            values_dict[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
            values_dict[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = my_usecase

        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection_xy
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection_z
        values_dict[f'{self.study_name}.DoE_Eval.design_space'] = dspace_xy

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        values_dict[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}

        values_dict[f'{self.study_name}.DoE_Eval.Hessian.x'] = x
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.y'] = y

        values_dict[f'{self.study_name}.DoE_Eval.Hessian.ax2'] = ax2
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.by2'] = by2
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.cx'] = cx
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.dy'] = dy
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.exy'] = exy

        return [values_dict]

    def setup_Hessian_usecase_from_sub_usecase(self, restricted=True, my_usecase='usecase'):
        """
        Define a set of data inputs with selected use_case
        """
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'

        ######### Numerical values   ####

        input_selection_xy = {'selected_input': [True, True, False, False, False, False, False],
                              'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                            'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                            'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
        input_selection_xy = pd.DataFrame(input_selection_xy)

        output_selection_z = {'selected_output': [True],
                              'full_name': ['DoE_Eval.Hessian.z']}
        output_selection_z = pd.DataFrame(output_selection_z)

        dspace_dict_xy = {'variable': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y'],
                          'lower_bnd': [-5., -5.],
                          'upper_bnd': [+5., +5.],
                          #'enable_variable': [True, True],
                          # 'activated_elem': [[True], [True]]
                          }
        my_doe_algo = "lhs"
        n_samples = 4

        dspace_xy = pd.DataFrame(dspace_dict_xy)

        ######### Fill the dictionary for dm   ####
        values_dict = {}
        if restricted == False:
            values_dict[f'{self.study_name}.DoE_Eval.repo_of_sub_processes'] = repo
            values_dict[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
            values_dict[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = my_usecase

        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection_xy
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection_z
        values_dict[f'{self.study_name}.DoE_Eval.design_space'] = dspace_xy

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        values_dict[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}

        return [values_dict]

    def setup_usecase_from_sub_usecase(self, restricted=True, my_test=1, my_usecase=1):
        """
        Define a set of data inputs with selected use_case
        """
        ######### Numerical values   ####
        repo = 'sos_trades_core.sos_processes.test'
        if my_test == 1:
            # SubProcess selection values
            mod_id = 'test_disc_hessian'
            if my_usecase == 1:
                my_usecase = 'usecase'
            elif my_usecase == 2:
                my_usecase = 'usecase2'
            elif my_usecase == 3:
                my_usecase = 'usecase3'
            input_selection = {'selected_input': [True, True, False, False, False, False, False],
                               'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                             'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                             'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
            input_selection = pd.DataFrame(input_selection)

            output_selection = {'selected_output': [True],
                                'full_name': ['DoE_Eval.Hessian.z']}
            output_selection = pd.DataFrame(output_selection)

            dspace_dict = {'variable': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y'],
                           'lower_bnd': [-5., -5.],
                           'upper_bnd': [+5., +5.],
                           }
            my_doe_algo = "lhs"
            n_samples = 4
            dspace = pd.DataFrame(dspace_dict)
        elif my_test == 2:
            mod_id = 'test_proc_build_disc0'  # discipline with ns_disc1 in outputs
            if my_usecase == 1:
                my_usecase = 'usecase1_int'
            elif my_usecase == 2:
                my_usecase = 'usecase2_float'
            input_selection = {'selected_input': [True, False],
                               'full_name': ['DoE_Eval.Disc0.r', 'DoE_Eval.Disc0.mod']}
            input_selection = pd.DataFrame(input_selection)

            output_selection = {'selected_output': [True, True],
                                'full_name': ['x', 'a']}
            output_selection = pd.DataFrame(output_selection)

            dspace_dict = {'variable': ['DoE_Eval.Disc0.r'],
                           'lower_bnd': [-5.],
                           'upper_bnd': [+5.],
                           }
            my_doe_algo = "lhs"
            n_samples = 4
            dspace = pd.DataFrame(dspace_dict)
        elif my_test == 3:
            mod_id = 'test_proc_build_disc1_all_types'
            if my_usecase == 1:
                my_usecase = 'usecase1'
            elif my_usecase == 2:
                my_usecase = 'usecase2'
            input_selection = {'selected_input': [True],
                               'full_name': ['DoE_Eval.Disc1.x']}
            input_selection = pd.DataFrame(input_selection)

            output_selection = {'selected_output': [True, True],
                                'full_name': ['DoE_Eval.Disc1.indicator', 'DoE_Eval.Disc1.y_dict']}  # cannot use 'DoE_Eval.Disc1.y' !
            output_selection = pd.DataFrame(output_selection)

            dspace_dict = {'variable': ['DoE_Eval.Disc1.x'],
                           'lower_bnd': [-5.],
                           'upper_bnd': [+5.],
                           }
            my_doe_algo = "lhs"
            n_samples = 4
            dspace = pd.DataFrame(dspace_dict)
        elif my_test == 4:
            mod_id = 'test_proc_build_disc1_grid'
            if my_usecase == 1:
                my_usecase = 'usecase1'
            elif my_usecase == 2:
                my_usecase = 'usecase2'
            input_selection = {'selected_input': [True],
                               'full_name': ['DoE_Eval.Disc1.x']}
            input_selection = pd.DataFrame(input_selection)

            output_selection = {'selected_output': [True, True, True],
                                'full_name': ['DoE_Eval.Disc1.indicator', 'DoE_Eval.Disc1.y', 'DoE_Eval.Disc1.y_dict2']}
            output_selection = pd.DataFrame(output_selection)

            dspace_dict = {'variable': ['DoE_Eval.Disc1.x'],
                           'lower_bnd': [-5.],
                           'upper_bnd': [+5.],
                           }
            my_doe_algo = "lhs"
            n_samples = 4
            dspace = pd.DataFrame(dspace_dict)
        elif my_test == 5:
            # here we have namespace and list of builders
            mod_id = 'test_disc1_disc2_coupling'
            if my_usecase == 1:
                my_usecase = 'usecase_coupling_2_disc_test'
            input_selection = {'selected_input': [True],
                               'full_name': ['DoE_Eval.Disc1.a']}  # Disc1.a, Disc1.b, Disc2.constant, Disc2.power Coupled x(ns_ac) and y(ns_ac)
            input_selection = pd.DataFrame(input_selection)

            output_selection = {'selected_output': [True],
                                'full_name': ['DoE_Eval.Disc1.indicator']}  # Disc1.indicator, z (ns_ac)
            output_selection = pd.DataFrame(output_selection)

            dspace_dict = {'variable': ['DoE_Eval.Disc1.indicator'],
                           'lower_bnd': [-5.],
                           'upper_bnd': [+5.],
                           }
            my_doe_algo = "lhs"
            n_samples = 4
            dspace = pd.DataFrame(dspace_dict)
        elif my_test == 6:
            mod_id = 'test_sellar_coupling'  # here we have namespace and a coupling
            if my_usecase == 1:
                my_usecase = 'usecase'
            input_selection = {'selected_input': [True, True],
                               'full_name': ['DoE_Eval.SellarCoupling.x', 'DoE_Eval.SellarCoupling.z']}
            input_selection = pd.DataFrame(input_selection)

            output_selection = {'selected_output': [False, False, True, True, True],
                                'full_name': ['DoE_Eval.SellarCoupling.c_1', 'DoE_Eval.SellarCoupling.c_2', 'DoE_Eval.SellarCoupling.obj',
                                              'DoE_Eval.SellarCoupling.y_1', 'DoE_Eval.SellarCoupling.y_2']}
            output_selection = pd.DataFrame(output_selection)

            dspace_dict = {'variable': ['DoE_Eval.SellarCoupling.x', 'DoE_Eval.SellarCoupling.z'],

                           'lower_bnd': [0., [-10., 0.]],
                           'upper_bnd': [10., [10., 10.]],
                           }

            my_doe_algo = "lhs"
            n_samples = 4
            dspace = pd.DataFrame(dspace_dict)
        ######### Fill the dictionary for dm   ####
        values_dict = {}
        if restricted == False:
            values_dict[f'{self.study_name}.DoE_Eval.repo_of_sub_processes'] = repo
            values_dict[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
            values_dict[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = my_usecase

        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection
        values_dict[f'{self.study_name}.DoE_Eval.design_space'] = dspace

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        values_dict[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}

        return [values_dict]

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
        self.assertListEqual(target_outputs_list, outputs_list_disc)

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
                my_disc, key, target_value=target_values_dict[key], print_flag=print_flag, ioType=ioType)
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
                my_disc, key, target_value=target_values_dict[key], print_flag=print_flag)
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

    def test_01_build_doe_eval_with_empty_disc(self):
        '''
        Test the creation of the empty doe (doe without sub_process) 
        It is done directly from DoE_eval class (i.e. without using the empty DOE process)
        '''
        print('test_01_build_doe_eval_with_empty_disc')
        builder_list = []

        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        inputs_list = ['repo_of_sub_processes', 'sub_process_folder_name']
        inputs_list = inputs_list + \
            ['n_processes', 'wait_time_between_fork']
        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = None
        target_values_dict['sub_process_folder_name'] = None
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'MISSING'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'repo_of_sub_processes', 'sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)

    def test_02_build_doe_eval_with_nested_proc_selection(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        '''
        print('test_02_build_doe_eval_with_nested_proc_selection')
        # Step 0: setup an empty doe
        print('Step 0: setup an empty doe')
        builder_list = []
        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        print_flag = False
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_last = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['repo_of_sub_processes', 'sub_process_folder_name']
        inputs_list = inputs_list + \
            ['n_processes', 'wait_time_between_fork']
        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = None
        target_values_dict['sub_process_folder_name'] = None
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'MISSING'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'repo_of_sub_processes', 'sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 1: provide a process (with disciplines) to the set doe
        print('Step 1: provide a process (with disciplines) to the set doe')
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        dict_values = {}
        dict_values['MyStudy.DoE_Eval.repo_of_sub_processes'] = repo
        dict_values['MyStudy.DoE_Eval.sub_process_folder_name'] = mod_id
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval',
                       f'\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = [
            'usecase_of_sub_process', 'sampling_algo', 'eval_inputs', 'eval_outputs']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = None
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.Hessian')[0]
        target_x = None
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check possible values for 'usecase_of_sub_process'
        my_data_name = 'usecase_of_sub_process'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'usecase_of_sub_process[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'MISSING'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['sampling_algo',
                                    'x', 'y', 'ax2', 'by2', 'cx', 'dy', 'exy']
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 2: provide inputs to the set doe with disciplines
        print('Step 2: provide inputs to the set doe with disciplines')
        values_dict = self.setup_Hessian_usecase_from_direct_input(restricted=True)[
            0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = ['algo_options', 'design_space']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict', 'DoE_Eval.Hessian.z_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = 'lhs'
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'USER'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 3: run
        skip_run = False
        if not skip_run:
            print('Step 3: run')
            my_result = self.exec_eng.execute()
            # print('Data')
            #print (my_result)
            # print(self.exec_eng.dm.get_data_dict_values())

            # print configuration state:
            if print_flag:
                self.print_config_state()
            # check configuration state
            self.check_status_state(target_status='DONE')

            # Check output
            target_values_dict = {}
            target_values_dict['z'] = 166.0
            self.check_discipline_values(
                hessian_disc, target_values_dict, print_flag=print_flag, ioType='out')

            my_data = doe_disc.get_data_io_from_key(
                'out', 'DoE_Eval.Hessian.z_dict')
            my_value = my_data['value']['scenario_1']
            tolerance = 1.e-6
            target_x = 252.8146117501509
            self.assertAlmostEqual(target_x, my_value, delta=tolerance)

    def test_03_build_doe_eval_with_nested_proc_selection_through_process_driver(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        Here : setup_usecase(restricted=True)
        It is then used (fill data and execute)
        '''
        print('test_03_build_doe_eval_with_nested_proc_selection_through_process_driver')
        # Step 0: setup an empty doe
        print('Step 0: setup an empty doe')
        driver_repo = 'sos_trades_core.sos_processes.test'
        driver_mod_id = 'test_driver_build_doe_eval_empty'
        doe_eval_builder = self.exec_eng.factory.get_builder_from_process(
            repo=driver_repo, mod_id=driver_mod_id)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_last = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['repo_of_sub_processes', 'sub_process_folder_name']
        inputs_list = inputs_list + \
            ['n_processes', 'wait_time_between_fork']
        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = None
        target_values_dict['sub_process_folder_name'] = None
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'MISSING'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'repo_of_sub_processes', 'sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 1: provide a process (with disciplines) to the set doe
        print('Step 1: provide a process (with disciplines) to the set doe')
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        dict_values = {}
        dict_values['MyStudy.DoE_Eval.repo_of_sub_processes'] = repo
        dict_values['MyStudy.DoE_Eval.sub_process_folder_name'] = mod_id
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval',
                       f'\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = [
            'usecase_of_sub_process', 'sampling_algo', 'eval_inputs', 'eval_outputs']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = None
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.Hessian')[0]
        target_x = None
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'MISSING'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['sampling_algo',
                                    'x', 'y', 'ax2', 'by2', 'cx', 'dy', 'exy']
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 2: provide inputs to the set doe with disciplines
        print('Step 2: provide inputs to the set doe with disciplines')
        values_dict = self.setup_Hessian_usecase_from_direct_input(restricted=True)[
            0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = ['algo_options', 'design_space']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict', 'DoE_Eval.Hessian.z_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = 'lhs'
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')
        # check possible values for 'usecase_of_sub_process'
        my_data_name = 'usecase_of_sub_process'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'usecase_of_sub_process[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'USER'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 3: run
        skip_run = False
        if not skip_run:
            print('Step 3: run')
            my_result = self.exec_eng.execute()
            # print('Data')
            #print (my_result)
            # print(self.exec_eng.dm.get_data_dict_values())

            # print configuration state:
            if print_flag:
                self.print_config_state()
            # check configuration state
            self.check_status_state(target_status='DONE')

            # Check output
            target_values_dict = {}
            target_values_dict['z'] = 166.0
            self.check_discipline_values(
                hessian_disc, target_values_dict, print_flag=print_flag, ioType='out')

            my_data = doe_disc.get_data_io_from_key(
                'out', 'DoE_Eval.Hessian.z_dict')
            my_value = my_data['value']['scenario_1']
            tolerance = 1.e-6
            target_x = 252.8146117501509
            self.assertAlmostEqual(target_x, my_value, delta=tolerance)

    def test_04_build_doe_eval_with_nested_proc_selection_through_process_driver(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        Here : setup_usecase(restricted=False)
        It is then used (fill data and execute)
        '''
        print('test_04_build_doe_eval_with_nested_proc_selection_through_process_driver')
        # Step 0: setup an empty doe
        print('Step 0: setup an empty doe')
        driver_repo = 'sos_trades_core.sos_processes.test'
        driver_mod_id = 'test_driver_build_doe_eval_empty'
        doe_eval_builder = self.exec_eng.factory.get_builder_from_process(
            repo=driver_repo, mod_id=driver_mod_id)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_last = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['repo_of_sub_processes', 'sub_process_folder_name']
        inputs_list = inputs_list + \
            ['n_processes', 'wait_time_between_fork']
        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = None
        target_values_dict['sub_process_folder_name'] = None
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'MISSING'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'repo_of_sub_processes', 'sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 1: Provide subprocess and provide data input
        print('Step 1: provide a process (with disciplines) to the set doe')
        values_dict = self.setup_Hessian_usecase_from_direct_input(restricted=False)[
            0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval',
                       f'\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = [
            'usecase_of_sub_process', 'sampling_algo', 'eval_inputs', 'eval_outputs', 'algo_options', 'design_space']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict', 'DoE_Eval.Hessian.z_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = 'lhs'
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'USER'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertListEqual(target_missing_variables, missing_variables)

        # Step 2: run
        skip_run = False
        if not skip_run:
            print('Step 2: run')
            my_result = self.exec_eng.execute()
            # print('Data')
            #print (my_result)
            # print(self.exec_eng.dm.get_data_dict_values())

            # print configuration state:
            if print_flag:
                self.print_config_state()
            # check configuration state
            self.check_status_state(target_status='DONE')

            # Check output
            target_values_dict = {}
            target_values_dict['z'] = 166.0
            self.check_discipline_values(
                hessian_disc, target_values_dict, print_flag=print_flag, ioType='out')

            my_data = doe_disc.get_data_io_from_key(
                'out', 'DoE_Eval.Hessian.z_dict')
            my_value = my_data['value']['scenario_1']
            tolerance = 1.e-6
            target_x = 252.8146117501509
            self.assertAlmostEqual(target_x, my_value, delta=tolerance)

    def test_05_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_subproc(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_05_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_subproc')
        # Step 0: setup an empty doe
        print('Step 0: setup an empty doe')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo = 'sos_trades_core.sos_processes.test'
        mod_id_empty_doe = 'test_driver_build_doe_eval_empty'
        self.study_name = 'MyStudy'

        study_dump = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()

        ################ Start checks ##########################
        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_last = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['repo_of_sub_processes', 'sub_process_folder_name']
        inputs_list = inputs_list + \
            ['n_processes', 'wait_time_between_fork']
        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = None
        target_values_dict['sub_process_folder_name'] = None
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'MISSING'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'repo_of_sub_processes', 'sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)
        ################ End checks ##########################

        # Step 1: Provide subprocess and provide data input
        print('Step 1: provide a process (with disciplines) to the set doe')
        dict_values = self.setup_Hessian_usecase_from_direct_input(restricted=False)[
            0]
        study_dump.load_data(from_input_dict=dict_values)

        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval',
                       f'\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = [
            'usecase_of_sub_process', 'sampling_algo', 'eval_inputs', 'eval_outputs', 'algo_options', 'design_space']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict', 'DoE_Eval.Hessian.z_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = 'lhs'
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'USER'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertListEqual(target_missing_variables, missing_variables)

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
            target_values_dict = {}
            target_values_dict['z'] = 166.0
            self.check_discipline_values(
                hessian_disc, target_values_dict, print_flag=print_flag, ioType='out')

            my_data = doe_disc.get_data_io_from_key(
                'out', 'DoE_Eval.Hessian.z_dict')
            my_value = my_data['value']['scenario_1']
            tolerance = 1.e-6
            target_x = 252.8146117501509
            self.assertAlmostEqual(target_x, my_value, delta=tolerance)

            ########################
            study_load = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
            study_load.load_data(from_path=dump_dir)
            # print(study_load.ee.dm.get_data_dict_values())
            study_load.run()
            from shutil import rmtree
            rmtree(dump_dir)

    def test_06_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_usecase(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        Same as test 05 but with bad usecase
        '''
        print('test_06_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_usecase')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo = 'sos_trades_core.sos_processes.test'
        mod_id_empty_doe = 'test_driver_build_doe_eval_empty'
        self.study_name = 'MyStudy'

        study_dump = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()

        print_flag = True
        #dict_values = self.setup_usecase(restricted=False)
        dict_values = self.setup_Hessian_usecase_from_sub_usecase(
            restricted=False, my_usecase='usecase')
        print('load usecase file')
        study_dump.load_data(from_input_dict=dict_values)
        # check input values (and print) of Hessian discipline
        hessian_disc = study_dump.ee.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.Hessian')[0]
        target_x = 2.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)
        # change of usecase
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'usecase2'
        print('load usecase2 file')
        study_dump.load_data(from_input_dict=dict_values)
        # check input values (and print) of Hessian discipline
        target_x = 12.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)
        # bad use case warning : (To be done)
        # In python we can provide 'usecase4' :
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'usecase4'
        print('load usecase4 file: does not exist!')
        study_dump.load_data(from_input_dict=dict_values)
        # Go on to finish study
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'usecase2'
        print('load usecase2 file')
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.dump_data(dump_dir)
        # print(study_dump.ee.dm.get_data_dict_values())

        study_dump.run()

        study_load = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        study_load.load_data(from_path=dump_dir)
        # print(study_load.ee.dm.get_data_dict_values())
        study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_07_build_doe_eval_with_nested_proc_selection_sellar(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_07_build_doe_eval_with_nested_proc_selection_sellar')
        # Step 0: setup an empty doe
        print('Step 0: setup an empty doe')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo = 'sos_trades_core.sos_processes.test'
        mod_id_empty_doe = 'test_driver_build_doe_eval_empty'
        self.study_name = 'MyStudy'

        study_dump = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()
        ################ Start checks ##########################
        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        print_flag = True
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_last = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['repo_of_sub_processes', 'sub_process_folder_name']
        inputs_list = inputs_list + \
            ['n_processes', 'wait_time_between_fork']
        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = None
        target_values_dict['sub_process_folder_name'] = None
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'MISSING'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'repo_of_sub_processes', 'sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)
        ################ End checks ##########################

        # Step 1: Provide subprocess
        print('Step 1: provide a process (with disciplines)')

        mod_id = 'test_sellar_coupling'  # here we have namespace and a coupling
        coupling_name = "SellarCoupling"
        ns = f'{self.study_name}'

        input_selection = {'selected_input': [True, True],
                           'full_name': ['DoE_Eval.SellarCoupling.x',
                                         'DoE_Eval.SellarCoupling.z']}
        input_selection = pd.DataFrame(input_selection)

        output_selection = {'selected_output': [False, False, True, True, True],
                            'full_name': ['DoE_Eval.SellarCoupling.c_1', 'DoE_Eval.SellarCoupling.c_2', 'DoE_Eval.SellarCoupling.obj',
                                          'DoE_Eval.SellarCoupling.y_1', 'DoE_Eval.SellarCoupling.y_2']}
        output_selection = pd.DataFrame(output_selection)

        dspace_dict = {'variable': ['DoE_Eval.SellarCoupling.x', 'DoE_Eval.SellarCoupling.z'],

                       'lower_bnd': [0., [-10., 0.]],
                       'upper_bnd': [10., [10., 10.]],
                       }
        my_doe_algo = "lhs"
        n_samples = 4
        dspace = pd.DataFrame(dspace_dict)
        ######### Fill the dictionary for dm   ####

        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.repo_of_sub_processes'] = repo
        dict_values[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
        dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'Empty'

        study_dump.load_data(from_input_dict=dict_values)
        # print(study_dump.ee.display_treeview_nodes(True))

        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval',
                       f'\t\t|_ SellarCoupling',
                       f'\t\t\t|_ Sellar_2',
                       f'\t\t\t|_ Sellar_1',
                       f'\t\t\t|_ Sellar_Problem']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = [
            'usecase_of_sub_process', 'sampling_algo', 'eval_inputs', 'eval_outputs']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = None
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        sellar_coupling_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.SellarCoupling')[0]
        target_x = None
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            sellar_coupling_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'MISSING'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'sampling_algo', 'z', 'x', 'local_dv', 'x', 'z', 'local_dv', 'z', 'x', 'z']
        self.assertListEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        # Step 2: Provide input data
        print('Step 2: provide a process (with disciplines)')

        dict_values[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection
        dict_values[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection
        dict_values[f'{self.study_name}.DoE_Eval.design_space'] = dspace

        dict_values[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        dict_values[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}

        from numpy import array
        dict_values[f'{ns}.DoE_Eval.{coupling_name}.x'] = 1.
        dict_values[f'{ns}.DoE_Eval.{coupling_name}.y_1'] = 1.
        dict_values[f'{ns}.DoE_Eval.{coupling_name}.y_2'] = 1.
        dict_values[f'{ns}.DoE_Eval.{coupling_name}.z'] = array([1., 1.])
        dict_values[f'{ns}.DoE_Eval.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        study_dump.load_data(from_input_dict=dict_values)
        # print(study_dump.ee.display_treeview_nodes(True))
        ################ Start checks ##########################
       # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = ['algo_options', 'design_space']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict', 'DoE_Eval.SellarCoupling.obj_dict', 'DoE_Eval.SellarCoupling.y_1_dict', 'DoE_Eval.SellarCoupling.y_2_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = 'lhs'
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values (and print) of Hessian discipline
        hessian_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval.SellarCoupling')[0]
        target_x = 1.0
        target_values_dict = {}
        target_values_dict['x'] = target_x
        self.check_discipline_values(
            hessian_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')
        # check possible values for 'usecase_of_sub_process'
        my_data_name = 'usecase_of_sub_process'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'usecase_of_sub_process[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'USER'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = []
        self.assertListEqual(target_missing_variables, missing_variables)
        ################ End checks ##########################
        # Step 3: run
        skip_run = False
        if not skip_run:
            print('Step 3: run')

            study_dump.dump_data(dump_dir)
            # print(study_dump.ee.dm.get_data_dict_values())
            # print(study_dump.ee.display_treeview_nodes(True))
            study_dump.run(my_data)

            my_data = doe_disc.get_data_io_from_key(
                'out', 'DoE_Eval.SellarCoupling.obj_dict')
            my_value = my_data['value']['scenario_1']
            tolerance = 1.e-6
            target_x = 42.37077735
            self.assertAlmostEqual(target_x, my_value[0], delta=tolerance)

            study_load = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
            study_load.load_data(from_path=dump_dir)
            # print(study_load.ee.dm.get_data_dict_values())
            study_load.run()
            from shutil import rmtree
            rmtree(dump_dir)

    def test_08_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_08_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo = 'sos_trades_core.sos_processes.test'
        mod_id_empty_doe = 'test_driver_build_doe_eval_empty'
        self.study_name = 'MyStudy'

        study_dump = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()

        # You can switch between tests (choose your sub_process) from 1 to 6
        my_test = 1
        dict_values = self.setup_usecase_from_sub_usecase(
            restricted=False, my_test=my_test, my_usecase=1)
        dict_values = dict_values[0]
        if my_test == 5:
            dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'Empty'
            dict_values[self.study_name + '.x'] = 10.
            dict_values[self.study_name + '.DoE_Eval.Disc1.a'] = 5.
            dict_values[self.study_name + '.DoE_Eval.Disc1.b'] = 25431.
            dict_values[self.study_name + '.y'] = 4.
            dict_values[self.study_name + '.DoE_Eval.Disc2.constant'] = 3.1416
            dict_values[self.study_name + '.DoE_Eval.Disc2.power'] = 2

        if my_test == 6:
            dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'Empty'
            coupling_name = "SellarCoupling"
            ns = f'{self.study_name}'
            from numpy import array
            dict_values[f'{ns}.DoE_Eval.{coupling_name}.x'] = 1.
            dict_values[f'{ns}.DoE_Eval.{coupling_name}.y_1'] = 1.
            dict_values[f'{ns}.DoE_Eval.{coupling_name}.y_2'] = 1.
            dict_values[f'{ns}.DoE_Eval.{coupling_name}.z'] = array([1., 1.])
            dict_values[f'{ns}.DoE_Eval.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        study_dump.load_data(from_input_dict=dict_values)
        # print(study_dump.ee.display_treeview_nodes(True))

        study_dump.dump_data(dump_dir)

        skip_run = False
        if skip_run == False:
            local_run = True
            if local_run == True:
                study_dump.run()
            else:
                study_load = BaseStudyManager(
                    repo, mod_id_empty_doe, 'MyStudy')
                study_load.load_data(from_path=dump_dir)
                study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_09_build_doe_eval_test_GUI_sequence(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_09_build_doe_eval_test_GUI_sequence')
        # Step 0: setup an empty doe
        print('Step 0: setup an empty doe')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo = 'sos_trades_core.sos_processes.test'
        mod_id_empty_doe = 'test_driver_build_doe_eval_empty'
        self.study_name = 'MyStudy'

        # create session with empty DoE
        print(
            '################################################################################')
        print('STEP_0: create session with empty DoE')
        study_dump = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
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
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_last = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_last = [elem for elem in full_inputs_list_last]
        # print("Full_inputs_list_last:")
        # print(full_inputs_list_last)
        inputs_list = ['repo_of_sub_processes', 'sub_process_folder_name']
        inputs_list = inputs_list + \
            ['n_processes', 'wait_time_between_fork']
        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df', 'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = None
        target_values_dict['sub_process_folder_name'] = None
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'MISSING'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = [
            'repo_of_sub_processes', 'sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        # Prepare inputs #########
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        my_usecase = 'usecase'
        x = 2.0
        y = 3.0

        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0

        input_selection_xy = {'selected_input': [True, True, False, False, False, False, False],
                              'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                            'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                            'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
        input_selection_xy = pd.DataFrame(input_selection_xy)

        output_selection_z = {'selected_output': [True],
                              'full_name': ['DoE_Eval.Hessian.z']}
        output_selection_z = pd.DataFrame(output_selection_z)

        dspace_dict_xy = {'variable': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y'],
                          'lower_bnd': [-5., -5.],
                          'upper_bnd': [+5., +5.],
                          #'enable_variable': [True, True],
                          # 'activated_elem': [[True], [True]]
                          }
        my_doe_algo = "lhs"
        n_samples = 4

        dspace_xy = pd.DataFrame(dspace_dict_xy)
        ######################## End of prepare inputs ########################

        print(
            '################################################################################')
        print(
            'STEP_1: update with subprocess Hessian selection and filled subprocess data')

        print("\n")
        print("1.1 Provide repo")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.repo_of_sub_processes'] = repo
        study_dump.load_data(from_input_dict=dict_values)
        # check multi-configure max 100 reached
        #
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new

        target_added_inputs_list = []
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = None
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)  # The possible value should be set !
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')
        target_possible_values = ['test_disc_hessian', 'test_disc1_disc2_coupling', 'test_sellar_coupling', 'test_proc_build_disc0',
                                  'test_proc_build_disc1_all_types', 'test_proc_build_disc1_grid', 'test_proc_build_disc_self_coupled']
        possible_values_list = my_data['possible_values']
        if isinstance(possible_values_list, type(None)):
            print('possible_values is None instead of :')
            print(target_possible_values)
        else:
            self.assertListEqual(target_possible_values, possible_values_list)
        print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'MISSING'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['sub_process_folder_name']
        self.assertListEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        print("\n")
        print("1.2 Provide process name")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
        study_dump.load_data(from_input_dict=dict_values)
        ##
        ################ Start checks ##########################
        # check created tree structure
        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ MyStudy',
                       f'\t|_ DoE_Eval',
                       f'\t\t|_ Hessian']
        self.check_created_tree_structure(exp_tv_list)
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        target_added_inputs_list = [
            'usecase_of_sub_process', 'sampling_algo', 'eval_inputs', 'eval_outputs']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = 'Empty'
        target_values_dict['sampling_algo'] = None
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')
        target_possible_values = ['test_disc_hessian', 'test_disc1_disc2_coupling', 'test_sellar_coupling', 'test_proc_build_disc0',
                                  'test_proc_build_disc1_all_types', 'test_proc_build_disc1_grid', 'test_proc_build_disc_self_coupled']
        possible_values_list = my_data['possible_values']
        if isinstance(possible_values_list, type(None)):
            print('possible_values is None instead of :')
            print(target_possible_values)
        else:
            self.assertListEqual(target_possible_values, possible_values_list)
        print('\n')
        # check possible values for 'usecase_of_sub_process'
        my_data_name = 'usecase_of_sub_process'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'usecase_of_sub_process[possible_values]:')
            print(my_data['possible_values'])
            print('\n')
        target_possible_values = ['Empty', 'usecase', 'usecase2', 'usecase3']
        possible_values_list = my_data['possible_values']
        if isinstance(possible_values_list, type(None)):
            print('possible_values is None instead of :')
            print(target_possible_values)
        else:
            self.assertListEqual(target_possible_values, possible_values_list)
        print('\n')
        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'MISSING'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['sampling_algo',
                                    'x', 'y', 'ax2', 'by2', 'cx', 'dy', 'exy']
        self.assertListEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        print("\n")
        print("1.3 Provide use case name")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = my_usecase
        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
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
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = my_usecase
        target_values_dict['sampling_algo'] = None
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'usecase_of_sub_process'
        my_data_name = 'usecase_of_sub_process'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'usecase_of_sub_process[possible_values]:')
            print(my_data['possible_values'])
            print('\n')
        target_possible_values = ['Empty', 'usecase', 'usecase2', 'usecase3']
        possible_values_list = my_data['possible_values']
        if isinstance(possible_values_list, type(None)):
            print('possible_values is None instead of :')
            print(target_possible_values)

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'MISSING'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        target_missing_variables = ['sampling_algo']
        self.assertListEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        print("\n")
        print("1.4 Provide sampling_algo")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        study_dump.load_data(from_input_dict=dict_values)
        ################ Start checks ##########################
        # print configuration state:
        if print_flag:
            self.print_config_state()
        # check configuration state
        self.check_status_state()

        # select doe_eval disc
        doe_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.DoE_Eval')[0]
        # check input parameter list and values of DoE_Eval discipline
        full_inputs_list_new = doe_disc.get_data_io_dict_keys('in')
        full_inputs_list_new = [elem for elem in full_inputs_list_new]
        added_inputs_list = [
            elem for elem in full_inputs_list_new if elem not in full_inputs_list_last]
        removed_inputs_list = [
            elem for elem in full_inputs_list_last if elem not in full_inputs_list_new]
        full_inputs_list_last = full_inputs_list_new
        target_added_inputs_list = ['algo_options']
        self.assertListEqual(target_added_inputs_list, added_inputs_list)
        target_removed_inputs_list = []
        self.assertListEqual(target_removed_inputs_list, removed_inputs_list)

        # print(doe_disc.get_data_io_dict_keys('in'))
        self.check_discipline_inputs_list(doe_disc, inputs_list)
        # check output parameter list  of DoE_Eval discipline
        # print(doe_disc.get_data_io_dict_keys('out'))
        outputs_list = ['samples_inputs_df',
                        'all_ns_dict']
        self.check_discipline_outputs_list(doe_disc, outputs_list)

        # check input values (and print) of DoE_Eval discipline
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = repo
        target_values_dict['sub_process_folder_name'] = mod_id
        target_values_dict['n_processes'] = 1
        target_values_dict['wait_time_between_fork'] = 0
        target_values_dict['usecase_of_sub_process'] = my_usecase
        target_values_dict['sampling_algo'] = 'lhs'
        self.check_discipline_values(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check possible values for 'sub_process_folder_name'
        my_data_name = 'sub_process_folder_name'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'sub_process_folder_name[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check possible values for 'usecase_of_sub_process'
        my_data_name = 'usecase_of_sub_process'
        my_data = doe_disc.get_data_io_from_key(
            'in', my_data_name)
        if print_flag:
            print(
                'usecase_of_sub_process[possible_values]:')
            print(my_data['possible_values'])
            print('\n')

        # check input values_types (and print) of DoE_Eval discipline
        target_values_dict = {}
        target_values_dict['repo_of_sub_processes'] = 'USER'
        target_values_dict['sub_process_folder_name'] = 'USER'
        target_values_dict['n_processes'] = 'USER'
        target_values_dict['wait_time_between_fork'] = 'USER'
        target_values_dict['usecase_of_sub_process'] = 'USER'
        target_values_dict['sampling_algo'] = 'USER'
        target_values_dict['eval_inputs'] = 'USER'
        target_values_dict['eval_outputs'] = 'USER'

        self.check_discipline_value_types(
            doe_disc, target_values_dict, print_flag=print_flag)

        # check start execution status (can be run if no mandatory value))
        missing_variables = self.start_execution_status(print_flag=False)
        #target_missing_variables = ['sampling_algo']
        target_missing_variables = []
        self.assertListEqual(target_missing_variables, missing_variables)

        ################ End checks ##########################

        print("1.5 Provide algo_options")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}
        study_dump.load_data(from_input_dict=dict_values)

        print("\n")
        print("1.6 Provide eval_inputs and eval_outputs")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection_xy
        dict_values[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection_z
        study_dump.load_data(from_input_dict=dict_values)

        print("\n")
        print("1.7 Provide algo_options")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}
        study_dump.load_data(from_input_dict=dict_values)

        print("\n")
        print("1.8 Provide design_space")
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.design_space'] = dspace_xy
        study_dump.load_data(from_input_dict=dict_values)

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
                    repo, mod_id_empty_doe, 'MyStudy')
                study_load.load_data(from_path=dump_dir)
                print(study_load.ee.dm.get_data_dict_values())
                study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_10_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc_and_updates(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        Test the Undo (go back to None) and Update. 
        Will be finalized in an associated US.
        '''
        print('test_10_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc_and_updates')
        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo = 'sos_trades_core.sos_processes.test'
        mod_id_empty_doe = 'test_driver_build_doe_eval_empty'
        self.study_name = 'MyStudy'

        # create session with empty DoE
        print(
            '################################################################################')
        print('STEP_1: create session with empty DoE')
        study_dump = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()  # configure
        study_dump.dump_data(dump_dir)
        print(study_dump.ee.dm.get_data_dict_values())
        # Check that repo_of_sub_processes and sub_process_folder_name are set
        value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
        print('repo_of_sub_processes: ' +
              str(study_dump.ee.dm.get_data(value_2_print)['value']))
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'

        print(
            '################################################################################')
        print(
            'STEP_2: update with subprocess Hessian selection and filled subprocess data')
        dict_values = self.setup_Hessian_usecase_from_direct_input(
            restricted=True)

        study_dump.load_data(from_input_dict=dict_values)
        study_dump.ee.configure()
        study_dump.dump_data(dump_dir)
        # print(study_dump.ee.dm.get_data_dict_values())
        # Check that repo_of_sub_processes and sub_process_folder_name are set
        value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
        print('repo_of_sub_processes: ')
        print(study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
        print('sub_process_folder_name: ')
        print(study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
        print('usecase_of_sub_process: ')  # Nexiste pas !
        # print(study_dump.ee.dm.get_data(value_2_print)['value'])

        # update with with data Hessian subprocess update from usecase
        print(
            '################################################################################')
        print(
            'STEP_3: update with with data Hessian subprocess update from usecase ')
        my_usecase = 'usecase'
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = my_usecase
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.dump_data(dump_dir)
        # print(study_dump.ee.dm.get_data_dict_values())
        # Check that repo_of_sub_processes and sub_process_folder_name are set
        value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
        print('repo_of_sub_processes: ')
        print(study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
        print('sub_process_folder_name: ')
        print(study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
        print('usecase_of_sub_process: ')
        # print(study_dump.ee.dm.get_data(value_2_print)['value']) #Nexiste pas !
        # update subprocess
        print(
            '################################################################################')
        if 0 == 1:  # Will be used in case of undo or update (cleaning)
            print(
                'STEP_4.1: update subprocess selection by come back to None')
            #
            mod_id = None
            dict_values = {}
            dict_values[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
            study_dump.load_data(from_input_dict=dict_values)
            study_dump.dump_data(dump_dir)
            # print(study_dump.ee.dm.get_data_dict_values())
            # Check that repo_of_sub_processes and sub_process_folder_name are
            # set
            value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
            print('repo_of_sub_processes: ' +
                  study_dump.ee.dm.get_data(value_2_print)['value'])
            value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
            print('sub_process_folder_name: ' +
                  study_dump.ee.dm.get_data(value_2_print)['value'])
            value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
        print(
            '################################################################################')
        print(
            'STEP_4.2: update subprocess selection by come back to Hessian')
        #
        skip = True
        if skip == False:
            mod_id = 'test_disc_hessian'
            dict_values = {}
            dict_values[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
            study_dump.load_data(from_input_dict=dict_values)
            study_dump.dump_data(dump_dir)
            print(study_dump.ee.dm.get_data_dict_values())
            # Check that repo_of_sub_processes and sub_process_folder_name are
            # set
            value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
            print('repo_of_sub_processes: ' +
                  study_dump.ee.dm.get_data(value_2_print)['value'])
            value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
            print('sub_process_folder_name: ' +
                  study_dump.ee.dm.get_data(value_2_print)['value'])
            value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
            print('usecase_of_sub_process: ')
            # print(study_dump.ee.dm.get_data(value_2_print)['value']) #Nexiste
            # pas !
        print(
            '################################################################################')
        print(
            'STEP_4.3: update subprocess selection by replacing by disc1_all_types')
        #
        mod_id = 'test_proc_build_disc1_all_types'
        #mod_id = 'test_sellar_coupling'
        my_usecase = 'usecase1'
        dict_values = {}
        dict_values[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
        dict_values[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = my_usecase
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.dump_data(dump_dir)
        # print(study_dump.ee.dm.get_data_dict_values())
        # Check that repo_of_sub_processes and sub_process_folder_name are set
        value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
        print('repo_of_sub_processes: ')
        print(study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
        print('sub_process_folder_name: ')
        print(study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
        print('usecase_of_sub_process: ')  # Nexiste pas !
        # print(study_dump.ee.dm.get_data(value_2_print)['value'])
        # Run
        flag_run = False  # Will be used in case of undo or update (cleaning)
        flag_local = True
        if flag_run:
            print(
                '################################################################################')
            print('STEP_5: run')
            if flag_local:
                study_dump.run()
            else:
                study_load = BaseStudyManager(
                    repo, mod_id_empty_doe, 'MyStudy')
                study_load.load_data(from_path=dump_dir)
                print(study_load.ee.dm.get_data_dict_values())
                study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)


if '__main__' == __name__:
    my_test = TestMultiScenarioOfDoeEval()
    test_selector = 9
    if test_selector == 1:
        my_test.setUp()
        my_test.test_01_build_doe_eval_with_empty_disc()
    elif test_selector == 2:
        my_test.setUp()
        my_test.test_02_build_doe_eval_with_nested_proc_selection()
    elif test_selector == 3:
        my_test.setUp()
        my_test.test_03_build_doe_eval_with_nested_proc_selection_through_process_driver()
    elif test_selector == 4:
        my_test.setUp()
        my_test.test_04_build_doe_eval_with_nested_proc_selection_through_process_driver()
    elif test_selector == 5:
        my_test.test_05_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_subproc()
    elif test_selector == 6:
        my_test.test_06_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_usecase()
    elif test_selector == 7:
        my_test.test_07_build_doe_eval_with_nested_proc_selection_sellar()
    elif test_selector == 8:
        my_test.test_08_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc()
    elif test_selector == 9:
        my_test.test_09_build_doe_eval_test_GUI_sequence()
    elif test_selector == 10:
        my_test.test_10_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc_and_updates()
