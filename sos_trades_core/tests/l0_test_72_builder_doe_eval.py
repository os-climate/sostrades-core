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

    def setup_usecase(self, restricted=True):
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

    def setup_usecase_bad(self, restricted=True):
        """
        Define a set of data inputs with selected use_case
        """
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        # my_usecase = 'usecase_toto'  # provide a warning if not existing
        # use_case
        my_usecase = 'usecase'
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

    def test_03_build_doe_eval_with_empty_disc(self):
        '''
        Test the creation of the doe without nested disciplines and directly from DoE_eval class
        '''
        print('test_03_build_doe_eval_with_empty_disc')
        builder_list = []

        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        # doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
        #    'DoE_Eval', 'build_doe_eval', builder_list)
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        print(self.exec_eng.display_treeview_nodes())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_04_build_doe_eval_with_nested_proc_selection(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        '''
        print('test_04_build_doe_eval_with_nested_proc_selection')
        # setup an empty doe
        builder_list = []
        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # provide a process (with disciplines) to the set doe
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        dict_values = {}
        dict_values['MyStudy.DoE_Eval.repo_of_sub_processes'] = repo
        dict_values['MyStudy.DoE_Eval.sub_process_folder_name'] = mod_id
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # provide inputs to the set doe with disciplines
        values_dict = self.setup_usecase(restricted=True)[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_05_1_build_doe_eval_with_nested_proc_selection_through_process_driver(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        '''
        print('test_05_1_build_doe_eval_with_nested_proc_selection_through_process_driver')
        # setup an empty doe and configure
        driver_repo = 'sos_trades_core.sos_processes.test'
        driver_mod_id = 'test_driver_build_doe_eval_empty'
        doe_eval_builder = self.exec_eng.factory.get_builder_from_process(
            repo=driver_repo, mod_id=driver_mod_id)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # provide a sub process (with disciplines) to the set doe and configure
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        dict_values = {}
        dict_values['MyStudy.DoE_Eval.repo_of_sub_processes'] = repo
        dict_values['MyStudy.DoE_Eval.sub_process_folder_name'] = mod_id
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # provide data inputs to the set doe with disciplines and configure
        values_dict = self.setup_usecase()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_05_2_build_doe_eval_with_nested_proc_selection_through_process_driver(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        '''
        print('test_05_2_build_doe_eval_with_nested_proc_selection_through_process_driver')
        # setup an empty doe and configure
        driver_repo = 'sos_trades_core.sos_processes.test'
        driver_mod_id = 'test_driver_build_doe_eval_empty'
        doe_eval_builder = self.exec_eng.factory.get_builder_from_process(
            repo=driver_repo, mod_id=driver_mod_id)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # Provide subprocess and provide data input
        values_dict = self.setup_usecase(restricted=False)[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_06_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_subproc(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_06_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_subproc')
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

        dict_values = self.setup_usecase(restricted=False)

        study_dump.load_data(from_input_dict=dict_values)
        study_dump.dump_data(dump_dir)
        print(study_dump.ee.dm.get_data_dict_values())
        # study_dump.ee.configure()
        # print(study_dump.ee.dm.get_data_dict_values())
        # study_dump.run()

        #study_load = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        # study_load.load_data(from_path=dump_dir)
        # print(study_load.ee.dm.get_data_dict_values())
        # study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_07_build_doe_eval_with_nested_proc_selection_through_process_driver_bad_usecase(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        Same as test 06 but with bad usecase
        '''
        print('test_07_build_doe_eval_with_nested_proc_selection_through_process_driver_bad_usecase')
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

        dict_values = self.setup_usecase(restricted=False)
        # dict_values = self.setup_usecase_bad(restricted=False) # Use this for
        # check

        study_dump.load_data(from_input_dict=dict_values)
        study_dump.dump_data(dump_dir)
        print(study_dump.ee.dm.get_data_dict_values())
        study_dump.ee.configure()
        print(study_dump.ee.dm.get_data_dict_values())
        study_dump.run()

        study_load = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
        study_load.load_data(from_path=dump_dir)
        print(study_load.ee.dm.get_data_dict_values())
        # study_load.run()
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

        print("toto")
        print(dict_values)
        study_dump.load_data(from_input_dict=dict_values)
        # print(study_dump.ee.display_treeview_nodes(True))

        study_dump.dump_data(dump_dir)
        print(study_dump.ee.dm.get_data_dict_values())
        print(study_dump.ee.display_treeview_nodes(True))
        skip_run = False
        if skip_run == False:
            local_run = True
            if local_run == True:
                study_dump.run()
                print(study_dump.ee.dm.get_data_dict_values())
            else:
                study_load = BaseStudyManager(
                    repo, mod_id_empty_doe, 'MyStudy')
                study_load.load_data(from_path=dump_dir)
                print(study_load.ee.dm.get_data_dict_values())
                study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_09_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc_and_updates(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_09_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc_and_updates')
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
              study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
        # print('sub_process_folder_name: ' +
        #      study_dump.ee.dm.get_data(value_2_print)['value'])
        # update with subprocess Hessian selection and filled subprocess data
        print(
            '################################################################################')
        print(
            'STEP_2: update with subprocess Hessian selection and filled subprocess data')
        dict_values = self.setup_usecase(restricted=False)
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.ee.configure()
        study_dump.dump_data(dump_dir)
        print(study_dump.ee.dm.get_data_dict_values())
        # Check that repo_of_sub_processes and sub_process_folder_name are set
        value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
        print('repo_of_sub_processes: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
        print('sub_process_folder_name: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
        print('usecase_of_sub_process: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])

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
        print(study_dump.ee.dm.get_data_dict_values())
        # Check that repo_of_sub_processes and sub_process_folder_name are set
        value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
        print('repo_of_sub_processes: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
        print('sub_process_folder_name: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
        print('usecase_of_sub_process: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        # update subprocess
        print(
            '################################################################################')
        if 0 == 1:
            print(
                'STEP_4.1: update subprocess selection by come back to None')
            #
            mod_id = 'None'
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
            print('usecase_of_sub_process: ' +
                  study_dump.ee.dm.get_data(value_2_print)['value'])
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
        print(study_dump.ee.dm.get_data_dict_values())
        # Check that repo_of_sub_processes and sub_process_folder_name are set
        value_2_print = f'{self.study_name}.DoE_Eval.repo_of_sub_processes'
        print('repo_of_sub_processes: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.sub_process_folder_name'
        print('sub_process_folder_name: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        value_2_print = f'{self.study_name}.DoE_Eval.usecase_of_sub_process'
        print('usecase_of_sub_process: ' +
              study_dump.ee.dm.get_data(value_2_print)['value'])
        # Run
        flag_run = False
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

    def test_10_build_doe_eval_with_nested_proc_selection_sellar(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        Here the study is used as in the study defined in the GUI (if test work then gui should work!)
        '''
        print('test_10_build_doe_eval_with_nested_proc_selection_sellar')
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
        print(study_dump.ee.display_treeview_nodes(True))

        skip = False
        if skip == False:
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
            print(study_dump.ee.display_treeview_nodes(True))

            study_dump.dump_data(dump_dir)
            # print(study_dump.ee.dm.get_data_dict_values())
            print(study_dump.ee.display_treeview_nodes(True))
            study_dump.run()
            print(study_dump.ee.dm.get_data_dict_values())

            #study_load = BaseStudyManager(repo, mod_id_empty_doe, 'MyStudy')
            # study_load.load_data(from_path=dump_dir)
            # print(study_load.ee.dm.get_data_dict_values())
            # study_load.run()
            from shutil import rmtree
            rmtree(dump_dir)


if '__main__' == __name__:
    my_test = TestMultiScenarioOfDoeEval()
    test_selector = 10
    if test_selector == 3:
        my_test.setUp()
        my_test.test_03_build_doe_eval_with_empty_disc()
    elif test_selector == 4:
        my_test.setUp()
        my_test.test_04_build_doe_eval_with_nested_proc_selection()
    elif test_selector == 5_1:
        my_test.setUp()
        my_test.test_05_1_build_doe_eval_with_nested_proc_selection_through_process_driver()
    elif test_selector == 5_2:
        my_test.setUp()
        my_test.test_05_2_build_doe_eval_with_nested_proc_selection_through_process_driver()
    elif test_selector == 6:
        my_test.test_06_build_doe_eval_with_nested_proc_selection_through_process_driver_Hessian_subproc()
    elif test_selector == 7:
        my_test.test_07_build_doe_eval_with_nested_proc_selection_through_process_driver_bad_usecase()
    elif test_selector == 8:
        my_test.test_08_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc()
    elif test_selector == 9:
        my_test.test_09_build_doe_eval_with_nested_proc_selection_through_process_driver_several_subproc_and_updates()
    elif test_selector == 10:
        my_test.test_10_build_doe_eval_with_nested_proc_selection_sellar()
