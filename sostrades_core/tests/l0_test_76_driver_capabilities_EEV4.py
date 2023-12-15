'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/03 Copyright 2023 Capgemini

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
import copy
import logging
from logging import Handler
from time import time

from pandas._testing import assert_frame_equal

from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""

import unittest
from numpy import array, std, NaN
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

import os
from os.path import dirname, join
import math

from importlib import import_module


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestSoSDOEScenario(unittest.TestCase):

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

    def setUp(self):

        self.sampling_method_doe = 'doe_algo'
        self.study_name = 'doe'
        self.ns = f'{self.study_name}'
        self.sc_name = "SellarDoeScenario"
        self.c_name = "SellarCoupling"
        dspace_dict = {'variable': ['x', 'z_in', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_dict_optim = {'variable': ['x', 'z_in', 'y_1', 'y_2'],
                             'value': [[1.], [5., 2.], [1.], [1.]],
                             'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                             'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                             'enable_variable': [True, True, True, True],
                             'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_dict_eval = {'variable': ['x', 'z_in'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }

        self.dspace = pd.DataFrame(dspace_dict)
        self.dspace_eval = pd.DataFrame(dspace_dict_eval)
        self.dspace_optim = pd.DataFrame(dspace_dict_optim)

        input_selection_local_dv_x = {'selected_input': [True, True, False, False, False],
                                      'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z_in']}
        self.input_selection_local_dv_x = pd.DataFrame(
            input_selection_local_dv_x)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z_in']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z_in']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        input_selection_local_dv = {'selected_input': [True, False, False, False, False],
                                    'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                  'y_2',
                                                  'z_in']}
        self.input_selection_local_dv = pd.DataFrame(input_selection_local_dv)

        output_selection_obj = {'selected_output': [False, False, True, False, False],
                                'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj = pd.DataFrame(output_selection_obj)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj_y1_y2 = pd.DataFrame(
            output_selection_obj_y1_y2)

        self.repo = 'sostrades_core.sos_processes.test'

    def test_1_simple_disc_output_to_input_driver_DoeEval(self):
        """
        This test checks that the coupling between the output of a simple discipline and the input of a driver
        subprocess works. The doe_eval will be made sith a lhs on x.
        """

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        repo_name = self.repo + '.tests_driver_eval.mono'
        proc_name = "test_mono_driver_sellar_simple_disc"
        doe_eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()
        builder_mode_input = {}
        exec_eng.load_study_from_input_dict(builder_mode_input)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = self.sampling_method_doe
        disc_dict[f'{self.ns}.SampleGenerator.sampling_generation_mode'] = "at_run_time"
        disc_dict[f'{self.ns}.SampleGenerator.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.SampleGenerator.design_space'] = dspace_x
        disc_dict[f'{self.ns}.SampleGenerator.algo_options'] = {
            'n_samples': n_samples}
        disc_dict[f'{self.ns}.Eval.with_sample_generator'] = True
        disc_dict[f'{self.ns}.Eval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.Eval.gather_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.Eval.x'] = array([1.])
        values_dict[f'{self.ns}.Eval.z'] = array([1.])
        values_dict[f'{self.ns}.Eval.y_1'] = array([1.])
        values_dict[f'{self.ns}.Eval.y_2'] = array([1.])
        values_dict[f'{self.ns}.Eval.z_in'] = 2 * array([1., 1.])
        values_dict[f'{self.ns}.Eval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ Simple_Disc',
                       f'\t|_ SampleGenerator',
                       f'\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1',
                       ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes(exec_display=True)
        eval_disc = exec_eng.dm.get_disciplines_with_name('doe.Eval')[0]

        # eval_disc_samples = eval_disc.get_sosdisc_outputs(
        #    'samples_inputs_df')
        eval_disc_obj = eval_disc.get_sosdisc_outputs('obj_dict')
        eval_disc_y1 = eval_disc.get_sosdisc_outputs('y_1_dict')
        eval_disc_y2 = eval_disc.get_sosdisc_outputs('y_2_dict')
        # self.assertEqual(len(eval_disc_samples), n_samples + 1)
        self.assertEqual(len(eval_disc_obj), n_samples + 1)
        reference_dict_eval_disc_y1 = {'scenario_1': array([10.491019856682016]),
                                       'scenario_2': array([7.247824531594309]),
                                       'scenario_3': array([2.9753409599263483]),
                                       'scenario_4': array([1.7522749587335193]),
                                       'scenario_5': array([9.384097972066053]),
                                       'scenario_6': array([8.36704386923391]),
                                       'scenario_7': array([4.479056921478663]),
                                       'scenario_8': array([5.286891081070988]),
                                       'scenario_9': array([3.240108355137796]),
                                       'scenario_10': array([6.194561090631401]),
                                       'reference_scenario': array([2.29689011157193])}
        reference_dict_eval_disc_y2 = {'scenario_1': array([5.238984386606706]),
                                       'scenario_2': array([4.692178398916815]),
                                       'scenario_3': array([3.7249176675790494]),
                                       'scenario_4': array([3.3237352298452736]),
                                       'scenario_5': array([5.063347510823095]),
                                       'scenario_6': array([4.892584289045681]),
                                       'scenario_7': array([4.116378255765888]),
                                       'scenario_8': array([4.2993240487306235]),
                                       'scenario_9': array([3.8000300983977455]),
                                       'scenario_10': array([4.488887520686984]),
                                       'reference_scenario': array([3.5155494421403515])}
        for key in eval_disc_y1.keys():
            self.assertAlmostEqual(
                eval_disc_y1[key][0], reference_dict_eval_disc_y1[key][0])
        for key in eval_disc_y2.keys():
            self.assertAlmostEqual(
                eval_disc_y2[key][0], reference_dict_eval_disc_y2[key][0])

    def _test_3_simple_custom_driver(self):
        # FIXME: Out of scope current US. This test will have to be adapted to
        # the new architecture in the future.

        study_name = 'root'
        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        proc_name = "test_disc1_custom_driver"
        driver_builder = factory.get_builder_from_process(repo=self.repo,
                                                          mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            driver_builder)

        exec_eng.configure()

        in_dict = {}
        in_dict[f'{study_name}.Driver1.output_full_name'] = f'{study_name}.y'
        in_dict[f'{study_name}.x'] = array([10.])
        in_dict[f'{study_name}.Driver1.Disc1.a'] = array([5.])
        in_dict[f'{study_name}.Driver1.Disc1.b'] = array([25431.])
        exec_eng.load_study_from_input_dict(in_dict)

        # check expected output from execution
        exec_eng.execute()
        self.assertEqual(exec_eng.dm.get_value(
            'root.Driver1.output_squared'), array([649281361.]))

        # check that the root process knows all the numerical inputs of the
        # entire subprocess
        root_inputs = exec_eng.root_process.get_input_data_names()
        self.assertIn('root.linearization_mode', root_inputs)
        self.assertIn('root.Driver1.linearization_mode', root_inputs)
        self.assertIn('root.Driver1.Disc1.linearization_mode', root_inputs)

    def _test_4_simple_discs_work_with_io_of_DoeEval_and_its_subdisciplines(self):
        """
        This test checks that the coupling between the output of a simple discipline and the input of a driver and its
        subprocess works, as well as the coupling of the Doe+Eval output and its subdisciplines outputs with another
        simple discipline.
        """
        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.mono"
        proc_name = "test_mono_driver_sample_generator_sellar_disc1_disc2"
        builders = factory.get_builder_from_process(repo=repo_name,
                                                    mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            builders)

        exec_eng.configure()
        builder_mode_input = {}
        exec_eng.load_study_from_input_dict(builder_mode_input)

        # -- set up disciplines in Scenario
        values_dict = {}
        # DoE + Eval inputs
        n_samples = 10
        values_dict[f'{self.ns}.SampleGenerator.sampling_method'] = self.sampling_method_doe
        values_dict[f'{self.ns}.SampleGenerator.sampling_generation_mode'] = "at_run_time"
        values_dict[f'{self.ns}.SampleGenerator.design_space'] = dspace_x
        # values_dict[f'{self.ns}.SampleGenerator.algo_options'] = {'n_samples': n_samples}
        values_dict[f'{self.ns}.Simple_Disc1.added_algo_options'] = {
            'n_samples': n_samples}
        values_dict[f'{self.ns}.eval_inputs'] = self.input_selection_x
        values_dict[f'{self.ns}.gather_outputs'] = self.output_selection_obj_y1_y2

        # Sellar inputs
        local_dv = 10.
        # values_dict = {}
        values_dict[f'{self.ns}.Eval.x'] = array([1.])
        values_dict[f'{self.ns}.Eval.y_1'] = array([1.])
        values_dict[f'{self.ns}.Eval.y_2'] = array([1.])
        values_dict[f'{self.ns}.Eval.subprocess.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.Eval.z_in'] = 2 * \
                                              array([1., 1.])  # Input of SimpleDisc1
        # values_dict[f'{self.ns}.c_1'] = array([1.])            #Input of
        # SimpleDisc2
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ Simple_Disc1',
                       '\t|_ SampleGenerator',
                       '\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1',
                       '\t|_ Simple_Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes(exec_display=True)
        eval_disc = exec_eng.dm.get_disciplines_with_name('doe.Eval')[0]

        assert exec_eng.root_process.proxy_disciplines[2].proxy_disciplines[0].is_sos_coupling

        z = exec_eng.dm.get_value('doe.z')
        if z[0] > 0.5:
            self.assertEqual(exec_eng.dm.get_value(
                'doe.SampleGenerator.sampling_algo'), "lhs")
        else:
            self.assertEqual(exec_eng.dm.get_value(
                'doe.SampleGenerator.sampling_algo'), "fullfact")

        # eval_disc_samples = eval_disc.get_sosdisc_outputs(
        #    'samples_inputs_df')
        eval_disc_obj = eval_disc.get_sosdisc_outputs('obj_dict')
        eval_disc_y1 = eval_disc.get_sosdisc_outputs('y_1_dict')
        eval_disc_y2 = eval_disc.get_sosdisc_outputs('y_2_dict')
        # self.assertEqual(len(eval_disc_samples), n_samples + 1)
        self.assertEqual(len(eval_disc_obj), n_samples + 1)
        reference_dict_eval_disc_y1 = {'scenario_1': array([10.491019856682016]),
                                       'scenario_2': array([7.247824531594309]),
                                       'scenario_3': array([2.9753409599263483]),
                                       'scenario_4': array([1.7522749587335193]),
                                       'scenario_5': array([9.384097972066053]),
                                       'scenario_6': array([8.36704386923391]),
                                       'scenario_7': array([4.479056921478663]),
                                       'scenario_8': array([5.286891081070988]),
                                       'scenario_9': array([3.240108355137796]),
                                       'scenario_10': array([6.194561090631401]),
                                       'reference_scenario': array([2.29689011157193])}
        reference_dict_eval_disc_y2 = {'scenario_1': array([5.238984386606706]),
                                       'scenario_2': array([4.692178398916815]),
                                       'scenario_3': array([3.7249176675790494]),
                                       'scenario_4': array([3.3237352298452736]),
                                       'scenario_5': array([5.063347510823095]),
                                       'scenario_6': array([4.892584289045681]),
                                       'scenario_7': array([4.116378255765888]),
                                       'scenario_8': array([4.2993240487306235]),
                                       'scenario_9': array([3.8000300983977455]),
                                       'scenario_10': array([4.488887520686984]),
                                       'reference_scenario': array([3.5155494421403515])}
        for key in eval_disc_y1.keys():
            self.assertAlmostEqual(
                eval_disc_y1[key][0], reference_dict_eval_disc_y1[key][0])
        for key in eval_disc_y2.keys():
            self.assertAlmostEqual(
                eval_disc_y2[key][0], reference_dict_eval_disc_y2[key][0])

        self.assertEqual(exec_eng.dm.get_value('doe.out_simple2'),
                         exec_eng.dm.get_value('doe.c_1') * std(
                             list(exec_eng.dm.get_value('doe.y_1_dict').values())[:-1]))

    def test_5_simple_disc_DoeEval_check_num_in_grammar_and_root_process(self):
        """
        This test checks that the coupling between the output of a simple discipline and the input of a driver
        subprocess works. The doe_eval will be made sith a lhs on x.
        """

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.mono"
        proc_name = "test_mono_driver_sellar_simple_disc"
        doe_eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()
        builder_mode_input = {}
        exec_eng.load_study_from_input_dict(builder_mode_input)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = self.sampling_method_doe
        disc_dict[f'{self.ns}.SampleGenerator.sampling_generation_mode'] = "at_run_time"
        disc_dict[f'{self.ns}.SampleGenerator.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.SampleGenerator.design_space'] = dspace_x
        disc_dict[f'{self.ns}.SampleGenerator.algo_options'] = {
            'n_samples': n_samples}
        disc_dict[f'{self.ns}.Eval.with_sample_generator'] = True
        disc_dict[f'{self.ns}.Eval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.Eval.gather_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.Eval.x'] = array([1.])
        values_dict[f'{self.ns}.Eval.y_1'] = array([1.])
        values_dict[f'{self.ns}.Eval.y_2'] = array([1.])
        values_dict[f'{self.ns}.Eval.z_in'] = 2 * array([1., 1.])
        values_dict[f'{self.ns}.Eval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ Simple_Disc',
                       f'\t|_ SampleGenerator',
                       f'\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1',
                       ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes(exec_display=True)
        proxy_disc2 = exec_eng.root_process.proxy_disciplines[
            2].proxy_disciplines[0].proxy_disciplines[1]
        ns_id_cache_disc2_own_data_structure = proxy_disc2._io_ns_map_in['cache_type']
        ns_id_cache_disc2_ns_manager = id(
            exec_eng.ns_manager.get_local_namespace(proxy_disc2))
        self.assertEqual(ns_id_cache_disc2_own_data_structure,
                         ns_id_cache_disc2_ns_manager)

        data_in_proxy_disc2 = proxy_disc2._data_in
        var_dict_dm_in = exec_eng.dm.get_data(
            'doe.Eval.subprocess.Sellar_2.cache_type')
        var_dict_data_in = data_in_proxy_disc2[(
            'cache_type', ns_id_cache_disc2_own_data_structure)]
        var_dict_data_in_root = exec_eng.root_process._data_in[(
            'cache_type', ns_id_cache_disc2_own_data_structure)]
        self.assertEqual(var_dict_dm_in, var_dict_data_in)
        self.assertEqual(var_dict_dm_in, var_dict_data_in_root)

        proxy_disc_sellar_problem = exec_eng.root_process.proxy_disciplines[
            2].proxy_disciplines[0].proxy_disciplines[0]
        ns_id_cache_disc_sellar_problem_own_data_structure = proxy_disc_sellar_problem._io_ns_map_out[
            'c_1']
        # ns_id_cache_disc_sellar_problem_ns_manager = id(
        #     exec_eng.ns_manager.get_shared_ns_dict()['ns_OptimSellar'])
        # self.assertEqual(ns_id_cache_disc_sellar_problem_own_data_structure,
        #                  ns_id_cache_disc_sellar_problem_ns_manager)

        data_out_proxy_disc_sellar_problem = proxy_disc_sellar_problem._data_out
        var_dict_dm_out = exec_eng.dm.get_data('doe.Eval.c_1')
        var_dict_data_out = data_out_proxy_disc_sellar_problem[(
            'c_1', ns_id_cache_disc_sellar_problem_own_data_structure)]
        var_dict_data_out_root = exec_eng.root_process._data_out[(
            'c_1', ns_id_cache_disc_sellar_problem_own_data_structure)]
        self.assertEqual(var_dict_dm_out, var_dict_data_out)
        self.assertEqual(var_dict_dm_out, var_dict_data_out_root)

    def test_6_Eval_User_Defined_samples_variables_not_in_root_process(self):
        """
        This test checks that the custom samples applied to an Eval driver delivers expected outputs. The user_defined
        sampling is applied to variables that are not in the root process, to check that namespacing works properly.
        It is a non regression test
        """

        study_name = 'root'
        ns = study_name

        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.mono"
        proc_name = "test_mono_driver_simple"
        eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()
        builder_mode_input = {}
        exec_eng.load_study_from_input_dict(builder_mode_input)

        exp_tv_list = [f'Nodes representation for Treeview {ns}',
                       '|_ root',
                       f'\t|_ Eval',
                       '\t\t|_ Disc1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes(exec_display=True)

        assert not exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].is_sos_coupling

        # -- Eval inputs
        input_selection_a = {'selected_input': [False, True, False],
                             'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {'selected_output': [False, True],
                                'full_name': ['y', 'Disc1.indicator']}
        output_selection_ind = pd.DataFrame(output_selection_ind)

        disc_dict = {f'{ns}.Eval.eval_inputs': input_selection_a,
                     f'{ns}.Eval.gather_outputs': output_selection_ind}

        a_values = [2.0, 4.0, 6.0, 8.0, 10.0]

        samples_dict = {SampleGeneratorWrapper.SELECTED_SCENARIO: [True] * 5,
                        SampleGeneratorWrapper.SCENARIO_NAME: [f'scenario_{i}' for i in range(1, 6)],
                        'Disc1.a': a_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{ns}.Eval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # -- Discipline inputs
        private_values = {
            f'{ns}.Eval.x': 10.,
            f'{ns}.Eval.Disc1.a': 5.,
            f'{ns}.Eval.Disc1.b': 25431.,
            f'{ns}.Eval.y': 4.,
            f'{ns}.Eval.Disc1.indicator': 53.}
        exec_eng.load_study_from_input_dict(private_values)

        exec_eng.execute()

        root_outputs = exec_eng.root_process.get_output_data_names()
        self.assertIn('root.Eval.Disc1.indicator_dict', root_outputs)

        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]

        # eval_disc_samples = eval_disc.get_sosdisc_outputs(
        #    'samples_inputs_df')
        # self.assertEqual(
        #    list(eval_disc_samples['Disc1.a'][0:-1]), a_values)

        eval_disc_ind = eval_disc.get_sosdisc_outputs(
            'Disc1.indicator_dict')

        self.assertEqual(len(eval_disc_ind), 6)
        i = 0
        # for key in eval_disc_ind.keys():
        #     self.assertAlmostEqual(eval_disc_ind[key],
        #                            private_values[f'{ns}.Eval.Disc1.b'] * eval_disc_samples['Disc1.a'][i])
        #     i += 1

    def test_7_Coupling_of_Coupling_to_check_data_io(self):
        """
        TO BE COMPLETED
        """

        study_name = 'root'

        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        proc_name = "test_disc1_disc2_coupling_of_coupling"
        coupling_of_coupling_builder = factory.get_builder_from_process(repo=self.repo,
                                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            coupling_of_coupling_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {study_name}',
                       '|_ root',
                       f'\t|_ UpperCoupling',
                       '\t\t|_ LowerCoupling',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes(exec_display=True)

        # -- set up disciplines
        private_values = {
            study_name + '.x': 10.,
            study_name + '.UpperCoupling.LowerCoupling.Disc1.a': 5.,
            study_name + '.UpperCoupling.LowerCoupling.Disc1.b': 7.,
            study_name + '.Eval.y': array([4.]),
            study_name + '.UpperCoupling.LowerCoupling.Disc2.power': 3,
            study_name + '.UpperCoupling.LowerCoupling.Disc2.constant': 4.,
        }
        exec_eng.load_study_from_input_dict(private_values)
        exec_eng.execute()

        for disc in [exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[0],
                     # discipline with no coupled inputs
                     exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0],
                     exec_eng.root_process.proxy_disciplines[0], exec_eng.root_process]:  # couplings
            io_ns_map_in = disc._io_ns_map_in
            for var, identifier in io_ns_map_in.items():
                var_tuple = (var, identifier)
                self.assertEqual(identifier, id(
                    exec_eng.root_process._data_in[var_tuple]['ns_reference']))

    def test_8_Eval_reconfiguration_adding_again_User_Defined_samples_if_still_in_eval_inputs(self):
        """
        This test checks that samples dataframe is properly modified and generated when eval_inputs is modified and
        , consequently, a reconfiguration is undertaken (since eval_inputs is a structuring variable).
        """
        # FIXME: review test with data integrity logic and overwrite_samples_df
        study_name = 'root'
        ns = study_name

        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.mono"
        proc_name = "test_mono_driver_simple"
        eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()
        initial_input = {f'{ns}.Eval.with_sample_generator': True,
                         f'{ns}.SampleGenerator.sampling_method': 'simple'}
        exec_eng.load_study_from_input_dict(initial_input)

        exp_tv_list = [f'Nodes representation for Treeview {ns}',
                       '|_ root',
                       '\t|_ SampleGenerator',
                       f'\t|_ Eval',
                       '\t\t|_ Disc1',
                       ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes(exec_display=True)

        assert not exec_eng.root_process.proxy_disciplines[1].proxy_disciplines[0].is_sos_coupling

        # -- Eval inputs
        input_selection_a = {'selected_input': [False, True, False],
                             'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {'selected_output': [False, True],
                                'full_name': ['y', 'Disc1.indicator']}
        output_selection_ind = pd.DataFrame(output_selection_ind)

        disc_dict = {f'{ns}.Eval.eval_inputs': input_selection_a,
                     f'{ns}.Eval.gather_outputs': output_selection_ind}

        exec_eng.load_study_from_input_dict(disc_dict)

        # a_values = [array([2.0]), array([4.0]), array(
        #     [6.0]), array([8.0]), array([10.0])]
        a_values = [2.0, 4.0, 6.0, 8.0, 10.0]

        samples_dict = {SampleGeneratorWrapper.SELECTED_SCENARIO: [True] * 5,
                        SampleGeneratorWrapper.SCENARIO_NAME: ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4',
                                                               'scenario_5'],
                        'Disc1.a': a_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{ns}.Eval.samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # -- Discipline inputs
        # private_values = {
        #     f'{ns}.x': array([10.]),
        #     f'{ns}.Eval.Disc1.a': array([5.]),
        #     f'{ns}.Eval.Disc1.b': array([25431.]),
        #     f'{ns}.y': array([4.]),
        #     f'{ns}.Eval.Disc1.indicator': array([53.])}
        private_values = {
            f'{ns}.Eval.x': 10.,
            f'{ns}.Eval.Disc1.a': 5.,
            f'{ns}.Eval.Disc1.b': 25431.,
            f'{ns}.Eval.y': 4.,
            f'{ns}.Eval.Disc1.indicator': 53.}
        exec_eng.load_study_from_input_dict(private_values)

        exec_eng.execute()

        root_outputs = exec_eng.root_process.get_output_data_names()
        self.assertIn('root.Eval.Disc1.indicator_dict', root_outputs)

        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]

        # eval_disc_samples = eval_disc.get_sosdisc_outputs(
        #    'samples_inputs_df')
        # self.assertEqual(
        #    list(eval_disc_samples['Disc1.a'][0:-1]), a_values)

        eval_disc_ind = eval_disc.get_sosdisc_outputs(
            'Disc1.indicator_dict')

        self.assertEqual(len(eval_disc_ind), 6)
        i = 0
        # for key in eval_disc_ind.keys():
        #     self.assertAlmostEqual(eval_disc_ind[key],
        #                            private_values[f'{ns}.Eval.Disc1.b'] * eval_disc_samples['Disc1.a'][i])
        #     i += 1

        # 1. Samples and eval_inputs equal
        input_selection_a_b = {'selected_input': [False, True, True],
                               'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_a_b = pd.DataFrame(input_selection_a_b)
        disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_a_b

        # Change of samples
        # b_values = [array([1.0]), array([3.0]), array(
        #     [5.0]), array([1.0]), array([1.0])]
        b_values = [1.0, 3.0, 5.0, 7.0, 9.0]
        new_samples_dict = {SampleGeneratorWrapper.SELECTED_SCENARIO: [True] * 5,
                            SampleGeneratorWrapper.SCENARIO_NAME: ['scenario_1', 'scenario_2', 'scenario_3',
                                                                   'scenario_4', 'scenario_5'],
                            'Disc1.a': a_values, 'Disc1.b': b_values}
        new_samples_df = pd.DataFrame(new_samples_dict)
        disc_dict[f'{ns}.Eval.samples_df'] = new_samples_df

        # Reconfigure et re-execute
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()

        # Check samples
        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]
        # eval_disc_samples = eval_disc.get_sosdisc_outputs(
        #    'samples_inputs_df')
        # self.assertEqual(
        #     list(eval_disc_samples['Disc1.a'][0:-1]), a_values)
        # self.assertEqual(
        #     list(eval_disc_samples['Disc1.b'][0:-1]), b_values)

        # 2. More eval_inputs than samples and sample included in eval_inputs
        # Change of eval_inputs
        input_selection_x_a = {'selected_input': [True, True, False],
                               'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_x_a = pd.DataFrame(input_selection_x_a)
        disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_x_a

        # Change of samples
        new_samples_dict2 = {SampleGeneratorWrapper.SELECTED_SCENARIO: [True] * 5,
                             SampleGeneratorWrapper.SCENARIO_NAME: ['scenario_1', 'scenario_2', 'scenario_3',
                                                                    'scenario_4', 'scenario_5'],
                             'Disc1.a': a_values}
        new_samples_df2 = pd.DataFrame(new_samples_dict2)
        disc_dict[f'{ns}.Eval.samples_df'] = new_samples_df2

        # Reconfigure
        exec_eng.load_study_from_input_dict(disc_dict)

        # Check samples
        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]
        eval_disc_samples = eval_disc.get_sosdisc_inputs(
            'samples_df')
        # self.assertEqual(list(eval_disc_samples['x'][0:-1]), x_values)
        self.assertEqual(list(eval_disc_samples['Disc1.a']), a_values)
        # TODO: [WIP] deactivate bc x is not in samples_df here --> change by data integrity check?
        # x_all_None = True
        # for element in list(eval_disc_samples['x']):
        #     if element is None or math.isnan(element):  # TODO: verify equivalence in gui.
        #         pass
        #     else:
        #         x_all_None = False
        #         break
        # assert x_all_None == True

        # 3. More eval_inputs than samples and samples not included in eval_inputs
        # Change of eval_inputs
        input_selection_x_b = {'selected_input': [True, False, True],
                               'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_x_b = pd.DataFrame(input_selection_x_b)
        disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_x_b

        # Change of samples
        new_samples_dict3 = {SampleGeneratorWrapper.SELECTED_SCENARIO: [True] * 5,
                             SampleGeneratorWrapper.SCENARIO_NAME: ['scenario_1', 'scenario_2', 'scenario_3',
                                                                    'scenario_4', 'scenario_5'],
                             'Disc1.a': a_values}
        new_samples_df3 = pd.DataFrame(new_samples_dict3)
        disc_dict[f'{ns}.Eval.samples_df'] = new_samples_df3

        # Reconfigure
        exec_eng.load_study_from_input_dict(disc_dict)

        # Check samples
        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]
        eval_disc_samples = eval_disc.get_sosdisc_inputs(
            'samples_df')
        # self.assertEqual(list(eval_disc_samples['x'][0:-1]), x_values)
        # self.assertEqual(list(eval_disc_samples['Eval.Disc1.a']), a_values)

        # TODO: [WIP] deactivate bc x and b are not in samples_df here --> change by data integrity check?
        # x_all_nan = True
        # for element in list(eval_disc_samples['x']):
        #     if element is None or math.isnan(element):
        #         pass
        #     else:
        #         x_all_nan = False
        #         break
        # assert x_all_nan == True
        #
        # b_all_nan = True
        # for element in list(eval_disc_samples['Disc1.b']):
        #     if element is None or math.isnan(element):
        #         pass
        #     else:
        #         b_all_nan = False
        #         break
        # assert b_all_nan == True

        # 4. More samples than eval_inputs and samples partially included in eval_inputs
        # Change of eval_inputs
        input_selection_x_a = {'selected_input': [False, True, False],
                               'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_x_a = pd.DataFrame(input_selection_x_a)
        disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_x_a

        # Change of samples
        new_samples_dict = {SampleGeneratorWrapper.SELECTED_SCENARIO: [True] * 5,
                            SampleGeneratorWrapper.SCENARIO_NAME: ['scenario_1', 'scenario_2', 'scenario_3',
                                                                   'scenario_4', 'scenario_5'],
                            'Disc1.a': a_values, 'Disc1.b': b_values}
        new_samples_df = pd.DataFrame(new_samples_dict)
        disc_dict[f'{ns}.Eval.samples_df'] = new_samples_df

        # Reconfigure
        exec_eng.load_study_from_input_dict(disc_dict)

        # Check samples
        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]
        eval_disc_samples = eval_disc.get_sosdisc_inputs(
            'samples_df')
        self.assertEqual(list(eval_disc_samples['Disc1.a']), a_values)

        # 5. Eval_inputs and samples do not coincide at all.
        # Change of eval_inputs
        input_selection_x_a = {'selected_input': [True, False, False],
                               'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_x_a = pd.DataFrame(input_selection_x_a)
        disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_x_a

        # Change of samples
        new_samples_dict = {SampleGeneratorWrapper.SELECTED_SCENARIO: [True] * 5,
                            SampleGeneratorWrapper.SCENARIO_NAME: ['scenario_1', 'scenario_2', 'scenario_3',
                                                                   'scenario_4', 'scenario_5'],
                            'Disc1.a': a_values, 'Disc1.b': b_values}
        new_samples_df = pd.DataFrame(new_samples_dict)
        disc_dict[f'{ns}.Eval.samples_df'] = new_samples_df

        # Reconfigure
        exec_eng.load_study_from_input_dict(disc_dict)

        # Check samples
        eval_disc = exec_eng.dm.get_disciplines_with_name(
            study_name + '.Eval')[0]
        eval_disc_samples = eval_disc.get_sosdisc_inputs(
            'samples_df')

        # TODO: [WIP] deactivate bc x IS not in samples_df here --> change by data integrity check?
        # x_all_nan = True
        # for element in list(eval_disc_samples['x']):
        #     if element is None or math.isnan(element):
        #         pass
        #     else:
        #         x_all_nan = False
        #         break
        # assert x_all_nan == True

    def test_9_nested_very_simple_multi_scenarios(self):
        from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver_of_multi_driver.usecase_without_ref import \
            Study
        study_name = 'root'
        ns = study_name
        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = "test_multi_driver_of_multi_driver"
        eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()

        usecase = Study(execution_engine=exec_eng)
        usecase.study_name = ns
        values_dict = usecase.setup_usecase()

        exec_eng.load_study_from_input_dict(values_dict[0])
        # print(exec_eng.display_treeview_nodes(exec_display=True))
        # print(exec_eng.root_process.display_proxy_subtree(
        #     callback=lambda x: x.is_configured()))
        exp_ns_tree = 'Nodes representation for Treeview root\n' \
                      '|_ root\n' \
                      '\t|_ outer_ms\n' \
                      '\t\t|_ scenario_1\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3\n' \
                      '\t\t\t|_ inner_ms_gather\n' \
                      '\t\t|_ scenario_2\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3\n' \
                      '\t\t\t|_ inner_ms_gather\n' \
                      '\t|_ outer_ms_gather'
        exp_proxy_tree = '|_ root  (ProxyCoupling) [True]\n' \
                         '    |_ root.outer_ms  (ProxyMultiInstanceDriver) [True]\n' \
                         '    |_ root.outer_ms_gather  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_1.inner_ms  (ProxyMultiInstanceDriver) [True]\n' \
                         '    |_ root.outer_ms.scenario_1.Disc3  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_2.inner_ms  (ProxyMultiInstanceDriver) [True]\n' \
                         '    |_ root.outer_ms.scenario_2.Disc3  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_1.inner_ms_gather  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_2.inner_ms_gather  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_1.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_1.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_2.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n' \
                         '    |_ root.outer_ms.scenario_2.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]'

        self.assertEqual(exec_eng.display_treeview_nodes(exec_display=True),
                         exp_ns_tree)
        self.assertEqual(exec_eng.root_process.display_proxy_subtree(callback=lambda x: x.is_configured()),
                         exp_proxy_tree)

        exec_eng.execute()

        scenario_list_outer = ['scenario_1', 'scenario_2']
        scenario_list_inner = ['name_1', 'name_2']
        for i, sc in enumerate(scenario_list_outer):
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.o'),
                             usecase.constant[i] + usecase.z[i] ** usecase.power[i])
            for j, name in enumerate(scenario_list_inner):
                self.assertEqual(
                    exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.indicator'),
                    usecase.a[i] * usecase.b[i][j])
                self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.y'),
                                 usecase.a[i] * usecase.x[i] + usecase.b[i][j])

    def test_10_nested_very_simple_multi_scenarios_with_archi_builder(self):
        """
        This test builds a nested multi scenario using the DriverEvaluator where the core subprocess is composed of two
        archi builders Business and Production. The outer multi scenario driver adds variations on the business process
        whereas the inner multi scenario driver represents scenarios on the Production process. The test is load from a
        usecase and it checks only the treeviews both for namespaces and for proxy objects.
        """
        from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver_of_multi_driver_archibuilder.usecase import \
            Study
        study_name = 'root'
        ns = study_name
        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = "test_multi_driver_of_multi_driver_archibuilder"
        eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()
        usecase = Study(execution_engine=exec_eng)
        usecase.study_name = ns
        values_dict = usecase.setup_usecase()

        exec_eng.load_study_from_input_dict(values_dict[0])
        # print(exec_eng.display_treeview_nodes(exec_display=True))
        # print('=====')
        # print(exec_eng.root_process.display_proxy_subtree(
        #     callback=lambda x: x.is_configured()))
        exp_ns_tree = 'Nodes representation for Treeview root\n' \
                      '|_ root\n' \
                      '\t|_ outer_ms\n' \
                      '\t\t|_ sc1_business\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ sc1_local_prod\n' \
                      '\t\t\t\t\t|_ Production\n' \
                      '\t\t\t\t\t\t|_ Abroad\n' \
                      '\t\t\t\t\t\t|_ Local\n' \
                      '\t\t\t\t\t\t\t|_ Road\n' \
                      '\t\t\t\t\t|_ Business\n' \
                      '\t\t\t\t\t\t|_ Remy\n' \
                      '\t\t\t\t\t\t\t|_ CAPEX\n' \
                      '\t\t\t\t|_ sc2_abroad_prod\n' \
                      '\t\t\t\t\t|_ Production\n' \
                      '\t\t\t\t\t\t|_ Abroad\n' \
                      '\t\t\t\t\t\t\t|_ Road\n' \
                      '\t\t\t\t\t\t\t|_ Plane\n' \
                      '\t\t\t\t\t\t|_ Local\n' \
                      '\t\t\t\t\t|_ Business\n' \
                      '\t\t\t\t\t\t|_ Remy\n' \
                      '\t\t\t\t\t\t\t|_ CAPEX\n' \
                      '\t\t\t|_ inner_ms_gather\n' \
                      '\t\t|_ sc2_business\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ sc1_local_prod\n' \
                      '\t\t\t\t\t|_ Production\n' \
                      '\t\t\t\t\t\t|_ Abroad\n' \
                      '\t\t\t\t\t\t|_ Local\n' \
                      '\t\t\t\t\t\t\t|_ Road\n' \
                      '\t\t\t\t\t|_ Business\n' \
                      '\t\t\t\t\t\t|_ Remy\n' \
                      '\t\t\t\t\t\t\t|_ CAPEX\n' \
                      '\t\t\t\t\t\t\t|_ OPEX\n' \
                      '\t\t\t\t|_ sc3_all_by_road\n' \
                      '\t\t\t\t\t|_ Production\n' \
                      '\t\t\t\t\t\t|_ Abroad\n' \
                      '\t\t\t\t\t\t\t|_ Road\n' \
                      '\t\t\t\t\t\t|_ Local\n' \
                      '\t\t\t\t\t\t\t|_ Road\n' \
                      '\t\t\t\t\t|_ Business\n' \
                      '\t\t\t\t\t\t|_ Remy\n' \
                      '\t\t\t\t\t\t\t|_ CAPEX\n' \
                      '\t\t\t\t\t\t\t|_ OPEX\n' \
                      '\t\t\t|_ inner_ms_gather\n' \
                      '\t|_ outer_ms_gather'

        exp_proxy_tree = '|_ root  (ProxyCoupling) [True]\n' \
                         '    |_ root.outer_ms  (ProxyMultiInstanceDriver) [True]\n' \
                         '        |_ root.outer_ms.sc1_business  (ProxyCoupling) [True]\n' \
                         '            |_ root.outer_ms.sc1_business.inner_ms  (ProxyMultiInstanceDriver) [True]\n' \
                         '                |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod  (ProxyCoupling) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod.Production  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod.Business  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod.Production.Abroad  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod.Production.Local  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod.Production.Local.Road  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod.Business.Remy  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc1_local_prod.Business.Remy.CAPEX  (ProxyDiscipline) [True]\n' \
                         '                |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod  (ProxyCoupling) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Production  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Business  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Production.Abroad  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Production.Local  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Production.Abroad.Road  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Production.Abroad.Plane  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Business.Remy  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc1_business.inner_ms.sc2_abroad_prod.Business.Remy.CAPEX  (ProxyDiscipline) [True]\n' \
                         '        |_ root.outer_ms.sc2_business  (ProxyCoupling) [True]\n' \
                         '            |_ root.outer_ms.sc2_business.inner_ms  (ProxyMultiInstanceDriver) [True]\n' \
                         '                |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod  (ProxyCoupling) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Production  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Business  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Production.Abroad  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Production.Local  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Production.Local.Road  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Business.Remy  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Business.Remy.CAPEX  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc1_local_prod.Business.Remy.OPEX  (ProxyDiscipline) [True]\n' \
                         '                |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road  (ProxyCoupling) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Production  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Business  (ArchiBuilder) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Production.Abroad  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Production.Local  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Production.Local.Road  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Production.Abroad.Road  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Business.Remy  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Business.Remy.CAPEX  (ProxyDiscipline) [True]\n' \
                         '                    |_ root.outer_ms.sc2_business.inner_ms.sc3_all_by_road.Business.Remy.OPEX  (ProxyDiscipline) [True]'

        self.assertEqual(exec_eng.display_treeview_nodes(exec_display=True),
                         exp_ns_tree)
        # self.assertEqual(exec_eng.root_process.display_proxy_subtree(callback=lambda x: x.is_configured()),
        #                  exp_proxy_tree)  # FIXME: update with flatten_subprocess

    def _test_11_usecase_import_multi_instances(self):
        """
        This test checks the usecase import capability in multi instance mode.
        """
        from os.path import join, dirname
        from sostrades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        with_coupling = True
        # TODO: n'existe plus voir Carlos
        # The generator eval process
        if with_coupling:
            proc_name = 'test_sellar_coupling_generator_eval_cp'
        else:
            proc_name = 'test_sellar_generator_eval_cp'

        usecase_name = 'usecase1_cp_multi'

        # Associated nested subprocess
        if with_coupling:
            sub_process_name = 'test_sellar_coupling'
        else:
            sub_process_name = 'test_sellar_list'

        sub_process_usecase_name = 'usecase'

        # Creation of the study from the associated usecase
        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([self.repo, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()

        study_dump.run()

        # Check the created study

        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        self.exec_eng.display_treeview_nodes(True)
        if with_coupling:
            ref_disc_sellar_1 = self.exec_eng.dm.get_disciplines_with_name(
                f'{self.study_name}.Eval.ReferenceScenario.SellarCoupling.Sellar_1')[0]
        else:
            ref_disc_sellar_1 = self.exec_eng.dm.get_disciplines_with_name(
                f'{self.study_name}.Eval.ReferenceScenario.Sellar_1')[0]

        # In the study creation it is provided x = array([2.])
        target_x = array([2.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc_sellar_1, target_values_dict, print_flag=print_flag)

        # Load the anonymized dict from associated selected sub_process

        anonymize_input_dict_from_usecase = study_dump.static_load_raw_usecase_data(
            self.repo, sub_process_name, sub_process_usecase_name)

        # Update the reference from the selected imported usecase anonymised
        # dict

        dict_values = {}
        dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase
        study_dump.load_data(from_input_dict=dict_values)

        # Check that the reference has been updated

        # In the anonymised dict of the selected usecase it is provided x =
        # array([2.])
        target_x = array([1.])
        target_values_dict = {}
        target_values_dict['x'] = target_x
        print_flag = False
        self.check_discipline_values(
            ref_disc_sellar_1, target_values_dict, print_flag=print_flag)

    def _test_12_nested_very_simple_multi_scenarios_with_reference(self):
        # FIXME: flatten_subprocess + instance_reference and maybe other issues
        '''
        Same as test 11 of nested very simple multi scenario but with reference. Let it be noted that all variables
        are non-trade variables.
        '''

        from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver_of_multi_driver._usecase_with_ref import \
            Study
        study_name = 'root'
        ns = study_name
        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = "test_multi_driver_of_multi_driver"
        eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()

        usecase = Study(execution_engine=exec_eng)
        usecase.study_name = ns
        values_dict = usecase.setup_usecase()

        exec_eng.load_study_from_input_dict(values_dict[0])

        # TODO: [to discuss] whether the scenario name reordering (that might come from a scatter_tool cleaning and that
        #  is at the source of ReferenceScenario appearing first after reconfiguration) is OK or should be handled both
        #  by applying reordering both to proxies and to scattered namespaces.

        # TREEVIEWS WITH REFERENCESCENARIO AT THE END
        exp_ns_tree = 'Nodes representation for Treeview root' \
                      '\n|_ root\n' \
                      '\t|_ outer_ms\n' \
                      '\t\t|_ scenario_1\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ ReferenceScenario\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3\n' \
                      '\t\t|_ scenario_2\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ ReferenceScenario\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3\n' \
                      '\t\t|_ ReferenceScenario\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ ReferenceScenario\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3'

        exp_proxy_tree = '|_ root  (ProxyCoupling) [True]\n    ' \
                         '|_ root.outer_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.Disc3  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.Disc3  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.Disc3  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms.ReferenceScenario.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms.ReferenceScenario.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms.ReferenceScenario.Disc1  (ProxyDiscipline) [True]'

        self.assertEqual(exec_eng.display_treeview_nodes(),
                         exp_ns_tree)
        self.assertEqual(exec_eng.root_process.display_proxy_subtree(callback=lambda x: x.is_configured()),
                         exp_proxy_tree)

        # Execute to check all inputs have been propagated from
        # ReferenceScenario (so execution would not give error)
        exec_eng.execute()

        self.constant = [1, 2]
        self.power = [1, 2]
        self.z = [1, 2]
        self.a = [0, 10]
        self.x = [0, 10]
        self.b = [[1, 2], [3, 4]]

        scenario_list_outer = ['scenario_1', 'scenario_2']
        scenario_list_inner = ['name_1', 'name_2']
        for i, sc in enumerate(scenario_list_outer):
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.constant'),
                             self.constant[0])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.power'),
                             self.power[0])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.z'),
                             self.z[0])
            for j, name in enumerate(scenario_list_inner):
                self.assertEqual(
                    exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b'),
                    self.b[0][0])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a'),
                    self.a[0])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x'),
                    self.x[0])

        # Now, values are given to all the variables to check that in that case, the dm has the added values and not the
        # values propagated from the ReferenceScenario
        for i, sc in enumerate(scenario_list_outer):
            values_dict[0][study_name + '.outer_ms.' +
                           sc + '.Disc3.constant'] = self.constant[i]
            values_dict[0][study_name + '.outer_ms.' +
                           sc + '.Disc3.power'] = self.power[i]
            values_dict[0][study_name + '.outer_ms.' +
                           sc + '.z'] = self.z[i]
            for j, name in enumerate(scenario_list_inner):
                values_dict[0][study_name + '.outer_ms.' + sc +
                               '.inner_ms.' + name + '.Disc1.b'] = self.b[i][j]
                values_dict[0][study_name + '.outer_ms.' +
                               sc + '.inner_ms.' + name + '.a'] = self.a[j]
                values_dict[0][study_name + '.outer_ms.' +
                               sc + '.inner_ms.' + name + '.x'] = self.x[j]
        exec_eng.load_study_from_input_dict(values_dict[0])

        for i, sc in enumerate(scenario_list_outer):
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.constant'),
                             self.constant[i])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.power'),
                             self.power[i])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.z'),
                             self.z[i])
            for j, name in enumerate(scenario_list_inner):
                self.assertEqual(
                    exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b'),
                    self.b[i][j])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a'),
                    self.a[j])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x'),
                    self.x[j])

        # Execute anc check outputs
        exec_eng.execute()
        for i, sc in enumerate(scenario_list_outer):
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.o'),
                             self.constant[i] + self.z[i] ** self.power[i])
            for j, name in enumerate(scenario_list_inner):
                self.assertEqual(
                    exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.indicator'),
                    self.a[j] * self.b[i][j])
                self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.y'),
                                 self.a[j] * self.x[j] + self.b[i][j])

        # Now, a variable would be modified in the inner_ms ReferenceScenario and its propagation inside
        # that inner_ms is going to be checked.
        values_dict_test = {}
        values_dict_test[f'{study_name}.outer_ms.scenario_1.inner_ms.ReferenceScenario.a'] = 80
        exec_eng.load_study_from_input_dict(values_dict_test)
        for name in scenario_list_inner:
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.scenario_1.inner_ms.' + name + '.a'), 80)

        # Now, editability propagation will be checked for differente reference modes combinatios of outer and inner ms
        # 1- Outer in linked and inner in linked (everything
        # non-editable/read-only)
        values_dict_test[f'{study_name}.outer_ms.instance_reference'] = True
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'linked_mode'
        values_dict_test[f'{study_name}.outer_ms.ReferenceScenario.inner_ms.instance_reference'] = True
        values_dict_test[f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'] = 'linked_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        for sc in scenario_list_outer:
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.constant', 'editable'),
                             False)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.power', 'editable'),
                             False)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.z', 'editable'),
                             False)
            for name in scenario_list_inner:
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b', 'editable'),
                    False)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a', 'editable'),
                    False)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x', 'editable'),
                    False)

        # 2- Outer in linked and inner in copy (outer should have priority;
        # everything non-editable/read-only)
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'linked_mode'
        values_dict_test[f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'] = 'copy_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(
            f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'), 'linked_mode')
        for sc in scenario_list_outer:
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.constant', 'editable'),
                             False)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.power', 'editable'),
                             False)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.z', 'editable'),
                             False)
            for name in scenario_list_inner:
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b', 'editable'),
                    False)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a', 'editable'),
                    False)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x', 'editable'),
                    False)

        # 3- Outer in copy and inner in copy (everything editable)
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'copy_mode'
        values_dict_test[f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'] = 'copy_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        for sc in scenario_list_outer:
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + sc + '.Disc3.constant', 'editable'),
                True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.power', 'editable'),
                             True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.z', 'editable'),
                             True)
            for name in scenario_list_inner:
                self.assertEqual(
                    exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b',
                                         'editable'),
                    True)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a', 'editable'),
                    True)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x', 'editable'),
                    True)

        # 4- Outer in copy and inner in linked (outer editable, inner
        # read-only)
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'copy_mode'
        values_dict_test[f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'] = 'linked_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        for sc in scenario_list_outer:
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + sc + '.Disc3.constant', 'editable'),
                True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.power', 'editable'),
                             True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.z', 'editable'),
                             True)
            for name in scenario_list_inner:
                self.assertEqual(
                    exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b',
                                         'editable'),
                    False)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a', 'editable'),
                    False)
                self.assertEqual(
                    exec_eng.dm.get_data(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x', 'editable'),
                    False)

        # 5- Outer in copy, ref-inner in linked and non-ref inner in linked
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'copy_mode'
        values_dict_test[
            f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'] = 'copy_mode'  # Will be propagated
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'copy_mode')
        values_dict_test[f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'] = 'linked_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'linked_mode')
        for sc in scenario_list_outer:
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + sc + '.Disc3.constant', 'editable'),
                True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.power', 'editable'),
                             True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.z', 'editable'),
                             True)
        for name in scenario_list_inner:
            self.assertEqual(
                exec_eng.dm.get_data(study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.Disc1.b',
                                     'editable'),
                False)
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.a', 'editable'),
                False)
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.x', 'editable'),
                False)
        # 6- ref-Outer in copy, ref-inner in linked and non-ref inner in copy
        # (outer editable, inner read-only)
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'copy_mode'
        values_dict_test[
            f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'] = 'linked_mode'  # Will be propagated
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'linked_mode')
        values_dict_test[f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'] = 'copy_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'copy_mode')
        for sc in scenario_list_outer:
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + sc + '.Disc3.constant', 'editable'),
                True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.power', 'editable'),
                             True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.z', 'editable'),
                             True)
        for name in scenario_list_inner:
            self.assertEqual(
                exec_eng.dm.get_data(study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.Disc1.b',
                                     'editable'),
                True)
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.a', 'editable'),
                True)
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.x', 'editable'),
                True)

    def _test_12_2_reproduction_error_GUI_multi_scenarios_with_reference(self):
        # FIXME: flatten_subprocess + instance_reference and maybe other issues
        '''
        Same as test 11 of nested very simple multi scenario but with reference. Let it be noted that all variables
        are non-trade variables.
        '''
        from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver_of_multi_driver._usecase_with_ref_2 import \
            Study
        study_name = 'root'
        ns = study_name
        exec_eng = ExecutionEngine(study_name)
        factory = exec_eng.factory
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = "test_multi_driver_of_multi_driver"
        eval_builder = factory.get_builder_from_process(repo=repo_name,
                                                        mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            eval_builder)

        exec_eng.configure()

        usecase = Study(execution_engine=exec_eng)
        usecase.study_name = ns
        values_dict = usecase.setup_usecase()
        exec_eng.load_study_from_input_dict(values_dict[0])
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.reference_mode'),
                         'copy_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'),
                         'copy_mode')

        # TODO: [to discuss] whether the scenario name reordering (that might come from a scatter_tool cleaning and that
        #  is at the source of ReferenceScenario appearing first after reconfiguration) is OK or should be handled both
        #  by applying reordering both to proxies and to scattered namespaces.

        # TREEVIEWS WITH REFERENCESCENARIO AT THE END
        exp_ns_tree = 'Nodes representation for Treeview root' \
                      '\n|_ root\n' \
                      '\t|_ outer_ms\n' \
                      '\t\t|_ scenario_1\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ ReferenceScenario\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3\n' \
                      '\t\t|_ scenario_2\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ ReferenceScenario\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3\n' \
                      '\t\t|_ ReferenceScenario\n' \
                      '\t\t\t|_ inner_ms\n' \
                      '\t\t\t\t|_ name_1\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ name_2\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t\t|_ ReferenceScenario\n' \
                      '\t\t\t\t\t|_ Disc1\n' \
                      '\t\t\t|_ Disc3'

        exp_proxy_tree = '|_ root  (ProxyCoupling) [True]\n    ' \
                         '|_ root.outer_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.Disc3  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.Disc3  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms  (ProxyMultiInstanceDriver) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.Disc3  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_1.inner_ms.ReferenceScenario.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.scenario_2.inner_ms.ReferenceScenario.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms.name_1.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms.name_2.Disc1  (ProxyDiscipline) [True]\n    ' \
                         '|_ root.outer_ms.ReferenceScenario.inner_ms.ReferenceScenario.Disc1  (ProxyDiscipline) [True]'

        self.assertEqual(exec_eng.display_treeview_nodes(),
                         exp_ns_tree)
        self.assertEqual(exec_eng.root_process.display_proxy_subtree(callback=lambda x: x.is_configured()),
                         exp_proxy_tree)

        # Execute to check all inputs have been propagated from
        # ReferenceScenario (so execution would not give error)
        exec_eng.execute()

        self.constant = [1, 2]
        self.power = [1, 2]
        self.z = [1, 2]
        self.a = [0, 10]
        self.x = [0, 10]
        self.b = [[1, 2], [3, 4]]

        scenario_list_outer = ['scenario_1', 'scenario_2']
        scenario_list_inner = ['name_1', 'name_2']
        for i, sc in enumerate(scenario_list_outer):
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.constant'),
                             self.constant[0])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.power'),
                             self.power[0])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.z'),
                             self.z[0])
            for j, name in enumerate(scenario_list_inner):
                self.assertEqual(
                    exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b'),
                    self.b[0][0])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a'),
                    self.a[0])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x'),
                    self.x[0])

        # Now, values are given to all the variables to check that in that case, the dm has the added values and not the
        # values propagated from the ReferenceScenario
        for i, sc in enumerate(scenario_list_outer):
            values_dict[0][study_name + '.outer_ms.' +
                           sc + '.Disc3.constant'] = self.constant[i]
            values_dict[0][study_name + '.outer_ms.' +
                           sc + '.Disc3.power'] = self.power[i]
            values_dict[0][study_name + '.outer_ms.' +
                           sc + '.z'] = self.z[i]
            for j, name in enumerate(scenario_list_inner):
                values_dict[0][study_name + '.outer_ms.' + sc +
                               '.inner_ms.' + name + '.Disc1.b'] = self.b[i][j]
                values_dict[0][study_name + '.outer_ms.' +
                               sc + '.inner_ms.' + name + '.a'] = self.a[j]
                values_dict[0][study_name + '.outer_ms.' +
                               sc + '.inner_ms.' + name + '.x'] = self.x[j]
        exec_eng.load_study_from_input_dict(values_dict[0])

        for i, sc in enumerate(scenario_list_outer):
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.constant'),
                             self.constant[i])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.Disc3.power'),
                             self.power[i])
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.z'),
                             self.z[i])
            for j, name in enumerate(scenario_list_inner):
                self.assertEqual(
                    exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.b'),
                    self.b[i][j])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a'),
                    self.a[j])
                self.assertEqual(
                    exec_eng.dm.get_value(
                        study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x'),
                    self.x[j])

        # Execute anc check outputs
        exec_eng.execute()
        for i, sc in enumerate(scenario_list_outer):
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.o'),
                             self.constant[i] + self.z[i] ** self.power[i])
            for j, name in enumerate(scenario_list_inner):
                self.assertEqual(
                    exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.Disc1.indicator'),
                    self.a[j] * self.b[i][j])
                self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.y'),
                                 self.a[j] * self.x[j] + self.b[i][j])

        # Now, editability propagation will be checked for differente reference modes combinatios of outer and inner ms
        # 1- Outer in copy, ref-inner in copy and non-ref inner in linked
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.reference_mode'),
                         'copy_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'),
                         'copy_mode')
        values_dict_test = {}
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'copy_mode')
        values_dict_test[f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'] = 'linked_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.reference_mode'),
                         'copy_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'),
                         'copy_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'linked_mode')  # TODO: THIS SHOULD LEAD TO ERROR as in GUI!!!!! In GUI, it does not change from copy to linked

        for sc in scenario_list_outer:
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + sc + '.Disc3.constant', 'editable'),
                True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.Disc3.power', 'editable'),
                             True)
            self.assertEqual(exec_eng.dm.get_data(study_name + '.outer_ms.' + sc + '.z', 'editable'),
                             True)
        for name in scenario_list_inner:
            self.assertEqual(
                exec_eng.dm.get_data(study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.Disc1.b',
                                     'editable'),
                False)
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.a', 'editable'),
                False)
            self.assertEqual(
                exec_eng.dm.get_data(
                    study_name + '.outer_ms.' + 'scenario_1' + '.inner_ms.' + name + '.x', 'editable'),
                False)

        values_dict_test[f'{study_name}.outer_ms.scenario_1.inner_ms.ReferenceScenario.a'] = 80
        exec_eng.load_study_from_input_dict(values_dict_test)
        for name in scenario_list_inner:
            self.assertEqual(exec_eng.dm.get_value(study_name + '.outer_ms.scenario_1.inner_ms.' + name + '.a'), 80)

        # Now, change of outer_ms to linked mode and verify linked mode propagation
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'linked_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'),
                         'linked_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'linked_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_2.inner_ms.reference_mode'),
                         'linked_mode')

        # Now, change of outer_ms back to copy mode and verify no mode propagation
        values_dict_test[f'{study_name}.outer_ms.reference_mode'] = 'copy_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.reference_mode'),
                         'copy_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'),
                         'linked_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'linked_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_2.inner_ms.reference_mode'),
                         'linked_mode')

        # Now, change of scenario_1 inner_ms to copy. In GUI this gave error and scenario_1 inner_ms stayed in linked
        values_dict_test[f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'] = 'copy_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'copy_mode')  # TODO: THIS SHOULD LEAD TO ERROR as in GUI!!!!!

        # Now, change of ReferenceScenario inner_ms to copy and propagate.
        values_dict_test[f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'] = 'copy_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.ReferenceScenario.inner_ms.reference_mode'),
                         'copy_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'copy_mode')
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_2.inner_ms.reference_mode'),
                         'copy_mode')

        # Now, change of scenario_1 inner_ms to linked. In GUI this gave error and scenario_1 inner_ms remained in copy
        values_dict_test[f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'] = 'linked_mode'
        exec_eng.load_study_from_input_dict(values_dict_test)
        self.assertEqual(exec_eng.dm.get_value(f'{study_name}.outer_ms.scenario_1.inner_ms.reference_mode'),
                         'linked_mode')  # TODO: THIS SHOULD LEAD TO ERROR as in GUI!!!!!

    def test_13_sellar_coupling_multi_instances_flatten(self):
        """
        This test checks the flatten_subprocess flag on a sellar coupling with cp gene and multi instances val. 
        """
        from os.path import join, dirname
        from sostrades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        # The generator eval process
        repo_name = self.repo + '.tests_driver_eval.multi'
        proc_name = 'test_multi_driver_sample_generator_sellar_coupling'
        usecase_name = 'usecase1_cp_multi_with_ref'

        self.study_name = usecase_name
        imported_module = import_module(
            '.'.join([repo_name, proc_name, usecase_name]))

        study_dump = imported_module.Study(run_usecase=True)

        study_dump.load_data()
        study_dump.run()

        # check that the DriverEvaluator has no i/o of the subprocess:
        driver_disc = study_dump.ee.dm.get_disciplines_with_name(
            f'{usecase_name}.Eval')[0]
        for var_name in driver_disc.get_input_data_names():
            self.assertFalse('ReferenceScenario' in var_name)
            self.assertFalse('scenario_1' in var_name)
            self.assertFalse('scenario_2' in var_name)
        self.assertEqual(driver_disc.get_output_data_names(), [])

        # check that the namespace treeview is proper
        exp_ns_tv = 'Nodes representation for Treeview usecase1_cp_multi_with_ref\n' \
                    '|_ usecase1_cp_multi_with_ref\n'  \
                    '\t|_ SampleGenerator\n' \
                    '\t|_ Eval\n' \
                    '\t\t|_ ReferenceScenario\n' \
                    '\t\t\t|_ SellarCoupling\n' \
                    '\t\t\t\t|_ Sellar_Problem\n' \
                    '\t\t\t\t|_ Sellar_2\n' \
                    '\t\t\t\t|_ Sellar_1\n' \
                    '\t\t|_ scenario_1\n' \
                    '\t\t\t|_ SellarCoupling\n' \
                    '\t\t\t\t|_ Sellar_Problem\n' \
                    '\t\t\t\t|_ Sellar_2\n' \
                    '\t\t\t\t|_ Sellar_1\n' \
                    '\t\t|_ scenario_2\n' \
                    '\t\t\t|_ SellarCoupling\n' \
                    '\t\t\t\t|_ Sellar_Problem\n' \
                    '\t\t\t\t|_ Sellar_2\n' \
                    '\t\t\t\t|_ Sellar_1'

        self.assertEqual(study_dump.ee.display_treeview_nodes(), exp_ns_tv)

        # check that the proxy tree has put the subprocess at the same level as
        # the evaluator and all proxies have run
        exp_proxy_tv_with_status = '|_ usecase1_cp_multi_with_ref  (ProxyCoupling) [DONE]\n' \
                                   '    |_ usecase1_cp_multi_with_ref.SampleGenerator  (ProxySampleGenerator) [DONE]\n' \
                                   '    |_ usecase1_cp_multi_with_ref.Eval_gather  (ProxyDiscipline) [DONE]\n' \
                                   '    |_ usecase1_cp_multi_with_ref.Eval  (ProxyMultiInstanceDriver) [DONE]\n' \
                                   '    |_ usecase1_cp_multi_with_ref.Eval.ReferenceScenario.SellarCoupling  (ProxyCoupling) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.ReferenceScenario.SellarCoupling.Sellar_Problem  (ProxyDiscipline) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.ReferenceScenario.SellarCoupling.Sellar_2  (ProxyDiscipline) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.ReferenceScenario.SellarCoupling.Sellar_1  (ProxyDiscipline) [DONE]\n' \
                                   '    |_ usecase1_cp_multi_with_ref.Eval.scenario_1.SellarCoupling  (ProxyCoupling) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.scenario_1.SellarCoupling.Sellar_Problem  (ProxyDiscipline) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.scenario_1.SellarCoupling.Sellar_2  (ProxyDiscipline) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.scenario_1.SellarCoupling.Sellar_1  (ProxyDiscipline) [DONE]\n' \
                                   '    |_ usecase1_cp_multi_with_ref.Eval.scenario_2.SellarCoupling  (ProxyCoupling) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.scenario_2.SellarCoupling.Sellar_Problem  (ProxyDiscipline) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.scenario_2.SellarCoupling.Sellar_2  (ProxyDiscipline) [DONE]\n' \
                                   '        |_ usecase1_cp_multi_with_ref.Eval.scenario_2.SellarCoupling.Sellar_1  (ProxyDiscipline) [DONE]'

        self.assertEqual(study_dump.ee.root_process.display_proxy_subtree(
            lambda x: x.status), exp_proxy_tv_with_status)


if __name__ == '__main__':
    test = TestSoSDOEScenario()
    test.setUp()
    test.test_1_simple_disc_output_to_input_driver_DoeEval()
