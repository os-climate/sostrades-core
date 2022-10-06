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
import copy
import logging
from logging import Handler
from time import time

from pandas._testing import assert_frame_equal

from gemseo.algos.doe.doe_factory import DOEFactory

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""

import unittest
from numpy import array, std
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
import os
from os.path import dirname, join


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

    def setUp(self):
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
                                      'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z_in']}
        self.input_selection_local_dv_x = pd.DataFrame(
            input_selection_local_dv_x)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z_in']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z_in']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        input_selection_local_dv = {'selected_input': [True, False, False, False, False],
                                    'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
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
        self.proc_name = 'test_sellar_doe'     # In each test proc_name will be redefined

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

        proc_name = "test_simple_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoEEval.design_space'] = dspace_x
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z_in'] = 2 * array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ Simple_Disc',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), n_samples + 1)
        self.assertEqual(len(doe_disc_obj), n_samples + 1)
        reference_dict_doe_disc_y1 = {'scenario_1': array([10.491019856682016]),
                                      'scenario_2': array([7.247824531594309]),
                                      'scenario_3': array([2.9753409599263483]),
                                      'scenario_4': array([1.7522749587335193]),
                                      'scenario_5': array([9.384097972066053]),
                                      'scenario_6': array([8.36704386923391]),
                                      'scenario_7': array([4.479056921478663]),
                                      'scenario_8': array([5.286891081070988]),
                                      'scenario_9': array([3.240108355137796]),
                                      'scenario_10': array([6.194561090631401]),
                                      'reference': array([2.29689011157193])}
        reference_dict_doe_disc_y2 = {'scenario_1': array([5.238984386606706]),
                                      'scenario_2': array([4.692178398916815]),
                                      'scenario_3': array([3.7249176675790494]),
                                      'scenario_4': array([3.3237352298452736]),
                                      'scenario_5': array([5.063347510823095]),
                                      'scenario_6': array([4.892584289045681]),
                                      'scenario_7': array([4.116378255765888]),
                                      'scenario_8': array([4.2993240487306235]),
                                      'scenario_9': array([3.8000300983977455]),
                                      'scenario_10': array([4.488887520686984]),
                                      'reference': array([3.5155494421403515])}
        for key in doe_disc_y1.keys():
            self.assertAlmostEqual(doe_disc_y1[key][0], reference_dict_doe_disc_y1[key][0])
        for key in doe_disc_y2.keys():
            self.assertAlmostEqual(doe_disc_y2[key][0], reference_dict_doe_disc_y2[key][0])

    def test_2_DoeEval_of_DoeEval(self):
        """ Here we test a DoeEval of a DoeEval process on a single sub-discipline to check that the transition of the
        ProxyDisciplineDriver from working with short names to working with tuples of short names and namespace (of the
        discipline to the local data variable belongs) is implemented. It is really a test of driver of a driver using
        DoeEval. The test demonstrates the capability to use a driver of a driver.
        """

        dspace_dict_upper = {'variable': ['DoEEvalUpper.DoEEvalLower.Disc1.b'],

                               'lower_bnd': [50.],
                               'upper_bnd': [200.],

                               }
        dspace_upper = pd.DataFrame(dspace_dict_upper)
        dspace_dict_lower = {'variable': ['DoEEvalUpper.DoEEvalLower.Disc1.a'],

                               'lower_bnd': [50.],
                               'upper_bnd': [200.],

                               }
        dspace_lower = pd.DataFrame(dspace_dict_lower)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        proc_name = "test_disc1_doe_eval_of_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEvalUpper',
                       '\t\t|_ DoEEvalLower',
                       '\t\t\t|_ Disc1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        assert exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].is_sos_coupling
        assert exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[0].is_sos_coupling

        # -- set up disciplines
        private_values = {
            self.study_name + '.x': array([10.]),
            self.study_name + '.DoEEvalUpper.DoEEvalLower.Disc1.a': array([5.]),
            self.study_name + '.DoEEvalUpper.DoEEvalLower.Disc1.b': array([25431.]),
            self.study_name + '.y': array([4.])}
        exec_eng.load_study_from_input_dict(private_values)

        input_selection_upper_b = {'selected_input': [False, False, True],
                               'full_name': ['x', 'DoEEvalUpper.DoEEvalLower.Disc1.a', 'DoEEvalUpper.DoEEvalLower.Disc1.b']}
        input_selection_upper_b = pd.DataFrame(input_selection_upper_b)
        output_selection_upper_y_dict = {'selected_output': [True, False, True],
                                'full_name': ['y_dict', 'DoEEvalUpper.DoEEvalLower.Disc1.indicator', 'DoEEvalUpper.DoEEvalLower.samples_inputs_df']}
        output_selection_upper_y_dict = pd.DataFrame(output_selection_upper_y_dict)

        input_selection_lower_a = {'selected_input': [False, True, False],
                                   'full_name': ['x', 'DoEEvalUpper.DoEEvalLower.Disc1.a', 'DoEEvalUpper.DoEEvalLower.Disc1.b']}
        input_selection_lower_a = pd.DataFrame(input_selection_lower_a)
        output_selection_lower_y = {'selected_output': [True, False],
                                    'full_name': ['y', 'DoEEvalUpper.DoEEvalLower.Disc1.indicator']}
        output_selection_lower_y = pd.DataFrame(output_selection_lower_y)

        disc_dict = {f'{self.ns}.DoEEvalUpper.sampling_algo': "lhs",
                     f'{self.ns}.DoEEvalUpper.eval_inputs': input_selection_upper_b,
                     f'{self.ns}.DoEEvalUpper.eval_outputs': output_selection_upper_y_dict,
                     f'{self.ns}.DoEEvalUpper.DoEEvalLower.sampling_algo': "lhs",
                     f'{self.ns}.DoEEvalUpper.DoEEvalLower.eval_inputs': input_selection_lower_a,
                     f'{self.ns}.DoEEvalUpper.DoEEvalLower.eval_outputs': output_selection_lower_y}


        n_samples = 3
        exec_eng.load_study_from_input_dict(disc_dict)
        disc_dict = {'doe.DoEEvalUpper.algo_options': {'n_samples': n_samples, 'face': 'faced'},
                     'doe.DoEEvalUpper.design_space': dspace_upper,
                     'doe.DoEEvalUpper.DoEEvalLower.algo_options': {'n_samples': n_samples, 'face': 'faced'},
                     'doe.DoEEvalUpper.DoEEvalLower.design_space': dspace_lower
                     }

        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()

        for var in ['doe.y_dict_dict', 'doe.y_dict', 'doe.y']:
            self.assertIn(var, exec_eng.root_process.get_output_data_names())

        proxy_disc = exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[0]
        mdo_disc = proxy_disc.mdo_discipline_wrapp.mdo_discipline
        reference_local_data = copy.deepcopy(mdo_disc.local_data)

        keys_upper = list(exec_eng.dm.get_value('doe.y_dict_dict').keys())
        i_upper = 0
        for b in exec_eng.dm.get_value('doe.DoEEvalUpper.samples_inputs_df')['DoEEvalUpper.DoEEvalLower.Disc1.b']:
            keys_lower = list(exec_eng.dm.get_value('doe.y_dict_dict')[keys_upper[i_upper]].keys())
            i_lower = 0
            samples_input_dataframe = exec_eng.dm.get_value('doe.DoEEvalUpper.DoEEvalLower.samples_inputs_df_dict')[keys_upper[i_upper]]
            for a in samples_input_dataframe['DoEEvalUpper.DoEEvalLower.Disc1.a']:
                y_output = exec_eng.dm.get_value('doe.y_dict_dict')[keys_upper[i_upper]][keys_lower[i_lower]]

                in_local_data = copy.deepcopy(reference_local_data)
                in_local_data['doe.DoEEvalUpper.DoEEvalLower.Disc1.a'] = a
                in_local_data['doe.DoEEvalUpper.DoEEvalLower.Disc1.b'] = b
                out_local_data = mdo_disc.execute(in_local_data)
                y_reference = out_local_data['doe.y']

                self.assertAlmostEqual(y_output, y_reference)

                i_lower += 1
            i_upper += 1

    def test_3_simple_custom_driver(self):

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
        self.assertEqual(exec_eng.dm.get_value('root.Driver1.output_squared'),array([649281361.]))

        # check that the root process knows all the numerical inputs of the entire subprocess
        root_inputs = exec_eng.root_process.get_input_data_names()
        self.assertIn('root.linearization_mode', root_inputs)
        self.assertIn('root.Driver1.linearization_mode', root_inputs)
        self.assertIn('root.Driver1.Disc1.linearization_mode', root_inputs)

    def test_4_simple_discs_work_with_io_of_DoeEval_and_its_subdisciplines(self):
        """
        This test checks that the coupling between the output of a simple discipline and the input of a driver and its
        subprocess works, as well as the coupling of the DoeEval output and its subdisciplines outputs with another
        simple discipline.
        """
        #FIXME: update asserts
        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_simple1_simple2_sellar_doe_eval"
        builders = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            builders)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        values_dict = {}
        # DoE inputs
        n_samples = 10
        values_dict[f'{self.ns}.DoEEval.design_space'] = dspace_x
        values_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
        values_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        values_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2

        # Sellar inputs
        local_dv = 10.
        # values_dict = {}
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.z_in'] = 2 * array([1., 1.])   #Input of SimpleDisc1
        # values_dict[f'{self.ns}.c_1'] = array([1.])            #Input of SimpleDisc2
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ Simple_Disc1',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t|_ Simple_Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        z = exec_eng.dm.get_value('doe.z')
        if z[0] > 0.5:
            self.assertEqual(exec_eng.dm.get_value('doe.DoEEval.sampling_algo'), "lhs")
        else:
            self.assertEqual(exec_eng.dm.get_value('doe.DoEEval.sampling_algo'), "fullfact")

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), n_samples + 1)
        self.assertEqual(len(doe_disc_obj), n_samples + 1)
        reference_dict_doe_disc_y1 = {'scenario_1': array([10.491019856682016]),
                                      'scenario_2': array([7.247824531594309]),
                                      'scenario_3': array([2.9753409599263483]),
                                      'scenario_4': array([1.7522749587335193]),
                                      'scenario_5': array([9.384097972066053]),
                                      'scenario_6': array([8.36704386923391]),
                                      'scenario_7': array([4.479056921478663]),
                                      'scenario_8': array([5.286891081070988]),
                                      'scenario_9': array([3.240108355137796]),
                                      'scenario_10': array([6.194561090631401]),
                                      'reference': array([2.29689011157193])}
        reference_dict_doe_disc_y2 = {'scenario_1': array([5.238984386606706]),
                                      'scenario_2': array([4.692178398916815]),
                                      'scenario_3': array([3.7249176675790494]),
                                      'scenario_4': array([3.3237352298452736]),
                                      'scenario_5': array([5.063347510823095]),
                                      'scenario_6': array([4.892584289045681]),
                                      'scenario_7': array([4.116378255765888]),
                                      'scenario_8': array([4.2993240487306235]),
                                      'scenario_9': array([3.8000300983977455]),
                                      'scenario_10': array([4.488887520686984]),
                                      'reference': array([3.5155494421403515])}
        for key in doe_disc_y1.keys():
            self.assertAlmostEqual(doe_disc_y1[key][0], reference_dict_doe_disc_y1[key][0])
        for key in doe_disc_y2.keys():
            self.assertAlmostEqual(doe_disc_y2[key][0], reference_dict_doe_disc_y2[key][0])

        self.assertEqual(exec_eng.dm.get_value('doe.out_simple2'),
                         exec_eng.dm.get_value('doe.c_1')*std(list(exec_eng.dm.get_value('doe.y_1_dict').values())[:-1]))
    def test_io2_Coupling_of_Coupling_to_check_data_io(self):
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
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines
        private_values = {
            study_name + '.x': array([10.]),
            study_name + '.UpperCoupling.LowerCoupling.Disc1.a': array([5.]),
            study_name + '.UpperCoupling.LowerCoupling.Disc1.b': array([7.]),
            study_name + '.y': array([4.]),
            study_name + '.UpperCoupling.LowerCoupling.Disc2.power': array([3.]),
            study_name + '.UpperCoupling.LowerCoupling.Disc2.constant': array([4.]),
        }
        exec_eng.load_study_from_input_dict(private_values)
        exec_eng.execute()


        for disc in [exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[0], # discipline with no coupled inputs
                     exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0], exec_eng.root_process.proxy_disciplines[0], exec_eng.root_process] : # couplings
            io_ns_map_in = disc._io_ns_map_in
            for var, identifier in io_ns_map_in.items():
                var_tuple = (var, identifier)
                self.assertEqual(identifier, id(exec_eng.root_process._data_in_ns_tuple[var_tuple]['ns_reference']))  #TODO: to be changed by _data_in


