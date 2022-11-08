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
import logging
from logging import Handler
from time import time

from pandas._testing import assert_frame_equal


"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""

import unittest
from numpy import array
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
import os
from os.path import dirname, join

from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import CartesianProductSampleGenerator

from sostrades_core.execution_engine.disciplines_wrappers.doe_wrapper import DoeWrapper


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestSampleGenerator(unittest.TestCase):
    """
    Sample Generator test classes
    """

    def setUp(self):
        self.study_name = 'doe'
        self.generator_name = 'doe_generator'
        self.sampling_algo = 'fullfact'

        full_fact_algo_options_desc_in = {  # default options
            'alpha': 'orthogonal',
            'face': 'faced',
            'criterion': None,
            'iterations': 5,
            'eval_jac': False,
            'center_bb': None,
            'center_cc': None,
            'n_samples': None,
            'levels': None,
            'n_processes': 1,
            'wait_time_between_samples': 0.0,
            'seed': 1,
            'max_time': 0}

        self.algo_options_desc_in = full_fact_algo_options_desc_in

        n_samples = 10

        #======================================================================
        # user_fullfact_algo_options = {
        #     'n_samples': n_samples,
        #     'alpha': 'orthogonal',
        #     'eval_jac': False,
        #     'face': 'faced',
        #     'iterations': 5,
        #     'max_time': 0,
        #     'seed': 1,
        #     'center_bb': 'default',
        #     'center_cc': 'default',
        #     'criterion': 'default',
        #     'levels': 'default'}
        #======================================================================

        # it is better to always explicit default options values and not use
        # the 'default' input !!

        # update only default n_samples in default options
        user_fullfact_algo_options = full_fact_algo_options_desc_in
        user_fullfact_algo_options['n_samples'] = n_samples

        self.algo_options = user_fullfact_algo_options

        dspace_dict_eval = {'variable': ['x', 'z'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }

        self.dspace_eval = pd.DataFrame(dspace_dict_eval)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}

        self.eval_inputs = pd.DataFrame(input_selection_x_z)
        # from eval_inputs to selected_inputs and eval_in_list
        self.selected_inputs = self.eval_inputs[self.eval_inputs['selected_input']
                                                == True]['full_name']
        self.selected_inputs = self.selected_inputs.tolist()
        self.eval_in_list = [
            f'{self.study_name}.{element}' for element in self.selected_inputs]
        #self.eval_in_list = ['doe.x', 'doe.z']
        ##########################

        target_samples_fullfact = [[array([0.]), array([-10., 0.])],
                                   [array([10.]), array([-10., 0.])],
                                   [array([0.]), array([10., 0.])],
                                   [array([10.]), array([10., 0.])],
                                   [array([0.]), array([-10., 10.])],
                                   [array([10.]), array([-10., 10.])],
                                   [array([0.]), array([10., 10.])],
                                   [array([10.]), array([10., 10.])]]

        self.target_samples_df = pd.DataFrame(data=target_samples_fullfact,
                                              columns=self.selected_inputs)

    def test_01_check_get_options_desc_in(self):
        '''
        Test that checks get_options_desc_in for DoeSampleGenerator
        '''
        sample_generator = DoeSampleGenerator()

        algo_names_list = sample_generator.get_available_algo_names()
        # print(algo_names_list)

        sampling_algo_name = 'fullfact'
        algo_options_desc_in, algo_options_descr_dict = sample_generator.get_options_desc_in(
            sampling_algo_name)

        # print(algo_options_desc_in)
        # print(algo_options_descr_dict)

        # check algo_options_desc_in
        targ_algo_options_desc_in = self.algo_options_desc_in
        self.assertDictEqual(self.algo_options_desc_in, targ_algo_options_desc_in,
                             "coupling algo_options_desc_in doesn't match")

        # check keys of algo_options_desc_in.keys()
        target_algo_options_descr_dict_keys = [
            elem for elem in algo_options_descr_dict.keys() if elem not in ['kwargs']]
        self.assertSetEqual(set(algo_options_desc_in.keys()), set(
            target_algo_options_descr_dict_keys))

        # test if it works with all algo samples names
        # print('\n')
        for sampling_algo_name in algo_names_list:
            algo_options_desc_in, algo_options_descr_dict = sample_generator.get_options_desc_in(
                sampling_algo_name)
            # print(sampling_algo_name)
            # print(algo_options_desc_in)
            # print('\n')

        # test the error message in case of wrong algo_name
        sampling_algo_name = 'toto'
        with self.assertRaises(Exception) as cm:
            algo_options_desc_in, algo_options_descr_dict = sample_generator.get_options_desc_in(
                sampling_algo_name)

        error_message = f'The provided algorithm name {sampling_algo_name} is not in the available algorithm list : {algo_names_list}'
        self.assertEqual(str(cm.exception), error_message)

        # test the error message in case of 'CustomDOE' and 'DiagonalDOE'
        # algo_names
        for sampling_algo_name in ['CustomDOE', 'DiagonalDOE']:
            with self.assertRaises(Exception) as cm:
                algo_options_desc_in, algo_options_descr_dict = sample_generator.get_options_desc_in(
                    sampling_algo_name)

            error_message = f'The provided algorithm name {sampling_algo_name} is not allowed in doe sample generator'
            self.assertEqual(str(cm.exception), error_message)

    def test_02_check_generate_samples_fullfact(self):
        '''
        Test that checks generate_samples for DoeSampleGenerator: it is tested on sampling_algo = 'fullfact'
        '''

        sampling_algo_name = self.sampling_algo
        algo_options = self.algo_options

        dspace_df = self.dspace_eval  # data_manager design space in df format

        selected_inputs = self.selected_inputs

        doe_wrapper = DoeWrapper(self.study_name)
        design_space = doe_wrapper.create_design_space(
            selected_inputs, dspace_df)  # gemseo DesignSpace

        sample_generator = DoeSampleGenerator()
        samples_df = sample_generator.generate_samples(
            sampling_algo_name, algo_options, selected_inputs, design_space)

        # print(samples_df)

        assert_frame_equal(samples_df, self.target_samples_df)

    def test_03_check_generate_samples_pydoe_algo_names(self):
        '''
        Test that checks generate_samples for DoeSampleGenerator: it is tested on pyDOE algo names
        '''

        pydoe_list_of_algo_names = ['fullfact', 'ff2n',
                                    'pbdesign', 'bbdesign',
                                    'ccdesign', 'lhs']

        pydoe_algo_options_desc_in = {  # default options
            'alpha': 'orthogonal',
            'face': 'faced',
            'criterion': None,
            'iterations': 5,
            'eval_jac': False,
            'center_bb': None,
            'center_cc': None,
            'n_samples': None,
            'levels': None,
            'n_processes': 1,
            'wait_time_between_samples': 0.0,
            'seed': 1,
            'max_time': 0}

        # update only default n_samples in default options
        n_samples = 10
        user_pydoe_algo_options = pydoe_algo_options_desc_in
        user_pydoe_algo_options['n_samples'] = n_samples

        #list_of_algo_names = [algo_names for algo_names in pydoe_list_of_algo_names if algo_names not in []]
        list_of_algo_names = pydoe_list_of_algo_names

        for sampling_algo_name in list_of_algo_names:
            # print('\n')
            # print(sampling_algo_name)

            algo_options = user_pydoe_algo_options

            #################################################################
            # 'ccdesign' gemseo : center_cc : tuple[int, int] | None, optional !
            # 'ccdesign' pydoe : center is a 2-tuple of center points (one for the factorial block, one for the star block, default (4, 4)) !
            # Problem in gemseo !:
            #          The default of 'ccdesign' in doc and provided default values should be (4, 4) and not None
            #          Should we provide this default value ?
            # or should we give a warning to the user in case of 'ccdesign'?
            if sampling_algo_name == 'ccdesign':
                algo_options['center_cc'] = (4, 4)

            dspace_df = self.dspace_eval  # data_manager design space in df format
            #################################################################

            selected_inputs = self.selected_inputs

            doe_wrapper = DoeWrapper(self.study_name)
            design_space = doe_wrapper.create_design_space(
                selected_inputs, dspace_df)  # gemseo DesignSpace

            sample_generator = DoeSampleGenerator()
            samples_df = sample_generator.generate_samples(
                sampling_algo_name, algo_options, selected_inputs, design_space)

            # print(samples_df)

            #assert_frame_equal(samples_df, self.target_samples_df)

    def test_04_check_generate_samples_openturns_algo_names(self):
        '''
        Test that checks generate_samples for DoeSampleGenerator: it is tested on openturns algo names
        '''

        openturns_list_of_algo_names = ['OT_SOBOL', 'OT_RANDOM', 'OT_HASELGROVE', 'OT_REVERSE_HALTON', 'OT_HALTON',
                                        'OT_FAURE', 'OT_MONTE_CARLO', 'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL',
                                        'OT_OPT_LHS', 'OT_LHS', 'OT_LHSC', 'OT_FULLFACT', 'OT_SOBOL_INDICES']

        #list_of_algo_names = [algo_names for algo_names in openturns_list_of_algo_names if algo_names not in []]
        list_of_algo_names = openturns_list_of_algo_names

        openturns_algo_options_desc_in = {  # default options
            'levels': None,
            'centers': None,
            'eval_jac': False,
            'n_samples': None,
            'n_processes': 1,
            'wait_time_between_samples': 0.0,
            'criterion': 'C2',
            'temperature': 'Geometric',
            'annealing': True,
            'n_replicates': 1000,
            'seed': 1,
            'max_time': 0}

        # update only default n_samples in default options
        n_samples = 10
        user_openturns_algo_options = openturns_algo_options_desc_in
        user_openturns_algo_options['n_samples'] = n_samples

        for sampling_algo_name in list_of_algo_names:
            # print('\n')
            # print(sampling_algo_name)

            algo_options = openturns_algo_options_desc_in

            #################################################################
            # 'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL' gemseo : int | Sequence[int] | None, optional !
            # 'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL' openturns :
            #         It has two constructors Factorial(center, levels) and Factorial(dimension, levels)
            #         How do gemseo choose between those two constructors? Does he choose Factorial(center, levels)?
            #         levels: The discretisation of directions (the same for each one), without any consideration of unit.
            # center: enter of the design of experiments. If not specified, the
            # design of experiments is centered on 0 of R^n
            # Problem in gemseo !:
            #          Levels and centers are not optionals in openturns !
            #          Should we provide this default value ?
            #          or should we give a warning to the user in case of
            #          'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL' ?
            if sampling_algo_name in ['OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL']:
                algo_options['levels'] = [0.1]
                algo_options['centers'] = (0, 0, 0)
                # Problem in SoSTrades !
                # Levels must belong to [0, 1] has we have a normalized sample !!
                # We create a normalized samples in SoSTrades that we unnormalized but what if user provide unnormalized input options !!!
            #################################################################

            dspace_df = self.dspace_eval  # data_manager design space in df format

            selected_inputs = self.selected_inputs

            doe_wrapper = DoeWrapper(self.study_name)
            design_space = doe_wrapper.create_design_space(
                selected_inputs, dspace_df)  # gemseo DesignSpace

            sample_generator = DoeSampleGenerator()
            samples_df = sample_generator.generate_samples(
                sampling_algo_name, algo_options, selected_inputs, design_space)
            # print(samples_df)

            #assert_frame_equal(samples_df, self.target_samples_df)

            #################################################################
            if sampling_algo_name in ['OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL']:
                algo_options['levels'] = None
                algo_options['centers'] = None
            #################################################################

    def test_05_check_big_n_samples(self):
        '''
        Test to check big values of n_samples and associated performances
        '''
        pass

    def test_06_check_generate_samples_cartesian_product(self):
        '''
        Test to check the cartesian product algorithm
        '''
        dict_of_list_values = {
            'x': [0., 3., 4., 5., 7.],
            'z': [[-10., 0.], [-5., 4.], [10, 10]]
        }
        variable_list = dict_of_list_values.keys()

        sample_generator = CartesianProductSampleGenerator(self)
        samples_df = sample_generator.generate_samples(dict_of_list_values)

        print(samples_df)

        targeted_samples = [
            [0.0, [-10.0, 0.0]],
            [0.0, [-5.0, 4.0]],
            [0.0, [10, 10]],
            [3.0, [-10.0, 0.0]],
            [3.0, [-5.0, 4.0]],
            [3.0, [10, 10]],
            [4.0, [-10.0, 0.0]],
            [4.0, [-5.0, 4.0]],
            [4.0, [10, 10]],
            [5.0, [-10.0, 0.0]],
            [5.0, [-5.0, 4.0]],
            [5.0, [10, 10]],
            [7.0, [-10.0, 0.0]],
            [7.0, [-5.0, 4.0]],
            [7.0, [10, 10]]]

        target_samples_df = pd.DataFrame(
            targeted_samples, columns=variable_list)

        assert_frame_equal(samples_df, target_samples_df)


if '__main__' == __name__:
    cls = TestSampleGenerator()
    cls.setUp()
