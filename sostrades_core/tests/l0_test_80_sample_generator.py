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

        full_fact_algo_options_desc_in = {
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
        user_fullfact_algo_options = {
            'n_samples': n_samples,
            'alpha': 'orthogonal',
            'eval_jac': False,
            'face': 'faced',
            'iterations': 5,
            'max_time': 0,
            'seed': 1,
            'center_bb': 'default',
                         'center_cc': 'default',
                         'criterion': 'default',
                         'levels': 'default'}
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

        target_samples = [[array([0.]), array([-10., 0.])],
                          [array([10.]), array([-10., 0.])],
                          [array([0.]), array([10., 0.])],
                          [array([10.]), array([10., 0.])],
                          [array([0.]), array([-10., 10.])],
                          [array([10.]), array([-10., 10.])],
                          [array([0.]), array([10., 10.])],
                          [array([10.]), array([10., 10.])]]
        self.target_samples = target_samples

    def test_01_check_get_options_desc_in(self):
        '''
        Test that checks get_options_desc_in for DoeSampleGenerator
        '''
        generator_name = 'doe_generator'
        sample_generator = DoeSampleGenerator(generator_name)

        algo_names_list = sample_generator.get_available_algo_names()
        print(algo_names_list)

        sampling_algo_name = 'fullfact'
        algo_options_desc_in, algo_options_descr_dict = sample_generator.get_options_desc_in(
            sampling_algo_name)

        print(algo_options_desc_in)
        print(algo_options_descr_dict)

        targ_algo_options_desc_in = self.algo_options_desc_in

        # check algo_options_desc_in equal targ_algo_options_desc_in
        # check set of algo_options_desc_in.keys() equal set of algo_options_descr_dict without 'kwargs'
        # test the error message in case of algo_name = 'toto'

        # test it works with all algo samples names

    def test_02_check_generate_samples(self):
        '''
        Test that checks generate_samples for DoeSampleGenerator
        '''
        generator_name = 'doe_generator'
        sample_generator = DoeSampleGenerator(generator_name)

        sampling_algo_name = self.sampling_algo
        algo_options = self.algo_options

        eval_in_list = self.eval_in_list

        dspace_df = self.dspace_eval  # data_manager design space in df format

        doe_wrapper = DoeWrapper(self.study_name)
        design_space = doe_wrapper.set_design_space(
            eval_in_list, dspace_df)  # gemseo DesignSpace
        # why do we have a design_space function creation in doe_wrapper ?

        samples = sample_generator.generate_samples(
            sampling_algo_name, algo_options, eval_in_list, design_space)

        print(samples)
        # check versus self.target_samples

        samples_df = sample_generator.put_samples_in_df_format(
            samples, eval_in_list)
        print(samples_df)
        # add assert


if '__main__' == __name__:
    cls = TestSampleGenerator()
    cls.setUp()
