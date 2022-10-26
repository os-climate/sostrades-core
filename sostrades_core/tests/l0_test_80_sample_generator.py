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
from sostrades_core.execution_engine.disciplines_wrappers.doe_eval import DoeEval


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

        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        eval_in_list_x_z = ['doe.x', 'doe.z']
        self.eval_in_list = eval_in_list_x_z

    def test_01_check_get_options_desc_in(self):
        '''
        Test that checks get_options_desc_in for DoeSampleGenerator
        '''
        generator_name = 'doe_generator'
        sample_generator = DoeSampleGenerator(generator_name)

        algo_names_list = sample_generator.get_available_algo_names()
        print(algo_names_list)

        algo_name = 'fullfact'
        algo_options_desc_in, algo_options_descr_dict = sample_generator.get_options_desc_in(
            algo_name)

        print(algo_options_desc_in)
        print(algo_options_descr_dict)

        targ_algo_options_desc_in = self.algo_options_desc_in

        # check algo_options_desc_in equal targ_algo_options_desc_in
        # check set of algo_options_desc_in.keys() equal set of algo_options_descr_dict without 'kwargs'
        # test the error message in case of algo_name = 'toto'

    def test_02_check_generate_samples(self):
        '''
        Test that checks generate_samples for DoeSampleGenerator
        '''
        generator_name = 'doe_generator'
        sample_generator = DoeSampleGenerator(generator_name)

        sampling_algo = self.sampling_algo
        algo_options = self.algo_options
        eval_in_list = self.eval_in_list
        design_space = self.dspace_eval

        #doe_eval = DoeEval
        #design_space = self.set_design_space(dspace_df)

        # samples = sample_generator.generate_samples(
        #    sampling_algo, algo_options, eval_in_list, design_space)

        # print(samples)


if '__main__' == __name__:
    cls = TestSampleGenerator()
    cls.setUp()
