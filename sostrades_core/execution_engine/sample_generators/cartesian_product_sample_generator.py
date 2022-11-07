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
from builtins import NotImplementedError

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator

import pandas as pd
import numpy as np

import itertools

import logging
LOGGER = logging.getLogger(__name__)

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class SampleTypeError(TypeError):
    pass


class CartesianProductSampleGenerator(AbstractSampleGenerator):
    '''
    Abstract class that generates sampling
    '''

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    N_PROCESSES = 'n_processes'
    WAIT_TIME_BETWEEN_SAMPLES = 'wait_time_between_samples'

    N_SAMPLES = "n_samples"

    # def __init__(self, generator_name):
    #     '''
    #     Constructor
    #     '''
    #     self.name = generator_name

    def generate_samples(self, selected_inputs, dict_of_list_values):
        '''
        Method that generate samples in a design space for a selected algorithm with its options 
        The method also checks the output formating

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dict_of_list_values (dict): for each selected variables it provides a list of values to be combined in a cartesian product


        Returns:
            samples_df (data_frame) : generated samples
                                      dataframe of a matrix of n raws  (each raw is an input point to be evaluated)  
                                      any variable of dim m is an array of dim m in a single column of the matrix            
        '''

        # generate the sampling by subclass
        samples_df = self._generate_samples(
            selected_inputs, dict_of_list_values)

        # check sample formatting
        self._check_samples(samples_df)

        return samples_df

    def _check_samples(self, samples_df):
        '''
        Method that checks the sample output type
        Arguments:
            samples_df (dataframe) : generated samples 
        Raises:
            Exception if samples_df is not a dataframe                   
        '''
        if not(isinstance(samples_df, pd.DataFrame)):
            msg = "Expected sampling output type should be pandas.core.frame.DataFrame"
            msg += "however sampling type of sampling generator <%s> " % str(
                self.__class__.__name__)
            msg += "is <%s> " % str(type(samples_df))
            raise SampleTypeError()

    def _generate_samples(self, selected_inputs, dict_of_list_values):
        '''
        Method that generate samples

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dict_of_list_values (dict): for each selected variables it provides a list of values to be combined in a cartesian product

        Returns:
            samples_df (dataframe) : generated samples
        '''

        vect_list = [dict_of_list_values[elem]
                     for elem in selected_inputs]

        def combvec(vect_list):
            my_sample = list(itertools.product(*vect_list))
            return my_sample
        my_res = combvec(vect_list)
        samples_df = pd.DataFrame(my_res, columns=selected_inputs)

        return samples_df
