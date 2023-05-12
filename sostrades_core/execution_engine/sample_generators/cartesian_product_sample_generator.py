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

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator,\
    SampleTypeError

import pandas as pd

import itertools

import logging
LOGGER = logging.getLogger(__name__)

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class CartesianProductSampleTypeError(SampleTypeError):
    pass


class CartesianProductSampleGenerator(AbstractSampleGenerator):
    '''
    Caresian Product class that generates sampling
    '''
    GENERATOR_NAME = "CARTESIAN_PRODUCT_GENERATOR"

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    N_PROCESSES = 'n_processes'
    WAIT_TIME_BETWEEN_SAMPLES = 'wait_time_between_samples'

    N_SAMPLES = "n_samples"

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__(self.GENERATOR_NAME)

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

    def _generate_samples(self, dict_of_list_values):
        '''
        Method that generate samples based as a cartesian product of list of values for selected variables.
        Selected variables are provided in the keys of "dict_of_list_values".

        Arguments:
            dict_of_list_values (dict): for each selected variables it provides a list of values to be combined in a cartesian product

        Returns:
            samples_df (dataframe) : generated samples
        '''

        variable_list = dict_of_list_values.keys()
        vect_list = [dict_of_list_values[elem]
                     for elem in variable_list]

        def combvec(vect_list):
            my_sample = list(itertools.product(*vect_list))
            return my_sample
        my_res = combvec(vect_list)
        samples_df = pd.DataFrame(my_res, columns=variable_list)

        return samples_df
