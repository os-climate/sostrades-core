'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/05/16 Copyright 2023 Capgemini

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

import itertools
import logging

import pandas as pd

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import (
    AbstractSampleGenerator,
    SampleTypeError,
)

LOGGER = logging.getLogger(__name__)


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

    def __init__(self, logger: logging.Logger):
        '''
        Constructor
        '''
        super().__init__(self.GENERATOR_NAME, logger=logger)

    def _check_samples(self, samples_df):
        '''
        Method that checks the sample output type
        Arguments:
            samples_df (dataframe) : generated samples 
        Raises:
            Exception if samples_df is not a dataframe                   
        '''
        if not (isinstance(samples_df, pd.DataFrame)):
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

    @staticmethod
    def filter_eval_inputs_cp(eval_inputs_cp, wrapper):
        """
        Method that reformat eval_input_cp depending on user's selection

        Arguments:
            eval_inputs_cp (dataframe):

        Returns:
            dict_of_list_values (dict[list]) : dictionary {'var': [var_cp_values]} ignoring empty lists

        """
        if eval_inputs_cp is None or eval_inputs_cp.empty:
            return {}
        logic_1 = eval_inputs_cp[wrapper.SELECTED_INPUT] == True
        logic_2 = eval_inputs_cp[wrapper.LIST_OF_VALUES].isin([[]])
        logic_3 = eval_inputs_cp[wrapper.FULL_NAME] is None
        logic_4 = eval_inputs_cp[wrapper.FULL_NAME] == ''
        eval_inputs_cp_filtered = eval_inputs_cp[logic_1 &
                                                 ~logic_2 & ~logic_3 & ~logic_4]
        eval_inputs_cp_filtered = eval_inputs_cp_filtered[[wrapper.FULL_NAME, wrapper.LIST_OF_VALUES]]
        return eval_inputs_cp_filtered.set_index(wrapper.FULL_NAME)[wrapper.LIST_OF_VALUES].to_dict()

    def get_arguments(self, wrapper):
        eval_inputs_cp = wrapper.get_sosdisc_inputs(wrapper.EVAL_INPUTS)
        dict_of_list_values = self.filter_eval_inputs_cp(eval_inputs_cp, wrapper)
        return [], {'dict_of_list_values': dict_of_list_values}

    def is_ready_to_sample(self, proxy):
        # the generator is ready to sample if there are items in the dict_of_list_values, which is a dictionary
        # {'var': [var_cp_values]} that is assured to contain no empty lists due to filter_eval_inputs_cp
        _args, _kwargs = self.get_arguments(proxy)
        return bool(_kwargs['dict_of_list_values'])
