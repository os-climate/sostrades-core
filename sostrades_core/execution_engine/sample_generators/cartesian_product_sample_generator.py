'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/03 Copyright 2023 Capgemini

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

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator,\
    SampleTypeError
from gemseo.utils.compare_data_manager_tooling import dict_are_equal

import pandas as pd
import numpy as np

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

    def __init__(self, logger: logging.Logger):
        '''
        Constructor
        '''
        super().__init__(self.GENERATOR_NAME, logger=logger)
        self.previous_eval_inputs_cp = None
        self.eval_inputs_cp_has_changed = False
        self.eval_inputs_cp_filtered = None
        self.eval_inputs_cp_validity = True

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

    # TODO: REFACTOR IF POSSIBLE W/O PROXY REFs
    def setup(self, proxy):
        """
        Method that setup the cp method
        """
        dynamic_inputs = {}
        dynamic_outputs = {}
        # Setup dynamic inputs which depend on EVAL_INPUTS_CP setting or
        # update: i.e. GENERATED_SAMPLES
        self.setup_dynamic_inputs_which_depend_on_eval_input_cp(dynamic_inputs, proxy)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_which_depend_on_eval_input_cp(self, dynamic_inputs, proxy):
        """
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        # TODO : why is it more complex as in doe_algo ? [???]
        self.eval_inputs_cp_has_changed = False
        disc_in = proxy.get_data_in()
        if proxy.EVAL_INPUTS in disc_in:
            eval_inputs_cp = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
            self.setup_eval_inputs_cp_and_generated_samples(dynamic_inputs, eval_inputs_cp, proxy)

    def setup_eval_inputs_cp_and_generated_samples(self, dynamic_inputs, eval_inputs_cp, proxy):
        """
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
            eval_inputs_cp (dataframe): the variables and possible values for the sample
        """
        # TODO: this way of working with setting attributes should be reviewed
        # 1. Manage update status of EVAL_INPUTS_CP
        # if not (eval_inputs_cp.equals(self.previous_eval_inputs_cp)):
        # if not dict_are_equal({'_': eval_inputs_cp},
        #                       {'_': self.previous_eval_inputs_cp}):
        if not dict_are_equal( eval_inputs_cp, self.previous_eval_inputs_cp):
            self.eval_inputs_cp_has_changed = True
            self.previous_eval_inputs_cp = eval_inputs_cp
        # 2. Manage selection in EVAL_INPUTS_CP
        if eval_inputs_cp is not None:
            # reformat eval_inputs_cp to take into account only useful
            # informations
            self.eval_inputs_cp_filtered = self.reformat_eval_inputs_cp(
                eval_inputs_cp)
            # Check selected input cp validity
            self.eval_inputs_cp_validity = self.check_eval_inputs_cp(
                self.eval_inputs_cp_filtered)
            # # Setup GENERATED_SAMPLES for cartesian product
            # if proxy.sampling_generation_mode == proxy.AT_CONFIGURATION_TIME:
            #     self.setup_generated_samples_for_cp(dynamic_inputs, proxy)
        else:
            self.eval_inputs_cp_validity = False


    # def setup_generated_sample(self, dynamic_inputs, proxy):
    #     """
    #     Method that setup GENERATED_SAMPLES for cartesian product at configuration time
    #     Arguments:
    #         dynamic_inputs (dict): the dynamic input dict to be updated
    #     """
    #     generated_samples_data_description = proxy.SAMPLES_DF_DESC_SHARED.copy()
    #
    #     # TODO: implement separation btw config. and sampling at config. time (remaining of the method should go away)
    #     if self.eval_inputs_cp_validity:
    #         if self.eval_inputs_cp_has_changed:
    #             proxy.set_sample()
    #
    #         # df_descriptor = {proxy.SELECTED_SCENARIO: ('bool', None, False),
    #         #                  proxy.SCENARIO_NAME: ('string', None, False)}
    #         # df_descriptor.update(
    #         #     {row['full_name']: (type(row['list_of_values'][0]).__name__, None, False) for index, row in
    #         #      self.eval_inputs_cp_filtered.iterrows()})  # FIXME: no good, handle DATAFRAME_DESCRIPTOR
    #         # generated_samples_data_description.update({proxy.DATAFRAME_DESCRIPTOR: df_descriptor,
    #         #                                            proxy.DYNAMIC_DATAFRAME_COLUMNS: False})
    #     else:
    #         # TODO: better handling of wrong input for CP
    #         proxy.samples_gene_df = pd.DataFrame(columns=[proxy.SELECTED_SCENARIO, proxy.SCENARIO_NAME])
    #
    #     # generated_samples_data_description.update({proxy.DEFAULT: proxy.samples_gene_df})
    #     dynamic_inputs.update({proxy.SAMPLES_DF: generated_samples_data_description})
    #
    #     # Set or update GENERATED_SAMPLES in line with selected
    #     # eval_inputs_cp
    #     disc_in = proxy.get_data_in()  #FIXME: pass disc_in
    #     if proxy.SAMPLES_DF in disc_in:
    #         # proxy.set_sample()
    #         if proxy.samples_gene_df is not None:
    #             proxy.dm.set_data(proxy.get_var_full_name(proxy.SAMPLES_DF, disc_in),
    #                               'value', proxy.samples_gene_df, check_value=False)
    #         proxy.sample_pending = False
    #         # disc_in[self.GENERATED_SAMPLES][self.VALUE] = self.samples_gene_df
    #     else:
    #         # TODO: generalise to all methods sampling at config-time (when decoupling setup from sampling) or
    #         #  otherwise there will be issues when generator tries to sample before samples_df is added in disc_in
    #         proxy.sample_pending = True

    def reformat_eval_inputs_cp(self, eval_inputs_cp):
        """
        Method that reformat eval_input_cp depending on user's selection

        Arguments:
            eval_inputs_cp (dataframe):

        Returns:
            eval_inputs_cp_filtered (dataframe) :

        """
        logic_1 = eval_inputs_cp['selected_input'] == True
        logic_2 = eval_inputs_cp['list_of_values'].isin([[]])
        logic_3 = eval_inputs_cp['full_name'] is None
        logic_4 = eval_inputs_cp['full_name'] == ''
        eval_inputs_cp_filtered = eval_inputs_cp[logic_1 &
                                                 ~logic_2 & ~logic_3 & ~logic_4]
        eval_inputs_cp_filtered = eval_inputs_cp_filtered[[
            'full_name', 'list_of_values']]
        return eval_inputs_cp_filtered

    def check_eval_inputs_cp(self, eval_inputs_cp_filtered):
        """
        Method that reformat eval_input_cp depending on user's selection

        Arguments:
            eval_inputs_cp (dataframe):

        Returns:
            validity (boolean) :

        """
        # TODO: better handling of CartesianProduct wrong input...
        is_valid = True
        selected_inputs_cp = list(eval_inputs_cp_filtered['full_name'])
        # n_min = 2
        n_min = 1
        if len(selected_inputs_cp) < n_min:
            self.logger.warning(
                f'Selected_inputs must have at least {n_min} variables to do a cartesian product')
            is_valid = False
        return is_valid

    def get_arguments(self, proxy):
        dict_of_list_values = self.eval_inputs_cp_filtered.set_index(
            'full_name').T.to_dict('records')[0]
        return [], {'dict_of_list_values': dict_of_list_values}

    def is_ready_to_sample(self, proxy):
        return self.eval_inputs_cp_validity and self.eval_inputs_cp_filtered is not None # and self.eval_inputs_cp_has_changed # should come from structuring checks
