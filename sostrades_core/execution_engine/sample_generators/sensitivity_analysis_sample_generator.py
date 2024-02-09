'''
Copyright 2024 Capgemini

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

import pandas as pd
from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator,\
    SampleTypeError

import itertools

import logging
LOGGER = logging.getLogger(__name__)

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class SensitivityAnalysisSampleTypeError(SampleTypeError):
    pass


class SensitivityAnalysisSampleGenerator(AbstractSampleGenerator):
    '''
    SensitivityAnalysis class that generates sampling
    '''
    GENERATOR_NAME = "SENSITIVITY_ANALYSIS_GENERATOR"

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    N_PROCESSES = 'n_processes'
    WAIT_TIME_BETWEEN_SAMPLES = 'wait_time_between_samples'
    VALUE = 'value'
    VARIATION_LIST = 'variation_list'
    SCENARIO_VARIABLE_VARIATIONS = 'scenario_variations'
    DICT_OF_VALUE = 'dict_of_value'
    N_SAMPLES = "n_samples"

    def __init__(self, logger: logging.Logger):
        '''
        Constructor
        '''
        self.selected_inputs = []
        self.selected_inputs_types = {}
        super().__init__(self.GENERATOR_NAME, logger=logger)

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

    def setup(self, proxy):
        """
        Method that setup the  method
        """
        dynamic_inputs = {}
        dynamic_outputs = {}

        # Setup dynamic inputs in case of sensitivity analysis selection:
        # i.e. PERCENTAGES
        self.setup_dynamic_inputs_for_sensitivity_analysis_method(dynamic_inputs, proxy)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_for_sensitivity_analysis_method(self, dynamic_inputs, proxy):
        """
        Method that setup dynamic inputs in case of sensitivity analysis selection: i.e. PERCENTAGES
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        """
        # save possible types in sample generator
        if proxy.eval_in_possible_types is not None:
            self.selected_inputs_types = proxy.eval_in_possible_types.copy()

        
                    
        # add variation list
        dynamic_inputs.update({self.VARIATION_LIST:
                                       {proxy.TYPE: 'list',
                                        proxy.DEFAULT: [-5, 5],
                                        proxy.STRUCTURING: True,
                                        proxy.UNIT:'%'}
                                   })
        
        disc_in = proxy.get_data_in()
        # Dynamic input of mapping of variable for variables each scenario
        if proxy.EVAL_INPUTS in disc_in:
            eval_inputs = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
            if eval_inputs is not None:
                selected_inputs = eval_inputs[eval_inputs['selected_input'] == True]['full_name'].tolist()

                # save selected inputs in sample generator
                if set(selected_inputs) != set(self.selected_inputs):
                    self.selected_inputs = selected_inputs

                # the dataframe containing the mapping of variation percentage for each variables for each scenario
                # it will be: {'var_1': [percentage_sc1, percentage_sc2], 'var_2': [percentage_sc1, percentage_sc2]}
                design_scenario_mapping_descriptor = {
                    name: ('float', None, False)
                    for name in selected_inputs
                }
                dynamic_inputs.update({self.SCENARIO_VARIABLE_VARIATIONS: {proxy.TYPE: 'dataframe',
                                                            proxy.STRUCTURING: False,
                                                            proxy.EDITABLE: False,
                                                            proxy.DATAFRAME_DESCRIPTOR: design_scenario_mapping_descriptor}})

            
    def _generate_samples(self, variation_list, dict_of_value, proxy):
        '''
        
        '''
        disc_in = proxy.get_data_in()
        dict_of_list_values = {}
        for name, value in dict_of_value.items():
            # compute all values with percentages
            if value is not None and len(value)> 0 and bool([isinstance(percent, float) for percent in variation_list]):
                dict_of_list_values[name] = [(1.0 + percent/100.0) * value[0] for percent in variation_list]
        variable_list = dict_of_list_values.keys()
        vect_list = [dict_of_list_values[elem]
                     for elem in variable_list]
        percentage_vect_list = [variation_list for elem in variable_list]

        def combvec(vect_list):
            my_sample = list(itertools.product(*vect_list))
            my_sample_percentages = list(itertools.product(*percentage_vect_list))
            return my_sample,my_sample_percentages
        my_res, my_percentages = combvec(vect_list)
        samples_df = pd.DataFrame(my_res, columns=variable_list)
        samples_percentage_df = pd.DataFrame(my_percentages, columns=variable_list)
        
        proxy.dm.set_data(proxy.get_var_full_name(self.SCENARIO_VARIABLE_VARIATIONS, disc_in),
                                      proxy.VALUE, samples_percentage_df, check_value=False)
        return samples_df


    def get_arguments(self, wrapper):
        desc_in = wrapper.get_data_in()
        arguments = {}
        # retrieve variation list
        if self.VARIATION_LIST in desc_in:
            arguments[self.VARIATION_LIST] = wrapper.get_sosdisc_inputs(self.VARIATION_LIST)

        # retrieve input values
        if self.selected_inputs is not None:
            samples_df = {}
            for selected_input in self.selected_inputs:
                #TODO recuperer la value avec le full name!!! check in simple_sample_generator
                samples_df[selected_input] = wrapper.dm.get_value(selected_input)
            arguments[self.DICT_OF_VALUE] = samples_df

        # set the proxy to set the scenario_variations into dm values
        arguments['proxy'] = wrapper

        return [], arguments

    def is_ready_to_sample(self, proxy):
        # the generator is ready to sample if there are items in the dict_of_list_values, which is a dictionary
        # {'var': [var_cp_values]} that is assured to contain no empty lists due to filter_eval_inputs_cp
        desc_in = proxy.get_data_in()
        return self.VARIATION_LIST in desc_in and proxy.EVAL_INPUTS in desc_in
