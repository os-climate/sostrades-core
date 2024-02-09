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
        dynamic_inputs.update({self.VARIATION_LIST:
                                       {proxy.TYPE: 'list',
                                        proxy.DEFAULT: [-5,0, 5],
                                        proxy.STRUCTURING: True}
                                   })
        
        disc_in = proxy.get_data_in()
        # Dynamic input of mapping of variable for variables each scenario
        if proxy.EVAL_INPUTS in disc_in:
            eval_inputs = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
            if eval_inputs is not None:
                selected_inputs = eval_inputs[eval_inputs['selected_input'] == True]['full_name'].tolist()

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

    def filter_eval_inputs(self, eval_inputs_cp, wrapper):
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
        logic_2 = eval_inputs_cp[wrapper.LIST_OF_VALUES] is not None
        eval_inputs_cp_filtered = eval_inputs_cp[logic_1 & logic_2]
        eval_inputs_cp_filtered = eval_inputs_cp_filtered[[wrapper.FULL_NAME, wrapper.LIST_OF_VALUES]]
        return eval_inputs_cp_filtered.set_index(wrapper.FULL_NAME)[wrapper.LIST_OF_VALUES].to_dict()

    def get_arguments(self, wrapper):
        desc_in = wrapper.get_data_in()
        arguments = {}
        if self.VARIATION_LIST in desc_in:
            arguments[self.VARIATION_LIST] = wrapper.get_sosdisc_inputs(self.VARIATION_LIST)
        eval_inputs = wrapper.get_sosdisc_inputs(wrapper.EVAL_INPUTS)
        dict_of_value = self.filter_eval_inputs(eval_inputs, wrapper)
        arguments[self.DICT_OF_VALUE] = dict_of_value
        arguments['proxy'] = wrapper
        return [], arguments

    def is_ready_to_sample(self, proxy):
        # the generator is ready to sample if there are items in the dict_of_list_values, which is a dictionary
        # {'var': [var_cp_values]} that is assured to contain no empty lists due to filter_eval_inputs_cp
        desc_in = proxy.get_data_in()
        return self.VARIATION_LIST in desc_in and proxy.EVAL_INPUTS in desc_in
