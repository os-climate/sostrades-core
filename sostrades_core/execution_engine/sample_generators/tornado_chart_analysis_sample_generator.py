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
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator,\
    SampleTypeError

import itertools

import logging
LOGGER = logging.getLogger(__name__)

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class TornadoChartAnalysisSampleTypeError(SampleTypeError):
    pass


class TornadoChartAnalysisSampleGenerator(AbstractSampleGenerator):
    '''
    Tornado chart Analysis class that generates sampling
    '''
    GENERATOR_NAME = "TORNADO_CHART_ANALYSIS_GENERATOR"

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
    NS_ANALYSIS = 'ns_analysis'
    REFERENCE_SCENARIO_NAME = 'reference_scenario'
    SCENARIO_NAMES = SampleGeneratorWrapper.SCENARIO_NAME

    def __init__(self, logger: logging.Logger):
        '''
        Constructor
        '''
        self.selected_inputs = []
        self.selected_inputs_types = {}
        self.ns_sampling = None
        
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
        # retrieve the namespace of the reference scenario
        self.ns_sampling = proxy.ee.ns_manager.get_shared_namespace_value(proxy, proxy.NS_SAMPLING)

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
                design_scenario_mapping_descriptor = {self.SCENARIO_NAMES:('string', None, False)}
                design_scenario_mapping_descriptor.update({
                    name: ('float', None, False)
                    for name in selected_inputs
                })
                dynamic_inputs.update({self.SCENARIO_VARIABLE_VARIATIONS: {proxy.TYPE: 'dataframe',
                                                            proxy.STRUCTURING: False,
                                                            proxy.EDITABLE: False,
                                                            proxy.DATAFRAME_DESCRIPTOR: design_scenario_mapping_descriptor,
                                                            proxy.VISIBILITY: proxy.SHARED_VISIBILITY,
                                                            proxy.NAMESPACE: self.NS_ANALYSIS}})
        samples_df_full_path = f'{self.ns_sampling}.{proxy.SAMPLES_DF}'
        if proxy.ee.dm.check_data_in_dm(samples_df_full_path):
            #set samples_df not editable
            proxy.ee.dm.set_data(samples_df_full_path, proxy.EDITABLE, False, check_value=False)

            
    def _generate_samples(self, variation_list, dict_of_value, proxy):
        '''
        generate samples: each scenario is a percentage applied to one of the variables.
        First scenario is the reference scenario (0% of variation)
        '''
        
        selected_scenario_str = SampleGeneratorWrapper.SELECTED_SCENARIO
        scenario_names_str = SampleGeneratorWrapper.SCENARIO_NAME
        
        samples_dict = {}
        scenario_variations_dict = {}
        # add the reference value(0%) at the begining of the list
        if 0.0 in variation_list:
            variation_list.remove(0.0)

        # add reference scenario
        samples_dict[selected_scenario_str] = [True]
        samples_dict[scenario_names_str] = [self.REFERENCE_SCENARIO_NAME]
        samples_dict.update({name:[value] for name, value in dict_of_value.items()})

        scenario_variations_dict[scenario_names_str] = [self.REFERENCE_SCENARIO_NAME]
        scenario_variations_dict.update({name:[0.0] for name in dict_of_value.keys()})

        nb_scenario = len(variation_list) # number of scenario per variable (without reference)
        current_sc = 1 # current scenario number

        for name, value in dict_of_value.items():
            
            if value is not None and isinstance(value, float) and bool([isinstance(percent, float) for percent in variation_list]):
                # compute all values with percentages, add them in samples_dict with scenario names
                samples_dict[selected_scenario_str].extend([True]*nb_scenario)
                samples_dict[scenario_names_str].extend([f'scenario_{i}' for i in range(current_sc, nb_scenario + current_sc)])
                samples_dict[name].extend([(1.0 + percent/100.0) * value for percent in variation_list])
                
                #update scenario variation with percentages per scenario
                scenario_variations_dict[scenario_names_str].extend([f'scenario_{i}' for i in range(current_sc, nb_scenario + current_sc)])
                scenario_variations_dict[name].extend(variation_list)
                
                # other values are reference values (only the current variable has changed)
                for other_name, other_value in dict_of_value.items():
                    if other_name != name:
                        samples_dict[other_name].extend([other_value]*nb_scenario)
                        scenario_variations_dict[other_name].extend([0.0]*nb_scenario)
                
                current_sc += nb_scenario
        
        # build dataframes
        samples_df = pd.DataFrame(samples_dict)
        samples_percentage_df = pd.DataFrame(scenario_variations_dict)
        
        disc_in = proxy.get_data_in()
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
                # check that the value is defined
                if f'{self.ns_sampling}.{selected_input}' in wrapper.dm.data_id_map.keys():
                    samples_df[selected_input] = wrapper.dm.get_value(f'{self.ns_sampling}.{selected_input}')
                else:
                    self.logger.info(f'the variable {selected_input} is not found in data manager')
            arguments[self.DICT_OF_VALUE] = samples_df

        # set the proxy to set the scenario_variations into dm values
        arguments['proxy'] = wrapper

        return [], arguments

    def is_ready_to_sample(self, proxy):
        # the generator is ready to sample if there are items in the dict_of_list_values, which is a dictionary
        # {'var': [var_cp_values]} that is assured to contain no empty lists due to filter_eval_inputs_cp
        desc_in = proxy.get_data_in()
        return self.VARIATION_LIST in desc_in and proxy.EVAL_INPUTS in desc_in
