'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/02 Copyright 2023 Capgemini

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
import copy
import re

import logging
import platform
from tqdm import tqdm
import time

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from gemseo.utils.compare_data_manager_tooling import dict_are_equal

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.sample_generators.simple_sample_generator import SimpleSampleGenerator
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import \
    CartesianProductSampleGenerator
import pandas as pd
import numpy as np
from collections import ChainMap
from gemseo.api import get_available_doe_algorithms


class SampleGeneratorWrapper(SoSWrapp):
    '''
    Generic SampleGenerator class
    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SAMPLING_METHOD (structuring)       
                |_ EVAL_INPUTS (namespace: 'ns_sampling', structuring, dynamic : SAMPLING_METHOD == self.DOE_ALGO) 
                        |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO != None) NB: default DESIGN_SPACE depends on EVAL_INPUTS (Has to be "Not empty")            
                |_ SAMPLING_ALGO (structuring, dynamic : SAMPLING_METHOD == self.DOE_ALGO)
                        |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
                            |_ GENERATED_SAMPLES (namespace: 'ns_sampling', structuring,                                               
                                                 dynamic: EVAL_INPUTS_CP != None and ALGO_OPTIONS set 
                                                     and SAMPLING_GENERATION_MODE == 'at_configuration_time')                          
                |_ EVAL_INPUTS_CP (namespace: 'ns_sampling', structuring, dynamic : SAMPLING_METHOD == self.CARTESIAN_PRODUCT)
                        |_ GENERATED_SAMPLES (namespace: 'ns_sampling', structuring,                                               
                                               dynamic: EVAL_INPUTS_CP != None and SAMPLING_GENERATION_MODE == 'at_configuration_time')                 
            |_ SAMPLING_GENERATION_MODE ('editable': False)
        |_ DESC_OUT
            |_ SAMPLES_DF (namespace: 'ns_sampling')

   2) Description of DESC parameters:
        |_ DESC_IN
            |_ SAMPLING_METHOD   
                |_ EVAL_INPUTS
                        |_ DESIGN_SPACE 
                |_ SAMPLING_ALGO 
                        |_ ALGO_OPTIONS 
                            |_ GENERATED_SAMPLES                        
                |_ EVAL_INPUTS_CP 
                        |_ GENERATED_SAMPLES 
        |_ DESC_OUT
            |_ SAMPLES_DF

    '''

    _ontology_data = {
        'label': 'Sample_Generator wrapper',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'Sample_Generator wrapper that implements the genearotion of a samples from a DoE (Design of Experiment) algorithm or from a cross product.',
        # icon for sample generator from
        # https://fontawesome.com/icons/grid-4?s=regular&f=classic
        'icon': 'fas fa-grid-4 fa-fw',
        'version': ''
    }

    VARIABLES = "variable"
    VALUES = "value"
    POSSIBLE_VALUES = 'possible_values'
    TYPE = "type"

    NS_SEP = '.'
    INPUT_TYPE = ['float', 'array', 'int', 'string']

    DESIGN_SPACE = "design_space"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"

    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"

    N_SAMPLES = "n_samples"
    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    USER_GRAD = 'user'

    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    SAMPLING_METHOD = 'sampling_method'
    SIMPLE_SAMPLING_METHOD = 'simple'
    DOE_ALGO = 'doe_algo'
    CARTESIAN_PRODUCT = 'cartesian_product'
    GRID_SEARCH = 'grid_search'
    FULLFACT = 'fullfact'
    available_sampling_methods = [SIMPLE_SAMPLING_METHOD, DOE_ALGO, CARTESIAN_PRODUCT, GRID_SEARCH]

    SAMPLE_GENERATOR_CLS = {
        SIMPLE_SAMPLING_METHOD: SimpleSampleGenerator,
        DOE_ALGO: DoeSampleGenerator,
        CARTESIAN_PRODUCT: CartesianProductSampleGenerator,
        GRID_SEARCH: CartesianProductSampleGenerator
    }

    SAMPLING_GENERATION_MODE = 'sampling_generation_mode'
    AT_CONFIGURATION_TIME = 'at_configuration_time'
    AT_RUN_TIME = 'at_run_time'
    available_sampling_generation_modes = [AT_CONFIGURATION_TIME, AT_RUN_TIME]

    EVAL_INPUTS_CP = 'eval_inputs_cp'
    GENERATED_SAMPLES = 'generated_samples'
    SAMPLES_DF = 'samples_df'
    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'

    NS_DRIVER = 'ns_driver'
    NS_SAMPLING = 'ns_sampling'
    REFERENCE_SCENARIO_NAME = 'Reference Scenario'
    SAMPLES_DF_DESC = {
        SoSWrapp.TYPE: 'dataframe',
        SoSWrapp.DEFAULT: pd.DataFrame({SELECTED_SCENARIO: [True],
                                        SCENARIO_NAME: REFERENCE_SCENARIO_NAME}),
        # SoSWrapp.DEFAULT: pd.DataFrame(
        #     columns=[SELECTED_SCENARIO, SCENARIO_NAME]),
        SoSWrapp.DATAFRAME_DESCRIPTOR: {SELECTED_SCENARIO: ('bool', None, True),
                                        SCENARIO_NAME: ('string', None, True)},
        SoSWrapp.DYNAMIC_DATAFRAME_COLUMNS: True,
        SoSWrapp.DATAFRAME_EDITION_LOCKED: False,
        SoSWrapp.EDITABLE: True,
        SoSWrapp.STRUCTURING: True,
        SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
        SoSWrapp.NAMESPACE: NS_DRIVER
    }

    DESC_IN = {SAMPLING_METHOD: {'type': 'string',
                                 'structuring': True,
                                 'possible_values': available_sampling_methods,
                                 'default': DOE_ALGO},
               SAMPLING_GENERATION_MODE: {'type': 'string',
                                          'structuring': False, # TODO: when editable also structuring
                                          'possible_values': available_sampling_generation_modes,
                                          'default': AT_RUN_TIME,
                                          'editable': False} # TODO: render editable
               }

    # DESC_OUT = {SAMPLES_DF: SAMPLES_DF_DESC}

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name=sos_name, logger=logger)
        self.sampling_method = None
        self.sample_generator = None

        self.sampling_generation_mode = None

        # self.previous_algo_name = ""

        self.selected_inputs = []
        self.dict_desactivated_elem = {}

        self.previous_eval_inputs_cp = None
        self.eval_inputs_cp_has_changed = False
        self.eval_inputs_cp_filtered = None
        self.eval_inputs_cp_validity = True
        self.samples_gene_df = None

    def setup_sos_disciplines(self):
        '''
        Overload of setup_sos_disciplines to specify the specific dynamic inputs of sample generator
        '''

        disc_in = self.get_data_in()

        if len(disc_in) != 0:
            self.sampling_method = self.get_sosdisc_inputs(
                self.SAMPLING_METHOD)
            self.instantiate_sampling_tool()

            # Switch between doe_algo method or cartesian_product method

            # TODO: Here we start from scratch each time we switch from one method to the other
            #       ... but maybe we would like to keep selected eval_inputs or eval_inputs_cp ?

            if self.sampling_method == self.SIMPLE_SAMPLING_METHOD:
                # Reset parameters of the other method to initial values
                # (cleaning)
                # TODO: move all of these to the corresponding tools !
                self.previous_eval_inputs_cp = None
                self.eval_inputs_cp_filtered = None
                self.eval_inputs_cp_validity = True
                self.selected_inputs = []
                self.dict_desactivated_elem = {}

                # 0. force config time sampling
                self.sampling_generation_mode = self.AT_CONFIGURATION_TIME
                disc_in[self.SAMPLING_GENERATION_MODE][self.VALUE] = self.AT_CONFIGURATION_TIME

                # 1. handle dynamic inputs of the mode
                # TODO: a dedicated dynamic io method but Q: should be moved to the tool ?
                dynamic_inputs, dynamic_outputs = {}, {}
                dynamic_inputs.update({'eval_inputs':
                                           {self.TYPE: 'dataframe',
                                            self.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
                                                                        'full_name': ('string', None, False)},
                                            self.DATAFRAME_EDITION_LOCKED: False,
                                            self.STRUCTURING: True,
                                            self.VISIBILITY: self.SHARED_VISIBILITY,
                                            self.NAMESPACE: self.NS_SAMPLING}
                                       })
                dynamic_inputs.update({'scenario_names':
                                           {self.TYPE: 'list',
                                            self.SUBTYPE: {'list': 'string'},
                                            self.STRUCTURING: True,
                                            self.VISIBILITY: self.SHARED_VISIBILITY,
                                            self.NAMESPACE: self.NS_SAMPLING}
                                       })
                dynamic_inputs.update({self.GENERATED_SAMPLES: {self.TYPE: 'dataframe',
                                                                self.DATAFRAME_DESCRIPTOR: {},
                                                                self.DYNAMIC_DATAFRAME_COLUMNS: True,
                                                                self.DATAFRAME_EDITION_LOCKED: True,
                                                                self.STRUCTURING: True,
                                                                self.UNIT: None,
                                                                self.VISIBILITY: self.SHARED_VISIBILITY,
                                                                self.NAMESPACE: self.NS_SAMPLING,
                                                                self.DEFAULT: pd.DataFrame()}})
                # 2. retrieve input that configures the sampling tool
                if 'scenario_names' in disc_in:
                    scenario_names = self.get_sosdisc_inputs('scenario_names')
                    eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
                    if scenario_names and eval_inputs is not None:
                        selected_inputs = self.reformat_eval_inputs(eval_inputs).tolist()
                        # 3. if sampling at config.time set the generated samples
                        self.samples_gene_df = self.sample_generator.generate_samples(selected_inputs)
                        self.samples_gene_df = self.set_scenario_columns(self.samples_gene_df,
                                                                         scenario_names=scenario_names)
                        disc_in[self.GENERATED_SAMPLES][self.VALUE] = self.samples_gene_df

            elif self.sampling_method == self.DOE_ALGO:
                # TODO: consider refactoring this in object-oriented fashion before implementing the more complex modes
                # Reset parameters of the other method to initial values
                # (cleaning)
                self.previous_eval_inputs_cp = None
                self.eval_inputs_cp_filtered = None
                self.eval_inputs_cp_validity = True

                # setup_doe_algo_method
                # TODO: configuration-time sampling not implemented yet for doe and gridsearch
                self.sampling_generation_mode = self.AT_RUN_TIME
                disc_in[self.SAMPLING_GENERATION_MODE][self.VALUE] = self.AT_RUN_TIME

                # self.sampling_generation_mode = self.AT_CONFIGURATION_TIME #
                # It was tested that it also works

                dynamic_inputs, dynamic_outputs = self.setup_doe_algo_method()

            elif self.sampling_method == self.CARTESIAN_PRODUCT:
                # Reset parameters of the other method to initial values
                # (cleaning)
                self.selected_inputs = []
                self.dict_desactivated_elem = {}

                # setup_cp_method

                self.sampling_generation_mode = self.AT_CONFIGURATION_TIME
                disc_in[self.SAMPLING_GENERATION_MODE][self.VALUE] = self.AT_CONFIGURATION_TIME
                # self.sampling_generation_mode = self.AT_RUN_TIME # It was
                # tested that it also works

                dynamic_inputs, dynamic_outputs = self.setup_cp_method()

            elif self.sampling_method == self.GRID_SEARCH:
                # setup_cp_method
                # TODO: configuration-time sampling not implemented yet for doe and gridsearch
                self.sampling_generation_mode = self.AT_RUN_TIME
                disc_in[self.SAMPLING_GENERATION_MODE][self.VALUE] = self.AT_RUN_TIME
                # most setup is similar to doe_algo
                dynamic_inputs, dynamic_outputs = self.setup_doe_algo_method()

                self.setup_gs(dynamic_inputs)

            elif self.sampling_method is not None:
                raise Exception(
                    f"The selected sampling method {self.sampling_method} is not allowed in the sample generator. Please "
                    f"introduce one of the available methods from {self.available_sampling_methods}.")
            else:
                dynamic_inputs = {}
                dynamic_outputs = {}
        else:
            dynamic_inputs = {}
            dynamic_outputs = {}

        # 4. if sampling at run-time add the corresponding output
        if self.sampling_generation_mode == self.AT_RUN_TIME:
            dynamic_outputs[self.SAMPLES_DF] = self.SAMPLES_DF_DESC.copy()

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def set_eval_in_possible_values(self, possible_values):
        pass

    def run(self):
        '''
            Overloaded class method
            The generation of samples_df as run time
        '''
        if self.sampling_generation_mode == self.AT_RUN_TIME:
            samples_df = None

            if self.sampling_method == self.DOE_ALGO:
                samples_df = self.run_doe()
            elif self.sampling_method in [self.CARTESIAN_PRODUCT, self.GRID_SEARCH]:
                samples_df = self.run_cp()

            # Loop to raise an error in case the sampling has not been made.
            # If samples' type is dataframe, that means that the previous loop has
            # been entered.
            if isinstance(samples_df, pd.DataFrame):
                pass
            else:
                raise Exception(
                    f"Sampling has not been made")

            # Add the scenario names and selected scenario columns
            samples_df = self.set_scenario_columns(samples_df)
            self.store_sos_outputs_values({self.SAMPLES_DF: samples_df})

    def set_scenario_columns(self, samples_df, scenario_names=None):
        '''
        Add the columns SELECTED_SCENARIO and SCENARIO_NAME to the samples_df dataframe
        '''
        if self.SELECTED_SCENARIO not in samples_df:
            ordered_columns = [self.SELECTED_SCENARIO, self.SCENARIO_NAME] + samples_df.columns.tolist()
            if scenario_names is None:
                samples_df[self.SCENARIO_NAME] = [f'scenario_{i}' for i in range(1, len(samples_df) + 1)]
            else:
                samples_df[self.SCENARIO_NAME] = scenario_names
            samples_df[self.SELECTED_SCENARIO] = [True] * len(samples_df)
            samples_df = samples_df[ordered_columns]
        return samples_df

    def instantiate_sampling_tool(self):
        """
           Instantiate SampleGenerator only if needed
        """
        if self.sampling_method:
            sample_generator_cls = self.SAMPLE_GENERATOR_CLS[self.sampling_method]
            if self.sample_generator.__class__ != sample_generator_cls:
                self.sample_generator = sample_generator_cls(logger=self.logger.getChild(sample_generator_cls.__name__))

    def get_algo_default_options(self, algo_name):
        """
            This algo generate the default options to set for a given doe algorithm
        """

        # In get_options_and_default_values, it is already checked whether the algo_name belongs to the list of possible Gemseo
        # DoE algorithms
        if algo_name in get_available_doe_algorithms():
            algo_options_desc_in, algo_options_descr_dict = self.sample_generator.get_options_and_default_values(
                algo_name)
            return algo_options_desc_in
        else:
            raise Exception(
                f"A DoE algorithm which is not available in GEMSEO has been selected.")

    def create_design_space(self, selected_inputs, dspace_df):
        """
        create_design_space with variables names based on selected_inputs (if dspace_df is not None)

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dspace_df (dataframe): design space in Desc_in format     

        Returns:
             design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
        """

        design_space = None
        if dspace_df is not None:
            dspace_df_updated = self.update_design_space(
                selected_inputs, dspace_df)
            design_space = self.create_gemseo_dspace_from_dspace_df(
                dspace_df_updated)
        return design_space

    def update_design_space(self, selected_inputs, dspace_df):
        """
        update dspace_df (design space in Desc_in format)   

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dspace_df (dataframe): design space in Desc_in format     

        Returns:
             dspace_df_updated (dataframe): updated dspace_df        

        """
        lower_bounds = dspace_df[self.LOWER_BOUND].tolist()
        upper_bounds = dspace_df[self.UPPER_BOUND].tolist()
        values = lower_bounds
        enable_variables = [True for _ in selected_inputs]
        dspace_df_updated = pd.DataFrame({self.VARIABLES: selected_inputs,
                                          self.VALUES: values,
                                          self.LOWER_BOUND: lower_bounds,
                                          self.UPPER_BOUND: upper_bounds,
                                          self.ENABLE_VARIABLE_BOOL: enable_variables,
                                          self.LIST_ACTIVATED_ELEM: [[True] for _ in selected_inputs]})
        # TODO: Hardcoded as in EEV3, but not differenciating between array or not.
        return dspace_df_updated

    def create_gemseo_dspace_from_dspace_df(self, dspace_df):
        """
        Create gemseo dspace from sostrades updated dspace_df 
        It parses the dspace_df DataFrame to create the gemseo DesignSpace

        Arguments:
            dspace_df (dataframe): updated dspace_df     

        Returns:
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
        """
        names = list(dspace_df[self.VARIABLES])
        values = list(dspace_df[self.VALUES])
        l_bounds = list(dspace_df[self.LOWER_BOUND])
        u_bounds = list(dspace_df[self.UPPER_BOUND])
        enabled_variable = list(dspace_df[self.ENABLE_VARIABLE_BOOL])
        list_activated_elem = list(dspace_df[self.LIST_ACTIVATED_ELEM])
        design_space = DesignSpace()
        for dv, val, lb, ub, l_activated, enable_var in zip(names, values, l_bounds, u_bounds, list_activated_elem,
                                                            enabled_variable):

            # check if variable is enabled to add it or not in the design var
            if enable_var:

                self.dict_desactivated_elem[dv] = {}
                name = dv
                if type(val) != list and type(val) != ndarray:
                    size = 1
                    var_type = ['float']
                    l_b = array([lb])
                    u_b = array([ub])
                    value = array([val])
                else:
                    # check if there is any False in l_activated
                    if not all(l_activated):
                        index_false = l_activated.index(False)
                        self.dict_desactivated_elem[dv] = {
                            'value': val[index_false], 'position': index_false}

                        val = delete(val, index_false)
                        lb = delete(lb, index_false)
                        ub = delete(ub, index_false)

                    size = len(val)
                    var_type = ['float'] * size
                    l_b = array(lb)
                    u_b = array(ub)
                    value = array(val)
                design_space.add_variable(
                    name, size, var_type, l_b, u_b, value)
        return design_space

    def setup_doe_algo_method(self):
        """        
        Method that setup the doe_algo method
        """
        dynamic_inputs = {}
        dynamic_outputs = {}

        # Setup dynamic inputs in case of DOE_ALGO selection:
        # i.e. EVAL_INPUTS and SAMPLING_ALGO
        self.setup_dynamic_inputs_for_doe_generator_method(dynamic_inputs)
        # Setup dynamic inputs when EVAL_INPUTS/SAMPLING_ALGO are already set
        self.setup_dynamic_inputs_algo_options_design_space(dynamic_inputs)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_for_doe_generator_method(self, dynamic_inputs):
        """
        Method that setup dynamic inputs in case of DOE_ALGO selection: i.e. EVAL_INPUTS and SAMPLING_ALGO
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        """
        if self.sampling_method == self.DOE_ALGO:
            # Get possible values for sampling algorithm name
            available_doe_algorithms = self.sample_generator.get_available_algo_names()
            dynamic_inputs.update({'sampling_algo':
                                       {self.TYPE: 'string',
                                        self.STRUCTURING: True,
                                        self.POSSIBLE_VALUES: available_doe_algorithms}
                                   })
        # elif self.sampling_method == self.GRID_SEARCH:
        #     dynamic_inputs.update({'sampling_algo':
        #                            {'type': 'string', 'structuring': True,
        #                             'default': self.FULLFACT,
        #                             'possible_values': [self.FULLFACT],
        #                             'editable': False,
        #                             'user_level': 99}
        #                            })

        dynamic_inputs.update({self.EVAL_INPUTS:
                                   {self.TYPE: 'dataframe',
                                    self.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
                                                                'full_name': ('string', None, False)},
                                    self.DATAFRAME_EDITION_LOCKED: False,
                                    self.STRUCTURING: True,
                                    self.VISIBILITY: self.SHARED_VISIBILITY,
                                    self.NAMESPACE: self.NS_SAMPLING}
                               })

    def setup_dynamic_inputs_algo_options_design_space(self, dynamic_inputs):
        """
            Setup dynamic inputs when EVAL_INPUTS/SAMPLING_ALGO are already set
            Create or update DESIGN_SPACE
            Create or update ALGO_OPTIONS
        """
        self.setup_design_space(dynamic_inputs)
        self.setup_algo_options(dynamic_inputs)
        # Setup GENERATED_SAMPLES for cartesian product
        if self.sampling_generation_mode == self.AT_CONFIGURATION_TIME:
            # TODO: manage config-time sample for grid search and test for DoE
            self.setup_generated_samples_for_doe(dynamic_inputs)

    def setup_algo_options(self, dynamic_inputs):
        """
            Method that setup 'algo_options'
            Arguments:
                dynamic_inputs (dict): the dynamic input dict to be updated
        """
        disc_in = self.get_data_in()
        # Dynamic input of algo_options
        if self.ALGO in disc_in:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            if algo_name is not None:  # and algo_name_has_changed:
                default_dict = self.get_algo_default_options(algo_name)
                algo_options_dict = {self.ALGO_OPTIONS: {self.TYPE: 'dict', self.DEFAULT: default_dict,
                                                         self.DATAFRAME_EDITION_LOCKED: False,
                                                         self.STRUCTURING: True,
                                                         self.DATAFRAME_DESCRIPTOR: {
                                                             self.VARIABLES: ('string', None, False),
                                                             self.VALUES: ('string', None, True)}}}
                dynamic_inputs.update(algo_options_dict)
                all_options = list(default_dict.keys())
                if self.ALGO_OPTIONS in disc_in and disc_in[self.ALGO_OPTIONS][self.VALUE] is not None and list(
                        disc_in[self.ALGO_OPTIONS][self.VALUE].keys()) != all_options:
                    options_map = ChainMap(
                        disc_in[self.ALGO_OPTIONS][self.VALUE], default_dict)
                    disc_in[self.ALGO_OPTIONS][self.VALUE] = {
                        key: options_map[key] for key in all_options}

    def setup_design_space(self, dynamic_inputs):
        """
        Method that setup 'design_space'
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        selected_inputs_has_changed = False
        disc_in = self.get_data_in()
        # Dynamic input of default design space
        if 'eval_inputs' in disc_in:
            eval_inputs = self.get_sosdisc_inputs('eval_inputs')

            if eval_inputs is not None:

                # selected_inputs = eval_inputs[eval_inputs['selected_input']
                #                               == True]['full_name']
                selected_inputs = self.reformat_eval_inputs(
                    eval_inputs).tolist()

                if set(selected_inputs) != set(self.selected_inputs):
                    selected_inputs_has_changed = True
                    self.selected_inputs = selected_inputs

                default_design_space = pd.DataFrame()
                design_space_dataframe_descriptor = {
                    self.VARIABLES: ('string', None, False),
                    self.VALUES: ('multiple', None, True),
                    self.LOWER_BOUND: ('multiple', None, True),
                    self.UPPER_BOUND: ('multiple', None, True),
                    self.ENABLE_VARIABLE_BOOL: (
                        'bool', None, True),
                    self.LIST_ACTIVATED_ELEM: (
                        'list', None, True), }

                if self.sampling_method == self.DOE_ALGO:
                    default_design_space = pd.DataFrame({self.VARIABLES: self.selected_inputs,
                                                         self.LOWER_BOUND: [None] * len(self.selected_inputs),
                                                         self.UPPER_BOUND: [None] * len(self.selected_inputs)
                                                         })
                elif self.sampling_method == self.GRID_SEARCH:
                    default_design_space = pd.DataFrame({self.VARIABLES: self.selected_inputs,
                                                         self.LOWER_BOUND: [0.0] * len(self.selected_inputs),
                                                         self.UPPER_BOUND: [100.0] * len(self.selected_inputs),
                                                         'nb_points': [2] * len(self.selected_inputs)
                                                         })
                    design_space_dataframe_descriptor.update({'nb_points': ('int', None, True)})
                dynamic_inputs.update({'design_space': {self.TYPE: 'dataframe',
                                                        self.DEFAULT: default_design_space,
                                                        self.STRUCTURING: True,
                                                        self.DATAFRAME_DESCRIPTOR: design_space_dataframe_descriptor}})

                # Next lines of code treat the case in which eval inputs change with a previously defined design space,
                # so that the bound are kept instead of set to default None.
                if 'design_space' in disc_in:
                    disc_in['design_space'][self.DEFAULT] = default_design_space
                    disc_in['design_space'][self.DATAFRAME_DESCRIPTOR] = design_space_dataframe_descriptor
                    if selected_inputs_has_changed:
                        from_design_space = list(
                            disc_in['design_space'][self.VALUE]['variable'])
                        from_eval_inputs = self.selected_inputs

                        df_cols = ['variable', 'lower_bnd', 'upper_bnd'] + (
                            ['nb_points'] if self.sampling_method == self.GRID_SEARCH else [])
                        final_dataframe = pd.DataFrame(
                            None, columns=df_cols)

                        for element in from_eval_inputs:
                            if element in from_design_space:
                                final_dataframe = final_dataframe.append(disc_in['design_space'][self.VALUE]
                                                                         [disc_in['design_space'][self.VALUE][
                                                                              'variable'] == element])
                            else:
                                elem_dict = {'variable': element, 'lower_bnd': None, 'upper_bnd': None}
                                if self.sampling_method == self.GRID_SEARCH:
                                    elem_dict['lower_bnd'] = 0.0
                                    elem_dict['upper_bnd'] = 100.0
                                    elem_dict['nb_points'] = 2
                                final_dataframe = final_dataframe.append(
                                    elem_dict, ignore_index=True)
                        disc_in['design_space'][self.VALUE] = final_dataframe

    def filter_eval_inputs_types_to_float(self, eval_inputs):
        allowed_types = ['float']
        driverevaluator_ns = self.get_var_full_name(self.EVAL_INPUTS, self.get_data_in()).split('.eval_inputs')[
            0]  # pylint: disable-msg=E1121
        to_filter = []
        for var in eval_inputs['full_name']:
            var_f_name = '.'.join([driverevaluator_ns, var])
            if var_f_name in self.dm.data_id_map and self.dm.get_data(var_f_name, self.TYPE) not in allowed_types:
                to_filter.append(False)
            else:
                to_filter.append(True)
        return eval_inputs[to_filter]

    def reformat_eval_inputs(self, eval_inputs):
        """
        Method that reformat eval_input depending on user's selection
        Arguments:
            eval_inputs (dataframe):
        Returns:
            eval_inputs_filtered (dataframe) :

        """
        logic_1 = eval_inputs['selected_input'] == True
        eval_inputs_filtered = eval_inputs[logic_1]
        eval_inputs_filtered = eval_inputs_filtered['full_name']
        return eval_inputs_filtered

    def setup_generated_samples_for_doe(self, dynamic_inputs):
        """
         Method that setup GENERATED_SAMPLES for doe_algo at configuration time
         Arguments:
             dynamic_inputs (dict): the dynamic input dict to be updated
         """
        # if self.eval_inputs_validity:
        #    if self.eval_inputs_has_changed:
        disc_in = self.get_data_in()
        if self.ALGO in disc_in and self.ALGO_OPTIONS in disc_in and self.DESIGN_SPACE in disc_in and self.selected_inputs is not None:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)

            self.samples_gene_df = self.generate_sample_for_doe(
                algo_name, algo_options, dspace_df)
            self.samples_gene_df = self.set_scenario_columns(self.samples_gene_df)

            dynamic_inputs.update({self.GENERATED_SAMPLES: {self.TYPE: 'dataframe',
                                                            self.DATAFRAME_DESCRIPTOR: {},
                                                            self.DYNAMIC_DATAFRAME_COLUMNS: True,
                                                            self.DATAFRAME_EDITION_LOCKED: True,
                                                            self.STRUCTURING: True,
                                                            self.UNIT: None,
                                                            self.VISIBILITY: self.SHARED_VISIBILITY,
                                                            self.NAMESPACE: self.NS_SAMPLING,
                                                            self.DEFAULT: self.samples_gene_df}})

        # Set or update GENERATED_SAMPLES in line with selected
        # eval_inputs_cp
        disc_in = self.get_data_in()
        if self.GENERATED_SAMPLES in disc_in:
            disc_in[self.GENERATED_SAMPLES][self.VALUE] = self.samples_gene_df

    def generate_sample_for_doe(self, algo_name, algo_options, dspace_df):
        """
        Outputs:
            samples_gene_df (dataframe) : prepared samples for evaluation
        """
        # Dynamic input of default design space
        design_space = self.create_design_space(
            self.selected_inputs, dspace_df)

        samples_gene_df = self.sample_generator.generate_samples(
            algo_name, algo_options, design_space)
        return samples_gene_df

    def run_doe(self):
        """
        Method that generates the desc_out self.SAMPLES_DF from desc_in self.ALGO, self.ALGO_OPTIONS and self.DESIGN_SPACE

        Inputs:
            self.GENERATED_SAMPLES (dataframe) : generated samples from sampling method
        Outputs:
            self.SAMPLES_DF (dataframe) : prepared samples for evaluation
        """
        if self.sampling_generation_mode == self.AT_RUN_TIME:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)
            samples_df = self.generate_sample_for_doe(
                algo_name, algo_options, dspace_df)
        elif self.sampling_generation_mode == self.AT_CONFIGURATION_TIME:
            generated_samples = self.get_sosdisc_inputs(self.GENERATED_SAMPLES)
            samples_df = generated_samples

        return samples_df

    def setup_cp_method(self):
        """
        Method that setup the cp method
        """
        dynamic_inputs = {}
        dynamic_outputs = {}
        # Setup dynamic inputs for CARTESIAN_PRODUCT method: i.e.
        # EVAL_INPUTS_CP
        self.setup_dynamic_inputs_for_cp_generator_method(dynamic_inputs)
        # Setup dynamic inputs which depend on EVAL_INPUTS_CP setting or
        # update: i.e. GENERATED_SAMPLES
        self.setup_dynamic_inputs_which_depend_on_eval_input_cp(dynamic_inputs)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_for_cp_generator_method(self, dynamic_inputs):
        """
        Method that setup dynamic inputs in case of CARTESIAN_PRODUCT selection: i.e. EVAL_INPUTS_CP

        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        """
        # TODO :  the i/o methods should belong to the generator for modularity
        default_in_eval_input_cp = pd.DataFrame({'selected_input': [False],
                                                 'full_name': [''],
                                                 'list_of_values': [[]]})
        dynamic_inputs.update({self.EVAL_INPUTS_CP: {self.TYPE: 'dataframe',
                                                     self.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
                                                                                 'full_name': ('string', None, True),
                                                                                 'list_of_values': (
                                                                                     'list', None, True)},
                                                     self.DATAFRAME_EDITION_LOCKED: False,
                                                     self.STRUCTURING: True,
                                                     self.VISIBILITY: self.SHARED_VISIBILITY,
                                                     self.NAMESPACE: self.NS_SAMPLING,
                                                     self.DEFAULT: default_in_eval_input_cp}})

    def setup_gs(self, dynamic_inputs):
        """
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        with specificities for the GridSearch sampling method.
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        disc_in = self.get_data_in()
        self.eval_inputs_cp_has_changed = False
        if self.DESIGN_SPACE in disc_in:
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            design_space = self.get_sosdisc_inputs(self.DESIGN_SPACE)
            # link doe-like inputs to cp attributes in the framework of GridSearch
            eval_inputs_cp = self.get_eval_inputs_cp_for_gs(eval_inputs, design_space)
            self.setup_eval_inputs_cp_and_generated_samples(dynamic_inputs, eval_inputs_cp)

    def setup_eval_inputs_cp_and_generated_samples(self, dynamic_inputs, eval_inputs_cp):
        """
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
            eval_inputs_cp (dataframe): the variables and possible values for the sample
        """
        # 1. Manage update status of EVAL_INPUTS_CP
        # if not (eval_inputs_cp.equals(self.previous_eval_inputs_cp)):
        if not dict_are_equal(eval_inputs_cp, self.previous_eval_inputs_cp):
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
            # Setup GENERATED_SAMPLES for cartesian product
            if self.sampling_generation_mode == self.AT_CONFIGURATION_TIME:
                self.setup_generated_samples_for_cp(dynamic_inputs)

    def get_eval_inputs_cp_for_gs(self, eval_inputs, design_space):
        """
        Method that modifies Doe-type eval_inputs into eval_inputs_cp to use CartesianProduct for GridSearch.

        Arguments:
            eval_inputs(dataframe): Doe-like eval_inputs.
            design_space(dataframe): GridSearch design space with nb_points.
        Returns:
            eval_inputs_cp(dataframe): with extra column with the values for CartesianProduct SampleGenerator.
        """
        if eval_inputs is not None and design_space is not None:
            lists_of_values = []
            for idx, var_row in eval_inputs.iterrows():
                if var_row['selected_input'] is True and var_row['full_name'] in design_space['variable'].tolist():
                    dspace_row = design_space[design_space['variable'] == var_row['full_name']].iloc[0]
                    lb = dspace_row['lower_bnd']
                    ub = dspace_row['upper_bnd']
                    nb_points = dspace_row['nb_points']
                    lists_of_values.append(np.linspace(lb, ub, nb_points).tolist())
                else:
                    lists_of_values.append([])

            eval_inputs_cp = eval_inputs.assign(list_of_values=lists_of_values)
            return eval_inputs_cp

    def setup_dynamic_inputs_which_depend_on_eval_input_cp(self, dynamic_inputs):
        """
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        # TODO : why is it more complex as in doe_algo ?
        self.eval_inputs_cp_has_changed = False
        disc_in = self.get_data_in()
        if self.EVAL_INPUTS_CP in disc_in:
            eval_inputs_cp = self.get_sosdisc_inputs(self.EVAL_INPUTS_CP)
            self.setup_eval_inputs_cp_and_generated_samples(dynamic_inputs, eval_inputs_cp)

    def setup_generated_samples_for_cp(self, dynamic_inputs):
        """
        Method that setup GENERATED_SAMPLES for cartesian product at configuration time
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        generated_samples_data_description = {self.TYPE: 'dataframe',
                                              self.DATAFRAME_EDITION_LOCKED: True,
                                              self.DATAFRAME_DESCRIPTOR: {},
                                              self.DYNAMIC_DATAFRAME_COLUMNS: True,
                                              self.STRUCTURING: True,
                                              self.UNIT: None,
                                              self.VISIBILITY: self.SHARED_VISIBILITY,
                                              self.NAMESPACE: self.NS_SAMPLING}
        if self.eval_inputs_cp_validity:
            if self.eval_inputs_cp_has_changed:
                self.samples_gene_df = self.generate_sample_for_cp()
            df_descriptor = {self.SELECTED_SCENARIO: ('bool', None, False),
                             self.SCENARIO_NAME: ('string', None, False)}
            df_descriptor.update(
                {row['full_name']: (type(row['list_of_values'][0]).__name__, None, False) for index, row in
                 self.eval_inputs_cp_filtered.iterrows()})
            generated_samples_data_description.update({self.DATAFRAME_DESCRIPTOR: df_descriptor,
                                                       self.DYNAMIC_DATAFRAME_COLUMNS: False})
        else:
            # if self.eval_inputs_cp_has_changed:
            self.samples_gene_df = pd.DataFrame()

        self.samples_gene_df = self.set_scenario_columns(self.samples_gene_df)
        generated_samples_data_description.update({self.DEFAULT: self.samples_gene_df})
        dynamic_inputs.update({self.GENERATED_SAMPLES: generated_samples_data_description})

        # Set or update GENERATED_SAMPLES in line with selected
        # eval_inputs_cp
        disc_in = self.get_data_in()
        if self.GENERATED_SAMPLES in disc_in and self.samples_gene_df is not None:
            disc_in[self.GENERATED_SAMPLES][self.VALUE] = self.samples_gene_df

    def generate_sample_for_cp(self):
        """
        Outputs:
            samples_gene_df (dataframe) : prepared samples for evaluation
        """
        dict_of_list_values = self.eval_inputs_cp_filtered.set_index(
            'full_name').T.to_dict('records')[0]
        samples_gene_df = self.sample_generator.generate_samples(
            dict_of_list_values)
        return samples_gene_df

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
        is_valid = True
        selected_inputs_cp = list(eval_inputs_cp_filtered['full_name'])
        # n_min = 2
        n_min = 1
        if len(selected_inputs_cp) < n_min:
            self.logger.warning(
                f'Selected_inputs must have at least {n_min} variables to do a cartesian product')
            is_valid = False
        return is_valid

    def run_cp(self):
        """
        Method that generates the desc_out self.SAMPLES_DF from desc_in self.GENERATED_SAMPLES
        Here no modification of the samples: but we can imagine that we may remove raws.
        Maybe we do not need a run method here.

        Inputs:
            self.GENERATED_SAMPLES (dataframe) : generated samples from sampling method
        Outputs:
            self.SAMPLES_DF (dataframe) : prepared samples for evaluation
        """

        if self.sampling_generation_mode == self.AT_CONFIGURATION_TIME:
            generated_samples = self.get_sosdisc_inputs(self.GENERATED_SAMPLES)
            samples_df = generated_samples
        elif self.sampling_generation_mode == self.AT_RUN_TIME:
            if self.eval_inputs_cp_validity:
                if self.eval_inputs_cp_has_changed:
                    samples_df = self.generate_sample_for_cp()
        return samples_df
