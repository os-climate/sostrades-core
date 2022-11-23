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
import copy
import re

import platform
from tqdm import tqdm
import time

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import CartesianProductSampleGenerator
import pandas as pd
from collections import ChainMap
from gemseo.api import get_available_doe_algorithms

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)


class CartesianProductWrapper(SoSWrapp):
    '''
    Cartesian Product class
    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SAMPLING_METHOD (namespace: 'ns_cp',structuring)
                |_ EVAL_INPUTS (namespace: 'ns_doe1', structuring, dynamic : SAMPLING_METHOD ==self.DOE_ALGO)             
                |_ SAMPLING_ALGO (structuring, dynamic : SAMPLING_METHOD ==self.DOE_ALGO)
                        |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO!=None) NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty")
                        |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
                |_ EVAL_INPUTS_CP (namespace: 'ns_cp', structuring, dynamic : SAMPLING_METHOD ==self.CARTESIAN_PRODUCT)
                        |_ GENERATED_SAMPLES(namespace: 'ns_cp', structuring, dynamic: EVAL_INPUTS_CP != None)                 
        |_ DESC_OUT
            |_ SAMPLES_DF (namespace: 'ns_cp') 
    '''

    # Design space dataframe headers
    VARIABLES = "variable"
    VALUES = "value"
    TYPE = "type"
    POSSIBLE_VALUES = 'possible_values'

    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    DOE_ALGO = 'doe_algo'
    CARTESIAN_PRODUCT = 'cartesian_product'
    SAMPLING_METHOD = 'sampling_method'
    EVAL_INPUTS_CP = 'eval_inputs_cp'
    GENERATED_SAMPLES = 'generated_samples'
    SAMPLES_DF = 'samples_df'

    # INPUT_MULTIPLIER_TYPE = []

    available_sampling_methods = [DOE_ALGO, CARTESIAN_PRODUCT]

    DESC_IN = {SAMPLING_METHOD: {'type': 'string', 'structuring': True,
                                 'possible_values': available_sampling_methods, 'namespace': 'ns_cp'}}

    DESC_OUT = {
        SAMPLES_DF: {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                     'namespace': 'ns_cp'}
    }

    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.generator_name = ""
        self.sample_generator_doe = None
        self.sample_generator_cp = None
        self.selected_inputs = None
        self.previous_eval_inputs = None
        self.previous_eval_inputs_cp = None
        self.eval_inputs_cp_filtered = None
        self.eval_inputs_cp_validity = True
        self.samples_gene_df = None

    def setup_sos_disciplines(self, proxy):
        '''
        Method that defines the specific dynamic updates depending on user(s selections
        Arguments:
            proxy :  
        '''
        dynamic_inputs = {}
        dynamic_outputs = {}
        disc_in = proxy.get_data_in()

        if len(disc_in) != 0:
            sampling_method = proxy.get_sosdisc_inputs(self.SAMPLING_METHOD)
            if sampling_method == self.DOE_ALGO:
                self.setup_doe_algo_method(proxy, dynamic_inputs)
            elif sampling_method == self.CARTESIAN_PRODUCT:
                self.setup_cp_method(proxy, dynamic_inputs)

        proxy.add_inputs(dynamic_inputs)
        proxy.add_outputs(dynamic_outputs)

    def setup_doe_algo_method(self, proxy, dynamic_inputs):
        '''
        Method that setup the doe algo method

        '''
        # Reset parameters of the other method to initial values
        self.previous_eval_inputs_cp = None
        # Define the selected method name as attribute and instantiate
        # the sample generator
        self.generator_name = 'doe_generator'
        if self.sample_generator_doe == None:
            self.sample_generator_doe = DoeSampleGenerator()

    def setup_cp_method(self, proxy, dynamic_inputs):
        '''
        Method that setup the cp method

        '''
        # Reset parameters of the other method to initial values
        self.previous_eval_inputs = None
        # Define the selected method name as attribute and instantiate
        # the sample generator
        self.generator_name = 'cp_generator'
        if self.sample_generator_cp == None:
            self.sample_generator_cp = CartesianProductSampleGenerator()
        # Setup dynamic inputs for cp_generator method: i.e. EVAL_INPUTS_CP
        self.setup_dynamic_inputs_for_cp_generator_method(dynamic_inputs)
        # Setup dynamic inputs which depend on EVAL_INPUTS_CP setting or
        # update: i.e. GENERATED_SAMPLES
        self.setup_dynamic_inputs_which_depend_on_eval_input_cp(
            proxy, dynamic_inputs)

    def setup_dynamic_inputs_for_cp_generator_method(self, dynamic_inputs):
        '''
        Method that setup dynamic inputs in case of cp_generator selection: i.e. EVAL_INPUTS_CP

        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        '''
        default_in_eval_input_cp = pd.DataFrame({'selected_input': [False],
                                                 'full_name': [''],
                                                 'list_of_values': [[]]})
        dynamic_inputs.update({self.EVAL_INPUTS_CP: {'type': 'dataframe',
                                                     'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                              'full_name': ('string', None, True),
                                                                              'list_of_values': ('list', None, True)},
                                                     'dataframe_edition_locked': False,
                                                     'structuring': True,
                                                     'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                     'namespace': 'ns_cp',
                                                     'default': default_in_eval_input_cp}})

    def setup_dynamic_inputs_which_depend_on_eval_input_cp(self, proxy, dynamic_inputs):
        '''
        Method that setup dynamic inputs which depend on EVAL_INPUTS_CP setting or update: i.e. GENERATED_SAMPLES
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        '''
        self.eval_inputs_cp_has_changed = False
        disc_in = proxy.get_data_in()
        if self.EVAL_INPUTS_CP in disc_in and self.generator_name == 'cp_generator':
            eval_inputs_cp = proxy.get_sosdisc_inputs(self.EVAL_INPUTS_CP)
            # 1. Manage update status of EVAL_INPUTS_CP
            if not (eval_inputs_cp.equals(self.previous_eval_inputs_cp)):
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
                self.setup_generated_samples_for_cp(proxy, dynamic_inputs)

    def setup_generated_samples_for_cp(self, proxy, dynamic_inputs):
        '''
        Method that setup GENERATED_SAMPLES for cartesian product
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        '''
        if self.eval_inputs_cp_validity:
            if self.eval_inputs_cp_has_changed:
                dict_of_list_values = self.eval_inputs_cp_filtered.set_index(
                    'full_name').T.to_dict('records')[0]
                self.samples_gene_df = self.sample_generator_cp.generate_samples(
                    dict_of_list_values)
            dynamic_inputs.update({self.GENERATED_SAMPLES: {'type': 'dataframe',
                                                            'dataframe_edition_locked': True,
                                                            'structuring': True,
                                                            'unit': None,
                                                            'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                            'namespace': 'ns_cp',
                                                            'default': self.samples_gene_df}})

        # Set or update GENERATED_SAMPLES in line with selected
        # eval_inputs_cp
        disc_in = proxy.get_data_in()
        if self.GENERATED_SAMPLES in disc_in:
            disc_in[self.GENERATED_SAMPLES]['value'] = self.samples_gene_df

    def reformat_eval_inputs_cp(self, eval_inputs_cp):
        '''
        Method that reformat eval_input_cp depending on user's selection

        Arguments:
            eval_inputs_cp (dataframe): 

        Returns:
            eval_inputs_cp_filtered (dataframe) :

        '''
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
        '''
        Method that reformat eval_input_cp depending on user's selection

        Arguments:
            eval_inputs_cp (dataframe): 

        Returns:
            validity (boolean) :

        '''
        is_valid = True
        selected_inputs_cp = list(eval_inputs_cp_filtered['full_name'])
        if len(selected_inputs_cp) < 2:
            LOGGER.warning(
                'Selected_inputs must have at least 2 variables to do a cartesian product')
            is_valid = False
        return is_valid

    def run(self):
        '''
        Method that generates the desc_out self.SAMPLES_DF from desc_in self.GENERATED_SAMPLES
        Here no modification of the samples: but we can imagine that we may remove raws.
        Maybe we do not need a run method here.

        Inputs:
            self.GENERATED_SAMPLES (dataframe) : generated samples from sampling method
        Outputs:
            self.SAMPLES_DF (dataframe) : prepared samples for evaluation
        '''

        GENERATED_SAMPLES = self.get_sosdisc_inputs(self.GENERATED_SAMPLES)

        samples_df = GENERATED_SAMPLES

        # prepared_samples = self.sample_generator_cp.prepare_samples_for_evaluation(
        #     samples, eval_in_list, design_space)

        self.store_sos_outputs_values({self.SAMPLES_DF: samples_df})
