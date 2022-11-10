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
from sostrades_core.execution_engine.disciplines_wrappers.eval_wrapper import EvalWrapper

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
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

    # INPUT_MULTIPLIER_TYPE = []

    available_sampling_methods = [DOE_ALGO, CARTESIAN_PRODUCT]

    DESC_IN = {SAMPLING_METHOD: {'type': 'string', 'structuring': True,
                                 'possible_values': available_sampling_methods, 'namespace': 'ns_cp'}}

    DESC_OUT = {
        'samples_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                       'namespace': 'ns_cp'}
    }

    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.sample_generator = None
        self.selected_inputs = None
        self.eval_in_list = None
        self.selected_inputs = None
        self.previous_eval_inputs_cp = None
        self.dict_desactivated_elem = {}
        self.samples_gene_df = None

    def setup_sos_disciplines(self, proxy):

        dynamic_inputs = {}
        dynamic_outputs = {}
        eval_inputs_cp_has_changed = False
        disc_in = proxy.get_data_in()

        default_in_eval_input_cp = pd.DataFrame({'selected_input': [False],
                                                 'full_name': [''],
                                                 'list_of_values': [[]]})

        if len(disc_in) != 0:
            sampling_method = proxy.get_sosdisc_inputs(self.SAMPLING_METHOD)
            if sampling_method == self.DOE_ALGO:
                self.previous_eval_inputs_cp = None
                self.sample_generator = None
            elif sampling_method == self.CARTESIAN_PRODUCT:
                generator_name = 'cp_generator'
                if self.sample_generator == None:
                    if generator_name == 'cp_generator':
                        self.sample_generator = CartesianProductSampleGenerator()
                dynamic_inputs.update({self.EVAL_INPUTS_CP: {'type': 'dataframe',
                                                             'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                                      'full_name': ('string', None, True),
                                                                                      'list_of_values': ('list', None, True)},
                                                             'dataframe_edition_locked': False,
                                                             'structuring': True,
                                                             'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                             'namespace': 'ns_cp',
                                                             'default': default_in_eval_input_cp}})

            # Dynamic set or update of GENERATED_SAMPLES for cartesian product
            # method
            eval_inputs_cp_has_changed = False
            if self.EVAL_INPUTS_CP in disc_in:
                eval_inputs_cp = proxy.get_sosdisc_inputs(self.EVAL_INPUTS_CP)
                # 1. Manage update status of EVAL_INPUTS_CP
                if not (eval_inputs_cp.equals(self.previous_eval_inputs_cp)):
                    eval_inputs_cp_has_changed = True
                    self.previous_eval_inputs_cp = eval_inputs_cp
                # 2. Manage empty selection in EVAL_INPUTS_CP
                if eval_inputs_cp is not None:
                    logic_1 = eval_inputs_cp['selected_input'] == True
                    logic_2 = eval_inputs_cp['list_of_values'].isin([[]])
                    eval_inputs_cp_filtered = eval_inputs_cp[logic_1 & ~logic_2]
                    eval_inputs_cp_filtered = eval_inputs_cp_filtered[[
                        'full_name', 'list_of_values']]
                    selected_inputs_cp = list(
                        eval_inputs_cp_filtered['full_name'])
                    if len(selected_inputs_cp) < 2:
                        LOGGER.warning(
                            'Selected_inputs must have at least 2 variables to do a cartesian product')
                    else:
                        if eval_inputs_cp_has_changed:
                            dict_of_list_values = eval_inputs_cp_filtered.set_index(
                                'full_name').T.to_dict('records')[0]
                            self.samples_gene_df = self.sample_generator.generate_samples(
                                dict_of_list_values)
                        dynamic_inputs.update({self.GENERATED_SAMPLES: {'type': 'dataframe', 'structuring': True, 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                                'namespace': 'ns_cp', 'value': self.samples_gene_df}})
                    # 3. Set or update GENERATED_SAMPLES in line with selected
                    # eval_inputs_cp
                    # if self.GENERATED_SAMPLES in disc_in:
                    #    proxy.dm.set_data(
                    #        f'{proxy.get_disc_full_name()}.{self.GENERATED_SAMPLES}',
                    #        self.VALUE, samples_gene_df)

        proxy.add_inputs(dynamic_inputs)
        proxy.add_outputs(dynamic_outputs)

    def run(self):

        GENERATED_SAMPLES = self.get_sosdisc_inputs(self.GENERATED_SAMPLES)

        samples_df = GENERATED_SAMPLES

        # prepared_samples = self.sample_generator.prepare_samples_for_evaluation(
        #     samples, eval_in_list, design_space)

        self.store_sos_outputs_values({'samples_df': samples_df})
