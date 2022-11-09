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
    Generic DOE evaluation class
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
    SCENARIO_SELECTION = 'scenario_selection'
    # INPUT_MULTIPLIER_TYPE = []

    available_sampling_methods = [DOE_ALGO, CARTESIAN_PRODUCT]

    DESC_IN = {SAMPLING_METHOD: {'type': 'string', 'structuring': True, 'possible_values': available_sampling_methods, 'namespace': 'ns_cp'},
               'eval_inputs': {'type': 'dataframe',
                               'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                        'full_name': ('string', None, False)},
                               'dataframe_edition_locked': False,
                               'structuring': True,
                               'visibility': SoSWrapp.SHARED_VISIBILITY,
                               'namespace': 'ns_cp'},
               SCENARIO_SELECTION: {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                    'namespace': 'ns_cp'}
               }

    DESC_OUT = {
        'samples_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                       'namespace': 'ns_cp'}
    }

    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.sample_generator = None
        self.previous_sampling_method = ""
        self.selected_inputs = None
        self.eval_in_list = None
        self.selected_inputs = None
        self.dict_desactivated_elem = {}

    def setup_sos_disciplines(self, proxy):

        dynamic_inputs = {}
        dynamic_outputs = {}
        selected_inputs_has_changed = False
        disc_in = proxy.get_data_in()

        if len(disc_in) != 0:
            # Dynamic input of ...
            sampling_method_has_changed = False
            sampling_method = proxy.get_sosdisc_inputs(self.SAMPLING_METHOD)
            if self.previous_sampling_method != sampling_method:
                sampling_method_has_changed = True
                self.previous_sampling_method = sampling_method

            dynamic_inputs = {}
            dynamic_inputs.update({'eval_inputs': {'type': 'dataframe',
                                                   'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                            'full_name': ('string', None, False)},
                                                   'dataframe_edition_locked': False,
                                                   'structuring': True,
                                                   'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                   'namespace': 'ns_cp'}})
            dynamic_inputs.update({self.SCENARIO_SELECTION: {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                             'namespace': 'ns_cp'}})

            # if selected_inputs_has_changed:

        #generator_name = 'cp_generator'
        # if self.sample_generator == None:
        #    if generator_name == 'cp_generator':
        #        self.sample_generator = CartesianProductSampleGenerator()
        #samples = self.sample_generator.generate_samples(dict_of_list_values)

        proxy.add_inputs(dynamic_inputs)
        proxy.add_outputs(dynamic_outputs)

    def run(self):

        scenario_selection = self.get_sosdisc_inputs(self.SCENARIO_SELECTION)

        samples_df = scenario_selection

        # prepared_samples = self.sample_generator.prepare_samples_for_evaluation(
        #     samples, eval_in_list, design_space)

        self.store_sos_outputs_values({'samples_df': samples_df})
