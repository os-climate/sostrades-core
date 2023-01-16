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
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from gemseo.utils.compare_data_manager_tooling import dict_are_equal

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import CartesianProductSampleGenerator
import pandas as pd
import numpy as np
from collections import ChainMap
from gemseo.api import get_available_doe_algorithms

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)


class MultipliersWrapper(SoSWrapp):
    '''

    '''

    _ontology_data = {
        'label': 'Multipliers wrapper',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'Multipliers wrapper that implements the __MULTIPLIERS__ capability to be used with grid search.',
        'icon': 'fas fa-grid-4 fa-fw',
        'version': ''
    }
    #TODO: add and refer class variables
    EVAL_INPUTS = 'eval_inputs'
    EVAL_INPUTS_CP = 'eval_inputs_cp'
    DISC_SHARED_NS = 'ns_sampling'

    INPUT_MULTIPLIER_TYPE = ['dict', 'dataframe', 'float']
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'

    DESC_IN = {EVAL_INPUTS:  {SoSWrapp.TYPE: 'dataframe',
                              SoSWrapp.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
                                                              'full_name': ('string', None, False)},
                              SoSWrapp.DATAFRAME_EDITION_LOCKED: False,
                              SoSWrapp.STRUCTURING: True,
                              SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                              SoSWrapp.NAMESPACE: 'ns_sampling'}
               }

    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.previous_eval_inputs = None

    def setup_sos_disciplines(self):
        '''
        Overload of setup_sos_disciplines to specify the specific dynamic inputs of multipliers disc
        '''
        disc_in = self.get_data_in()
        dynamic_inputs = {}
        dynamic_outputs = {}

        self.add_multipliers(dynamic_inputs, dynamic_outputs, disc_in)
        self.apply_multipliers(dynamic_inputs, dynamic_outputs, disc_in)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def add_multipliers(self, dynamic_inputs, dynamic_outputs, disc_in):
        if self.EVAL_INPUTS in disc_in:
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            eval_disc = None
            eval_ns = self.get_var_full_name(self.EVAL_INPUTS, disc_in).rsplit('.'+self.EVAL_INPUTS, 1)[0]
            disc_list = self.dm.get_disciplines_with_name(eval_ns)
            if disc_list:
                eval_disc = disc_list[0]
            if eval_inputs is not None and eval_disc is not None:
                raise NotImplementedError()

        # dynamic_inputs.update({self.EVAL_INPUTS_CP: {self.TYPE: 'dataframe',
        #                                              self.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
        #                                                                          'full_name': ('string', None, True),
        #                                                                          'list_of_values': ('list', None, True)},
        #                                              self.DATAFRAME_EDITION_LOCKED: False,
        #                                              self.STRUCTURING: True,
        #                                              self.VISIBILITY: self.SHARED_VISIBILITY,
        #                                              self.NAMESPACE: 'ns_sampling',
        #                                              self.DEFAULT: default_in_eval_input_cp}})
    #
    #
    # def set_eval_possible_values(self):
    #     '''
    #         Once all disciplines have been run through,
    #         set the possible values for eval_inputs and eval_outputs in the DM
    #     '''
    #     analyzed_disc = self.proxy_disciplines[0]
    #     possible_in_values_full, possible_out_values_full = self.fill_possible_values(
    #         analyzed_disc)
    #     possible_in_values_full, possible_out_values_full = self.find_possible_values(analyzed_disc,
    #                                                                                   possible_in_values_full,
    #                                                                                   possible_out_values_full)
    #
    #     # Take only unique values in the list
    #     possible_in_values = list(set(possible_in_values_full))
    #     possible_out_values = list(set(possible_out_values_full))
    #
    #     # these sorts are just for aesthetics
    #     possible_in_values.sort()
    #     possible_out_values.sort()
    #
    #     default_in_dataframe = pd.DataFrame({'selected_input': [False for _ in possible_in_values],
    #                                          'full_name': possible_in_values})
    #     default_out_dataframe = pd.DataFrame({'selected_output': [False for _ in possible_out_values],
    #                                           'full_name': possible_out_values})
    #
    #     eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
    #     eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')
    #     my_ns_eval_path = self._get_disc_shared_ns_value()
    #
    #     if eval_input_new_dm is None:
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
    #                          'value', default_in_dataframe, check_value=False)
    #     # check if the eval_inputs need to be updated after a subprocess
    #     # configure
    #     elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
    #         self.check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
    #                            is_eval_input=True)
    #         default_dataframe = copy.deepcopy(default_in_dataframe)
    #         already_set_names = eval_input_new_dm['full_name'].tolist()
    #         already_set_values = eval_input_new_dm['selected_input'].tolist()
    #         for index, name in enumerate(already_set_names):
    #             default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
    #                 index]
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
    #                          'value', default_dataframe, check_value=False)
    #
    #     if eval_output_new_dm is None:
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
    #                          'value', default_out_dataframe, check_value=False)
    #     # check if the eval_inputs need to be updated after a subprocess
    #     # configure
    #     elif set(eval_output_new_dm['full_name'].tolist()) != (set(default_out_dataframe['full_name'].tolist())):
    #         self.check_eval_io(eval_output_new_dm['full_name'].tolist(), default_out_dataframe['full_name'].tolist(),
    #                            is_eval_input=False)
    #         default_dataframe = copy.deepcopy(default_out_dataframe)
    #         already_set_names = eval_output_new_dm['full_name'].tolist()
    #         already_set_values = eval_output_new_dm['selected_output'].tolist()
    #         for index, name in enumerate(already_set_names):
    #             default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_output'] = already_set_values[
    #                 index]
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
    #                          'value', default_dataframe, check_value=False)

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''
        poss_in_values_full = []
        poss_out_values_full = []
        disc_in = disc.get_data_in()
        for data_in_key in disc_in.keys():
            is_structuring = disc_in[data_in_key].get(
                self.STRUCTURING, False)
            in_coupling_numerical = data_in_key in list(
                ProxyCoupling.DESC_IN.keys())
            full_id = disc.get_var_full_name(
                data_in_key, disc_in)
            is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
                                           ]['io_type'] == 'in'
            is_input_multiplier_type = disc_in[data_in_key][self.TYPE] in self.INPUT_MULTIPLIER_TYPE
            is_editable = disc_in[data_in_key]['editable']
            is_None = disc_in[data_in_key]['value'] is None
            if is_in_type and not in_coupling_numerical and not is_structuring and is_editable:
                if is_input_multiplier_type and not is_None:
                    poss_in_values_list = self.set_multipliers_values(
                        disc, full_id, data_in_key)
                    for val in poss_in_values_list:
                        poss_in_values_full.append(val)
        return poss_in_values_full, poss_out_values_full

    # def find_possible_values(self, disc, possible_in_values, possible_out_values):
    #     return ProxyDriverEvaluator.find_possible_values(self, disc, possible_in_values, possible_out_values)

    def find_possible_values(
            self, disc, possible_in_values, possible_out_values):
        '''
            Run through all disciplines and sublevels
            to find possible values for eval_inputs and eval_outputs
        '''
        # TODO: copy-pasted code should b refactored (see above)
        # configuration ? (<-> config. graph)
        if len(disc.proxy_disciplines) != 0:
            for sub_disc in disc.proxy_disciplines:
                sub_in_values, sub_out_values = self.fill_possible_values(
                    sub_disc)
                possible_in_values.extend(sub_in_values)
                possible_out_values.extend(sub_out_values)
                self.find_possible_values(
                    sub_disc, possible_in_values, possible_out_values)
        return possible_in_values, possible_out_values