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
import numpy as np

import platform
from tqdm import tqdm
import time

from sostrades_core.tools.base_functions.compute_len import compute_len
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_new_type_into_array, convert_array_into_new_type

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
import pandas as pd
from collections import ChainMap
from gemseo.core.parallel_execution import ParallelExecution

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)

class AbstractEvalWrapper(SoSWrapp):
    '''
    Generic Wrapper with SoSEval functions
    '''

    _maturity = 'Fake'
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'
    # DESC_IN = {
    #     'eval_inputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
    #     'eval_outputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
    #     'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
    #     'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0},
    #
    # }

    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.custom_samples = None  # input samples dataframe
        self.samples = None         # samples to evaluate as list[list[Any]] or ndarray
        self.input_data_for_disc = None
        self.subprocesses_to_eval = None

    def _init_input_data(self):
        n_subprocs = len(self.attributes['sub_mdo_disciplines'])
        self.input_data_for_disc = [{}]*n_subprocs
        #TODO: deepcopy option? [discuss]
        for i_subprocess in self.subprocesses_to_eval or range(n_subprocs):
            self.input_data_for_disc[i_subprocess] = self.get_input_data_for_gems(self.attributes['sub_mdo_disciplines'][i_subprocess])

    def _get_input_data(self, delta_dict, i_subprocess=0):
        #TODO: deepcopy option? [discuss]
        self.input_data_for_disc[i_subprocess].update(delta_dict)
        return self.input_data_for_disc[i_subprocess]

    def _select_output_data(self, raw_data, eval_out_data_names):
        output_data_dict = {key: value for key,value in raw_data.items()
                          if key in eval_out_data_names}
        return output_data_dict

    def get_input_data_for_gems(self, disc):
        '''
        Get input_data for linearize sosdiscipline
        '''
        input_data = {}
        input_data_names = disc.input_grammar.get_data_names()
        if len(input_data_names) > 0:
            input_data = self.get_sosdisc_inputs(keys=input_data_names, in_dict=True, full_name_keys=True)
        return input_data


    def subproc_evaluation(self, var_delta_dict, i_subprocess, convert_to_array=True):
        local_data = self.attributes['sub_mdo_disciplines'][i_subprocess]\
                         .execute(self._get_input_data(var_delta_dict, i_subprocess))

        out_local_data = self._select_output_data(local_data, self.attributes['eval_out_list'][i_subprocess])
        if convert_to_array:
            out_local_data_converted = convert_new_type_into_array(
                out_local_data, self.attributes['reduced_dm'])
            out_values = np.concatenate(
                list(out_local_data_converted.values())).ravel()
        else:
            out_values = []
            # get back out_local_data is not enough because some variables
            # could be filtered for unsupported type for gemseo  TODO: is this case relevant??
            for y_id in self.attributes['eval_out_list'][i_subprocess]:
                y_val = out_local_data[y_id]
                out_values.append(y_val)
        return out_values

