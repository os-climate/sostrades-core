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
    DESC_IN = {
        'eval_inputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
        'eval_outputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
        'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
        'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0},

    }

    # def __init__(self, sos_name):
    #
    #     super().__init__(sos_name)
