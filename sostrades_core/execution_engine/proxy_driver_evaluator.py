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
import platform
import pandas as pd
import re

from tqdm import tqdm
import time

from gemseo.core.parallel_execution import ParallelExecution
from sostrades_core.tools.base_functions.compute_len import compute_len

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import numpy as np
from pandas.core.frame import DataFrame

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.ns_manager import NS_SEP
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_driver import ProxyDisciplineDriver


class ProxyDriverEvaluatorException(Exception):
    pass


class ProxyDriverEvaluator(ProxyDisciplineDriver):
    '''
        SOSEval class which creates a sub process to evaluate
        with different methods (Gradient,FORM,Sensitivity ANalysis, DOE, ...)
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Eval Model',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    def __init__(self, sos_name, ee, cls_builder, driver_wrapper_cls, associated_namespaces=None):
        super().__init__(sos_name, ee, cls_builder, driver_wrapper_cls,
                         associated_namespaces=associated_namespaces)
        self.eval_process_builder = None
        self.scatter_process_builder = None

    def setup_sos_disciplines(self):
        if 'builder_mode' in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs('builder_mode')
            if builder_mode == 'multi_instance':
                # TODO: addressing only the very simple multiscenario case
                if 'map_name' not in self.get_data_in():
                    dynamic_inputs = {'map_name': {self.TYPE: 'string',
                                                   self.DEFAULT: 'scenario_list',
                                                   self.STRUCTURING: True}}
                    self.add_inputs(dynamic_inputs)
            elif builder_mode == 'mono_instance':
                pass #TODO: to merge with Eval

    def build(self):
        if len(self.cls_builder) == 0: # added condition for proc build
            pass
        elif 'builder_mode' in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs('builder_mode')
            if builder_mode == 'multi_instance':
                self.multi_instance_build()
            elif builder_mode == 'mono_instance':
                self.mono_instance_build()
            elif builder_mode == 'custom':
                super().build()
            else:
                raise ValueError(f'Wrong builder mode input in {self.sos_name}')

    # MULTI INSTANCE PROCESS
    def _set_scatter_process_builder(self, map_name):
        if len(self.cls_builder) == 0:  # added condition for proc build
            scatter_builder = None
        else:
            # builder of the scatter in aggregation
            scatter_builder = self.ee.factory.create_scatter_builder('scatter_temp', map_name, self.cls_builder, # TODO: nice to remove scatter node
                                                                     coupling_per_scatter=True) #NB: is hardcoded also in VerySimpleMS/SimpleMS
        self.scatter_process_builder = scatter_builder

    def multi_instance_build(self):
        # TODO: will need to include options for MultiScenario other than VerySimple
        if 'map_name' in self.get_data_in():
            if self.scatter_process_builder is None:
                map_name = self.get_sosdisc_inputs('map_name')
                if map_name is not None:
                    self._set_scatter_process_builder(map_name)
            if self.scatter_process_builder is not None:
                super()._custom_build([self.scatter_process_builder])
            else:
                self.logger.warn(f'Scatter builder not configured in {self.sos_name}, map_name missing?')

    # MONO INSTANCE PROCESS
    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary
        '''
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        elif len(self.cls_builder) == 1:
            # Note no distinction is made whether the builder is executable or not; old implementation used to put
            # scatter builds under a coupling automatically too. # TODO: check if necessary for gather implementation.
            disc_builder = self.cls_builder[0]
        else:
            # If eval process is a list of builders then we build a coupling containing the eval process
            disc_builder = self.ee.factory.create_builder_coupling('subprocess')
            disc_builder.set_builder_info('cls_builder', self.cls_builder)
        self.eval_process_builder = disc_builder

    def mono_instance_build(self):
        '''
        Method copied from SoSCoupling: build and store disciplines in sos_disciplines
        '''
        if self.eval_process_builder is None:
            self._set_eval_process_builder()
        super()._custom_build([self.eval_process_builder])

