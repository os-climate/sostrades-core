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

    def _set_scatter_process_builder(self, map_name):
        # builder of the composition scatter
        scatter_builder = self.ee.factory.create_scatter_builder('scatter_temp', map_name, self.cls_builder)
        scatter_builder.set_builder_info('coupling_per_scatter', True) #TODO: is hardcoded also in VerySimpleMS/SimpleMS
        return scatter_builder

    def multi_instance_build(self): #TODO: factorize better
        if 'map_name' in self.get_data_in():
            old_current_discipline = self.ee.factory.current_discipline
            self.ee.factory.current_discipline = self
            if self.scatter_process_builder is None:
                map_name = self.get_sosdisc_inputs('map_name')
                if map_name is not None:
                    self.scatter_process_builder = self._set_scatter_process_builder(map_name)
            self.build_scatter_process()
            # If the old_current_discipline is None that means that it is the first build of a coupling then self is the high
            # level coupling and we do not have to restore the current_discipline
            if old_current_discipline is not None:
                self.ee.factory.current_discipline = old_current_discipline

    def build_scatter_process(self): #TODO: factorize better
        if self.scatter_process_builder is not None:
            subprocess_disc = self.scatter_process_builder.build()
            # store coupling in the children
            if subprocess_disc not in self.proxy_disciplines:
                self.ee.factory.add_discipline(subprocess_disc)
        else:
            pass #TODO: else add a warning ?

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

    # MONO INSTANCE STUFF
    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary
        '''
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        elif len(self.cls_builder) == 1:
            disc_builder = self.cls_builder[0]
        else:
            # If eval process is a list of builders or a non executable builder,
            # then we build a coupling containing the eval process

            disc_builder = self.ee.factory.create_builder_coupling(
                self.sos_name+'.subprocess')
            disc_builder.set_builder_info('cls_builder', self.cls_builder)

        return disc_builder

    def mono_instance_build(self):
        '''
        Method copied from SoSCoupling: build and store disciplines in sos_disciplines
        '''
        if self.eval_process_builder is None:
            self.eval_process_builder = self._set_eval_process_builder()
        # set current_discipline to self to build and store eval process in the
        # children of SoSEval
        old_current_discipline = self.ee.factory.current_discipline
        self.ee.factory.current_discipline = self

        # if we want to build an eval coupling containing eval process,
        # we have to remove SoSEval name in current_ns to build eval coupling
        # at the same node as SoSEval
        if len(self.cls_builder) == 0:  # added condition for proc build
            pass
        elif self.cls_builder[0] != self.eval_process_builder:
            current_ns = self.ee.ns_manager.current_disc_ns
            self.ee.ns_manager.set_current_disc_ns(
                current_ns.split(f'.{self.sos_name}')[0])
            self.build_eval_process()
            # reset current_ns after build
            self.ee.ns_manager.set_current_disc_ns(current_ns)
        else:
            self.build_eval_process()

        # If the old_current_discipline is None that means that it is the first build of a coupling then self is the high
        # level coupling and we do not have to restore the current_discipline
        if old_current_discipline is not None:
            self.ee.factory.current_discipline = old_current_discipline

    def build_eval_process(self):
        # build coupling containing eval process if self.cls_builder[0] != self.eval_process_builder
        # or build and store eval process in the children of SoSEval
        eval_process_disc = self.eval_process_builder.build()
        # store coupling in the children of SoSEval
        if eval_process_disc not in self.proxy_disciplines:
            self.ee.factory.add_discipline(eval_process_disc)

    def build(self):
        # TODO: better factorization
        if 'builder_mode' in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs('builder_mode')
            if builder_mode == 'multi_instance':
                self.multi_instance_build()
            elif builder_mode == 'mono_instance':
                self.mono_instance_build()
            elif builder_mode == 'custom':
                super().build()
            else:
                raise ValueError(f'Wrong builder mode input in {self.sos_name}')