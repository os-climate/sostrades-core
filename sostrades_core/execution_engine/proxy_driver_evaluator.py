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
        self.eval_process_builder = self._set_eval_process_builder()

    def _set_eval_process_builder(self):
        return self.cls_builder #TODO : to include mono instance case within this class not daughter etc.

    # MONO INSTANCE STUFF
    def mono_instance_build(self):
        '''
        Method copied from SoSCoupling: build and store disciplines in sos_disciplines
        '''
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