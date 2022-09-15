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

from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.mdo_discipline_driver_wrapp import MDODisciplineDriverWrapp

class ProxyDisciplineDriverException(Exception):
    pass


class ProxyDisciplineDriver(ProxyDisciplineBuilder):

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''
        super().__init__(sos_name, ee)
        self.cls_builder = cls_builder
        self.eval_process_builder = self._set_eval_process_builder()

    def create_mdo_discipline_wrap(self,name, wrapper, wrapping_mode):
        """
        creation of mdo_discipline_wrapp by the proxy
        To be overloaded by proxy without MDODisciplineWrapp (eg scatter...)
        """
        self.mdo_discipline_wrapp = MDODisciplineDriverWrapp(name, wrapper, wrapping_mode)

    def prepare_execution(self):
        '''
        Preparation of the GEMSEO process, including GEMSEO objects instanciation
        '''
        # prepare_execution of proxy_disciplines as in coupling
        # TODO: move to builder ?

        for disc in self.proxy_disciplines:
            disc.prepare_execution()
        # FIXME : cache mgmt of children necessary ? here or in SoSMDODisciplineDriver ?
        super().prepare_execution()

    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary
        '''
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        # elif len(self.cls_builder) > 1 or not self.cls_builder[0]._is_executable:
        else:
            # if eval process is a list of builders or a non executable builder,
            # then we build a coupling containing the eval process
            # In the case of a single sub-disc for sos_eval, although len(self.cls_builder) = 1 and it is an
            # executable discipline, a coupling is also wanted to contain the eval process: TODO this method only used in SoSEval???
            disc_builder = self.ee.factory.create_builder_coupling(
                self.sos_name)
            disc_builder.set_builder_info('cls_builder', self.cls_builder)

        return disc_builder
