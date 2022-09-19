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
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline

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

    def configure(self):
        '''
        Configure the SoSEval and its children sos_disciplines + set eval possible values for the GUI
        '''
        # configure eval process stored in children
        for disc in self.get_disciplines_to_configure():
            disc.configure()

        if self._data_in == {} or (self.get_disciplines_to_configure() == [] and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0:
            # Explanation:
            # 1. self._data_in == {} : if the discipline as no input key it should have and so need to be configured
            # 2. Added condition compared to SoSDiscipline(as sub_discipline or associated sub_process builder)
            # 2.1 (self.get_disciplines_to_configure() == [] and len(self.proxy_disciplines) != 0) : sub_discipline(s) exist(s) but all configured
            # 2.2 len(self.cls_builder) == 0 No yet provided builder but we however need to configure (as in 2.1 when we have sub_disciplines which no need to be configured)
            # Remark: condition "(   and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0" added for proc build
            #
            # Call standard configure methods to set the process discipline
            # tree
            ProxyDiscipline.configure(self)
            self.configure_driver()

        if len(self.get_disciplines_to_configure()) == 0:
            if len(self.proxy_disciplines) == 1 and self.proxy_disciplines[0].is_sos_coupling:
                self.update_data_io_with_subprocess_io() # only for 1 subcoupling, so not handling cases like driver of driver
            else:
                raise NotImplementedError
            self.set_children_cache_inputs()

    def update_data_io_with_subprocess_io(self):
        # FIXME: working with short names is problematic for driver of driver example bi-level optimization
        # only for 1 subcoupling
        # self._data_in_with_full_name = {f'{self.get_disc_full_name()}.{key}': value for key, value in
        #                         self._data_in.items()
        #                         if key in self.DESC_IN or key in self.NUM_DESC_IN}
        # self._data_out_with_full_name = {f'{self.get_disc_full_name()}.{key}': value for key, value in
        #                         self._data_out.items()}
        # self._data_in_with_full_name.update(self.proxy_disciplines[0].get_data_io_with_full_name(self.IO_TYPE_IN)) # the subcoupling num_desc_in is crushed
        # self._data_out_with_full_name.update(self.proxy_disciplines[0].get_data_io_with_full_name(self.IO_TYPE_OUT))
        #
        self._data_in.update({key:value
                              for key,value in self.proxy_disciplines[0].get_data_in().items()
                              if key not in self.NUM_DESC_IN.keys()}) # the subcoupling num_desc_in is crushed
        self._data_out.update(self.proxy_disciplines[0].get_data_out())


    def configure_driver(self):
        """
        To be overload by drivers with specific configuration actions
        """

    # def reload_io(self):
    #     '''
    #     Create the data_in and data_out of the discipline with the DESC_IN/DESC_OUT, inst_desc_in/inst_desc_out
    #     and initialize GEMS grammar with it (with a filter for specific variable types)
    #     '''
    #     ProxyDiscipline.reload_io(self)


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

    # def get_input_data_names(self):
    #     '''
    #     Returns:
    #         (List[string]) of input data full names based on i/o and namespaces declarations in the user wrapper
    #     '''
    #     return list(self._data_in_with_full_name.keys())
    #
    # def get_output_data_names(self):
    #     '''
    #     Returns:
    #         (List[string]) outpput data full names based on i/o and namespaces declarations in the user wrapper
    #     '''
    #     return list(self._data_out_with_full_name.keys())