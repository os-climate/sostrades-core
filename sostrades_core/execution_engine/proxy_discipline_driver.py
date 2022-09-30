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
    """
    **ProxyDiscipline** is a proxy class for a  discipline driver on the SoSTrades side...

    Attributes:

    """
    def __init__(self, sos_name, ee, cls_builder, driver_wrapper_cls=None, associated_namespaces=[]):
        '''
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (List[SoSBuilder]): list of the sub proxy builders
            driver_wrapper_cls (Class): class constructor of the driver wrapper (user-defined wrapper or SoSTrades wrapper or None)
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
        '''
        super().__init__(sos_name, ee, driver_wrapper_cls, associated_namespaces=associated_namespaces)
        self.cls_builder = cls_builder #TODO: Move to ProxyDisciplineBuilder?

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
            if len(self.proxy_disciplines) == 1:
                self.update_data_io_with_subprocess_io() # only for 1 subcoupling, so not handling cases like driver of driver
            else:
                raise NotImplementedError
            self.set_children_cache_inputs()

    def get_data_io_with_full_name(self, io_type, as_namespaced_tuple=False):
        # TODO: code is duplicated in driver and coupling... move to builder ? [discuss]
        if io_type == self.IO_TYPE_IN:
            if as_namespaced_tuple:
                return self._data_in_ns_tuple
            else:
                return self.namespaced_tuples_to_full_names(self._data_in_ns_tuple)
        elif io_type == self.IO_TYPE_OUT:
            if as_namespaced_tuple:
                return self._data_out_ns_tuple
            else:
                return self.namespaced_tuples_to_full_names(self._data_out_ns_tuple)
        else:
            raise ValueError('Unknown io type')

    def update_data_io_with_subprocess_io(self):

        # - data_i/o setup
        # self._data_in_with_full_name = dict(zip(self._convert_list_of_keys_to_namespace_name(list(self._data_in.keys()), self.IO_TYPE_IN), self._data_in.values()))
        # self._data_out_with_full_name = dict(zip(self._convert_list_of_keys_to_namespace_name(list(self._data_out.keys()), self.IO_TYPE_OUT), self._data_out.values()))

        self._data_in_ns_tuple = {(key, id(value[self.NS_REFERENCE])) : value for key, value in self._data_in.items()}
        self._data_out_ns_tuple = {(key, id(value[self.NS_REFERENCE])) : value for key, value in self._data_out.items()}

        for proxy_disc in self.proxy_disciplines:
            self._data_in_ns_tuple.update(proxy_disc.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple=True))
            self._data_out_ns_tuple.update(proxy_disc.get_data_io_with_full_name(self.IO_TYPE_OUT, as_namespaced_tuple=True))

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
        # TODO : cache mgmt of children necessary ? here or in SoSMDODisciplineDriver ?
        super().prepare_execution()


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

    def set_wrapper_attributes(self, wrapper):
        super().set_wrapper_attributes(wrapper)
        wrapper.attributes.update({'sub_mdo_disciplines': [proxy.mdo_discipline_wrapp.mdo_discipline for proxy in self.proxy_disciplines]})

    def is_configured(self):
        '''
        Return False if discipline is not configured or structuring variables have changed or children are not all configured
        '''
        return ProxyDiscipline.is_configured(self) and (
            (self.get_disciplines_to_configure() == [] and len(self.proxy_disciplines) != 0) or len(
                self.cls_builder) == 0)
        # Explanation:
        # 1. SoSDiscipline.is_configured(self) : as in discipline (i.e. conf status of Eval = True and no change in structuring variables of Eval
        # 2. Added condition compared to ProxSDiscipline :
        # 2.1 (self.get_disciplines_to_configure() == [] and len(self.proxy_disciplines) != 0) : sub_discipline exist but all configured
        # 2.2 len(self.cls_builder) == 0 No yet provided builder and so Is configured is True
        # Remark: condition "and len(self.proxy_disciplines) != 0) or
        # len(self.cls_builder) == 0)" added for proc build

    def get_desc_in_out(self, io_type):
        """
        get the desc_in or desc_out. if a wrapper exists get it from the wrapper, otherwise get it from the proxy class
        """
        if self.mdo_discipline_wrapp.wrapper is not None:
            return ProxyDiscipline.get_desc_in_out(self, io_type) #ProxyDiscipline gets the DESC from the wrapper
        else:
            return super().get_desc_in_out(io_type) #ProxyDisciplineBuilder expects the DESC on the proxies