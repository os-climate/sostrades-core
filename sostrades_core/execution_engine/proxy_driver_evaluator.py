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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.mdo_discipline_driver_wrapp import MDODisciplineDriverWrapp


class ProxyDriverEvaluatorException(Exception):
    pass


class ProxyDriverEvaluator(ProxyDisciplineBuilder):
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

    def __init__(self, sos_name, ee, cls_builder, driver_wrapper_cls=None, associated_namespaces=None):
        '''
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (List[SoSBuilder]): list of the sub proxy builders
            driver_wrapper_cls (Class): class constructor of the driver wrapper (user-defined wrapper or SoSTrades wrapper or None)
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
        '''
        super().__init__(sos_name, ee, driver_wrapper_cls,
                         associated_namespaces=associated_namespaces)
        if cls_builder is None:
            cls_builder = []
        self.cls_builder = cls_builder  # TODO: Move to ProxyDisciplineBuilder?
        self.eval_process_builder = None
        self.scatter_process_builder = None

    def get_desc_in_out(self, io_type):
        """
        get the desc_in or desc_out. if a wrapper exists get it from the wrapper, otherwise get it from the proxy class
        """
        if self.mdo_discipline_wrapp.wrapper is not None:
            # ProxyDiscipline gets the DESC from the wrapper
            return ProxyDiscipline.get_desc_in_out(self, io_type)
        else:
            # ProxyDisciplineBuilder expects the DESC on the proxies e.g. Coupling
            # TODO: move to coupling ?
            return super().get_desc_in_out(io_type)

    def create_mdo_discipline_wrap(self, name, wrapper, wrapping_mode):
        """
        creation of mdo_discipline_wrapp by the proxy
        To be overloaded by proxy without MDODisciplineWrapp (eg scatter...)
        """
        self.mdo_discipline_wrapp = MDODisciplineDriverWrapp(
            name, wrapper, wrapping_mode)

    def configure(self):
        '''
        Configure the SoSEval and its children sos_disciplines + set eval possible values for the GUI
        '''
        # configure eval process stored in children
        for disc in self.get_disciplines_to_configure():
            disc.configure()

        # if self._data_in == {} or (self.get_disciplines_to_configure() == [] and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0:
        if self._data_in == {} or self.subprocess_is_configured():
            # Call standard configure methods to set the process discipline tree
            ProxyDiscipline.configure(self)
            self.configure_driver()

        if self.subprocess_is_configured():
            self.update_data_io_with_subprocess_io()
            # if len(self.proxy_disciplines) == 1:
                # only for 1 subcoupling, so not handling cases like driver of
                # driver
                # self.update_data_io_with_subprocess_io()
            # else:
            #     raise NotImplementedError
            self.set_children_cache_inputs()

    def update_data_io_with_subprocess_io(self):
        self._restart_data_io_to_disc_io()
        #TODO: working because no two different discs share a local ns
        for proxy_disc in self.proxy_disciplines:
            # if not isinstance(proxy_disc, ProxyDisciplineGather):
            subprocess_data_in = proxy_disc.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple=True)
            subprocess_data_out = proxy_disc.get_data_io_with_full_name(self.IO_TYPE_OUT, as_namespaced_tuple=True)
            self._update_data_io(subprocess_data_in, self.IO_TYPE_IN)
            self._update_data_io(subprocess_data_out, self.IO_TYPE_OUT)

    def configure_driver(self):
        """
        To be overload by drivers with specific configuration actions
        """

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

    def set_wrapper_attributes(self, wrapper):
        #TODO: needs to accommodate the eval attributes in the mono instance case
        super().set_wrapper_attributes(wrapper)
        wrapper.attributes.update({'sub_mdo_disciplines': [
                                  proxy.mdo_discipline_wrapp.mdo_discipline for proxy in self.proxy_disciplines
                                  if proxy.mdo_discipline_wrapp is not None]})

    def is_configured(self):
        '''
        Return False if discipline is not configured or structuring variables have changed or children are not all configured
        '''
        return ProxyDiscipline.is_configured(self) and self.subprocess_is_configured()

    def subprocess_is_configured(self):
        # Explanation:
        # 1. self._data_in == {} : if the discipline as no input key it should have and so need to be configured
        # 2. Added condition compared to SoSDiscipline(as sub_discipline or associated sub_process builder)
        # 2.1 (self.get_disciplines_to_configure() == [] and len(self.proxy_disciplines) != 0) : sub_discipline(s) exist(s) but all configured
        # 2.2 len(self.cls_builder) == 0 No yet provided builder but we however need to configure (as in 2.1 when we have sub_disciplines which no need to be configured)
        # Remark1: condition "(   and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0" added for proc build
        # Remark2: /!\ REMOVED the len(self.proxy_disciplines) == 0 condition to accommodate the DriverEvaluator that holds te build until inputs are available
        return self.get_disciplines_to_configure() == [] or len(self.cls_builder) == 0

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

