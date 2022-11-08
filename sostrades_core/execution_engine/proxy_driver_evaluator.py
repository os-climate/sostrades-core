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
import pandas as pd

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.mdo_discipline_driver_wrapp import MDODisciplineDriverWrapp
from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper


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
    
    BUILDER_MODE = DriverEvaluatorWrapper.BUILDER_MODE
    MONO_INSTANCE = DriverEvaluatorWrapper.MONO_INSTANCE
    MULTI_INSTANCE = DriverEvaluatorWrapper.MULTI_INSTANCE
    REGULAR_BUILD = DriverEvaluatorWrapper.REGULAR_BUILD
    BUILDER_MODE_POSSIBLE_VALUES = DriverEvaluatorWrapper.BUILDER_MODE_POSSIBLE_VALUES
    SCENARIO_DF = 'scenario_df'
    SCATTER_NODE_NAME = 'scatter_temp'

    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'

    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 map_name=None,
                 associated_namespaces=None):
        """
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (List[SoSBuilder]): list of the sub proxy builders
            driver_wrapper_cls (Class): class constructor of the driver wrapper (user-defined wrapper or SoSTrades wrapper or None)
            map_name (string): name of the map associated to the scatter builder in case of multi-instance build
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
        """
        super().__init__(sos_name, ee, driver_wrapper_cls,
                         associated_namespaces=associated_namespaces)
        if cls_builder is None:
            cls_builder = []
        self.cls_builder = cls_builder  # TODO: Move to ProxyDisciplineBuilder?
        self.eval_process_builder = None
        self.scatter_process_builder = None
        self.map_name = map_name
        self.scatter_list = None

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
        creation of mdo_discipline_wrapp by the proxy which in this case is a MDODisciplineDriverWrapp that will create
        a SoSMDODisciplineDriver at prepare_execution, i.e. a driver node that knows its subprocesses but manipulates
        them in a different way than a coupling.
        """
        self.mdo_discipline_wrapp = MDODisciplineDriverWrapp(
            name, wrapper, wrapping_mode)

    def configure(self):
        """
        Configure the DriverEvaluator layer
        """
        # configure al processes stored in children
        for disc in self.get_disciplines_to_configure():
            disc.configure()

        # configure current discipline DriverEvaluator
        # if self._data_in == {} or (self.get_disciplines_to_configure() == [] and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0:
        if self._data_in == {} or self.subprocess_is_configured():
            # Call standard configure methods to set the process discipline tree
            ProxyDiscipline.configure(self)
            self.configure_driver()

        if self.subprocess_is_configured():
            self.update_data_io_with_subprocess_io()
            self.set_children_cache_inputs()

    def update_data_io_with_subprocess_io(self):
        """
        Update the DriverEvaluator _data_in and _data_out with subprocess i/o so that grammar of the driver can be
        exploited for couplings etc.
        """
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
        """
        Dynamic inputs and outputs of the DriverLayer
        """
        if self.BUILDER_MODE in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs(self.BUILDER_MODE)
            if builder_mode == self.MULTI_INSTANCE:
                self.build_inst_desc_io_with_scenario_df()
            elif builder_mode == self.MONO_INSTANCE:
                pass  # TODO: to merge with Eval
            elif builder_mode == self.REGULAR_BUILD:
                pass  # regular build requires no specific dynamic inputs
            else:
                raise ValueError(f'Wrong builder mode input in {self.sos_name}')
        # after managing the different builds inputs, we do the setup_sos_disciplines of the wrapper in case it is
        # overload, e.g. in the case of a custom driver_wrapper_cls (with DriverEvaluatorWrapper this does nothing)
        # super().setup_sos_disciplines() # TODO: manage custom driver wrapper case

    def prepare_build(self):
        """
        Get the actual drivers of the subprocesses of the DriverEvaluator.
        """
        # TODO: make me work with custom driver
        # TODO: test proper cleaning when changing builder mode
        builder_list = []
        if len(self.cls_builder) == 0: # added condition for proc build
            pass
        elif self.BUILDER_MODE in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs(self.BUILDER_MODE)
            if builder_mode == self.MULTI_INSTANCE:
                builder_list = self.prepare_multi_instance_build()
            elif builder_mode == self.MONO_INSTANCE:
                builder_list = self.prepare_mono_instance_build()
            elif builder_mode == self.REGULAR_BUILD:
                builder_list = super().prepare_build()
            else:
                raise ValueError(f'Wrong builder mode input in {self.sos_name}')
        return builder_list

    def prepare_execution(self):
        """
        Preparation of the GEMSEO process, including GEMSEO objects instantiation
        """
        # prepare_execution of proxy_disciplines as in coupling
        # TODO: move to builder ?
        for disc in self.proxy_disciplines:
            disc.prepare_execution()
        # TODO : cache mgmt of children necessary ? here or in SoSMDODisciplineDriver ?
        super().prepare_execution()

    def set_wrapper_attributes(self, wrapper):
        """
        set the attribute ".attributes" of wrapper which is used to provide the wrapper with information that is
        figured out at configuration time but needed at runtime. The DriverEvaluator in particular needs to provide
        its wrapper with a reference to the subprocess GEMSEO objets so they can be manipulated at runtime.
        """
        #TODO: needs to accommodate the eval attributes in the mono instance case
        super().set_wrapper_attributes(wrapper)
        wrapper.attributes.update({'sub_mdo_disciplines': [
                                  proxy.mdo_discipline_wrapp.mdo_discipline for proxy in self.proxy_disciplines
                                  if proxy.mdo_discipline_wrapp is not None]}) # discs and couplings but not scatters

    def is_configured(self):
        """
        Return False if discipline is not configured or structuring variables have changed or children are not all configured
        """
        return ProxyDiscipline.is_configured(self) and self.subprocess_is_configured()

    def subprocess_is_configured(self):
        """
        Return True if the subprocess is configured or the builder is empty.
        """
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
        """
        Create and set the scatter builder that will allow multi-instance builds.
        """
        if len(self.cls_builder) == 0:  # added condition for proc build
            scatter_builder = None
        else:
            # builder of the scatter in aggregation with references to self.cls_builder builders
            scatter_builder = self.ee.factory.create_scatter_builder(self.SCATTER_NODE_NAME, map_name, self.cls_builder, # TODO: nice to remove scatter node
                                                                     coupling_per_scatter=True) #NB: is hardcoded also in VerySimpleMS/SimpleMS
        self.scatter_process_builder = scatter_builder

    def prepare_multi_instance_build(self):
        """
        Get the scatter builder for the subprocesses in multi-instance builder mode.
        """
        # TODO: will need to include options for MultiScenario other than VerySimple
        if self.map_name is not None:
            # set the scatter builder that allows to scatter the subprocess
            if self.scatter_process_builder is None:
                self._set_scatter_process_builder(self.map_name)
            # if the scatter builder exists, use it to build the process
            if self.scatter_process_builder is not None:
                return [self.scatter_process_builder]
            else:
                self.logger.warn(f'Scatter builder not configured in {self.sos_name}, map_name missing?')
        else:
            self.logger.warn(f'Attempting multi-instance build without a map_name in {self.sos_name}')
        return []

    def build_inst_desc_io_with_scenario_df(self):
        '''
        Complete inst_desc_in with scenario_df
        '''
        # get a reference to the scatter discipline
        # TODO: refactor code below when scatter as a tool is ready /!\
        driver_evaluator_node = self.ee.ns_manager.get_local_namespace_value(self)
        scatter_node = self.ee.ns_manager.compose_ns([driver_evaluator_node, self.SCATTER_NODE_NAME])
        scatter_disc_list = self.dm.get_disciplines_with_name(scatter_node)
        if scatter_disc_list: # otherwise nothing is possible
            # get scatter disc
            scatter_disc = scatter_disc_list[0]
            if self.SCENARIO_DF not in self.get_data_in():
                # add scenario_df to inst_desc_in in the same namespace defined by the scatter map
                input_ns = scatter_disc.sc_map.get_input_ns()
                scenario_df_input = {self.SCENARIO_DF: {
                    self.TYPE: 'dataframe',
                    self.DEFAULT: pd.DataFrame(columns=[self.SELECTED_SCENARIO, self.SCENARIO_NAME]),
                    self.DATAFRAME_DESCRIPTOR: {self.SELECTED_SCENARIO: ('bool', None, True),
                                                self.SCENARIO_NAME: ('string', None, True)},
                    self.DATAFRAME_EDITION_LOCKED: False,
                    self.VISIBILITY: self.SHARED_VISIBILITY,
                    self.NAMESPACE: input_ns,
                    self.EDITABLE: True,
                    self.STRUCTURING: True}} #TODO: manage variable columns for (non-very-simple) multiscenario cases
                self.add_inputs(scenario_df_input)
            else:
                # TODO: refactor code below when scatter as a tool is ready /!\
                # brutally set the scatter node parameters to comply with scenario_df, which implies that scenario_df
                # has priority over the dynamic input of the scatter node (which is bound to disappear)
                scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
                self.scatter_list = scenario_df[scenario_df[self.SELECTED_SCENARIO] == True][self.SCENARIO_NAME].values.tolist()
                scatter_input_name = scatter_disc.sc_map.get_input_name()
                scatter_disc_in = scatter_disc.get_data_in()
                if scatter_input_name in scatter_disc_in:
                    self.dm.set_data(scatter_disc.get_var_full_name(scatter_input_name, scatter_disc_in), self.VALUE,
                                     self.scatter_list, check_value=False)

    # MONO INSTANCE PROCESS
    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary, which will allow mono-instance builds.
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

    def prepare_mono_instance_build(self):
        '''
        Get the builder of the single subprocesses in mono-instance builder mode.
        '''
        if self.eval_process_builder is None:
            self._set_eval_process_builder()
        return [self.eval_process_builder] if self.eval_process_builder is not None else []
