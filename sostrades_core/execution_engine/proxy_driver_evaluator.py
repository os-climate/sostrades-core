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

import pandas as pd
import copy
from numpy import NaN

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
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

    EVAL_INPUT_TYPE = ['float', 'array', 'int', 'string']

    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 map_name=None,
                 associated_namespaces=None,
                 builder_tool=None):
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
        # if 'ns_doe' does not exist in ns_manager, we create this new
        # namespace to store output dictionaries associated to eval_outputs
        if 'ns_doe' not in ee.ns_manager.shared_ns_dict.keys():
            ee.ns_manager.add_ns('ns_doe', ee.study_name)

        super().__init__(sos_name, ee, driver_wrapper_cls,
                         associated_namespaces=associated_namespaces)
        if cls_builder is None:
            cls_builder = []

        if builder_tool:
            self.builder_tool_cls = builder_tool
            self.cls_builder = cls_builder
            self.builder_tool = None
            self.scatter_list_name = None
        else:
            self.cls_builder = cls_builder  # TODO: Move to ProxyDisciplineBuilder?
            self.builder_tool_cls = None
        self.eval_process_builder = None
        self.scatter_process_builder = None
        self.map_name = map_name
        self.scatter_list = None
        self.eval_in_list = None
        self.eval_out_list = None
        self.selected_outputs = []
        self.selected_inputs = []
        self.eval_out_type = []
        self.eval_out_list_size = []
        self.logger = get_sos_logger(f'{self.ee.logger.name}.DriverEvaluator')

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
        if self.builder_tool_cls:
            self.configure_tool()
        # configure al processes stored in children
        for disc in self.get_disciplines_to_configure():
            disc.configure()

        # configure current discipline DriverEvaluator
        # if self._data_in == {} or (self.get_disciplines_to_configure() == []
        # and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0:
        if self._data_in == {} or self.subprocess_is_configured():
            # Call standard configure methods to set the process discipline
            # tree
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
        # TODO: working because no two different discs share a local ns
        for proxy_disc in self.proxy_disciplines:
            # if not isinstance(proxy_disc, ProxyDisciplineGather):
            subprocess_data_in = proxy_disc.get_data_io_with_full_name(
                self.IO_TYPE_IN, as_namespaced_tuple=True)
            subprocess_data_out = proxy_disc.get_data_io_with_full_name(
                self.IO_TYPE_OUT, as_namespaced_tuple=True)
            self._update_data_io(subprocess_data_in, self.IO_TYPE_IN)
            self._update_data_io(subprocess_data_out, self.IO_TYPE_OUT)

    def configure_driver(self):
        """
        To be overload by drivers with specific configuration actions
        """
        # Extract variables for eval analysis in mono instance mode
        disc_in = self.get_data_in()
        if self.BUILDER_MODE in disc_in \
                and self.get_sosdisc_inputs(self.BUILDER_MODE) == self.MONO_INSTANCE \
                and 'eval_inputs' in disc_in \
                and len(self.proxy_disciplines) > 0:
            self.set_eval_possible_values()

    def setup_sos_disciplines(self):
        """
        Dynamic inputs and outputs of the DriverEvaluator
        """
        if self.BUILDER_MODE in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs(self.BUILDER_MODE)
            if builder_mode == self.MULTI_INSTANCE:
                self.build_inst_desc_io_with_scenario_df()
            elif builder_mode == self.MONO_INSTANCE:
                # pass  # TODO: to merge with Eval WIP

                # TODO: clean code below with class variables etc.
                dynamic_inputs = {'eval_inputs': {'type': 'dataframe',
                                                  'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                           'full_name': ('string', None, False)},
                                                  'dataframe_edition_locked': False,
                                                  'structuring': True,
                                                  'visibility': self.SHARED_VISIBILITY,
                                                  'namespace': 'ns_eval'},
                                  'eval_outputs': {'type': 'dataframe',
                                                   'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                                            'full_name': ('string', None, False)},
                                                   'dataframe_edition_locked': False,
                                                   'structuring': True, 'visibility': self.SHARED_VISIBILITY,
                                                   'namespace': 'ns_eval'},
                                  'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
                                  'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0}
                                  }
                dynamic_outputs = {'samples_inputs_df': {'type': 'dataframe', 'unit': None, 'visibility': self.SHARED_VISIBILITY,
                                                         'namespace': 'ns_eval'}
                                   }

                selected_inputs_has_changed = False
                disc_in = self.get_data_in()
                if 'eval_inputs' in disc_in:
                    # if len(disc_in) != 0:

                    eval_outputs = self.get_sosdisc_inputs('eval_outputs')
                    eval_inputs = self.get_sosdisc_inputs('eval_inputs')

                    # we fetch the inputs and outputs selected by the user
                    selected_outputs = eval_outputs[eval_outputs['selected_output']
                                                    == True]['full_name']
                    selected_inputs = eval_inputs[eval_inputs['selected_input']
                                                  == True]['full_name']
                    if set(selected_inputs.tolist()) != set(self.selected_inputs):
                        selected_inputs_has_changed = True
                        self.selected_inputs = selected_inputs.tolist()
                    self.selected_outputs = selected_outputs.tolist()

                    if len(selected_inputs) > 0 and len(selected_outputs) > 0:
                        # TODO: is it OK that it crashes with empty input ? also, might want an eval without outputs ?
                        # we set the lists which will be used by the evaluation
                        # function of sosEval
                        self.set_eval_in_out_lists(
                            self.selected_inputs, self.selected_outputs)

                        # setting dynamic outputs. One output of type dict per selected
                        # output
                        for out_var in self.eval_out_list:
                            dynamic_outputs.update(
                                {f'{out_var.split(self.ee.study_name + ".", 1)[1]}_dict': {'type': 'dict',
                                                                                           'visibility': 'Shared',
                                                                                           'namespace': 'ns_doe'}})
                        dynamic_inputs.update(self._get_dynamic_inputs_doe(
                            disc_in, selected_inputs_has_changed))
                self.add_inputs(dynamic_inputs)
                self.add_outputs(dynamic_outputs)
            elif builder_mode == self.REGULAR_BUILD:
                pass  # regular build requires no specific dynamic inputs
            else:
                raise ValueError(
                    f'Wrong builder mode input in {self.sos_name}')
        # after managing the different builds inputs, we do the setup_sos_disciplines of the wrapper in case it is
        # overload, e.g. in the case of a custom driver_wrapper_cls (with DriverEvaluatorWrapper this does nothing)
        # super().setup_sos_disciplines() # TODO: manage custom driver wrapper
        # case

    def prepare_build(self):
        """
        Get the actual drivers of the subprocesses of the DriverEvaluator.
        """
        # TODO: make me work with custom driver
        # TODO: test proper cleaning when changing builder mode
        builder_list = []
        if len(self.cls_builder) == 0:  # added condition for proc build
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
                raise ValueError(
                    f'Wrong builder mode input in {self.sos_name}')
        return builder_list

    def prepare_execution(self):
        """
        Preparation of the GEMSEO process, including GEMSEO objects instantiation
        """
        # prepare_execution of proxy_disciplines as in coupling
        # TODO: move to builder ?
        for disc in self.proxy_disciplines:
            disc.prepare_execution()
        # TODO : cache mgmt of children necessary ? here or in
        # SoSMDODisciplineDriver ?
        super().prepare_execution()

    def set_wrapper_attributes(self, wrapper):
        """
        set the attribute ".attributes" of wrapper which is used to provide the wrapper with information that is
        figured out at configuration time but needed at runtime. The DriverEvaluator in particular needs to provide
        its wrapper with a reference to the subprocess GEMSEO objets so they can be manipulated at runtime.
        """
        # TODO: needs to accommodate the eval attributes in the mono instance
        # case
        # io full name maps set by ProxyDiscipline
        super().set_wrapper_attributes(wrapper)

        wrapper.attributes.update({'sub_mdo_disciplines': [
                                  proxy.mdo_discipline_wrapp.mdo_discipline for proxy in self.proxy_disciplines
                                  if proxy.mdo_discipline_wrapp is not None]})  # discs and couplings but not scatters

        if self.BUILDER_MODE in self.get_data_in() and self.get_sosdisc_inputs(self.BUILDER_MODE) == self.MONO_INSTANCE:
            eval_attributes = {'eval_in_list': self.eval_in_list,
                               'eval_out_list': self.eval_out_list,
                               'reference_scenario': self.get_x0(),
                               'activated_elems_dspace_df': [[True, True]
                                                             if self.ee.dm.get_data(var, 'type') == 'array' else [True]
                                                             for var in self.eval_in_list],  # TODO: Array dimensions greater than 2??? TEST
                               'study_name': self.ee.study_name,
                               'reduced_dm': self.ee.dm.reduced_dm,  # for conversions
                               'selected_inputs': self.selected_inputs,
                               'selected_outputs': self.selected_outputs,
                               }
            wrapper.attributes.update(eval_attributes)

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
        # Remark2: /!\ REMOVED the len(self.proxy_disciplines) == 0 condition
        # to accommodate the DriverEvaluator that holds te build until inputs
        # are available
        return self.get_disciplines_to_configure() == [] or len(self.cls_builder) == 0

    # MULTI INSTANCE PROCESS
    def _set_scatter_process_builder(self, map_name):
        """
        Create and set the scatter builder that will allow multi-instance builds.
        """
        if len(self.cls_builder) == 0:  # added condition for proc build
            scatter_builder = None
        else:

            # builder of the scatter in aggregation with references to
            # self.cls_builder builders
            scatter_builder = self.ee.factory.create_scatter_builder('scatter_temp', map_name, self.cls_builder,  # TODO: nice to remove scatter node
                                                                     coupling_per_scatter=True)  # NB: is hardcoded also in VerySimpleMS/SimpleMS
        self.scatter_process_builder = scatter_builder

    def prepare_multi_instance_build(self):
        """
        Get the scatter builder for the subprocesses in multi-instance builder mode.
        """
        # TODO: will need to include options for MultiScenario other than
        # VerySimple
        if self.map_name is not None:
            # set the scatter builder that allows to scatter the subprocess
            if self.scatter_process_builder is None:
                if self.builder_tool_cls:
                    self.scatter_process_builder = self.build_tool()

                else:
                    self._set_scatter_process_builder(self.map_name)
            # if the scatter builder exists, use it to build the process
            if self.scatter_process_builder is not None:
                return [self.scatter_process_builder]
            else:
                self.logger.warn(
                    f'Scatter builder not configured in {self.sos_name}, map_name missing?')
        else:
            self.logger.warn(
                f'Attempting multi-instance build without a map_name in {self.sos_name}')
        return []

    def build_inst_desc_io_with_scenario_df(self):
        '''
        Complete inst_desc_in with scenario_df
        '''
        # get a reference to the scatter discipline
        # TODO: refactor code below when scatter as a tool is ready /!\
        driver_evaluator_node = self.ee.ns_manager.get_local_namespace_value(
            self)
        scatter_node = self.ee.ns_manager.compose_ns(
            [driver_evaluator_node, self.SCATTER_NODE_NAME])
        scatter_disc_list = self.dm.get_disciplines_with_name(scatter_node)
        if scatter_disc_list:  # otherwise nothing is possible
            # get scatter disc
            scatter_disc = scatter_disc_list[0]
            if self.SCENARIO_DF not in self.get_data_in():
                # add scenario_df to inst_desc_in in the same namespace defined
                # by the scatter map
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
                    self.STRUCTURING: True}}  # TODO: manage variable columns for (non-very-simple) multiscenario cases
                self.add_inputs(scenario_df_input)
            else:
                # TODO: refactor code below when scatter as a tool is ready /!\
                # brutally set the scatter node parameters to comply with scenario_df, which implies that scenario_df
                # has priority over the dynamic input of the scatter node
                # (which is bound to disappear)
                scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
                self.scatter_list = scenario_df[scenario_df[self.SELECTED_SCENARIO]
                                                == True][self.SCENARIO_NAME].values.tolist()
                scatter_input_name = scatter_disc.sc_map.get_input_name()
                scatter_disc_in = scatter_disc.get_data_in()
                if scatter_input_name in scatter_disc_in:
                    self.dm.set_data(scatter_disc.get_var_full_name(scatter_input_name, scatter_disc_in), self.VALUE,
                                     self.scatter_list, check_value=False)

    def configure_tool(self):
        if self.builder_tool is None:
            self.builder_tool = self.builder_tool_cls(
                'scatter_tool', self.ee, self.map_name, self.cls_builder, coupling_per_scatter=False)
            scatter_list_desc_in = self.builder_tool.get_scatter_list_desc_in()
            self.add_inputs(scatter_list_desc_in)

        self.builder_tool.prepare_tool(self)

    def build_tool(self):

        builder_list = self.builder_tool.build()

        return builder_list

    # MONO INSTANCE PROCESS

    def _get_disc_shared_ns_value(self):
        # TODO: better factorization, rename?
        return self.ee.ns_manager.disc_ns_dict[self]['others_ns']['ns_eval'].get_value()

    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary, which will allow mono-instance builds.
        '''
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        elif len(self.cls_builder) == 1:
            # Note no distinction is made whether the builder is executable or not; old implementation used to put
            # scatter builds under a coupling automatically too. # TODO: check
            # if necessary for gather implementation.
            disc_builder = self.cls_builder[0]
        else:
            # If eval process is a list of builders then we build a coupling
            # containing the eval process
            disc_builder = self.ee.factory.create_builder_coupling(
                'subprocess')
            disc_builder.set_builder_info('cls_builder', self.cls_builder)
        self.eval_process_builder = disc_builder

    def prepare_mono_instance_build(self):
        '''
        Get the builder of the single subprocesses in mono-instance builder mode.
        '''
        if self.eval_process_builder is None:
            self._set_eval_process_builder()

        return [self.eval_process_builder] if self.eval_process_builder is not None else []

    def set_eval_in_out_lists(self, in_list, out_list, inside_evaluator=False):
        '''
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        '''
        self.eval_in_list = [
            f'{self.ee.study_name}.{element}' for element in in_list]
        self.eval_out_list = [
            f'{self.ee.study_name}.{element}' for element in out_list]

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        analyzed_disc = self.proxy_disciplines[0]
        possible_in_values_full, possible_out_values_full = self.fill_possible_values(
            analyzed_disc)
        possible_in_values_full, possible_out_values_full = self.find_possible_values(analyzed_disc,
                                                                                      possible_in_values_full, possible_out_values_full)

        # Take only unique values in the list
        possible_in_values = list(set(possible_in_values_full))
        possible_out_values = list(set(possible_out_values_full))

        # these sorts are just for aesthetics
        possible_in_values.sort()
        possible_out_values.sort()

        default_in_dataframe = pd.DataFrame({'selected_input': [False for _ in possible_in_values],
                                             'full_name': possible_in_values})
        default_out_dataframe = pd.DataFrame({'selected_output': [False for _ in possible_out_values],
                                              'full_name': possible_out_values})

        eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
        eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')
        my_ns_eval_path = self._get_disc_shared_ns_value()

        if eval_input_new_dm is None:
            self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
                             'value', default_in_dataframe, check_value=False)
        # check if the eval_inputs need to be updated after a subprocess
        # configure
        elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
            self.check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
                               is_eval_input=True)
            default_dataframe = copy.deepcopy(default_in_dataframe)
            already_set_names = eval_input_new_dm['full_name'].tolist()
            already_set_values = eval_input_new_dm['selected_input'].tolist()
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
                    index]
            self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
                             'value', default_dataframe, check_value=False)

        if eval_output_new_dm is None:
            self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
                             'value', default_out_dataframe, check_value=False)
        # check if the eval_inputs need to be updated after a subprocess
        # configure
        elif set(eval_output_new_dm['full_name'].tolist()) != (set(default_out_dataframe['full_name'].tolist())):
            self.check_eval_io(eval_output_new_dm['full_name'].tolist(), default_out_dataframe['full_name'].tolist(),
                               is_eval_input=False)
            default_dataframe = copy.deepcopy(default_out_dataframe)
            already_set_names = eval_output_new_dm['full_name'].tolist()
            already_set_values = eval_output_new_dm['selected_output'].tolist()
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_output'] = already_set_values[
                    index]
            self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
                             'value', default_dataframe, check_value=False)

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''
        poss_in_values_full = []
        poss_out_values_full = []
        disc_in = disc.get_data_in()
        for data_in_key in disc_in.keys():
            is_input_type = disc_in[data_in_key][self.TYPE] in self.EVAL_INPUT_TYPE
            is_structuring = disc_in[data_in_key].get(
                self.STRUCTURING, False)
            in_coupling_numerical = data_in_key in list(
                ProxyCoupling.DESC_IN.keys())
            full_id = disc.get_var_full_name(
                data_in_key, disc_in)
            is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
                                           ]['io_type'] == 'in'
            # is_input_multiplier_type = disc_in[data_in_key][self.TYPE] in self.INPUT_MULTIPLIER_TYPE
            is_editable = disc_in[data_in_key]['editable']
            is_None = disc_in[data_in_key]['value'] is None
            if is_in_type and not in_coupling_numerical and not is_structuring and is_editable:
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                if is_input_type:
                    poss_in_values_full.append(
                        full_id.split(self.ee.study_name + ".", 1)[1])
                    # poss_in_values_full.append(full_id)

                # if is_input_multiplier_type and not is_None:
                #     poss_in_values_list = self.set_multipliers_values(
                #         disc, full_id, data_in_key)
                #     for val in poss_in_values_list:
                #         poss_in_values_full.append(val)

        disc_out = disc.get_data_out()
        for data_out_key in disc_out.keys():
            # Caution ! This won't work for variables with points in name
            # as for ac_model
            in_coupling_numerical = data_out_key in list(
                ProxyCoupling.DESC_IN.keys()) or data_out_key == 'residuals_history'
            full_id = disc.get_var_full_name(
                data_out_key, disc_out)
            if not in_coupling_numerical:
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                poss_out_values_full.append(
                    full_id.split(self.ee.study_name + ".", 1)[1])
                # poss_out_values_full.append(full_id)
        return poss_in_values_full, poss_out_values_full

    def find_possible_values(
            self, disc, possible_in_values, possible_out_values):
        '''
            Run through all disciplines and sublevels
            to find possible values for eval_inputs and eval_outputs
        '''
        # TODO: does this involve avoidable, recursive back and forths during
        # configuration ?
        if len(disc.proxy_disciplines) != 0:
            for sub_disc in disc.proxy_disciplines:
                sub_in_values, sub_out_values = self.fill_possible_values(
                    sub_disc)
                possible_in_values.extend(sub_in_values)
                possible_out_values.extend(sub_out_values)
                self.find_possible_values(
                    sub_disc, possible_in_values, possible_out_values)

        return possible_in_values, possible_out_values

    def get_x0(self):
        '''
        Get initial values for input values decided in the evaluation
        '''
        return dict(zip(self.eval_in_list,
                        map(self.dm.get_value, self.eval_in_list)))

    def _get_dynamic_inputs_doe(self, disc_in, selected_inputs_has_changed):
        default_custom_dataframe = pd.DataFrame(
            [[NaN for _ in range(len(self.selected_inputs))]], columns=self.selected_inputs)
        dataframe_descriptor = {}
        for i, key in enumerate(self.selected_inputs):
            cle = key
            var = tuple([self.ee.dm.get_data(
                self.eval_in_list[i], 'type'), None, True])
            dataframe_descriptor[cle] = var

        dynamic_inputs = {'samples_df': {'type': 'dataframe', self.DEFAULT: default_custom_dataframe,
                                         'dataframe_descriptor': dataframe_descriptor,
                                         'dataframe_edition_locked': False,
                                         'visibility': SoSWrapp.SHARED_VISIBILITY,
                                         'namespace': 'ns_eval'
                                         }}

        if 'samples_df' in disc_in and selected_inputs_has_changed:
            disc_in['samples_df']['value'] = default_custom_dataframe
            disc_in['samples_df']['dataframe_descriptor'] = dataframe_descriptor
        return dynamic_inputs

    def check_eval_io(self, given_list, default_list, is_eval_input):
        """
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        """

        for given_io in given_list:
            if given_io not in default_list:
                if is_eval_input:
                    error_msg = f'The input {given_io} in eval_inputs is not among possible values. Check if it is an ' \
                                f'input of the subprocess with the correct full name (without study name at the ' \
                                f'beginning) and within allowed types (int, array, float). Dynamic inputs might  not ' \
                                f'be created. '

                else:
                    error_msg = f'The output {given_io} in eval_outputs is not among possible values. Check if it is an ' \
                                f'output of the subprocess with the correct full name (without study name at the ' \
                                f'beginning). Dynamic inputs might  not be created. '

                self.logger.warning(error_msg)
