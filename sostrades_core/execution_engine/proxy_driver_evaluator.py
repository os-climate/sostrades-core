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

import logging
from typing import Optional
import copy
import pandas as pd
from numpy import NaN
import numpy as np

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.mdo_discipline_driver_wrapp import MDODisciplineDriverWrapp
from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
from gemseo.utils.compare_data_manager_tooling import dict_are_equal
from sostrades_core.tools.builder_info.builder_info_functions import get_ns_list_in_builder_list
from gemseo.utils.compare_data_manager_tooling import compare_dict


class ProxyDriverEvaluatorException(Exception):
    pass


class ProxyDriverEvaluator(ProxyDisciplineBuilder):
    '''
        SOSEval class which creates a sub process to evaluate
        with different methods (Gradient,FORM,Sensitivity Analysis, DOE, ...)

    1) Structure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ INSTANCE_REFERENCE (structuring, dynamic : builder_mode == self.MULTI_INSTANCE)
                |_ REFERENCE_MODE (structuring, dynamic :instance_referance == TRUE) 
                |_ REFERENCE_SCENARIO_NAME (structuring, dynamic :instance_referance == TRUE) #TODO
            |_ EVAL_INPUTS (namespace: NS_EVAL, structuring, dynamic : builder_mode == self.MONO_INSTANCE)
            |_ EVAL_OUTPUTS (namespace: NS_EVAL, structuring, dynamic : builder_mode == self.MONO_INSTANCE)
            |_ GENERATED_SAMPLES ( structuring,dynamic: self.builder_tool == True)
            |_ SCENARIO_DF (structuring,dynamic: self.builder_tool == True)
            |_ SAMPLES_DF (namespace: NS_EVAL, dynamic: len(selected_inputs) > 0 and len(selected_outputs) > 0 )    
            |_ 'n_processes' (dynamic : builder_mode == self.MONO_INSTANCE)         
            |_ 'wait_time_between_fork' (dynamic : builder_mode == self.MONO_INSTANCE)

        |_ DESC_OUT
            |_ samples_inputs_df (namespace: NS_EVAL, dynamic: builder_mode == self.MONO_INSTANCE)
            |_ <var>_dict (internal namespace 'ns_doe', dynamic: len(selected_inputs) > 0 and len(selected_outputs) > 0
            and eval_outputs not empty, for <var> in eval_outputs)

    2) Description of DESC parameters:
        |_ DESC_IN
            |_ INSTANCE_REFERENCE 
                |_ REFERENCE_MODE 
                |_ REFERENCE_SCENARIO_NAME  #TODO
            |_ EVAL_INPUTS
            |_ EVAL_OUTPUTS
            |_ GENERATED_SAMPLES
            |_ SCENARIO_DF
            |_ SAMPLES_DF
            |_ 'n_processes' 
            |_ 'wait_time_between_fork'            
       |_ DESC_OUT
            |_ samples_inputs_df
            |_ <var observable name>_dict':     for each selected output observable doe result
                                                associated to sample and the selected observable

    '''

    # ontology information
    _ontology_data = {
        'label': 'Driver Evaluator',
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

    MONO_INSTANCE = DriverEvaluatorWrapper.MONO_INSTANCE
    MULTI_INSTANCE = DriverEvaluatorWrapper.MULTI_INSTANCE
    REGULAR_BUILD = DriverEvaluatorWrapper.REGULAR_BUILD
    SUB_PROCESS_INPUTS = DriverEvaluatorWrapper.SUB_PROCESS_INPUTS
    GATHER_DEFAULT_SUFFIX = DriverEvaluatorWrapper.GATHER_DEFAULT_SUFFIX

    INSTANCE_REFERENCE = 'instance_reference'
    LINKED_MODE = 'linked_mode'
    COPY_MODE = 'copy_mode'
    REFERENCE_MODE = 'reference_mode'
    REFERENCE_MODE_POSSIBLE_VALUES = [LINKED_MODE, COPY_MODE]
    REFERENCE_SCENARIO_NAME = 'ReferenceScenario'

    SCENARIO_DF = 'scenario_df'

    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'
    # with SampleGenerator, whether to activate and build all the sampled
    MAX_SAMPLE_AUTO_BUILD_SCENARIOS = 1024
    # scenarios by default or not. Set to None to always build.

    SUBCOUPLING_NAME = 'subprocess'
    EVAL_INPUTS = 'eval_inputs'
    EVAL_OUTPUTS = 'eval_outputs'
    EVAL_INPUT_TYPE = ['float', 'array', 'int', 'string']

    GENERATED_SAMPLES = SampleGeneratorWrapper.GENERATED_SAMPLES

    USECASE_DATA = 'usecase_data'

    # shared namespace of the mono-instance evaluator for eventual couplings
    NS_EVAL = 'ns_eval'

    MULTIPLIER_PARTICULE = '__MULTIPLIER__'

    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 associated_namespaces=None,
                 map_name=None,
                 flatten_subprocess=False,
                 display_options=None,
                 ):
        """
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (List[SoSBuilder]): list of the sub proxy builders
            driver_wrapper_cls (Class): class constructor of the driver wrapper (user-defined wrapper or SoSTrades wrapper or None)
            map_name (string): name of the map associated to the scatter builder in case of multi-instance build
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
            logger (logging.Logger): Logger to use
        """
        super().__init__(sos_name, ee, driver_wrapper_cls, associated_namespaces=associated_namespaces)
        if cls_builder is not None:
            self.cls_builder = cls_builder
            self.sub_builder_namespaces = get_ns_list_in_builder_list(
                self.cls_builder)
        else:
            raise Exception(
                'The driver evaluator builder must have a cls_builder to work')

        self.builder_tool = None

        self.map_name = map_name
        self.flatten_subprocess = flatten_subprocess
        self.scenarios = []  # to keep track of subdisciplines in a flatten_subprocess case

        self.display_options = display_options

        self.eval_process_builder = None
        self.eval_in_list = None
        self.eval_out_list = None
        self.selected_inputs = []
        self.selected_outputs = []
        self.eval_out_names = []
        self.eval_out_type = []
        self.eval_out_list_size = []

        self.old_samples_df, self.old_scenario_df = ({}, {})
        self.scatter_list_valid = True
        self.scatter_list_integrity_msg = ''

        self.previous_sub_process_usecase_name = 'Empty'
        self.previous_sub_process_usecase_data = {}
        # Possible values: 'No_SP_UC_Import', 'SP_UC_Import'
        self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

        self.old_ref_dict = {}
        self.old_scenario_names = []
        self.save_editable_attr = True
        self.original_editability_dict = {}
        self.original_editable_dict_ref = {}
        self.original_editable_dict_non_ref = {}
        self.there_are_new_scenarios = False

        self.gather_names = None

    def _add_optional_shared_ns(self):
        """
        Add the shared namespace NS_EVAL should it not exist.
        """
        # do the same for the shared namespace for coupling with the DriverEvaluator
        # also used to store gathered variables in multi-instance
        if self.NS_EVAL not in self.ee.ns_manager.shared_ns_dict.keys():
            self.ee.ns_manager.add_ns(
                self.NS_EVAL, self.ee.ns_manager.compose_local_namespace_value(self))

    def _get_disc_shared_ns_value(self):
        """
        Get the namespace ns_eval used in the mono-instance case.
        """
        return self.ee.ns_manager.disc_ns_dict[self]['others_ns'][self.NS_EVAL].get_value()

    def get_desc_in_out(self, io_type):
        """
        get the desc_in or desc_out. if a wrapper exists get it from the wrapper, otherwise get it from the proxy class
        """
        # TODO : check if the following logic could be OK and implement it
        # according to what we want to do : DESC_IN of Proxy is updated by SoSWrapp if exists
        # thus no mixed calls to n-1 and n-2

        if self.mdo_discipline_wrapp.wrapper is not None:
            # ProxyDiscipline gets the DESC from the wrapper
            return ProxyDiscipline.get_desc_in_out(self, io_type)
        else:
            # ProxyDisciplineBuilder expects the DESC on the proxies e.g. Coupling
            # TODO: move to coupling ?
            return super().get_desc_in_out(io_type)

    def create_mdo_discipline_wrap(self, name, wrapper, wrapping_mode, logger:logging.Logger):
        """
        creation of mdo_discipline_wrapp by the proxy which in this case is a MDODisciplineDriverWrapp that will create
        a SoSMDODisciplineDriver at prepare_execution, i.e. a driver node that knows its subprocesses but manipulates
        them in a different way than a coupling.
        """
        self.mdo_discipline_wrapp = MDODisciplineDriverWrapp(name, logger.getChild("MDODisciplineDriverWrapp"), wrapper, wrapping_mode)

    def configure(self):
        """
        Configure the DriverEvaluator layer
        """
        # set the scenarios references, for flattened subprocess configuration
        if self.flatten_subprocess and self.builder_tool:
            self.scenarios = self.builder_tool.get_all_built_disciplines()
        else:
            self.scenarios = self.proxy_disciplines

        # configure al processes stored in children
        for disc in self.get_disciplines_to_configure():
            disc.configure()

        # configure current discipline DriverEvaluator
        # if self._data_in == {} or (self.get_disciplines_to_configure() == []
        # and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0:
        if self._data_in == {} or self.subprocess_is_configured():
            # Call standard configure methods to set the process discipline
            # tree
            super().configure()
            self.configure_driver()

        if self.subprocess_is_configured():
            self.update_data_io_with_subprocess_io()
            self.set_children_numerical_inputs()

    def update_data_io_with_subprocess_io(self):
        """
        Update the DriverEvaluator _data_in and _data_out with subprocess i/o so that grammar of the driver can be
        exploited for couplings etc.
        """
        self._restart_data_io_to_disc_io()
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
        To be overloaded by drivers with specific configuration actions
        """
        pass

    # def setup_sos_disciplines(self):
    #     """
    #     Dynamic inputs and outputs of the DriverEvaluator
    #     """
    #     super().setup_sos_disciplines()  #TODO: this actually does nothing unless there exists a custom driver wrapper

    def prepare_build(self):
        """
        Get the actual drivers of the subprocesses of the DriverEvaluator.
        """
        # NB: custom driver wrapper not implemented
        # FIXME: clean the code that used to clean after builder mode change
        # TODO: feels like the class hierarchy coherence of this method could be improved..
        return []

    def prepare_execution(self):
        """
        Preparation of the GEMSEO process, including GEMSEO objects instantiation
        """
        # prepare_execution of proxy_disciplines as in coupling
        for disc in self.scenarios:
            disc.prepare_execution()
        # TODO : cache mgmt of children necessary ? here or in  SoSMDODisciplineDriver ?
        super().prepare_execution()
        self.reset_subdisciplines_of_wrapper()

    def reset_subdisciplines_of_wrapper(self):
        self.mdo_discipline_wrapp.reset_subdisciplines(self)

    def set_wrapper_attributes(self, wrapper):
        """
        set the attribute ".attributes" of wrapper which is used to provide the wrapper with information that is
        figured out at configuration time but needed at runtime. The DriverEvaluator in particular needs to provide
        its wrapper with a reference to the subprocess GEMSEO objets so they can be manipulated at runtime.
        """
        # io full name maps set by ProxyDiscipline
        super().set_wrapper_attributes(wrapper)

        # driverevaluator subprocess # TODO: actually no longer necessary in multi-instance (gather capabilities)
        wrapper.attributes.update({'sub_mdo_disciplines': [
            proxy.mdo_discipline_wrapp.mdo_discipline for proxy in self.proxy_disciplines
            if proxy.mdo_discipline_wrapp is not None]})  # discs and couplings but not scatters

    def is_configured(self):
        """
        Return False if discipline is not configured or structuring variables have changed or children are not all configured
        """
        return super().is_configured() and self.subprocess_is_configured()

    def subprocess_is_configured(self):
        """
        Return True if the subprocess is configured or the builder is empty.
        """
        return self.get_disciplines_to_configure() == []

    def get_disciplines_to_configure(self):
        return self._get_disciplines_to_configure(self.scenarios)

    def check_data_integrity(self):
        # checking for duplicates
        disc_in = self.get_data_in()
        if self.SCENARIO_DF in disc_in and not self.scatter_list_valid:
            self.dm.set_data(
                self.get_var_full_name(self.SCENARIO_DF, disc_in),
                self.CHECK_INTEGRITY_MSG, self.scatter_list_integrity_msg)

    def fill_possible_values(self, disc, io_type_in=True, io_type_out=True):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''
        poss_in_values_full = set()
        poss_out_values_full = set()
        if io_type_in: # TODO: edit this code if adding multi-instance eval_inputs in order to take structuring vars
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
                is_a_multiplier = self.MULTIPLIER_PARTICULE in data_in_key
                if is_in_type and not in_coupling_numerical and not is_structuring and is_editable:
                    # Caution ! This won't work for variables with points in name
                    # as for ac_model
                    # we remove the study name from the variable full  name for a
                    # sake of simplicity
                    if is_input_type and not is_a_multiplier:
                        poss_in_values_full.add(
                            full_id.split(f'{self.get_disc_full_name()}.', 1)[1])
                        # poss_in_values_full.append(full_id)

                    # if is_input_multiplier_type and not is_None:
                    #     poss_in_values_list = self.set_multipliers_values(
                    #         disc, full_id, data_in_key)
                    #     for val in poss_in_values_list:
                    #         poss_in_values_full.append(val)

        if io_type_out:
            disc_out = disc.get_data_out()
            for data_out_key in disc_out.keys():
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                in_coupling_numerical = data_out_key in list(
                    ProxyCoupling.DESC_IN.keys()) or data_out_key == 'residuals_history'
                full_id = disc.get_var_full_name(
                    data_out_key, disc_out)
                if not in_coupling_numerical:
                    # we anonymize wrt. driver evaluator node namespace
                    poss_out_values_full.add(
                        full_id.split(f'{self.get_disc_full_name()}.', 1)[1])
                    # poss_out_values_full.append(full_id)
        return poss_in_values_full, poss_out_values_full

    def find_possible_values(
            self, disc, possible_in_values, possible_out_values,
            io_type_in=True, io_type_out=True):
        '''
            Run through all disciplines and sublevels
            to find possible values for eval_inputs and eval_outputs
        '''
        # TODO: does this involve avoidable, recursive back and forths during  configuration ? (<-> config. graph)
        if len(disc.proxy_disciplines) != 0:
            for sub_disc in disc.proxy_disciplines:
                sub_in_values, sub_out_values = self.fill_possible_values(
                    sub_disc, io_type_in=io_type_in, io_type_out=io_type_out)
                possible_in_values.update(sub_in_values)
                possible_out_values.update(sub_out_values)
                self.find_possible_values(
                    sub_disc, possible_in_values, possible_out_values,
                    io_type_in=io_type_in, io_type_out=io_type_out)

        return possible_in_values, possible_out_values

    def check_eval_io(self, given_list, default_list, is_eval_input):
        """
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        """

        for given_io in given_list:
            if given_io not in default_list and not self.MULTIPLIER_PARTICULE in given_io:
                if is_eval_input:
                    error_msg = f'The input {given_io} in eval_inputs is not among possible values. Check if it is an ' \
                                f'input of the subprocess with the correct full name (without study name at the ' \
                                f'beginning) and within allowed types (int, array, float). Dynamic inputs might  not ' \
                                f'be created. should be in {default_list} '

                else:
                    error_msg = f'The output {given_io} in eval_outputs is not among possible values. Check if it is an ' \
                                f'output of the subprocess with the correct full name (without study name at the ' \
                                f'beginning). Dynamic inputs might  not be created. should be in {default_list}'

                self.logger.warning(error_msg)

    def manage_import_inputs_from_sub_process(self, ref_discipline_full_name):
        """
        """
        # Set sub_proc_import_usecase_status
        with_modal = True
        self.set_sub_process_usecase_status_from_user_inputs(with_modal)

        # Treat the case of SP_UC_Import
        if self.sub_proc_import_usecase_status == 'SP_UC_Import':
            # Get the anonymized dict
            if with_modal:  # TODO (when use of Modal)
                anonymize_input_dict_from_usecase = self.get_sosdisc_inputs(
                    self.SUB_PROCESS_INPUTS)[ProcessBuilderParameterType.USECASE_DATA]
            else:
                # (without use of Modal)
                anonymize_input_dict_from_usecase = self.get_sosdisc_inputs(
                    self.USECASE_DATA)
            # LOAD REFERENCE
            if self.update_reference():
                self.update_reference_from_anonymised_dict(
                    anonymize_input_dict_from_usecase, ref_discipline_full_name, with_modal)

    def update_reference(self):
        # TODO: quick fix for split of ref. instance, method is to refactor
        # TODO: currently inactive in ProxyOptim, need overload to activate
        return False

    def update_reference_from_anonymised_dict(self, anonymize_input_dict_from_usecase, ref_discipline_full_name,
                                              with_modal):
        """
        """
        # 1. Put anonymized dict in context (unanonymize) of the reference
        # First identify the reference scenario
        input_dict_from_usecase = self.put_anonymized_input_dict_in_sub_process_context(
            anonymize_input_dict_from_usecase, ref_discipline_full_name)
        # print(input_dict_from_usecase)
        # self.ee.display_treeview_nodes(True)
        # 2. load data in dm (# Push the data to the reference
        # instance)

        # ======================================================================
        # if method == 'load_study':  # We sometimes in multi instance get an infinite loop and never do the last in the sequence
        #     self.ee.load_study_from_input_dict(
        #         input_dict_from_usecase)
        # elif method =='set_values':  # This is what was done before the bellow correction. It doesn't work with dynamic subproc or if a data kay is not yet in the dm
        #     self.ee.dm.set_values_from_dict(
        #         input_dict_from_usecase)
        # self.ee.dm.set_values_from_dict(filtered_import_dict)
        # ======================================================================

        # Here is a NEW method : with filtering. With this method something is
        # added in is_configured function
        filtered_import_dict = {}
        for key in input_dict_from_usecase:
            if self.ee.dm.check_data_in_dm(key):
                filtered_import_dict[key] = input_dict_from_usecase[key]

        self.ee.dm.set_values_from_dict(filtered_import_dict)

        are_all_data_set = len(filtered_import_dict.keys()) == len(
            input_dict_from_usecase.keys())

        # Remark 1: This condition will be a problem if the users is putting a bad key of variable in its anonymized dict
        # It may be ok if the anonymized dict comes from a uses case ? --> so
        # having wrong keys may be not needed to be treated

        # Remark 2: however with this filtering we should verify that we will always have all the variable pushed at the end
        # (we should not miss data that were provided in the anonymized dict) : but this will be the case all valid keys
        # will be set in the dm if it is a appropriate key (based on the
        # dynamic configuration)

        # Remark 3: What could be done is: if we reach the 100 iterations limit because are_all_data_set is still not True
        # then provide a warning with the list of variables keys that makes are_all_data_set still be False

        # Remark 4: (Better improvement)  next Provide another mechanism at eev4 level in which you always can push data
        # in dm provide check and warning when you reach the end of the configuration.

        if are_all_data_set:
            # TODO Bug if 100 config reached ( a bad key in anonymised dict) .
            # In this case are_all_data_set always False and we do not reset all parameters as it should !
            # 3. Update parameters
            #     Set the status to 'No_SP_UC_Import'
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
            if with_modal:  # TODO (when use of Modal)
                # Empty the anonymized dict in (when use of Modal)
                sub_process_inputs_dict = self.get_sosdisc_inputs(
                    self.SUB_PROCESS_INPUTS)
                sub_process_inputs_dict[ProcessBuilderParameterType.USECASE_DATA] = {
                }
                # Consequently update the previous_sub_process_usecase_data
                sub_process_usecase_name = sub_process_inputs_dict[
                    ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_NAME]
                sub_process_usecase_data = sub_process_inputs_dict[
                    ProcessBuilderParameterType.USECASE_DATA]
                self.previous_sub_process_usecase_name = sub_process_usecase_name
                self.previous_sub_process_usecase_data = sub_process_usecase_data
                self.previous_sub_process_usecase_data = {}
            else:
                # Consequently update the previous_sub_process_usecase_data
                sub_process_usecase_data = self.get_sosdisc_inputs(
                    self.USECASE_DATA)
                self.previous_sub_process_usecase_data = sub_process_usecase_data

    def set_sub_process_usecase_status_from_user_inputs(self, with_modal):
        """
            State subprocess usecase import status
            The uscase is defined by its name and its anonimized dict
            Function needed in manage_import_inputs_from_sub_process()
        """
        disc_in = self.get_data_in()

        if with_modal:
            if self.SUB_PROCESS_INPUTS in disc_in:  # and self.sub_proc_build_status != 'Empty_SP'
                sub_process_inputs_dict = self.get_sosdisc_inputs(
                    self.SUB_PROCESS_INPUTS)
                sub_process_usecase_name = sub_process_inputs_dict[
                    ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_NAME]
                sub_process_usecase_data = sub_process_inputs_dict[
                    ProcessBuilderParameterType.USECASE_DATA]
                if self.previous_sub_process_usecase_name != sub_process_usecase_name or self.previous_sub_process_usecase_data != sub_process_usecase_data:
                    # not not sub_process_usecase_data True means it is not an
                    # empty dictionary
                    if sub_process_usecase_name != 'Empty' and not not sub_process_usecase_data:
                        self.sub_proc_import_usecase_status = 'SP_UC_Import'
                else:
                    self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
            else:
                self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
        else:
            if self.USECASE_DATA in disc_in:
                sub_process_usecase_data = self.get_sosdisc_inputs(
                    self.USECASE_DATA)
                if self.previous_sub_process_usecase_data != sub_process_usecase_data:
                    # not not sub_process_usecase_data True means it is not an
                    # empty dictionary
                    if not not sub_process_usecase_data:
                        self.sub_proc_import_usecase_status = 'SP_UC_Import'
                else:
                    self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
            else:
                self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

    def put_anonymized_input_dict_in_sub_process_context(self, anonymize_input_dict_from_usecase,
                                                         ref_discipline_full_name):
        """
            Put_anonymized_input_dict in sub_process context
            Function needed in manage_import_inputs_from_sub_process()
        """
        # Get unanonymized dict (i.e. dict of subprocess in driver context)
        # from anonymized dict and context
        # Following treatment of substitution of the new_study_placeholder of value self.ref_discipline_full_name
        # may not to be done for all variables (see vsMS with ns_to_update that
        # has not all the ns keys)

        input_dict_from_usecase = {}
        new_study_placeholder = ref_discipline_full_name
        for key_to_unanonymize, value in anonymize_input_dict_from_usecase.items():
            converted_key = key_to_unanonymize.replace(
                self.ee.STUDY_PLACEHOLDER_WITHOUT_DOT, new_study_placeholder)
            # see def __unanonymize_key  in execution_engine
            uc_d = {converted_key: value}
            input_dict_from_usecase.update(uc_d)
        return input_dict_from_usecase

    def set_eval_possible_values(self, io_type_in=True, io_type_out=True, strip_first_ns=False):
        '''
        Check recursively the disciplines in the subprocess in order to detect their inputs and outputs.
        Once all disciplines have been run through, set the possible values for eval_inputs and eval_outputs in the DM
        These are the variables names anonymized wrt driver-evaluator node (mono-instance) or scenario node
        (multi-instance).

        Arguments:
            io_type_in (bool): whether to take inputs into account
            io_type_out (bool): whether to take outputs into account
            strip_first_ns (bool): whether to strip the scenario name (multi-instance case) from the variable name
        '''

        possible_in_values, possible_out_values = set(), set()
        # scenarios contains all the built sub disciplines (proxy_disciplines does NOT in flatten mode)
        for scenario_disc in self.scenarios:
            analyzed_disc = scenario_disc
            possible_in_values_full, possible_out_values_full = self.fill_possible_values(
                analyzed_disc, io_type_in=io_type_in, io_type_out=io_type_out)
            possible_in_values_full, possible_out_values_full = self.find_possible_values(analyzed_disc,
                                                                                          possible_in_values_full,
                                                                                          possible_out_values_full,
                                                                                          io_type_in=io_type_in,
                                                                                          io_type_out=io_type_out)
            # strip the scenario name to have just one entry for repeated variables in scenario instances
            if strip_first_ns:
                possible_in_values_full = [_var.split('.', 1)[-1] for _var in possible_in_values_full]
                possible_out_values_full = [_var.split('.', 1)[-1] for _var in possible_out_values_full]
            possible_in_values.update(possible_in_values_full)
            possible_out_values.update(possible_out_values_full)

        disc_in = self.get_data_in()
        if possible_in_values and io_type_in:

            # Convert sets into lists
            possible_in_values = list(possible_in_values)
            # these sorts are just for aesthetics
            possible_in_values.sort()
            default_in_dataframe = pd.DataFrame({'selected_input': [False for _ in possible_in_values],
                                                 'full_name': possible_in_values})

            eval_input_new_dm = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            eval_inputs_f_name = self.get_var_full_name(self.EVAL_INPUTS, disc_in)

            if eval_input_new_dm is None:
                self.dm.set_data(eval_inputs_f_name,
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
                        index]  # this will filter variables that are not inputs of the subprocess
                    if self.MULTIPLIER_PARTICULE in name:
                        default_dataframe = default_dataframe.append(
                            pd.DataFrame({'selected_input': [already_set_values[index]],
                                          'full_name': [name]}), ignore_index=True)
                self.dm.set_data(eval_inputs_f_name,
                                 'value', default_dataframe, check_value=False)

        if possible_out_values and io_type_out:
            possible_out_values = list(possible_out_values)
            possible_out_values.sort()
            default_out_dataframe = pd.DataFrame({'selected_output': [False for _ in possible_out_values],
                                                  'full_name': possible_out_values,
                                                  'output_name': [None for _ in possible_out_values]})
            eval_output_new_dm = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
            eval_outputs_f_name = self.get_var_full_name(self.EVAL_OUTPUTS, disc_in)
            if eval_output_new_dm is None:
                self.dm.set_data(eval_outputs_f_name,
                                 'value', default_out_dataframe, check_value=False)
            # check if the eval_inputs need to be updated after a subprocess configure
            elif set(eval_output_new_dm['full_name'].tolist()) != (set(default_out_dataframe['full_name'].tolist())):
                self.check_eval_io(eval_output_new_dm['full_name'].tolist(), default_out_dataframe['full_name'].tolist(),
                                   is_eval_input=False)
                default_dataframe = copy.deepcopy(default_out_dataframe)
                already_set_names = eval_output_new_dm['full_name'].tolist()
                already_set_values = eval_output_new_dm['selected_output'].tolist()
                if 'output_name' in eval_output_new_dm.columns:
                    # TODO: maybe better to repair tests than to accept default, in particular for data integrity check
                    already_set_out_names = eval_output_new_dm['output_name'].tolist()
                else:
                    already_set_out_names = [None for _ in already_set_names]
                for index, name in enumerate(already_set_names):
                    default_dataframe.loc[default_dataframe['full_name'] == name,
                    ['selected_output', 'output_name']] = \
                        (already_set_values[index], already_set_out_names[index])
                self.dm.set_data(eval_outputs_f_name,
                                 'value', default_dataframe, check_value=False)
