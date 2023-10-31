'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/03-2023/11/08 Copyright 2023 Capgemini

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
import copy
import pandas as pd

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.proxy_sample_generator import ProxySampleGenerator
from sostrades_core.execution_engine.mdo_discipline_driver_wrapp import MDODisciplineDriverWrapp
from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.tools.gather.gather_tool import check_eval_io, get_eval_output
from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
from sostrades_core.tools.builder_info.builder_info_functions import get_ns_list_in_builder_list
from sostrades_core.tools.eval_possible_values.eval_possible_values import find_possible_values
from sostrades_core.execution_engine.gather_discipline import GatherDiscipline


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
            |_ EVAL_OUTPUTS (namespace: NS_DRIVER, structuring, dynamic : builder_mode == self.MONO_INSTANCE)
            |_ SAMPLES_DF (namespace: NS_DRIVER, dynamic: len(selected_inputs) > 0 and len(selected_outputs) > 0 )
            |_ 'n_processes' (dynamic : builder_mode == self.MONO_INSTANCE)         
            |_ 'wait_time_between_fork' (dynamic : builder_mode == self.MONO_INSTANCE)

        |_ DESC_OUT
            |_ samples_inputs_df (namespace: NS_DRIVER, dynamic: builder_mode == self.MONO_INSTANCE)
            |_ <var>_dict (internal namespace 'ns_doe', dynamic: len(selected_inputs) > 0 and len(selected_outputs) > 0
            and gather_outputs not empty, for <var> in gather_outputs)

    2) Description of DESC parameters:
        |_ DESC_IN
            |_ INSTANCE_REFERENCE 
                |_ REFERENCE_MODE 
                |_ REFERENCE_SCENARIO_NAME  #TODO
            |_ EVAL_INPUTS  #NO NEED
            |_ EVAL_OUTPUTS #FOR GATHER MODE
            |_ GENERATED_SAMPLES #TO DELETE
            |_ SCENARIO_DF #TO DELETE
            |_ SAMPLES_DF
            |_ 'n_processes' 
            |_ 'wait_time_between_fork'            
       |_ DESC_OUT
            |_ samples_inputs_df  #TO DELETE
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

    EVAL_INPUTS = SampleGeneratorWrapper.EVAL_INPUTS

    NS_DRIVER = SampleGeneratorWrapper.NS_DRIVER

    SAMPLES_DF = SampleGeneratorWrapper.SAMPLES_DF
    SAMPLES_DF_DESC = SampleGeneratorWrapper.SAMPLES_DF_DESC
    SELECTED_SCENARIO = SampleGeneratorWrapper.SELECTED_SCENARIO
    SCENARIO_NAME = SampleGeneratorWrapper.SCENARIO_NAME
    WITH_SAMPLE_GENERATOR = 'with_sample_generator'
    WITH_SAMPLE_GENERATOR_DESC = {
        ProxyDiscipline.TYPE: 'bool',
        ProxyDiscipline.DEFAULT: False,
        ProxyDiscipline.STRUCTURING: True,
    }

    GATHER_DEFAULT_SUFFIX = GatherDiscipline.GATHER_SUFFIX
    EVAL_OUTPUTS = GatherDiscipline.EVAL_OUTPUTS

    GENERATED_SAMPLES = SampleGeneratorWrapper.GENERATED_SAMPLES

    DESC_IN = {SAMPLES_DF: SAMPLES_DF_DESC,
               WITH_SAMPLE_GENERATOR: WITH_SAMPLE_GENERATOR_DESC}

    ##
    ## To refactor instancce reference and subprocess import
    ##

    INSTANCE_REFERENCE = 'instance_reference'
    LINKED_MODE = 'linked_mode'
    COPY_MODE = 'copy_mode'
    REFERENCE_MODE = 'reference_mode'
    REFERENCE_MODE_POSSIBLE_VALUES = [LINKED_MODE, COPY_MODE]
    REFERENCE_SCENARIO_NAME = 'ReferenceScenario'

    MULTIPLIER_PARTICULE = '__MULTIPLIER__'
    with_modal = True
    default_process_builder_parameter_type = ProcessBuilderParameterType(
        None, None, 'Empty')
    SUB_PROCESS_INPUTS = DriverEvaluatorWrapper.SUB_PROCESS_INPUTS
    USECASE_DATA = 'usecase_data'
    if with_modal:
        DESC_IN[SUB_PROCESS_INPUTS] = {'type': ProxyDiscipline.PROC_BUILDER_MODAL,
                                       'structuring': True,
                                       'default': default_process_builder_parameter_type.to_data_manager_dict(),
                                       'user_level': 1,
                                       'optional': False}
    else:

        DESC_IN[USECASE_DATA] = {'type': 'dict',
                                 'structuring': True,
                                 'default': {},
                                 'user_level': 1,
                                 'optional': False}

    ##
    ## End to refactor
    ##
    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 associated_namespaces=None,
                 map_name=None,
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
        self.scenarios = []

        self.samples = None
        self.sample_generator_disc = None

        self.eval_process_builder = None
        self.eval_in_possible_values = []
        self.eval_in_list = None
        self.eval_out_list = None
        self.selected_inputs = []
        self.selected_outputs = []
        self.eval_out_names = []
        self.eval_out_type = []

        self.eval_out_list_size = []

        self.old_samples_df, self.old_scenario_df = ({}, {})
        self.scenario_list_valid = True
        self.scenario_list_integrity_msg = ''

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
        self.original_editable_dict_trade_variables = {}
        self.there_are_new_scenarios = False

        self.gather_names = None

    def _add_optional_shared_ns(self):
        """
        Add the shared namespace NS_DRIVER should it not exist.
        """
        # do the same for the shared namespace for coupling with the DriverEvaluator
        # also used to store gathered variables in multi-instance
        if self.NS_DRIVER not in self.ee.ns_manager.shared_ns_dict.keys():
            self.ee.ns_manager.add_ns(
                self.NS_DRIVER, self.ee.ns_manager.compose_local_namespace_value(self))

    def _get_disc_shared_ns_value(self):
        """
        Get the namespace NS_DRIVER used in the mono-instance case.
        """
        return self.ee.ns_manager.disc_ns_dict[self]['others_ns'][self.NS_DRIVER].get_value()

    def create_mdo_discipline_wrap(self, name, wrapper, wrapping_mode, logger: logging.Logger):
        """
        creation of mdo_discipline_wrapp by the proxy which in this case is a MDODisciplineDriverWrapp that will create
        a SoSMDODisciplineDriver at prepare_execution, i.e. a driver node that knows its subprocesses but manipulates
        them in a different way than a coupling.
        """
        self.mdo_discipline_wrapp = MDODisciplineDriverWrapp(name, logger.getChild("MDODisciplineDriverWrapp"), wrapper,
                                                             wrapping_mode)

    def configure(self):
        """
        Configure the DriverEvaluator layer
        """
        # set the scenarios references, for flattened subprocess configuration
        if self.builder_tool:
            self.scenarios = self.builder_tool.get_all_built_disciplines()
        else:
            self.scenarios = self.proxy_disciplines

        # 1. configure all children which need to be configured
        for disc in self.get_disciplines_to_configure():
            disc.configure()

        # configure current discipline DriverEvaluator
        # if self._data_in == {} or (self.get_disciplines_to_configure() == []
        # and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0:
        if self._data_in == {} or self.subprocess_is_configured():
            # Call standard configure methods to set the process discipline
            # tree
            # 2. configure first as a generic discipline
            super().configure()
            # 3. specific configure depending on the driver
            self.configure_driver()
            self.configure_sample_generator()

        if self.subprocess_is_configured():
            self.update_data_io_with_subprocess_io()
            self.set_children_numerical_inputs()

    def configure_sample_generator(self):
        if self.sample_generator_disc:
            # TODO: remove eval_inputs from driver evaluator and activate this line
            self.sample_generator_disc.set_eval_in_possible_values(self.eval_in_possible_values)
            if not self.sample_generator_disc.is_configured():
                self.sample_generator_disc.configure()

    def update_data_io_with_subprocess_io(self):
        """
        Update the DriverEvaluator _data_in and _data_out with subprocess i/o so that grammar of the driver can be
        exploited for couplings etc.
        """
        # FIXME: check if move to mono-instance side as no longer really useful in multi
        self._restart_data_io_to_disc_io()
        for proxy_disc in self.proxy_disciplines:
            # if not isinstance(proxy_disc, ProxyDisciplineGather):
            subprocess_data_in = proxy_disc.get_data_io_with_full_name(
                self.IO_TYPE_IN, as_namespaced_tuple=True)
            subprocess_data_out = proxy_disc.get_data_io_with_full_name(
                self.IO_TYPE_OUT, as_namespaced_tuple=True)
            self.update_data_io_with_child(subprocess_data_in, subprocess_data_out)

    def configure_driver(self):
        """
        To be overloaded by drivers with specific configuration actions
        """
        pass

    def prepare_build(self):
        """
        Get the actual drivers of the subprocesses of the DriverEvaluator.
        """
        # NB: custom driver wrapper not implemented
        disc_in = self.get_data_in()
        if self.WITH_SAMPLE_GENERATOR in disc_in and self.get_sosdisc_inputs(self.WITH_SAMPLE_GENERATOR):
            if self.sample_generator_disc is None:
                self.sample_generator_disc = self.build_sample_generator_disc()
        elif self.sample_generator_disc is not None:
            self.clean_children([
                self.sample_generator_disc])  # TODO: check whether sufficient for removal of shared ns NS_SAMPLING --> cleaning test or GUI test
            self.sample_generator_disc = None
        return []

    def build_sample_generator_disc(self):
        # create the builder of a ProxySampleGenerator
        sampling_builder = self.ee.factory.create_sample_generator('SampleGenerator')
        # associate ns_sampling and ns_driver
        ns_sampling = self.ee.ns_manager.add_ns(SampleGeneratorWrapper.NS_SAMPLING, self._get_disc_shared_ns_value())
        sampling_builder.associate_namespaces(ns_sampling)
        # create discipline in factory as sister not daughter
        self.ee.factory.current_discipline = self.father_executor
        sampling_disc = sampling_builder.build()
        self.ee.factory.add_discipline(sampling_disc)
        # return discipline for association in driver
        return sampling_disc

    def prepare_execution(self):
        """
        Preparation of the GEMSEO process, including GEMSEO objects instantiation
        """
        # prepare_execution of proxy_disciplines as in coupling
        for disc in self.scenarios:
            disc.prepare_execution()
        # TODO : cache mgmt of children necessary ? here or in  SoSMDODisciplineDriver ?
        if self.mdo_discipline_wrapp is not None:
            super().prepare_execution()
            self.reset_subdisciplines_of_wrapper()
        else:
            self._update_status_dm(self.STATUS_DONE)

    def reset_subdisciplines_of_wrapper(self):
        '''

        Reset subdisciplines of the wrapper

        '''
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
            if proxy.mdo_discipline_wrapp is not None]})  # discs and couplings but not scenarios

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
        '''

        Get the disciplines to configure which are the self.scenarios in driver world

        '''
        return self._get_disciplines_to_configure(self.scenarios)

    def check_data_integrity(self):
        '''
        Check the data integrity of the input variables of the driver
        '''
        # checking for duplicates
        self.check_integrity_msg_list = []
        disc_in = self.get_data_in()
        if self.SAMPLES_DF in disc_in and not self.scenario_list_valid:
            self.dm.set_data(
                self.get_var_full_name(self.SAMPLES_DF, disc_in),
                self.CHECK_INTEGRITY_MSG, self.scenario_list_integrity_msg)

    def manage_import_inputs_from_sub_process(self, ref_discipline_full_name):
        """
        Method for import usecase option which will be refactored
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
                # TODO: Refactor with the US refactor reference mode
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
            splitted_key = key_to_unanonymize.split('.')
            if not (len(splitted_key) == 2 and splitted_key[-1] in ProxyCoupling.NUMERICAL_VAR_LIST) and splitted_key[
                -1] != 'residuals_history':
                converted_key = key_to_unanonymize.replace(
                    self.ee.STUDY_PLACEHOLDER_WITHOUT_DOT, new_study_placeholder)
                # see def __unanonymize_key  in execution_engine
                uc_d = {converted_key: value}
                input_dict_from_usecase.update(uc_d)
        return input_dict_from_usecase

    def set_eval_possible_values(self, io_type_in=True, io_type_out=True, strip_first_ns=False):
        '''
        Check recursively the disciplines in the subprocess in order to detect their inputs and outputs.
        Once all disciplines have been run through, set the possible values for eval_inputs and gather_outputs in the DM
        These are the variables names anonymized wrt driver-evaluator node (mono-instance) or scenario node
        (multi-instance).

        Arguments:
            io_type_in (bool): whether to take inputs into account
            io_type_out (bool): whether to take outputs into account
            strip_first_ns (bool): whether to strip the scenario name (multi-instance case) from the variable name
        '''

        possible_in_values, possible_out_values = find_possible_values(self, io_type_in=io_type_in,
                                                                       io_type_out=io_type_out,
                                                                       strip_first_ns=strip_first_ns)

        disc_in = self.get_data_in()
        if possible_in_values and io_type_in:

            # Convert sets into lists
            possible_in_values = list(possible_in_values)
            # these sorts are just for aesthetics
            possible_in_values.sort()
            self.eval_in_possible_values = possible_in_values

            eval_input_new_dm = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            eval_inputs_f_name = self.get_var_full_name(self.EVAL_INPUTS, disc_in)

            if eval_input_new_dm is None:
                self.dm.set_data(eval_inputs_f_name,
                                 'value', default_in_dataframe, check_value=False)
            # check if the eval_inputs need to be updated after a subprocess
            # configure
            elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
                error_msg = check_eval_io(eval_input_new_dm['full_name'].tolist(),
                                          default_in_dataframe['full_name'].tolist(),
                                          is_eval_input=True)
                if len(error_msg) > 0:
                    for msg in error_msg:
                        self.logger.warning(msg)
                default_dataframe = copy.deepcopy(default_in_dataframe)
                already_set_names = eval_input_new_dm['full_name'].tolist()
                already_set_values = eval_input_new_dm['selected_input'].tolist()
                for index, name in enumerate(already_set_names):
                    default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = \
                        already_set_values[
                            index]  # this will filter variables that are not inputs of the subprocess
                    if self.MULTIPLIER_PARTICULE in name:
                        default_dataframe = default_dataframe.append(
                            pd.DataFrame({'selected_input': [already_set_values[index]],
                                          'full_name': [name]}), ignore_index=True)
                self.dm.set_data(eval_inputs_f_name,
                                 'value', default_dataframe, check_value=False)

        if possible_out_values and io_type_out:
            # NB: if io_type_out then we are in mono_instance so it's driver's responsibility to do this
            # get already set eval_output
            eval_output_new_dm = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
            eval_outputs_f_name = self.get_var_full_name(self.EVAL_OUTPUTS, disc_in)

            # get all possible outputs and merge with current eval_output
            eval_output_df, error_msg = get_eval_output(possible_out_values, eval_output_new_dm)
            if len(error_msg) > 0:
                for msg in error_msg:
                    self.logger.warning(msg)
            self.dm.set_data(eval_outputs_f_name,
                             'value', eval_output_df, check_value=False)
