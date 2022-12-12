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

import copy
import pandas as pd
from numpy import NaN
import numpy as np

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.mdo_discipline_driver_wrapp import MDODisciplineDriverWrapp
from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
from gemseo.utils.compare_data_manager_tooling import dict_are_equal


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

    BUILDER_MODE = DriverEvaluatorWrapper.BUILDER_MODE
    MONO_INSTANCE = DriverEvaluatorWrapper.MONO_INSTANCE
    MULTI_INSTANCE = DriverEvaluatorWrapper.MULTI_INSTANCE
    REGULAR_BUILD = DriverEvaluatorWrapper.REGULAR_BUILD
    BUILDER_MODE_POSSIBLE_VALUES = DriverEvaluatorWrapper.BUILDER_MODE_POSSIBLE_VALUES
    SUB_PROCESS_INPUTS = DriverEvaluatorWrapper.SUB_PROCESS_INPUTS

    INSTANCE_REFERENCE = 'instance_reference'
    LINKED_MODE = 'linked_mode'
    COPY_MODE = 'copy_mode'
    REFERENCE_MODE = 'reference_mode'
    REFERENCE_MODE_POSSIBLE_VALUES = [LINKED_MODE, COPY_MODE]

    SCENARIO_DF = 'scenario_df'

    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'
    # with SampleGenerator, whether to activate and build all the sampled
    MAX_SAMPLE_AUTO_BUILD_SCENARIOS = 1024
    # scenarios by default or not. Set to None to always build.

    SUBCOUPLING_NAME = 'subprocess'
    EVAL_INPUT_TYPE = ['float', 'array', 'int', 'string']

    GENERATED_SAMPLES = SampleGeneratorWrapper.GENERATED_SAMPLES

    USECASE_DATA = 'usecase_data'

    # namespace for the [var]_dict outputs of the mono-instance evaluator.
    # since [var] are anonymized
    NS_DOE = 'ns_doe'
    # full names, set to root node to have [var]_dict appear in same node as [var]
    # shared namespace of the mono-instance evaluator for eventual couplings
    NS_EVAL = 'ns_eval'

    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 associated_namespaces=None,
                 map_name=None,
                 flatten_subprocess=False,
                 hide_coupling_in_driver=False):
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
        if cls_builder is not None:
            self.cls_builder = cls_builder
        else:
            raise Exception(
                'The driver evaluator builder must have a cls_builder to work')

        self.builder_tool = None

        self.map_name = map_name
        self.flatten_subprocess = flatten_subprocess
        self.hide_coupling_in_driver = hide_coupling_in_driver
        self.old_builder_mode = None
        self.eval_process_builder = None
        self.eval_in_list = None
        self.eval_out_list = None
        self.selected_outputs = []
        self.selected_inputs = []
        self.eval_out_type = []
        self.eval_out_list_size = []
        self.logger = get_sos_logger(f'{self.ee.logger.name}.DriverEvaluator')

        self.old_samples_df, self.old_scenario_df = ({}, {})
        self.scatter_list_valid = True

        self.previous_sub_process_usecase_name = 'Empty'
        self.previous_sub_process_usecase_data = {}
        # Possible values: 'No_SP_UC_Import', 'SP_UC_Import'
        self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

        self.old_ref_dict = {}
        # self.ref_changes_dict = {}

    def _add_optional_shared_ns(self):
        """
        Add the shared namespaces NS_DOE and NS_EVAL should they not exist.
        """
        # if NS_DOE does not exist in ns_manager, we create this new
        # namespace to store output dictionaries associated to eval_outputs
        if self.NS_DOE not in self.ee.ns_manager.shared_ns_dict.keys():
            self.ee.ns_manager.add_ns(self.NS_DOE, self.ee.study_name)
        # do the same for the shared namespace for coupling with the
        # DriverEvaluator
        if self.NS_EVAL not in self.ee.ns_manager.shared_ns_dict.keys():
            self.ee.ns_manager.add_ns(
                self.NS_EVAL, self.ee.ns_manager.compose_local_namespace_value(self))

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
        # if self._data_in == {} or (self.get_disciplines_to_configure() == []
        # and len(self.proxy_disciplines) != 0) or len(self.cls_builder) == 0:
        if self._data_in == {} or self.subprocess_is_configured():
            # Call standard configure methods to set the process discipline
            # tree
            super().configure()
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
        if self.BUILDER_MODE in disc_in:
            if self.get_sosdisc_inputs(self.BUILDER_MODE) == self.MONO_INSTANCE \
                    and 'eval_inputs' in disc_in and len(self.proxy_disciplines) > 0:
                self.set_eval_possible_values()
            elif self.get_sosdisc_inputs(self.BUILDER_MODE) == self.MULTI_INSTANCE and self.SCENARIO_DF in disc_in:
                self.configure_tool()
                self.configure_subprocesses_with_driver_input()

    def prepare_variables_to_propagate(self):

        # TODO: code below might need refactoring after reference_scenario
        # configuration fashion is decided upon
        scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
        instance_reference = self.get_sosdisc_inputs(self.INSTANCE_REFERENCE)
        # sce_df = copy.deepcopy(scenario_df)

        if instance_reference:
            # Addition of Reference Scenario
            scenario_df = scenario_df.append(
                {self.SELECTED_SCENARIO: True,
                    self.SCENARIO_NAME: 'ReferenceScenario'},
                ignore_index=True)

        # NB assuming that the scenario_df entries are unique otherwise there
        # is some intelligence to be added
        scenario_names = scenario_df[scenario_df[self.SELECTED_SCENARIO]
                                     == True][self.SCENARIO_NAME].values.tolist()
        trade_vars = []
        # check that all the input scenarios have indeed been built
        # (configuration sequence allows the opposite)
        if self.subprocesses_built(scenario_names):
            trade_vars = [col for col in scenario_df.columns if col not in
                          [self.SELECTED_SCENARIO, self.SCENARIO_NAME]]

        return scenario_df, instance_reference, trade_vars, scenario_names

    def configure_subprocesses_with_driver_input(self):
        """
        This function forces the trade variables values of the subprocesses in function of the driverevaluator input df.
        """

        scenario_df, instance_reference, trade_vars, scenario_names = self.prepare_variables_to_propagate()
        # PROPAGATE NON-TRADE VARIABLES VALUES FROM REFERENCE TO SUBDISCIPLINES
        if self.subprocesses_built(scenario_names):
            # PROPAGATE NON-TRADE VARIABLES VALUES FROM REFERENCE TO
            # SUBDISCIPLINES
            if instance_reference:
                scenario_names = scenario_names[:-1]
                ref_discipline = self.proxy_disciplines[self.get_reference_scenario_index()]
                ref_changes_dict, ref_dict = self.get_reference_non_trade_variables_changes(trade_vars)

                # Modification of read-only or editable depending on LINKED_MODE or COPY_MODE
                self.modify_editable_attribute_according_to_reference_mode(ref_discipline, scenario_names, ref_dict)
                # Propagation to other scenarios if necessary
                if ref_changes_dict:
                    self.propagate_reference_non_trade_variables_changes(ref_changes_dict, ref_dict, ref_discipline, scenario_names)
            # else:
            #     scenario_names = scenario_names

            # PROPAGATE TRADE VARIABLES VALUES FROM scenario_df
            # check that there are indeed variable changes input, with respect
            # to reference scenario
            if trade_vars:
                driver_evaluator_ns = self.ee.ns_manager.get_local_namespace_value(
                    self)
                scenarios_data_dict = {}
                for sc in scenario_names:
                    # assuming it is unique # TODO: as index?
                    sc_row = scenario_df[scenario_df[self.SCENARIO_NAME]
                                         == sc].iloc[0]
                    for var in trade_vars:
                        var_full_name = self.ee.ns_manager.compose_ns(
                            [driver_evaluator_ns, sc, var])
                        scenarios_data_dict[var_full_name] = sc_row.loc[var]
                if scenarios_data_dict and self.subprocess_is_configured():
                    # push to dm
                    # TODO: should also alter associated disciplines' reconfig.
                    # flags for structuring ? TO TEST
                    self.ee.dm.set_values_from_dict(scenarios_data_dict)

    # def set_reference_trade_variables_in_scenario_df(self, sce_df):
    #
    #     var_names = [col for col in sce_df.columns if col not in
    #                  [self.SELECTED_SCENARIO, self.SCENARIO_NAME]]
    #
    #     index_ref_disc = self.get_reference_scenario_index()
    #     for var in var_names:
    #         short_name_var = var.split(".")[-1]
    #         for subdisc in self.proxy_disciplines[index_ref_disc].proxy_disciplines:
    #             if short_name_var in subdisc.get_data_in():
    #                 value_var = subdisc.get_sosdisc_inputs(short_name_var)
    #                 sce_df.at[sce_df.loc[sce_df[self.SCENARIO_NAME] == 'ReferenceScenario'].index, var] = value_var
    #
    #     return sce_df
    # def set_reference_trade_variables_in_scenario_df(self, sce_df):
    #
    #     var_names = [col for col in sce_df.columns if col not in
    #                  [self.SELECTED_SCENARIO, self.SCENARIO_NAME]]
    #
    #     index_ref_disc = self.get_reference_scenario_index()
    #     # for var in var_names:
    #     #    short_name_var = var.split(".")[-1]
    #     #    for subdisc in self.proxy_disciplines[index_ref_disc].proxy_disciplines:
    #     #        if short_name_var in subdisc.get_data_in():
    #     #            value_var = subdisc.get_sosdisc_inputs(short_name_var)
    #     #            sce_df.at[sce_df.loc[sce_df[self.SCENARIO_NAME]
    #     #                                 == 'ReferenceScenario'].index, var] = value_var
    #     # TODO
    #     # This is with error in case value_var is a list-like object (numpy array, list, set, tuple etc.)
    #     # https://stackoverflow.com/questions/48000225/must-have-equal-len-keys-and-value-when-setting-with-an-iterable
    #     # Example variable z = array([1., 1.]) of sellar put in trade variables
    #     return sce_df

    def get_reference_scenario_index(self):
        index_ref = 0
        for disc in self.proxy_disciplines:
            if disc.sos_name == 'ReferenceScenario':
                break
            else:
                index_ref += 1
        return index_ref

    def check_if_there_are_reference_variables_changes(self):

        scenario_df, instance_reference, trade_vars, scenario_names = self.prepare_variables_to_propagate()

        ref_changes_dict = {}
        if self.subprocesses_built(scenario_names):
            if instance_reference:
                ref_changes_dict, ref_dict = self.get_reference_non_trade_variables_changes(
                    trade_vars)

        return ref_changes_dict

    def get_reference_non_trade_variables_changes(self, trade_vars):

        ref_discipline = self.proxy_disciplines[self.get_reference_scenario_index(
        )]

        # Take reference scenario non-trade variables (num and non-num) and its
        # values
        ref_dict = {}
        for key in ref_discipline.get_input_data_names():
            if all(key.split(ref_discipline.sos_name + '.')[-1] != trade_var for trade_var in trade_vars):
                ref_dict[key] = ref_discipline.ee.dm.get_value(key)

        # Check if reference values have changed and select only those which
        # have changed
        ref_changes_dict = {}
        for key in ref_dict.keys():
            if key in self.old_ref_dict.keys():
                if isinstance(ref_dict[key], pd.DataFrame):
                    if not ref_dict[key].equals(self.old_ref_dict[key]):
                        ref_changes_dict[key] = ref_dict[key]
                elif isinstance(ref_dict[key], np.ndarray):
                    if not (np.array_equal(ref_dict[key], self.old_ref_dict[key])):
                        ref_changes_dict[key] = ref_dict[key]
                else:
                    if ref_dict[key] != self.old_ref_dict[key]:
                        ref_changes_dict[key] = ref_dict[key]
            else:
                ref_changes_dict[key] = ref_dict[key]

        return ref_changes_dict, ref_dict

    def propagate_reference_non_trade_variables_changes(self, ref_changes_dict, ref_dict, ref_discipline, scenario_names_to_propagate):

        if ref_changes_dict:
            self.old_ref_dict = copy.deepcopy(ref_dict)

        # Build other scenarios variables and values dict from reference
        dict_to_propagate = self.transform_dict_from_reference_to_other_scenarios(ref_discipline,
                                                                                  scenario_names_to_propagate,
                                                                                  ref_changes_dict)
        # Propagate other scenarios variables and values
        self.ee.dm.set_values_from_dict(dict_to_propagate)

    def modify_editable_attribute_according_to_reference_mode(self,ref_discipline,scenario_names_to_propagate,ref_dict):

        scenarios_non_trade_vars_dict = self.transform_dict_from_reference_to_other_scenarios(ref_discipline,
                                                                                              scenario_names_to_propagate,
                                                                                              ref_dict)
        if self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.LINKED_MODE:
            for key in scenarios_non_trade_vars_dict.keys():
                # if 'ReferenceScenario' in key:
                #     pass
                # else:
                self.ee.dm.set_data(key, 'editable', False)
        elif self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.COPY_MODE:
            for key in scenarios_non_trade_vars_dict.keys():
                # if 'ReferenceScenario' in key:
                #     pass
                # else:
                self.ee.dm.set_data(key, 'editable', True)

    def transform_dict_from_reference_to_other_scenarios(self, ref_discipline, scenario_names, dict_from_ref):

        transformed_to_other_scenarios_dict = {}
        for key in dict_from_ref.keys():
            for sc in scenario_names:
                if ref_discipline.sos_name in key and self.sos_name in key:
                    new_key = key.split(self.sos_name, 1)[0] + self.sos_name + '.' + sc + \
                        key.split(self.sos_name,
                                  1)[-1].split(ref_discipline.sos_name, 1)[-1]
                elif ref_discipline.sos_name in key and not self.sos_name in key:
                    new_key = key.split(ref_discipline.sos_name, 1)[
                        0] + sc + key.split(ref_discipline.sos_name, 1)[-1]
                else:
                    new_key = key
                if self.dm.check_data_in_dm(new_key):
                    transformed_to_other_scenarios_dict[new_key] = dict_from_ref[key]

        return transformed_to_other_scenarios_dict

        # # Take non-trade variables values from subdisciplines of reference scenario
        # for subdisc in self.proxy_disciplines[index_ref_disc].proxy_disciplines:
        #     if subdisc.__class__.__name__ == 'ProxyDiscipline':
        #         # For ProxyDiscipline --> Propagation of non-trade variables
        #         self.propagate_non_trade_variables_of_proxy_discipline(subdisc, trade_vars)
        #     elif subdisc.__class__.__name__ == 'ProxyDriverEvaluator':
        #         # For ProxyDriverEvaluator --> Propagation of non-trade variables from ReferenceScenario (recursivity)
        #         subdisc.set_non_trade_variables_from_reference_scenario(trade_vars)
        #     else:
        #         # For ProxyCoupling... --> Propagation of its subdisciplines variables (recursively)
        #         self.propagate_non_trade_variables_of_proxy_coupling(subdisc, trade_vars)

    # def propagate_non_trade_variables_of_proxy_discipline(self, subdiscipline, trade_vars):
    #
    #     non_trade_var_dict_ref_to_propagate = {}
    #     non_trade_var_dict_not_ref_scenario = {}
    #     # Get non-numerical variables full name and values from reference
    #     non_num_var_dict = subdiscipline.get_non_numerical_variables_and_values_dict()
    #
    #     # If non-numerical variables have been set, select non-trade variables from them
    #     if all(value == None for value in non_num_var_dict.values()):
    #         pass
    #     else:
    #         for key in non_num_var_dict:  # Non-numerical variables
    #             if all(key.split('.ReferenceScenario.')[-1] != trade_var for trade_var in
    #                    trade_vars):  # Here non-trade variables are taken from non-numerical values
    #                 non_trade_var_dict_ref_to_propagate[key] = non_num_var_dict[key]
    #
    #     # Adapt non-trade variables and values from reference to full name of other scenarios
    #     if non_trade_var_dict_ref_to_propagate:
    #         for key in non_trade_var_dict_ref_to_propagate.keys():
    #             for sc in self.scenario_names[:-1]:
    #                 if 'ReferenceScenario' in key:
    #                     new_key = key.rsplit('ReferenceScenario', 1)[0] + sc + key.rsplit('ReferenceScenario', 1)[-1]
    #                     # new_key = driver_evaluator_ns + "." + sc + key.split('ReferenceScenario')[-1]
    #                 else:
    #                     new_key = key
    #                 non_trade_var_dict_not_ref_scenario[new_key] = non_trade_var_dict_ref_to_propagate[key]
    #
    #     if non_trade_var_dict_not_ref_scenario:
    #         self.ee.dm.set_values_from_dict(non_trade_var_dict_not_ref_scenario)
    #
    # def propagate_non_trade_variables_of_proxy_coupling(self, subcoupling, trade_vars):
    #     for subsubdisc in subcoupling.proxy_disciplines:
    #         if subsubdisc.__class__.__name__ == 'ProxyDiscipline':
    #             # For ProxyDiscipline --> Propagation of non-trade variables
    #             self.propagate_non_trade_variables_of_proxy_discipline(subsubdisc, trade_vars)
    #         elif subsubdisc.__class__.__name__ == 'ProxyDriverEvaluator':
    #             # For ProxyDriverEvaluator --> Propagation of non-trade variables from ReferenceScenario (recursivity)
    #             subsubdisc.set_non_trade_variables_from_reference_scenario(trade_vars)
    #         else:
    #             # For ProxyCoupling... --> Propagation of its subdisciplines variables (recursively)
    #             self.propagate_non_trade_variables_of_proxy_coupling(subsubdisc, trade_vars)

    def subprocesses_built(self, scenario_names):
        """
        Check whether the subproxies built are coherent with the input list scenario_names.

        Arguments:
            scenario_names (list[string]): expected names of the subproxies.
        """
        # TODO: if scenario_names is None get it?
        proxies_names = [disc.sos_name for disc in self.proxy_disciplines]
        # # assuming self.coupling_per_scenario is true so bock below commented
        # if self.coupling_per_scenario:
        #     builder_names = [b.sos_name for b in self.cls_builder]
        #     expected_proxies_names = []
        #     for sc_name in scenario_names:
        #         for builder_name in builder_names:
        #             expected_proxies_names.append(self.ee.ns_manager.compose_ns([sc_name, builder_name]))
        #     return set(expected_proxies_names) == set(proxies_names)
        # else:
        # return set(proxies_names) == set(scenario_names)
        return proxies_names != [] and set(proxies_names) == set(scenario_names)

    def setup_sos_disciplines(self):
        """
        Dynamic inputs and outputs of the DriverEvaluator
        """
        if self.BUILDER_MODE in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs(self.BUILDER_MODE)
            disc_in = self.get_data_in()
            if builder_mode == self.MULTI_INSTANCE:
                self.build_inst_desc_io_with_scenario_df()
                if self.GENERATED_SAMPLES in disc_in:
                    generated_samples = self.get_sosdisc_inputs(
                        self.GENERATED_SAMPLES)
                    generated_samples_dict = {
                        self.GENERATED_SAMPLES: generated_samples}
                    scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
                    # checking whether generated_samples has changed
                    # NB also doing nothing with an empty dataframe, which means sample needs to be regenerated to renew
                    # scenario_df on 2nd config. The reason of this choice is that using an optional generated_samples
                    # gives problems with structuring variables checks leading
                    # to incomplete configuration sometimes
                    if not (generated_samples.empty and not self.old_samples_df) \
                            and not dict_are_equal(generated_samples_dict, self.old_samples_df):
                        # checking whether the dataframes are already coherent in which case the changes come probably
                        # from a load and there is no need to crush the truth
                        # values
                        if not generated_samples.equals(
                                scenario_df.drop([self.SELECTED_SCENARIO, self.SCENARIO_NAME], 1)):
                            # TODO: could overload struct. var. check to spare this deepcopy (only if generated_samples
                            # remains as a DriverEvaluator input, othrwise
                            # another sample change check logic is needed)
                            self.old_samples_df = copy.deepcopy(
                                generated_samples_dict)
                            # we crush old scenario_df and propose a df with
                            # all scenarios imposed by new sample, all
                            # de-activated
                            scenario_df = pd.DataFrame(
                                columns=[self.SELECTED_SCENARIO, self.SCENARIO_NAME])
                            scenario_df = pd.concat(
                                [scenario_df, generated_samples], axis=1)
                            n_scenarios = len(scenario_df.index)
                            # check whether the number of generated scenarios
                            # is not too high to auto-activate them
                            if self.MAX_SAMPLE_AUTO_BUILD_SCENARIOS is None or n_scenarios <= self.MAX_SAMPLE_AUTO_BUILD_SCENARIOS:
                                scenario_df[self.SELECTED_SCENARIO] = True
                            else:
                                self.logger.warn(
                                    f'Sampled over {self.MAX_SAMPLE_AUTO_BUILD_SCENARIOS} scenarios, please select which to build. ')
                                scenario_df[self.SELECTED_SCENARIO] = False
                            scenario_name = scenario_df[self.SCENARIO_NAME]
                            for i in scenario_name.index.tolist():
                                scenario_name.iloc[i] = 'scenario_' + \
                                                        str(i + 1)
                            self.logger.info(
                                'Generated sample has changed, updating scenarios to select.')
                            self.dm.set_data(self.get_var_full_name(self.SCENARIO_DF, disc_in),
                                             'value', scenario_df, check_value=False)

            elif builder_mode == self.MONO_INSTANCE:
                # TODO: clean code below with class variables etc.
                dynamic_inputs = {'eval_inputs': {'type': 'dataframe',
                                                  'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                           'full_name': ('string', None, False)},
                                                  'dataframe_edition_locked': False,
                                                  'structuring': True,
                                                  'visibility': self.SHARED_VISIBILITY,
                                                  'namespace': self.NS_EVAL},
                                  'eval_outputs': {'type': 'dataframe',
                                                   'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                                            'full_name': ('string', None, False)},
                                                   'dataframe_edition_locked': False,
                                                   'structuring': True, 'visibility': self.SHARED_VISIBILITY,
                                                   'namespace': self.NS_EVAL},
                                  'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
                                  'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0}
                                  }

                dynamic_outputs = {
                    'samples_inputs_df': {'type': 'dataframe', 'unit': None, 'visibility': self.SHARED_VISIBILITY,
                                          'namespace': self.NS_EVAL}
                }

                selected_inputs_has_changed = False
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
                                                                                           'namespace': self.NS_DOE}})
                        dynamic_inputs.update(self._get_dynamic_inputs_doe(
                            disc_in, selected_inputs_has_changed))
                self.add_inputs(dynamic_inputs)
                self.add_outputs(dynamic_outputs)
            elif builder_mode == self.REGULAR_BUILD:
                pass  # regular build requires no specific dynamic inputs
            elif builder_mode is None:
                pass
            else:
                raise ValueError(
                    f'Wrong builder mode input in {self.sos_name}')
        # after managing the different builds inputs, we do the setup_sos_disciplines of the wrapper in case it is
        # overload, e.g. in the case of a custom driver_wrapper_cls (with DriverEvaluatorWrapper this does nothing)
        # super().setup_sos_disciplines() # TODO: manage custom driver wrapper
        # case

        # check and import usecase
        self.manage_import_inputs_from_sub_process()

    def prepare_build(self):
        """
        Get the actual drivers of the subprocesses of the DriverEvaluator.
        """
        # TODO: make me work with custom driver
        builder_list = []
        if self.BUILDER_MODE in self.get_data_in():
            builder_mode = self.get_sosdisc_inputs(self.BUILDER_MODE)
            builder_mode_has_changed = builder_mode != self.old_builder_mode
            if builder_mode_has_changed:
                self.clean_children()
                self.clean_sub_builders()
                if self.old_builder_mode == self.MONO_INSTANCE:
                    self.eval_process_builder = None
                elif self.old_builder_mode == self.MULTI_INSTANCE:
                    self.builder_tool = None
                self.old_builder_mode = copy.copy(builder_mode)
            if builder_mode == self.MULTI_INSTANCE:
                builder_list = self.prepare_multi_instance_build()
            elif builder_mode == self.MONO_INSTANCE:
                builder_list = self.prepare_mono_instance_build()
            elif builder_mode == self.REGULAR_BUILD:
                builder_list = super().prepare_build()
            elif builder_mode is None:
                pass
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

        # driverevaluator subprocess
        wrapper.attributes.update({'sub_mdo_disciplines': [
            proxy.mdo_discipline_wrapp.mdo_discipline for proxy in self.proxy_disciplines
            if proxy.mdo_discipline_wrapp is not None]})  # discs and couplings but not scatters

        # specific to mono-instance
        if self.BUILDER_MODE in self.get_data_in() and self.get_sosdisc_inputs(self.BUILDER_MODE) == self.MONO_INSTANCE and self.eval_in_list is not None:
            eval_attributes = {'eval_in_list': self.eval_in_list,
                               'eval_out_list': self.eval_out_list,
                               'reference_scenario': self.get_x0(),
                               'activated_elems_dspace_df': [[True, True]
                                                             if self.ee.dm.get_data(var, 'type') == 'array' else [True]
                                                             for var in self.eval_in_list],
                               # TODO: Array dimensions greater than 2?
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
        disc_in = self.get_data_in()
        if self.BUILDER_MODE in disc_in:
            if self.get_sosdisc_inputs(self.BUILDER_MODE) == self.MULTI_INSTANCE:
                if self.INSTANCE_REFERENCE in disc_in and self.get_sosdisc_inputs(self.INSTANCE_REFERENCE):
                    if self.SCENARIO_DF in disc_in:
                        # and self.sub_proc_import_usecase_status ==
                        # 'No_SP_UC_Import'
                        return super().is_configured() and self.subprocess_is_configured() and not self.check_if_there_are_reference_variables_changes()

        return super().is_configured() and self.subprocess_is_configured()

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
        return self.get_disciplines_to_configure() == []  # or len(self.cls_builder) == 0

    def prepare_multi_instance_build(self):
        """
        Call the tool to build the subprocesses in multi-instance builder mode.
        """
        self.build_tool()
        # Tool is building disciplines for the driver on behalf of the driver name
        # no further disciplines needed to be builded by the evaluator
        return []

    def build_inst_desc_io_with_scenario_df(self):
        '''
        Complete inst_desc_in with scenario_df
        '''
        dynamic_inputs = {self.SCENARIO_DF: {
            self.TYPE: 'dataframe',
            self.DEFAULT: pd.DataFrame(columns=[self.SELECTED_SCENARIO, self.SCENARIO_NAME]),
            self.DATAFRAME_DESCRIPTOR: {self.SELECTED_SCENARIO: ('bool', None, True),
                                        self.SCENARIO_NAME: ('string', None, True)},
            self.DATAFRAME_EDITION_LOCKED: False,
            self.EDITABLE: True,
            self.STRUCTURING: True}}  # TODO: manage variable columns for (non-very-simple) multiscenario cases

        dynamic_inputs.update({self.INSTANCE_REFERENCE:
                               {SoSWrapp.TYPE: 'bool',
                                SoSWrapp.DEFAULT: False,
                                SoSWrapp.POSSIBLE_VALUES: [True, False],
                                SoSWrapp.STRUCTURING: True}})

        disc_in = self.get_data_in()
        if self.INSTANCE_REFERENCE in disc_in:
            instance_reference = self.get_sosdisc_inputs(
                self.INSTANCE_REFERENCE)
            if instance_reference:
                dynamic_inputs.update({self.REFERENCE_MODE:
                                       {SoSWrapp.TYPE: 'string',
                                        SoSWrapp.DEFAULT: self.LINKED_MODE,
                                        SoSWrapp.POSSIBLE_VALUES: self.REFERENCE_MODE_POSSIBLE_VALUES,
                                        SoSWrapp.STRUCTURING: True}})

        dynamic_inputs.update({self.GENERATED_SAMPLES: {'type': 'dataframe',
                                                        'dataframe_edition_locked': True,
                                                        'structuring': True,
                                                        'unit': None,
                                                        # 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                        # 'namespace': 'ns_sampling',
                                                        'default': pd.DataFrame(),
                                                        # self.OPTIONAL:
                                                        # True,
                                                        self.USER_LEVEL: 3
                                                        }})
        self.add_inputs(dynamic_inputs)
        # so that eventual mono-instance outputs get clear
        self.add_outputs({})

    def configure_tool(self):
        '''
        Instantiate the tool if it does not and prepare it with data that he needs (the tool know what he needs)
        '''
        if self.builder_tool is None:
            builder_tool_cls = self.ee.factory.create_scatter_tool_builder(
                'scatter_tool', map_name=self.map_name, hide_coupling_in_driver=self.hide_coupling_in_driver)
            self.builder_tool = builder_tool_cls.instantiate()
            self.builder_tool.associate_tool_to_driver(
                self, cls_builder=self.cls_builder, associated_namespaces=self.associated_namespaces)
        self.scatter_list_valid = self.check_scatter_list_validity()
        if self.scatter_list_valid:
            self.builder_tool.prepare_tool()

    def build_tool(self):
        if self.builder_tool is not None and self.scatter_list_valid:
            self.builder_tool.build()

    def check_scatter_list_validity(self):
        # TODO: include as a case of check data integrity ?
        # checking for duplicates
        if self.SCENARIO_DF in self.get_data_in():
            scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
            scenario_names = scenario_df[scenario_df[self.SELECTED_SCENARIO]
                                         == True][self.SCENARIO_NAME].values.tolist()
            set_sc_names = set(scenario_names)
            if len(scenario_names) != len(set_sc_names):
                repeated_elements = [
                    sc for sc in set_sc_names if scenario_names.count(sc) > 1]
                msg = 'Cannot activate several scenarios with the same name (' + \
                      repeated_elements[0]
                for sc in repeated_elements[1:]:
                    msg += ', ' + sc
                msg += ').'
                self.logger.error(msg)
                # raise Exception(msg)
                return False
        # in any other case the list is valid
        return True

    # MONO INSTANCE PROCESS
    def _get_disc_shared_ns_value(self):
        """
        Get the namespace ns_eval used in the mono-instance case.
        """
        return self.ee.ns_manager.disc_ns_dict[self]['others_ns'][self.NS_EVAL].get_value()

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
            if self.flatten_subprocess:
                disc_builder = None
            else:
                disc_builder = self.create_sub_builder_coupling(
                    self.SUBCOUPLING_NAME, self.cls_builder)
                self.hide_coupling_in_driver_for_display(disc_builder)

        self.eval_process_builder = disc_builder

    def hide_coupling_in_driver_for_display(self, disc_builder):
        '''
        Set the display_value of the sub coupling to the display_value of the driver 
        (if no display_value filled the display_value is the simulation value)
        '''
        driver_display_value = self.ee.ns_manager.get_local_namespace(
            self).get_display_value()
        self.ee.ns_manager.add_display_ns_to_builder(
            disc_builder, driver_display_value)

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
                                                                                      possible_in_values_full,
                                                                                      possible_out_values_full)

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
        # configuration ? (<-> config. graph)
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
                                         'namespace': self.NS_EVAL
                                         }}

        # This reflects 'samples_df' dynamic input has been configured and that
        # eval_inputs have changed
        if 'samples_df' in disc_in and selected_inputs_has_changed:

            if disc_in['samples_df']['value'] is not None:
                from_samples = list(disc_in['samples_df']['value'].keys())
                from_eval_inputs = list(default_custom_dataframe.keys())
                final_dataframe = pd.DataFrame(
                    None, columns=self.selected_inputs)

                len_df = 1
                for element in from_eval_inputs:
                    if element in from_samples:
                        len_df = len(disc_in['samples_df']['value'])

                for element in from_eval_inputs:
                    if element in from_samples:
                        final_dataframe[element] = disc_in['samples_df']['value'][element]

                    else:
                        final_dataframe[element] = [NaN for _ in range(len_df)]

                disc_in['samples_df']['value'] = final_dataframe
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

    def clean_sub_builders(self):
        '''
        Clean sub_builders as they were at initialization especially for their associated namespaces
        '''
        for builder in self.cls_builder:
            # delete all associated namespaces
            builder.delete_all_associated_namespaces()
            # set back all associated namespaces that was at the init of the
            # evaluator
            builder.add_namespace_list_in_associated_namespaces(
                self.associated_namespaces)

    def manage_import_inputs_from_sub_process(self):
        """
            Function needed in setup_sos_disciplines()
        """
        # Set sub_proc_import_usecase_status
        self.set_sub_process_usecase_status_from_user_inputs()

        disc_in = self.get_data_in()

        # Treat the case of SP_UC_Import
        if self.sub_proc_import_usecase_status == 'SP_UC_Import':
            # Get the anonymized dict
            if 1 == 0:  # TODO (when use of Modal)
                anonymize_input_dict_from_usecase = self.get_sosdisc_inputs(
                    self.SUB_PROCESS_INPUTS)[ProcessBuilderParameterType.USECASE_DATA]
            # (without use of Modal)
            anonymize_input_dict_from_usecase = self.get_sosdisc_inputs(
                self.USECASE_DATA)

            # LOAD REFERENCE of MULTI-INSTANCE MODE WITH USECASE DATA
            if self.INSTANCE_REFERENCE in disc_in:
                instance_reference = self.get_sosdisc_inputs(
                    self.INSTANCE_REFERENCE)

                if instance_reference:

                    is_ref_disc = False
                    ref_disc_name = ''
                    for disc in self.proxy_disciplines:  # PB : in flatten mode self.proxy_disciplines =[]
                        if disc.sos_name == 'ReferenceScenario':
                            is_ref_disc = True
                            ref_discipline_full_name = disc.get_disc_full_name()

                    if is_ref_disc:
                        # 1. Put anonymized dict in context (unanonymize) of the reference
                        # First identify the reference scenario
                        input_dict_from_usecase = self.put_anonymized_input_dict_in_sub_process_context(
                            anonymize_input_dict_from_usecase, ref_discipline_full_name)
                        # print(input_dict_from_usecase)
                        # self.ee.display_treeview_nodes(True)
                        # 2. load data in dm (# Push the data to the reference
                        # instance)

                        as_in_eev3 = False
                        if as_in_eev3:  # We get an infinite loop et never do the last in the sequence
                            self.ee.load_study_from_input_dict(
                                input_dict_from_usecase)
                        else:  # This is what was done before the bellow correction. It doesn't work with dynamic subproc or if a data kay is not yet in the dm
                            self.ee.dm.set_values_from_dict(
                                input_dict_from_usecase)

                        # 3. Update parameters
                        #     Set the status to 'No_SP_UC_Import'
                        self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
                        if 1 == 0:  # TODO (when use of Modal)
                            # Empty the anonymized dict in (when use of Modal)
                            sub_process_inputs_dict = self.get_sosdisc_inputs(
                                self.SUB_PROCESS_INPUTS)
                            sub_process_inputs_dict[ProcessBuilderParameterType.USECASE_DATA] = {
                            }
                            self.dm.set_data(f'{self.get_disc_full_name()}.{self.SUB_PROCESS_INPUTS}',
                                             self.VALUES, sub_process_inputs_dict, check_value=False)
                        if 1 == 0:  # TODO (when use of Modal)
                            # Consequently update the previous_sub_process_usecase_data
                            #     Empty the previous_sub_process_usecase_data
                            self.previous_sub_process_usecase_data = {}
                        else:
                            sub_process_usecase_data = self.get_sosdisc_inputs(
                                self.USECASE_DATA)
                            self.previous_sub_process_usecase_data = sub_process_usecase_data
                    else:  # TODO Should we have USECASE_DATA only available in Mono or Multi with instance_reference =True ??
                        pass
            else:
                # LOAD REFERENCE of MONO-INSTANCE MODE WITH USECASE DATA
                # LOAD ALL SCENARIOS of MULTI-INSTANCE MODE WITH USECASE DATA
                pass

    def set_sub_process_usecase_status_from_user_inputs(self):
        """
            State subprocess usecase import status
            The uscase is defined by its name and its anonimized dict
            Function needed in manage_import_inputs_from_sub_process()
        """
        disc_in = self.get_data_in()
        # With modal #TODO Activate it when proc builder web_api eev3 migrated
        # on eev4 and remove USECASE_DATA
        if self.SUB_PROCESS_INPUTS in disc_in:  # and self.sub_proc_build_status != 'Empty_SP'
            sub_process_inputs_dict = self.get_sosdisc_inputs(
                self.SUB_PROCESS_INPUTS)
            sub_process_usecase_name = sub_process_inputs_dict[
                ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_NAME]
            sub_process_usecase_data = sub_process_inputs_dict[ProcessBuilderParameterType.USECASE_DATA]
            if self.previous_sub_process_usecase_name != sub_process_usecase_name or self.previous_sub_process_usecase_data != sub_process_usecase_data:
                self.previous_sub_process_usecase_name = sub_process_usecase_name
                self.previous_sub_process_usecase_data = sub_process_usecase_data
                # not not sub_process_usecase_data True means it is not an
                # empty dictionary
                if sub_process_usecase_name != 'Empty' and not not sub_process_usecase_data:
                    self.sub_proc_import_usecase_status = 'SP_UC_Import'
            else:
                self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
        else:
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

        # Without modal #TODO Remove when proc builder web_api eev3 migrated on
        # eev4
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

    def put_anonymized_input_dict_in_sub_process_context(self, anonymize_input_dict_from_usecase, ref_discipline_full_name):
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

    def get_disc_label(self):

        return 'DriverEvaluator'
