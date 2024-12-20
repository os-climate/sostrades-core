'''
Copyright 2023 Capgemini

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
import copy

import numpy as np
import pandas as pd

from sostrades_core.execution_engine.builder_tools.scatter_tool import ScatterTool
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator


class ProxyMultiInstanceDriverException(Exception):
    pass


class ProxyMultiInstanceDriver(ProxyDriverEvaluator):
    '''
    Class for driver on multi instance mode
    '''
    _ontology_data = {
        'label': ' Multi-Instance Driver',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    DISPLAY_OPTIONS_POSSIBILITIES = ScatterTool.DISPLAY_OPTIONS_POSSIBILITIES

    #            display_options (optional): Dictionary of display_options for multiinstance mode (value True or False) with options :
    #             'hide_under_coupling' : Hide all disciplines created under the coupling at scenario name node for display purpose
    #             'hide_coupling_in_driver': Hide the coupling (scenario_name node) under the driver for display purpose
    #             'group_scenarios_under_disciplines' : Invert the order of scenario and disciplines for display purpose
    #                                                   Scenarios will be under discipline for the display treeview

    DISPLAY_OPTIONS_DEFAULT = {disp_option: False for disp_option in DISPLAY_OPTIONS_POSSIBILITIES}
    DISPLAY_OPTIONS = 'display_options'

    # with SampleGenerator, whether to activate and build all the sampled
    MAX_SAMPLE_AUTO_BUILD_SCENARIOS = 1024

    DESC_IN = {
        ProxyDriverEvaluator.INSTANCE_REFERENCE: {
            ProxyDriverEvaluator.TYPE: 'bool',
            ProxyDriverEvaluator.DEFAULT: False,
            ProxyDriverEvaluator.POSSIBLE_VALUES: [True, False],
            ProxyDriverEvaluator.STRUCTURING: True
        },
        DISPLAY_OPTIONS: {ProxyDriverEvaluator.TYPE: 'dict',
                          ProxyDriverEvaluator.STRUCTURING: True,
                          ProxyDriverEvaluator.DEFAULT: DISPLAY_OPTIONS_DEFAULT,
                          ProxyDriverEvaluator.SUBTYPE: {'dict': 'bool'}
                          }
    }

    DESC_IN.update(ProxyDriverEvaluator.DESC_IN)

    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 associated_namespaces=None,
                 map_name=None,
                 process_display_options=None
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
            process_display_options  [dict] still keep the possibility to modify display options through the process for archibuilder
        """
        super().__init__(sos_name, ee, cls_builder, driver_wrapper_cls, associated_namespaces, map_name)
        self.driver_eval_mode = self.DRIVER_EVAL_MODE_MULTI
        self.display_options = None
        if process_display_options is not None:
            self.display_options = process_display_options

    def setup_sos_disciplines(self):
        disc_in = self.get_data_in()
        dynamic_inputs_ref_mode = self.add_reference_mode(disc_in)
        dynamic_inputs_scatter_map = self.add_scatter_map_inputs()
        self.add_inputs(dynamic_inputs_ref_mode | dynamic_inputs_scatter_map)
    def configure_sample_generator(self):
        """
        Configuration of the associated sample generator discipline for multi-instance driver is as in the general case
        except that run-time sampling is not allowed as the sample is needed for scenarios configuration.
        """
        self.sample_generator_disc.force_sampling_at_configuration_time = True
        super().configure_sample_generator()

    def configure_driver(self):
        disc_in = self.get_data_in()
        if self.SAMPLES_DF in disc_in:
            self.configure_tool()
            self.configure_subprocesses_with_driver_input()
            self.set_eval_possible_values(
                io_type_out=False,
                strip_first_ns=True)

    def clean_children(self, list_children=None):
        '''

        Args:
            list_children: list of children to clean

        Overload clean_children method to clean self.scenarios instead of self.proxy_disciplines

        '''
        if list_children is None:
            list_children = self.scenarios
            list_children.extend(self.builder_tool.get_all_gather_disciplines())

        super().clean_children(list_children)

        self.scenarios = [disc for disc in self.scenarios if disc not in list_children]

    def create_discipline_wrap(self, name, wrapper, wrapping_mode, logger):
        """
        No need to create a DisciplineWrap in the multi instance case , the computation is delegated to the coupling discipline above the driver
        """
        pass

    def prepare_build(self):
        """
        Call the tool to build the subprocesses in multi-instance builder mode.
        """
        if self.get_data_in():
            self.build_tool()
            # Tool is building disciplines for the driver on behalf of the driver name
            # no further disciplines needed to be built by the evaluator
        return super().prepare_build()

    def is_configured(self):
        config_status = super().is_configured()
        disc_in = self.get_data_in()
        # TODO: to be improved with ref. instance refacto
        if self.INSTANCE_REFERENCE in disc_in and self.INSTANCE_REFERENCE in self.get_data_in():
            config_status = config_status and (
                not self.check_if_there_are_reference_variables_changes()) and (
                                    self.sub_proc_import_usecase_status == 'No_SP_UC_Import'
                            )
        return config_status

    def update_reference(self):
        return self.INSTANCE_REFERENCE in self.get_data_in()

    def add_reference_mode(self, disc_in):
        '''
        Add reference mode as dynamic input if we are using instance reference option
        '''
        dynamic_inputs = {}

        if self.INSTANCE_REFERENCE in disc_in:
            instance_reference = self.get_sosdisc_inputs(
                self.INSTANCE_REFERENCE)
            if instance_reference:
                dynamic_inputs.update({self.REFERENCE_MODE:
                                           {self.TYPE: 'string',
                                            # SoSWrapp.DEFAULT: self.LINKED_MODE,
                                            self.POSSIBLE_VALUES: self.REFERENCE_MODE_POSSIBLE_VALUES,
                                            self.STRUCTURING: True}})

        return dynamic_inputs

    def add_scatter_map_inputs(self):
        '''

        Add scatter map input variables to dynamic desc_in

        '''
        dynamic_inputs = {}
        # so that eventual mono-instance outputs get clear
        if self.builder_tool is not None:
            dynamic_inputs = self.builder_tool.get_dynamic_input_from_tool()

        return dynamic_inputs

    def configure_tool(self):
        '''
        Instantiate the tool if it does not and prepare it with data that he needs (the tool know what he needs)
        Prepare the tool only if data integrity is OK
        '''
        if self.builder_tool is None:
            builder_tool_cls = self.ee.factory.create_scatter_tool_builder(
                'scatter_tool', map_name=self.map_name)
            self.builder_tool = builder_tool_cls.instantiate()
            self.builder_tool.associate_tool_to_driver(
                self, cls_builder=self.cls_builder, associated_namespaces=self.associated_namespaces)
        self.check_data_integrity()
        if self.scatter_list_validity:
            self.builder_tool.prepare_tool()

    def build_tool(self):
        '''

        Build the tool if the driver data integrity is OK

        '''
        if self.builder_tool is not None and self.scatter_list_validity:
            self.builder_tool.build()

    def subprocesses_built(self, scenario_names):
        """
        Check whether the subproxies built are coherent with the input list scenario_names.

        Arguments:
            scenario_names (list[string]): expected names of the subproxies.
        """
        if self.builder_tool:
            proxies_names = self.builder_tool.get_all_built_disciplines_names()
            # return self.builder_tool.has_built and proxies_names
            if self.get_sosdisc_inputs('samples_df').empty:
                return True
            else:
                if self.builder_tool.has_built:
                    return proxies_names != [] and set(proxies_names) == set(scenario_names)
                else:
                    self.set_configure_status(False)
                    return False
        else:
            proxies_names = [disc.sos_name for disc in self.scenarios]
            return proxies_names != [] and set(proxies_names) == set(scenario_names)

    def prepare_variables_to_propagate(self):
        # TODO: code below might need refactoring after reference_scenario
        # configuration fashion is decided upon
        samples_df = self.get_sosdisc_inputs(self.SAMPLES_DF)
        instance_reference = self.get_sosdisc_inputs(self.INSTANCE_REFERENCE)

        if instance_reference:
            # Addition of Reference Scenario
            ref_series = pd.Series(
                {self.SELECTED_SCENARIO: True, self.SCENARIO_NAME: 'ReferenceScenario'})
            samples_df = pd.concat([samples_df, pd.DataFrame([ref_series])], ignore_index=True)

        # NB assuming that the samples_df entries are unique otherwise there
        # is some intelligence to be added
        scenario_names = samples_df.loc[samples_df[self.SELECTED_SCENARIO]][self.SCENARIO_NAME].values.tolist()
        trade_vars = []
        # check that all the input scenarios have indeed been built
        # (configuration sequence allows the opposite)

        trade_vars = [col for col in samples_df.columns if col not in
                      [self.SELECTED_SCENARIO, self.SCENARIO_NAME] and col in self.eval_in_possible_values]
        return samples_df, instance_reference, trade_vars, scenario_names

    def configure_subprocesses_with_driver_input(self):
        """
        This method forces the trade variables values of the subprocesses in function of the driverevaluator input df.
        """

        samples_df, instance_reference, trade_vars, scenario_names = self.prepare_variables_to_propagate()

        if self.subprocesses_built(scenario_names):
            if instance_reference:
                # propagate non trade variables values from reference scenario to other scenarios
                scenario_names = scenario_names[:-1]
                self.manage_reference_scenario_features(trade_vars, scenario_names)
            else:
                self.set_variables_editability(scenario_names, self.original_editable_dict_non_ref)

            # PROPAGATE TRADE VARIABLES VALUES FROM samples_df
            # check that there are indeed variable changes input, with respect
            # to reference scenario
            if trade_vars:
                driver_evaluator_ns = self.ee.ns_manager.get_local_namespace_value(
                    self)
                scenarios_data_dict = {}
                for sc in scenario_names:
                    # assuming it is unique
                    sc_row = samples_df[samples_df[self.SCENARIO_NAME]
                                        == sc].iloc[0]
                    for var in trade_vars:
                        var_full_name = self.ee.ns_manager.compose_ns(
                            [driver_evaluator_ns, sc, var])
                        scenarios_data_dict[var_full_name] = sc_row.loc[var]
                if scenarios_data_dict and self.subprocess_is_configured():
                    # push to dm
                    self.ee.dm.set_values_from_dict(scenarios_data_dict)

                    # save trades variable original editable state and set them into not editable
                    self.change_editability_state_for_trade_variables(scenario_names, scenarios_data_dict.keys())

    def change_editability_state_for_trade_variables(self, scenario_names, trades_variables):
        '''
            save trade variables editable state if it is not already saved
            set old original editable state for variables no more trade variables
        '''
        # get variables that have not been already saved
        new_trades_variables = trades_variables - self.original_editable_dict_trade_variables.keys()
        # save editable property of those variables
        self.original_editable_dict_trade_variables.update(
            self.save_original_editable_attr_from_variables(new_trades_variables))

        # get variables that are no more trades variables
        old_trades_variables = {key: value for key, value in self.original_editable_dict_trade_variables.items() if
                                key not in trades_variables}

        # set '_' after scenario name to avoid scenario_1 in scenario12
        scenario_names_ = [f'.{sc}.' for sc in scenario_names]
        # set old editable state to those variables
        self.set_variables_editability(scenario_names_, old_trades_variables)
        # remove those trades variables
        for old_trade_var in old_trades_variables.keys():
            del self.original_editable_dict_trade_variables[old_trade_var]

        # set new trades variables not editable by giving a dict with editable = False
        not_editable_dict = {key: False for key in new_trades_variables}
        self.set_variables_editability(scenario_names_, not_editable_dict)

    '''
    All methods below are here for reference scenario and need to be cleaned

    '''

    def manage_reference_scenario_features(self, trade_vars, scenario_names):
        '''

        Args:
            trade_vars: trade variables
            scenario_names: scenario names

        Header TO DO when dealing with reference scenario

        '''
        # ref_discipline_full_name =
        # ref_discipline.get_disc_full_name() # do provide the sting
        # path of data in flatten
        reference_scenario_ns = self._compose_with_driver_ns(self.REFERENCE_SCENARIO_NAME)
        # ref_discipline_full_name may need to be renamed has it is not
        # true in flatten mode
        ref_discipline_full_name = reference_scenario_ns

        # Manage usecase import
        self.manage_import_inputs_from_sub_process(
            ref_discipline_full_name)
        ref_changes_dict, ref_dict = self.get_reference_non_trade_variables_changes(
            trade_vars)

        scenarios_non_trade_vars_dict = self.transform_dict_from_reference_to_other_scenarios(scenario_names,
                                                                                              ref_dict)

        # Update of original editability state in case modification
        # scenario df
        if set(scenario_names) != set(self.old_scenario_names):
            new_scenarios = set(scenario_names) - set(self.old_scenario_names)
            self.there_are_new_scenarios = True
            for new_scenario in new_scenarios:
                new_scenario_non_trade_vars_dict = {key: value
                                                    for key, value in scenarios_non_trade_vars_dict.items()
                                                    if new_scenario in key}

                new_scenario_editable_dict = self.save_original_editable_attr_from_variables(
                    new_scenario_non_trade_vars_dict)
                self.original_editable_dict_non_ref.update(
                    new_scenario_editable_dict)
        self.old_scenario_names = scenario_names

        # Save the original editability state in case reference is
        # un-instantiated.
        self.save_original_editability_state(
            ref_dict, scenarios_non_trade_vars_dict)
        # Modification of read-only or editable depending on
        # LINKED_MODE or COPY_MODE
        self.modify_editable_attribute_according_to_reference_mode(
            scenarios_non_trade_vars_dict)
        # Propagation to other scenarios if necessary
        self.propagate_reference_non_trade_variables(
            ref_changes_dict, ref_dict, scenario_names)

    def set_variables_editability(self, scenario_names, original_editable_dict):
        '''

        Args:
            scenario_names (list) : List of scenario names

        Turn Editable all the variables that needs to be editable in the original_editable_dict
        TO DO : better explain why

        '''
        if original_editable_dict:
            for sc in scenario_names:
                for key in original_editable_dict.keys():
                    if sc in key:
                        try:
                            self.ee.dm.set_data(
                                key, 'editable', original_editable_dict[key])
                        except:
                            self.logger.debug(f'the variable {key} could not beeing found to reset its editability')

    # def set_reference_trade_variables_in_samples_df(self, sce_df):
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
    # def set_reference_trade_variables_in_samples_df(self, sce_df):
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

    # These dicts are of non-trade variables
    def save_original_editability_state(self, ref_dict, non_ref_dict):

        if self.save_editable_attr:
            self.original_editable_dict_non_ref = self.save_original_editable_attr_from_variables(
                non_ref_dict)
            self.save_editable_attr = False

    def get_reference_scenario_disciplines(self):
        reference_scenario_root_name = self._compose_with_driver_ns(self.REFERENCE_SCENARIO_NAME)
        return [disc for disc in self.scenarios if reference_scenario_root_name in disc.get_disc_full_name()]

    # def get_reference_scenario_index(self):
    #     """
    #     """
    #     index_ref = 0
    #     my_root = self.ee.ns_manager.compose_ns(
    #         [self.sos_name, self.REFERENCE_SCENARIO_NAME])
    #
    #     for disc in self.scenarios:
    #         if disc.sos_name == self.REFERENCE_SCENARIO_NAME \
    #                 or my_root in disc.sos_name:  # for flatten_subprocess
    #             # TODO: better implement this 2nd condition ?
    #             break
    #         else:
    #             index_ref += 1
    #     return index_ref

    def check_if_there_are_reference_variables_changes(self):

        samples_df, instance_reference, trade_vars, scenario_names = self.prepare_variables_to_propagate()

        ref_changes_dict = {}
        if self.subprocesses_built(scenario_names):
            if instance_reference:
                ref_changes_dict, ref_dict = self.get_reference_non_trade_variables_changes(
                    trade_vars)

        return ref_changes_dict

    def get_reference_non_trade_variables_changes(self, trade_vars):
        # Take reference scenario non-trade variables (num and non-num) and its
        # values
        ref_dict = {}
        for ref_discipline in self.get_reference_scenario_disciplines():
            for key in ref_discipline.get_input_data_names():
                if all(key.split(self.REFERENCE_SCENARIO_NAME + '.')[-1] != trade_var for trade_var in trade_vars):
                    ref_dict[key] = ref_discipline.ee.dm.get_value(key)

        # Check if reference values have changed and select only those which
        # have changed

        ref_changes_dict = {}
        for key in ref_dict.keys():
            if key in self.old_ref_dict.keys():
                if isinstance(ref_dict[key], pd.DataFrame):
                    if not ref_dict[key].equals(self.old_ref_dict[key]):
                        ref_changes_dict[key] = ref_dict[key]
                elif isinstance(ref_dict[key], (np.ndarray)):
                    if not (np.array_equal(ref_dict[key], self.old_ref_dict[key])):
                        ref_changes_dict[key] = ref_dict[key]
                elif isinstance(ref_dict[key], (list)):
                    if not (np.array_equal(ref_dict[key], self.old_ref_dict[key])):
                        ref_changes_dict[key] = ref_dict[key]
                else:
                    if ref_dict[key] != self.old_ref_dict[key]:
                        ref_changes_dict[key] = ref_dict[key]
            else:
                ref_changes_dict[key] = ref_dict[key]

        # TODO: replace the above code by a more general function ...
        # ======================================================================
        # ref_changes_dict = {}
        # if self.old_ref_dict == {}:
        #     ref_changes_dict = ref_dict
        # else:
        #     # See Test 01 of test_69_compare_dict_compute_len
        #     compare_dict(ref_dict, self.old_ref_dict, '',
        #                  ref_changes_dict, df_equals=True)
        #     # We cannot use compare_dict as if: maybe we choude add a diff_compare_dict as an adaptation of compare_dict
        # ======================================================================

        return ref_changes_dict, ref_dict

    def propagate_reference_non_trade_variables(self, ref_changes_dict, ref_dict,
                                                scenario_names_to_propagate):

        if ref_changes_dict:
            self.old_ref_dict = copy.deepcopy(ref_dict)

        # Build other scenarios variables and values dict from reference
        dict_to_propagate = {}
        # Propagate all reference
        if self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.LINKED_MODE:
            dict_to_propagate = self.transform_dict_from_reference_to_other_scenarios(scenario_names_to_propagate,
                                                                                      ref_dict)
        # Propagate reference changes
        elif self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.COPY_MODE and ref_changes_dict:
            dict_to_propagate = self.transform_dict_from_reference_to_other_scenarios(scenario_names_to_propagate,
                                                                                      ref_changes_dict)
        # Propagate other scenarios variables and values
        if self.there_are_new_scenarios:
            if dict_to_propagate:
                self.ee.dm.set_values_from_dict(dict_to_propagate)
        else:
            if ref_changes_dict and dict_to_propagate:
                self.ee.dm.set_values_from_dict(dict_to_propagate)

    def get_other_evaluators_names_and_mode_under_current_one(self):
        other_evaluators_names_and_mode = []
        for disc in self.scenarios:
            name_and_modes = self.search_evaluator_names_and_modify_mode_iteratively(
                disc)
            if name_and_modes != []:
                other_evaluators_names_and_mode.append(name_and_modes)

        return other_evaluators_names_and_mode

    def search_evaluator_names_and_modify_mode_iteratively(self, disc):

        retval = []
        for subdisc in disc.proxy_disciplines:
            if subdisc.__class__.__name__ == 'ProxyMultiInstanceDriver':
                if subdisc.get_sosdisc_inputs(self.INSTANCE_REFERENCE):
                    # If upper ProxyDriverEvaluator is in linked mode, all
                    # lower ProxyDriverEvaluator shall be as well.
                    if self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.LINKED_MODE:
                        subdriver_full_name = self.ee.ns_manager.get_local_namespace_value(
                            subdisc)
                        if 'ReferenceScenario' in subdriver_full_name:
                            self.ee.dm.set_data(
                                subdriver_full_name + '.reference_mode', 'value', self.LINKED_MODE)
                    retval = [subdisc.sos_name]
                else:
                    retval = [subdisc.sos_name]
            elif subdisc.__class__.__name__ != 'ProxyDiscipline':
                name_and_modes = self.search_evaluator_names_and_modify_mode_iteratively(
                    subdisc)
                retval.append(name_and_modes)

        return retval

    def modify_editable_attribute_according_to_reference_mode(self, scenarios_non_trade_vars_dict):

        other_evaluators_names_and_mode = self.get_other_evaluators_names_and_mode_under_current_one()

        if self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.LINKED_MODE:
            for key in scenarios_non_trade_vars_dict.keys():
                self.ee.dm.set_data(key, 'editable', False)
        elif self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.COPY_MODE:
            for key in scenarios_non_trade_vars_dict.keys():
                if other_evaluators_names_and_mode != []:  # This means there are evaluators under current one
                    for element in other_evaluators_names_and_mode:
                        if element[0] in key:  # Ignore variables from inner ProxyDriverEvaluators
                            pass
                        else:
                            if self.original_editable_dict_non_ref[key]:
                                self.ee.dm.set_data(key, 'editable', True)
                else:
                    if self.original_editable_dict_non_ref[key]:
                        self.ee.dm.set_data(key, 'editable', True)

    def save_original_editable_attr_from_variables(self, dict):

        dict_out = {}
        for key in dict:
            dict_out[key] = self.dm.get_data(key, 'editable')

        return dict_out

    def transform_dict_from_reference_to_other_scenarios(self, scenario_names, dict_from_ref):

        transformed_to_other_scenarios_dict = {}
        for key in dict_from_ref.keys():
            for sc in scenario_names:
                if self.REFERENCE_SCENARIO_NAME in key and self.sos_name in key:
                    new_key = key.split(self.sos_name, 1)[0] + self.sos_name + '.' + sc + \
                              key.split(self.sos_name,
                                        1)[-1].split(self.REFERENCE_SCENARIO_NAME, 1)[-1]
                elif self.REFERENCE_SCENARIO_NAME in key and self.sos_name not in key:
                    new_key = key.split(self.REFERENCE_SCENARIO_NAME, 1)[
                                  0] + sc + key.split(self.REFERENCE_SCENARIO_NAME, 1)[-1]
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
