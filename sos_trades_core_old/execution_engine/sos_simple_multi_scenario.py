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
import pandas as pd
from copy import deepcopy

from sos_trades_core.execution_engine.sos_discipline_scatter import SoSDisciplineScatter
from sos_trades_core.execution_engine.sos_discipline_gather import SoSDisciplineGather
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.sos_eval import SoSEval
from sos_trades_core.execution_engine.sos_very_simple_multi_scenario import SoSVerySimpleMultiScenario


class SoSSimpleMultiScenarioException(Exception):
    pass


class SoSSimpleMultiScenario(SoSVerySimpleMultiScenario):
    ''' 
    Class that build scatter discipline and linked scatter data from scenario defined in scenario_df
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Simple Multi-Scenario',
        'type': 'Test',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-stream fa-fw',
        'version': '',
    }
    TRADE_VARIABLES = 'trade_variables'
    SCENARIO_DF = 'scenario_df'
    SCENARIO_NAME = 'scenario_name'
    NS_BUSINESS_OUTPUTS = 'ns_business_outputs'

    DESC_IN = {'trade_variables': {'type': 'dict', 'value': {}, 'dataframe_descriptor': {'variable':  (
        'string',  None, True), 'value': ('string',  None, True), }, 'dataframe_edition_locked': False, 'structuring': True}}

    def __init__(self, sos_name, ee, map_name, cls_builder, autogather, gather_node, business_post_proc):
        '''
        Constructor
        '''
        self.__scenario_dict = {}
        self.__trade_variables = {}
        self.__linked_scatter_data = {}
        SoSVerySimpleMultiScenario.__init__(
            self, sos_name, ee, map_name, cls_builder, autogather, gather_node, business_post_proc)

        self._maturity = ''

    def get_scenario_dict(self):
        return self.__scenario_dict

    def get_trade_variables(self):
        return self.__trade_variables

    def get_linked_scatter_data(self):
        return self.__linked_scatter_data

    def set_scenario_dict(self, scenario_dict):
        self.__scenario_dict = deepcopy(scenario_dict)

    def get_scenario_dict_for_parameter(self, parameter):
        '''
        Return a dict of parameter values for each scenario
        '''
        values_dict = {}
        for scenario, values in self.get_scenario_dict().items():
            values_dict[scenario] = values[parameter]
        return values_dict

    def build_scatter_data_maps(self, trade_variables):

        scenario_parameters = {}

        for output_name, output_type in trade_variables.items():

            if output_name not in self.get_trade_variables():

                if output_type == 'dataframe':
                    input_type = 'df_dict'
                    self._df_in_trade_variables = True
                else:
                    input_type = 'dict'

                trade_var_map = {'input_name': f'{output_name}_dict',
                                 'input_type': input_type,
                                 'input_ns': self.ee.smaps_manager.get_input_ns_from_build_map(self.sc_map.get_input_name()),
                                 'output_name': output_name,
                                 'output_type': output_type,
                                 'scatter_var_name': self.sc_map.get_input_name()}

                s_map = self.ee.smaps_manager.add_data_map(
                    f'{output_name}_dict', trade_var_map)

                scenario_parameters[output_name] = s_map

        return scenario_parameters

    def build_inst_desc_io_with_scenario_parameters(self):
        '''
        Complete inst_desc_in with scenario_df
        '''
        # add scenario_df to inst_desc_in
        if self.SCENARIO_DF not in self.inst_desc_in:
            input_ns = self.sc_map.get_input_ns()
            scenario_df_input = {self.SCENARIO_DF: {
                SoSDiscipline.TYPE: 'dataframe', SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: input_ns, SoSDiscipline.EDITABLE: True, SoSDiscipline.STRUCTURING: True}}
            self.inst_desc_in.update(scenario_df_input)

        # add scenario_dict to inst_desc_in
        if self.SCENARIO_DICT not in self.inst_desc_in:
            input_ns = self.sc_map.get_input_ns()
            scenario_dict_input = {self.SCENARIO_DICT: {
                SoSDiscipline.TYPE: 'dict', SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: input_ns, SoSDiscipline.EDITABLE: False, self.USER_LEVEL: 3}}
            self.inst_desc_in.update(scenario_dict_input)

    def build_scenarios(self):
        '''
        Generated scenarios and build inst_desc_io
        '''
        # if multi-scenario class is already configured so that
        # 'trade_variables' is in data_in
        if self.TRADE_VARIABLES in self._data_in:
            trade_variables = self.get_sosdisc_inputs(
                self.TRADE_VARIABLES)

            # get scenario parameters
            if len(self.get_trade_variables()) == 0 or list(set(self.get_trade_variables().keys())) != list(set(trade_variables.keys())):
                # build dict of scatter maps of trade variables
                self.get_trade_variables().update(self.build_scatter_data_maps(
                    trade_variables))
                self.build_inst_desc_io_with_scenario_parameters()

            # generate combined scenarios
            self.generate_scenarios()

            # store new generated scenarios in dm
            self.update_dm_with_scenario_parameters()

    def clean_trade_variables(self):
        ''' 
        Remove trade variables not in data_in anymore
        '''
        if self.TRADE_VARIABLES in self._data_in:
            trade_variables = self.get_sosdisc_inputs(
                self.TRADE_VARIABLES)
            if list(set(self.get_trade_variables().keys())) != list(set(trade_variables.keys())):
                trade_var_list = deepcopy(
                    list(self.get_trade_variables().keys()))
                for trade_var_name in trade_var_list:
                    if trade_var_name not in trade_variables:
                        del self.get_trade_variables()[trade_var_name]
                        self.ee.factory.clean_discipline_list([
                            self.get_linked_scatter_data()[trade_var_name]])
                        del self.get_linked_scatter_data()[trade_var_name]
                        self.clean_scattered_disciplines([])
                        self.ee.smaps_manager.remove_data_map(
                            f'{trade_var_name}_dict')

                        # remove keys in gather disciplines
                        if self.get_autogather():
                            for sub_builder in self.get_cls_builder():
                                if self.get_gather_node() is None:
                                    complete_name = f'{self.ee.ns_manager.current_disc_ns.split(self.sos_name)[0]}sub_builder.sos_name'
                                else:
                                    complete_name = f'{self.ee.ns_manager.current_disc_ns.split(self.sos_name)[0]}{self.get_gather_node()}.{sub_builder.sos_name}'

                                gather_disc_list = [disc for disc in self.ee.dm.get_disciplines_with_name(
                                    complete_name) if isinstance(disc, SoSDisciplineGather)]
                                for gather in gather_disc_list:
                                    self.ee.dm.clean_keys(gather.disc_id)
                                    gather.reset_data()

    def build(self):
        '''
        Overwrite scatter discipline method to configure scenarios
        '''
        self.clean_trade_variables()
        self.build_linked_scatter_data()
        self.build_scenarios()
        self.coupling_per_scatter = True
        SoSDisciplineScatter.build(self)

        self.build_business_io()

    def build_linked_scatter_data(self):
        '''
        Create the scatter_data builder associated to the multiscenario 
        to scatter the scenario matrix into all scenarios
        Build the scatter data with the same_name as the sosmultiscenario at the same namespace
        Add it to the factory and to the dict __linked_scatter_data
        '''
        for trade_var in self.get_trade_variables():
            # if the scatter data has not been build yet
            if trade_var not in self.get_linked_scatter_data():
                # Get the builder
                scatter_builder = self.ee.factory.create_scatter_data_builder(
                    self.sos_name, f'{trade_var}_dict')
                # Change the namespace to be the same as self
                old_current_ns = self.ee.ns_manager.current_disc_ns
                self.ee.ns_manager.change_disc_ns('..')
                # build the builder to get back the discipline
                scatter_disc = scatter_builder.build()
                # Add the discipline to the factory
                self.ee.factory.add_discipline(scatter_disc)
                self.ee.ns_manager.set_current_disc_ns(old_current_ns)
                # Add the discipline to the dict of linked scatter data
                self.get_linked_scatter_data()[trade_var] = scatter_disc

    def generate_scenarios(self):
        '''
        Generate combined scenarios
        '''
        if self.SCENARIO_DF in self._data_in.keys():
            scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
            columns = [self.SCENARIO_NAME]
            columns.extend(
                list(self.get_sosdisc_inputs(self.TRADE_VARIABLES).keys()))

            if scenario_df is None:
                scenario_df = pd.DataFrame(columns=columns)
                self.ee.dm.set_data(self.get_var_full_name(
                    self.SCENARIO_DF, self._data_in), self.VALUE, scenario_df, check_value=False)
            elif scenario_df.columns.to_list() != columns:
                to_remove = [
                    c for c in scenario_df.columns if c not in columns]
                scenario_df = scenario_df.drop(columns=to_remove)
                to_add = [c for c in columns if c not in scenario_df.columns]
                empty_df = pd.DataFrame(columns=to_add)
                scenario_df = pd.concat([scenario_df, empty_df], axis=1)
                self.ee.dm.set_data(self.get_var_full_name(
                    self.SCENARIO_DF, self._data_in), self.VALUE, scenario_df, check_value=False)

            self.set_scenario_dict(scenario_df.set_index(
                self.SCENARIO_NAME).to_dict(orient='index'))

    def update_dm_with_scenario_parameters(self):
        '''
        Store scenario parameters in data_io 
        '''
        if self.SCENARIO_DICT in self._data_in.keys():

            # store scenario_dict and scenario_list in data manager
            self.ee.dm.set_data(self.get_var_full_name(
                self.SCENARIO_DICT, self._data_in), self.VALUE, self.get_scenario_dict(), check_value=False)
            self.ee.dm.set_data(self.get_var_full_name(self.sc_map.get_input_name(
            ), self._data_in), SoSDiscipline.VALUE, list(self.get_scenario_dict().keys()))
            self.ee.dm.set_data(self.get_var_full_name(
                self.sc_map.get_input_name(), self._data_in), SoSDiscipline.EDITABLE, False)
            self.ee.dm.set_data(self.get_var_full_name(
                self.sc_map.get_input_name(), self._data_in), SoSDiscipline.USER_LEVEL, 3)

            # store linked scatter data input in dm (dict of values for each
            # scenario)
            for trade_var_map in self.get_trade_variables().values():
                input_name = trade_var_map.get_input_name()[0]
                input_ns = trade_var_map.get_input_ns()
                output_name = trade_var_map.get_output_name()[0]
                input_full_name = self.ee.ns_manager.compose_ns(
                    [self.ee.ns_manager.get_shared_namespace_value(self, input_ns), input_name])

                if input_full_name in self.ee.dm.data_id_map.keys():
                    input_value = self.get_scenario_dict_for_parameter(
                        output_name)
                    self.ee.dm.set_data(input_full_name, SoSDiscipline.EDITABLE,
                                        False)
                    self.ee.dm.set_data(input_full_name, SoSDiscipline.USER_LEVEL,
                                        3)
                    # store new dict of values in dm
                    self.ee.dm.set_data(input_full_name, SoSDiscipline.VALUE,
                                        deepcopy(input_value), check_value=False)

    def update_trade_variables_referencing(self, disc_list):
        '''
        Update trade variables referencing with correct namespace built by linked scatter data
        '''
        # loop on sub disciplines
        for disc in disc_list:
            # loop on trade variables
            for trade_var_map in self.get_trade_variables().values():

                # namespace associated to multi-scenario scatter
                scatter_ns = self.sc_map.get_scatter_ns()
                output_name = trade_var_map.get_output_name()[0]
                keys_to_update = [output_name]
                if self.ee.smaps_manager.get_build_map_with_input_name(output_name) is not None:
                    # get associated inputs if trade variable is used to build
                    # multi scatter builder discipline
                    keys_to_update.extend((list(set(self.ee.smaps_manager.get_build_map_with_input_name(
                        output_name).get_associated_inputs()))))

                for key in keys_to_update:
                    short_key = key.split(
                        self.ee.ns_manager.NS_SEP)[-1]
                    # if short_key is an input of disc
                    if short_key in disc._data_in.keys() and key in disc.get_var_full_name(short_key, disc._data_in):
                        full_key = disc.get_var_full_name(
                            short_key, disc.get_data_in())
                        new_full_name = self.ee.ns_manager.compose_ns(
                            [self.ee.ns_manager.get_disc_others_ns(disc)[scatter_ns].value, key])

                        if full_key != new_full_name:

                            if new_full_name in self.ee.dm.data_id_map:
                                # update disciplines dependencies in dm
                                self.ee.dm.get_data(
                                    new_full_name, SoSDiscipline.DISCIPLINES_DEPENDENCIES).extend(disc._data_in[short_key][SoSDiscipline.DISCIPLINES_DEPENDENCIES])
                                # update referencing in data_in
                                disc._data_in[short_key] = self.ee.dm.get_data(
                                    new_full_name)
                                # remove disc in full_key dependencies
                                self.ee.dm.remove_keys(
                                    disc.disc_id, [full_key])
                            else:
                                # change namespace in data_in
                                disc._data_in[short_key][self.NAMESPACE] = scatter_ns
                                disc._data_in[short_key][self.NS_REFERENCE] = self.ee.ns_manager.get_disc_others_ns(disc)[
                                    scatter_ns]

    def configure(self):
        '''
        Configure scenario scatter discipline after scattered disciplines configuration
        '''
        if not all([input in self._data_in.keys() for input in self.inst_desc_in.keys()]):
            SoSDiscipline.configure(self)

        if all([scatter_data.is_configured() for scatter_data in self.get_linked_scatter_data().values()]):
            sub_disc_list = self.get_sub_disciplines_list()
            self.update_trade_variables_referencing(sub_disc_list)
            self.ee.dm.generate_data_id_map()
            if self.check_structuring_variables_changes():
                self.set_structuring_variables_values()

    def check_all_scenarios_configured(self):
        '''
        Check if all coupling scenarios are configured
        Return False if at least one need to be configured
        '''
        for scenario in self.get_scattered_disciplines().values():
            if not scenario[0].is_configured():
                return False
        return True

    def get_sub_disciplines_list(self):
        '''
        Get all sub disciplines of multi-scenarios with recursive search
        '''
        sub_disc_list = []
        for disc in self.ee.factory.sos_disciplines:
            if not isinstance(disc, (SoSSimpleMultiScenario, SoSCoupling, SoSEval)):
                sub_disc_list.append(disc)
        return sub_disc_list

    def is_configured(self):
        '''
        Return False if at least one sub discipline needs to be configured, True if not
        '''
        return SoSDiscipline.is_configured(self) and self.check_all_scenarios_configured()
