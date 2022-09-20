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
from sos_trades_core.tools.scenario.scenario_generator import ScenarioGenerator
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_simple_multi_scenario import SoSSimpleMultiScenario


class SoSMultiScenarioException(Exception):
    pass


class SoSMultiScenario(SoSSimpleMultiScenario):
    ''' 
    Class that build scatter discipline and linked scatter data using a scenario generator
    The scenarios generated are the combinations of the trade variables
    '''

    # ontology information
    _ontology_data = {
        'label': 'Multi-Scenario Model',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-stream fa-fw',
        'version': '',
    }

    def __init__(self, sos_name, ee, map_name, cls_builder, autogather, gather_node, business_post_proc, associated_namespaces=[]):
        '''
        Constructor
        '''
        SoSSimpleMultiScenario.__init__(
            self, sos_name, ee, map_name, cls_builder, autogather, gather_node, business_post_proc, associated_namespaces=associated_namespaces)

    def build_inst_desc_io_with_scenario_parameters(self):
        '''
        Complete inst_desc_in with trade variables and scenario_dict
        '''
        # add trade variables to inst_desc_in (list)
        for trade_var_name, trade_var_map in self.get_trade_variables().items():
            if trade_var_name not in self.inst_desc_in:
                output_type = trade_var_map.get_output_type()[0]
                input_ns = trade_var_map.get_input_ns()
                if output_type not in ['list', 'dict']:
                    trade_var_input = {f'{trade_var_name}_trade': {
                        SoSDiscipline.TYPE: 'list', SoSDiscipline.SUBTYPE: output_type,
                        SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: input_ns,
                        SoSDiscipline.STRUCTURING: True}}
                else:

                    trade_var_input = {f'{trade_var_name}_trade': {
                        SoSDiscipline.TYPE: 'list', SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: input_ns, SoSDiscipline.STRUCTURING: True}}
                self.inst_desc_in.update(trade_var_input)

        # add scenario_dict to inst_desc_in
        if self.SCENARIO_DICT not in self.inst_desc_in:
            input_ns = self.sc_map.get_input_ns()
            scenario_dict_input = {self.SCENARIO_DICT: {
                SoSDiscipline.TYPE: 'dict', SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: input_ns, SoSDiscipline.EDITABLE: False, self.USER_LEVEL: 3}}
            self.inst_desc_in.update(scenario_dict_input)

    def generate_scenarios(self):
        '''
        Generate combined scenarios
        '''
        if self.SCENARIO_DICT in self._data_in and len(self.get_trade_variables().keys()) >= 1:

            scenario_generator = ScenarioGenerator()
            dict_parameters = {}
            for trade_var_name, trade_var_map in self.get_trade_variables().items():
                if f'{trade_var_name}_trade' in self._data_in.keys():
                    dict_parameters[trade_var_map.get_output_name()[0]] = self.get_sosdisc_inputs(
                        f'{trade_var_name}_trade')

            self.set_scenario_dict(scenario_generator.generate_scenarios(
                dict_parameters))
