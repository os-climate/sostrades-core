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
from sos_trades_core.execution_engine.sos_discipline_scatter import SoSDisciplineScatter
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.sos_wrapping.old_sum_value_block_discipline import OldSumValueBlockDiscipline


class SoSSimpleMultiScenarioException(Exception):
    pass


class SoSVerySimpleMultiScenario(SoSDisciplineScatter):
    ''' 
    Class that build scatter discipline and linked scatter data from scenario defined in a usecase
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Very Simple Multi-Scenario',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-stream fa-fw',
        'version': '',
    }
    SCENARIO_DICT = 'scenario_dict'
    NS_BUSINESS_OUTPUTS = 'ns_business_outputs'

    def __init__(self, sos_name, ee, map_name, cls_builder, autogather, gather_node, business_post_proc, associated_namespaces=[]):
        '''
        Constructor
        '''
        self.__autogather = autogather
        self.__gather_node = gather_node
        self.__build_business_io = business_post_proc
        self.__cls_builder = cls_builder
        SoSDisciplineScatter.__init__(
            self, sos_name, ee, map_name, cls_builder, associated_namespaces=associated_namespaces)

        self._maturity = ''

    def get_autogather(self):
        return self.__autogather

    def get_gather_node(self):
        return self.__gather_node

    def get_build_business_io(self):
        return self.__build_business_io

    def get_cls_builder(self):
        return self.__cls_builder

    def build(self):
        '''
        Overwrite scatter discipline method to configure scenarios
        '''
        self.coupling_per_scatter = True
        SoSDisciplineScatter.build(self)

        self.build_business_io()

    def build_business_io(self):
        '''
        Add SumValueBlockDiscipline ouputs in INST_DESC_IN and dict in INST_DESC_OUT if autogather is True
        '''
        if self.get_build_business_io():

            if self.get_gather_node() is None:
                ns_value = f'{self.ee.study_name}.Business'
            else:
                ns_value = f'{self.ee.study_name}.{self.get_gather_node()}.Business'

            if self.NS_BUSINESS_OUTPUTS not in self.ee.ns_manager.shared_ns_dict:
                self.ee.ns_manager.add_ns(self.NS_BUSINESS_OUTPUTS, ns_value)
                self.ee.ns_manager.disc_ns_dict[self]['others_ns'].update(
                    {self.NS_BUSINESS_OUTPUTS: self.ee.ns_manager.shared_ns_dict[self.NS_BUSINESS_OUTPUTS]})

            ns_to_cut = self.ee.ns_manager.get_shared_namespace_value(
                self, self.sc_map.get_gather_ns()) + self.ee.ns_manager.NS_SEP

            # In the case of old sum value block disciplines (used in business cases processes)
            # we needed a gather_data for multi scenario post processing
            # This capability will be replaced by generic gather data created
            # from archi builder when all business processes will be refactored
            for disc in self.ee.factory.sos_disciplines:
                if isinstance(disc, OldSumValueBlockDiscipline):
                    for key in disc._data_out.keys():
                        full_key = disc.get_var_full_name(key, disc._data_out)
                        end_key = full_key.split(ns_to_cut)[-1]
                        if end_key not in self._data_in:
                            self.inst_desc_in.update({end_key: {SoSDiscipline.TYPE: disc._data_out[key][SoSDiscipline.TYPE],
                                                                SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: self.sc_map.get_gather_ns(), SoSDiscipline.USER_LEVEL: 3}})
                        if f'{key}_dict' not in self._data_out:
                            self.inst_desc_out.update({f'{key}_dict': {SoSDiscipline.TYPE: 'dict',
                                                                       SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: self.NS_BUSINESS_OUTPUTS, SoSDiscipline.USER_LEVEL: 2}})

            # modify SCENARIO_DICT input namespace to store it in
            # NS_BUSINESS_OUTPUTS node
            if self.SCENARIO_DICT in self._data_in and self.ee.dm.get_data(self.get_var_full_name(self.SCENARIO_DICT, self._data_in), SoSDiscipline.NAMESPACE) != self.NS_BUSINESS_OUTPUTS:
                full_key = self.get_var_full_name(
                    self.SCENARIO_DICT, self._data_in)
                self.ee.dm.set_data(
                    full_key, self.NAMESPACE, self.NS_BUSINESS_OUTPUTS)
                self.ee.dm.set_data(
                    full_key, self.NS_REFERENCE, self.get_ns_reference(SoSDiscipline.SHARED_VISIBILITY, self.NS_BUSINESS_OUTPUTS))
                self.ee.dm.generate_data_id_map()

    def run(self):
        '''
        Store business outputs in dictionaries if autogather is True
        '''
        if self.get_autogather():

            new_values_dict = {}

            for long_key in self._data_in.keys():
                for key in self._data_out.keys():
                    if long_key.endswith(key.split('_dict')[0]):
                        if key in new_values_dict:
                            new_values_dict[key].update({long_key.rsplit('.', 1)[0]: self.ee.dm.get_value(
                                self.get_var_full_name(long_key, self._data_in))})
                        else:
                            new_values_dict[key] = {long_key.rsplit('.', 1)[0]: self.ee.dm.get_value(
                                self.get_var_full_name(long_key, self._data_in))}

            self.store_sos_outputs_values(new_values_dict)
