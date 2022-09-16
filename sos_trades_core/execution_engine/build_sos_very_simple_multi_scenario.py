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
from sos_trades_core.execution_engine.build_sos_discipline_scatter import BuildSoSDisciplineScatter
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.sos_wrapping.old_sum_value_block_discipline import OldSumValueBlockDiscipline


class BuildSoSSimpleMultiScenarioException(Exception):
    pass


class BuildSoSVerySimpleMultiScenario(BuildSoSDisciplineScatter):
    ''' 
    Class that build scatter discipline and linked scatter data from scenario defined in a usecase


    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SUB_PROCESS_INPUTS (structuring)
            |_ SCENARIO_MAP (structuring)            
               |_ SCENARIO_MAP['input_name'] (namespace: INPUT_NS if INPUT_NS in SCENARIO_MAP keys / if not then  local, structuring, dynamic : valid SUB_PROCESS_INPUTS and SCENARIO_MAP)
        |_ DESC_OUT

    2) Description of DESC parameters:
        |_ DESC_IN
           |_ SUB_PROCESS_INPUTS:               All inputs for driver builder in the form of a dictionary of four keys
                                                    PROCESS_REPOSITORY:   folder root of the sub processes to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    PROCESS_NAME:         selected process name (in repository) to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    USECASE_NAME:         either empty or an available usecase of the sub_process
                                                    USECASE_DATA:         anonymized dictionary of usecase inputs to be nested in context
                                                                          it is a temporary input: it will be put to None as soon as                                                                        
                                                                          its content is 'loaded' in the dm. We will have it has editable                                                                             
                                                It is in dict type (specific 'proc_builder_modale' type to have a specific GUI widget) 
           |_ SCENARIO_MAP:                     All inputs for driver builder in the form of a dictionary of four keys
                                                    INPUT_NAME:           name of the variable to scatter
                                                    INPUT_NS:             namespace of the variable to scatter
                                                    OUTPUT_NAME:          name of the variable to overwrite
                                                    SCATTER_NS:           namespace associated to the scatter discipline
                                                                          it is a temporary input: it will be put to None as soon as                                                                        
                                                    GATHER_NS:            namespace of the gather discipline associated to the scatter discipline 
                                                                          (input_ns by default and optional)
                                                    NS_TO_UPDATE:         list of namespaces depending on the scatter namespace (can be optional)                                                                                                                                            

         |_ DESC_OUT
    '''

#################### Begin: Ontology of the discipline ###################
    # ontology information
    _ontology_data = {
        'label': 'Proc Builder Core Very Simple Multi-Scenario',
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
#################### End: Ontology of the discipline #####################
#################### Begin: Constants and parameters #####################
    # -- Disciplinary attributes
    SUB_PROCESS_INPUTS = 'sub_process_inputs'
    PROCESS_NAME = 'process_name'
    USECASE_NAME = 'usecase_name'
    USECASE_DATA = 'usecase_data'
    PROCESS_REPOSITORY = 'process_repository'

    SCENARIO_MAP = 'scenario_map'
    INPUT_NAME = 'input_name'
    INPUT_NS = 'input_ns'
    OUTPUT_NAME = 'output_name'
    SCATTER_NS = 'scatter_ns'
    GATHER_NS = 'gather_ns'
    NS_TO_UPDATE = 'ns_to_update'

    DRIVER_NAME = 'driver_name'  # Default vs_MS
    AUTHOGATHER = 'autogather'  # Default  False
    GATHER_NODE = 'gather_node'  # Default None
    BUSINESS_POST_PROC = 'business_post_proc'  # Default False

    SCENARIO_DICT = 'scenario_dict'
    NS_BUSINESS_OUTPUTS = 'ns_business_outputs'

    default_sub_process_inputs_dict = {}
    default_sub_process_inputs_dict[PROCESS_REPOSITORY] = None
    default_sub_process_inputs_dict[PROCESS_NAME] = None
    default_sub_process_inputs_dict[USECASE_NAME] = 'Empty'
    default_sub_process_inputs_dict[USECASE_DATA] = {}

    default_scenario_map = {}
    default_scenario_map[INPUT_NAME] = None
    default_scenario_map[INPUT_NS] = ''
    default_scenario_map[OUTPUT_NAME] = ''
    default_scenario_map[SCATTER_NS] = ''
    default_scenario_map[GATHER_NS] = ''
    default_scenario_map[NS_TO_UPDATE] = []

    DESC_IN = {
        SUB_PROCESS_INPUTS: {'type': 'dict',
                             'structuring': True,
                             'default': default_sub_process_inputs_dict,
                             'user_level': 1,
                             'optional': False
                             },
        # SUB_PROCESS_INPUTS: {'type': SoSDiscipline.PROC_BUILDER_MODAL,
        #                     'structuring': True,
        #                     'default': default_sub_process_inputs_dict,
        #                     'user_level': 1,
        #                     'optional': False
        #                     },
        SCENARIO_MAP: {'type': 'dict',
                       'structuring': True,
                       'default': default_scenario_map,
                       'user_level': 1,
                       'optional': False
                       }
    }

    #DESC_OUT = {}

#################### End: Constants and parameters #######################
#################### Begin: Main methods ################################

    def __init__(self, sos_name, ee, map_name, cls_builder, autogather, gather_node, business_post_proc):
        '''
        Constructor
        '''
        self.__autogather = autogather
        self.__gather_node = gather_node
        self.__build_business_io = business_post_proc
        self.__cls_builder = cls_builder
        BuildSoSDisciplineScatter.__init__(
            self, sos_name, ee, map_name, cls_builder)
        self._maturity = ''

        self.previous_sub_process_repo = None
        self.previous_sub_process_name = None
        self.previous_sub_process_usecase_name = 'Empty'
        self.previous_sub_process_usecase_data = {}
        self.dyn_var_sp_from_import_dict = {}
        self.previous_algo_name = ""
        self.sub_process_ns_in_build = None
        # Possible values: 'Empty_SP', 'Create_SP', 'Replace_SP',
        # 'Unchanged_SP'
        self.sub_proc_build_status = 'Empty_SP'
        # Possible values: 'No_SP_UC_Import', 'SP_UC_Import'
        self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

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
            Overloaded scatter discipline method to "configure" scenarios
            Get and build builder from sub_process of vs_MS driver
            Added to provide proc builder capability
            Reached from __configure_io in ee.py: self.factory.build() is going from build to build starting from root
            It comes before configuring()
        '''
        self.coupling_per_scatter = True
        # Remark: cls_builder and sc_map added in __init__ of sos_discipline_scatter
        # Remark: Nested build is done by creating (in SoSDisciplineScatter.build/build_sub_coupling)
        # as many coupling as names in the provided list of the data key
        # SCENARIO_MAP['input_name']
        BuildSoSDisciplineScatter.build(self)
        self.build_business_io()

    def configure(self):
        """
            Overloaded SoSDiscipline method
            Configuration of the Build vsMS
            Reached from __configure_io in ee.py: self.root_process.configure_io() is going from confiure to configure starting from root
            It comes after build()
        """
        BuildSoSDisciplineScatter.configure(self)
        # Remark: the dynamic input SCENARIO_MAP['input_name'] is set in
        # build_inst_desc_in_with_map() in __init__ of sos_discipline_scatter

    def setup_sos_disciplines(self):
        """
           Overload setup_sos_disciplines to create a dynamic desc_in
           Reached from configure() of sos_discipline [SoSDiscipline.configure in config() of scatter discipline].
           It is done upstream of set_eval_possible_values()
        """
        dynamic_inputs = {}
        dynamic_outputs = {}
        if self.sub_proc_build_status != 'Empty_SP':
            sub_process_inputs_dict = self.get_sosdisc_inputs(
                self.SUB_PROCESS_INPUTS)
            sub_process_repo = sub_process_inputs_dict[self.PROCESS_REPOSITORY]
            sub_process_name = sub_process_inputs_dict[self.PROCESS_NAME]
            # 1. provide driver inputs based on selected subprocess
            # self.setup_sos_disciplines_driver_inputs_depend_on_sub_process(
            #    dynamic_inputs)
            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)
            # 2. import data from selected sub_process_usecase

    def run(self):
        '''
            Overloaded scatter discipline method
            Store business outputs in dictionaries if autogather is True
        '''
        if self.get_autogather():
            self.run_autogather()


#################### End: Main methods ################################
##################### Begin: Sub methods ################################

    def run_autogather(self):
        '''
           Store business outputs in dictionaries if autogather is True
        '''
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

    def build_business_io(self):
        '''
           Add SumValueBlockDiscipline ouputs in INST_DESC_IN and dict in INST_DESC_OUT if autogather is True
           Function needed in build(self)
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
##################### End: Sub methods ################################
