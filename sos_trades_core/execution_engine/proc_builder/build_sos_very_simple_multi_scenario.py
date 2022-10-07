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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.proc_builder.sos_add_subproc_to_driver import AddSubProcToDriver
from sos_trades_core.execution_engine.proc_builder.build_sos_discipline_scatter import BuildSoSDisciplineScatter
from sos_trades_core.sos_wrapping.old_sum_value_block_discipline import OldSumValueBlockDiscipline
from sos_trades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
import pandas as pd
from importlib import import_module


class BuildSoSSimpleMultiScenarioException(Exception):
    pass


class BuildSoSVerySimpleMultiScenario(BuildSoSDisciplineScatter):
    ''' 
    Class that build scatter discipline and linked scatter data from scenario defined in a usecase


    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SUB_PROCESS_INPUTS (structuring)
            |_ SCENARIO_MAP (structuring)            
               |_ SCENARIO_MAP[INPUT_NAME] (namespace: INPUT_NS if INPUT_NS in SCENARIO_MAP keys / if not then  local, structuring, dynamic : SCENARIO_MAP[INPUT_NAME] != '' or is not None)
               |_ NS_IN_DF (dynamic: if sub_process_ns_in_build is not None)
        |_ DESC_OUT

    2) Description of DESC parameters:
        |_ DESC_IN
           |_ SUB_PROCESS_INPUTS:               All inputs for driver builder in the form of ProcessBuilderParameterType type
           |_ SCENARIO_MAP:                     All inputs for driver builder in the form of a dictionary of four keys
                                                    INPUT_NAME:           name of the variable to scatter
                                                    INPUT_NS:             Optional key: namespace of the variable to scatter if the INPUT_NS key is this scenario map. 
                                                                          If the key is not here then it is local to the driver.
                                                    OUTPUT_NAME:          name of the variable to overwrite
                                                    SCATTER_NS:           Optional key: Internal namespace associated to the scatter discipline
                                                                          it is a temporary input: its value is put to None as soon as scenario disciplines are instantiated
                                                    GATHER_NS:            Optional key: namespace of the gather discipline associated to the scatter discipline 
                                                                          (input_ns by default). Only used if autogather = True
                                                    NS_TO_UPDATE:         list of namespaces depending on the scatter namespace 
                                                                          (by default, we have the list of namespaces of the nested sub_process)                                                                                                                                          
                |_ SCENARIO_MAP[INPUT_NAME]     select your list of scenario names
                |_ NS_IN_DF :                   a map of ns name: value
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

    VALUES = "value"

    SUB_PROCESS_INPUTS = 'sub_process_inputs'
    REFERENCE = 'reference'

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

    NS_IN_DF = 'ns_in_df'

    default_process_builder_parameter_type = ProcessBuilderParameterType(
        None, None, 'Empty')

    default_scenario_map = {}
    default_scenario_map[INPUT_NAME] = None
    #default_scenario_map[INPUT_NS] = ''
    #default_scenario_map[OUTPUT_NAME] = ''
    #default_scenario_map[SCATTER_NS] = ''
    #default_scenario_map[GATHER_NS] = ''
    default_scenario_map[NS_TO_UPDATE] = []

    default_full_scenario_map = {}
    default_full_scenario_map[INPUT_NAME] = None
    default_full_scenario_map[INPUT_NS] = ''
    default_full_scenario_map[OUTPUT_NAME] = ''
    default_full_scenario_map[SCATTER_NS] = ''
    default_full_scenario_map[GATHER_NS] = ''
    default_full_scenario_map[NS_TO_UPDATE] = []

    DESC_IN = {
        SCENARIO_MAP: {'type': 'dict',
                       'structuring': True,
                       'default': default_scenario_map,
                       'user_level': 1,
                       'optional': False
                       }}

    DESC_IN.update(AddSubProcToDriver.DESC_IN)

    #DESC_OUT = {}

#################### End: Constants and parameters #######################

#################### Begin: Main methods ################################
    def __init__(self, sos_name, ee, map_name, cls_builder, autogather, gather_node, business_post_proc, associated_namespaces=[]):
        '''
        Constructor
        '''
        self.__autogather = autogather
        self.__gather_node = gather_node
        self.__build_business_io = business_post_proc
        self.__cls_builder = cls_builder
        BuildSoSDisciplineScatter.__init__(
            self, sos_name, ee, map_name, cls_builder, associated_namespaces=associated_namespaces)
        self._maturity = ''

        self.previous_sc_map_dict = None
        self.previous_sc_map_name = None

        # Possible values: 'Empty', 'Create', 'Replace', 'Unchanged'
        self.sc_map_build_status = 'Empty'

    def get_autogather(self):
        return self.__autogather

    def get_gather_node(self):
        return self.__gather_node

    def get_build_business_io(self):
        return self.__build_business_io

    def build(self):
        '''
            Overloaded scatter discipline method to "configure" scenarios
            Get and build builder from sub_process of driver
            Added to provide proc builder capability
            Reached from __configure_io in ee.py: self.factory.build() is going from build to build starting from root
            It comes before configuring()
        '''
        self.coupling_per_scatter = True
        # Remark: cls_builder and sc_map added in __init__ of sos_discipline_scatter
        # Remark: Nested build is done by creating (in SoSDisciplineScatter.build/build_sub_coupling)
        # as many coupling as names in the provided list of the data key
        # SCENARIO_MAP['input_name']

        # 1. Create and add scenario_map
        self.create_and_add_scenario_map()
        # 2. Create and add cls_builder
        AddSubProcToDriver.build(self)
        # 3. Associate map to discipline and build scatter
        self.associate_map_to_discipline_and_build_scatter()

    def configure(self):
        """
            Overloaded SoSDiscipline method
            Configuration of the Build driver
            Reached from __configure_io in ee.py: self.root_process.configure_io() is going from confiure to configure starting from root
            It comes after build()
        """

        # if self._data_in == {} or len(self.get_cls_builder()) == 0:

        BuildSoSDisciplineScatter.configure(self)

    def is_configured(self):
        '''
        Function to modify is_configured if an added condition is needed
        '''
        return SoSDiscipline.is_configured(self)

    def setup_sos_disciplines(self):
        """
           Overload setup_sos_disciplines to create a dynamic desc_in
           Reached from configure() of sos_discipline [SoSDiscipline.configure in config() of scatter discipline].
        """

        # Update of non dynamic desc_in
        # Update default_scenario_map with list of subprocess ns and provide it
        # to  sc_map_name dynamic input if self.SCENARIO_MAP is still with
        # default values

        # 0. Update list of all ns of subprocess in default ns_to_update
        self.provide_possible_list_of_sub_process_ns()

        AddSubProcToDriver.setup_sos_disciplines(self)

        if self.check_if_ready_to_import_usecase():
            # 0. Dynamically add INST_DESC_IN and  INST_DESC_OUT if autogather is True
            # self.build_business_io()  # should be put in
            # setup_sos_disciplines !
            pass

    def run(self):
        '''
            Overloaded scatter discipline method
            Store business outputs in dictionaries if autogather is True
        '''
        if self.get_autogather():
            self.run_autogather()

#################### End: Main methods ################################

##################### Begin: Sub methods ################################
# Remark: those sub methods should be private functions

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

# Begin: Sub methods for proc builder to wrap specific driver dynamic
# inputs ####
    def setup_desc_in_dict_of_driver(self):
        """
            Create desc_in_dict for dynamic inputs of the driver depending on sub process
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sub_process()
            Function to be specified per driver
            Update of SCENARIO_MAP['input_name']
        """
        # Remark: the dynamic input SCENARIO_MAP['input_name'] is initially set in
        # build_inst_desc_in_with_map() in __init__ of sos_discipline_scatter
        desc_in_dict = BuildSoSDisciplineScatter.build_inst_desc_in_with_map(
            self)
        return desc_in_dict
# End: Sub methods for proc builder  to wrap specific driver dynamic
# inputs ####

##################### Begin: Sub methods for proc builder ################

    def create_and_add_scenario_map(self):
        '''
            Create and add scenario map
            Function needed in build(self)
            Function for proc builder only in vsMS
        '''
        if self.SCENARIO_MAP in self._data_in:
            sc_map_dict = self.get_sosdisc_inputs(self.SCENARIO_MAP)
            sc_map_name = sc_map_dict[self.INPUT_NAME]

            sc_map_ready = False
            sc_map_ns = None
            if self.INPUT_NS in sc_map_dict.keys():  # use of shared ns
                sc_map_ns = sc_map_dict[self.INPUT_NS]
                sc_map_ready = sc_map_name is not None and sc_map_name != '' and sc_map_ns is not None and sc_map_ns != ''
            else:  # use of local ns
                sc_map_ready = sc_map_name is not None and sc_map_name != ''

            if sc_map_ready:  # a filled sc_map is available
                # either Unchanged or Create or Replace
                # 1. set_sc_map_status
                self.set_sc_map_status(sc_map_dict)
                # 2. create and add map
                # in case of replace then we should update/clean the previous existing sc_map.
                # In case of cleaning, can this map be used by other and need
                # not to be clean ?
                if (self.sc_map_build_status == 'Create' or self.sc_map_build_status == 'Replace'):
                    # 1. Take care of the Replace (if any)
                    if (self.sc_map_build_status == 'Replace'):
                        self.clean_all_for_rebuild()
                    # 2. Do the create
                    # 2.1 add sc_map
                    self.ee.smaps_manager.add_build_map(
                        sc_map_name, sc_map_dict)
                    # 2.2 adapt the sc_map_ns namespace (if any)
                    if sc_map_ns is not None:
                        current_ns = self.ee.ns_manager.current_disc_ns
                        self.ee.ns_manager.add_ns(sc_map_ns, current_ns)
                        self.ee.ns_manager.update_others_ns_with_shared_ns(
                            self, sc_map_ns)

    def associate_map_to_discipline_and_build_scatter(self):
        '''
            Associate map to discipline and build scatter
            Automatically done when structuring variable has changed and OK here even if empty proc or map
            Function needed in build(self)
            Function for proc builder only in vsMS
        '''
        if self.SCENARIO_MAP in self._data_in:
            sc_map_dict = self.get_sosdisc_inputs(self.SCENARIO_MAP)
            sc_map_name = sc_map_dict[self.INPUT_NAME]
            # print(sc_map_dict[self.INPUT_NAME])
            # print(sc_map_dict[self.NS_TO_UPDATE])
            cls_builder = self.get_cls_builder()
            BuildSoSDisciplineScatter._associate_map_to_discipline(self,
                                                                   self.ee, sc_map_name, cls_builder)
            BuildSoSDisciplineScatter.build(self)
            # Dynamically add INST_DESC_IN and  INST_DESC_OUT if autogather is
            # True
            # self.build_business_io()  # should be put in
            # setup_sos_disciplines !

    def set_sc_map_status(self, sc_map_dict):
        '''
            State sc_map CRUD status
            The sc_map is defined by its dictionary
            Function needed in build(self)
            Function for proc builder only in vsMS
        '''
        # We come from outside driver process
        if sc_map_dict != self.previous_sc_map_dict:
            if self.previous_sc_map_dict is None:
                self.previous_sc_map_name = None
            else:
                self.previous_sc_map_name = self.previous_sc_map_dict[self.INPUT_NAME]
            self.previous_sc_map_dict = sc_map_dict
            # driver process with provided sc_map
            if self.previous_sc_map_name == '' or self.previous_sc_map_name is None:
                self.sc_map_build_status = 'Create'
            else:
                self.sc_map_build_status = 'Replace'
        else:
            self.sc_map_build_status = 'Unchanged'
        return self.sc_map_build_status

    def clean_all_for_rebuild(self):
        """
            Create_nested builders from their nested process.
            Function needed in build (if import map first) and build_driver_subproc
            Function for proc builder only in vsMS
        """
        # 1 remove all previously instantiated disciplines
        sub_names = []
        BuildSoSDisciplineScatter.clean_scattered_disciplines(self, sub_names)
        # 2 remove previous created scatter map of name
        # "self.previous_sc_map_name"

        # Remark: self.previous_sc_map_name is also self.sc_map.get_input_name()
        # print(self.previous_sc_map_name)
        # print(self.sc_map.get_input_name())
        #sc_map_name_to_remove = self.previous_sc_map_name
        # Why self.previous_sc_map_name is not with the good value (None
        # instead of scenario_list) in Test 05/Step 02 ?

        sc_map_name_to_remove = self.sc_map.get_input_name()
        self.ee.smaps_manager.remove_build_map(sc_map_name_to_remove)
        # 3 We "clean" also all dynamic inputs
        self.add_inputs({})

    def provide_possible_list_of_sub_process_ns(self):
        """
            Provide possible list of sub process ns
            Function needed in setup_sos_disciplines()
            Function for proc builder only in vsMS
        """
        if (self.SCENARIO_MAP in self._data_in and self.sc_map is None):
            sc_map_dict = self.get_sosdisc_inputs(self.SCENARIO_MAP)
            sc_map_name = sc_map_dict[self.INPUT_NAME]
            if sc_map_name == '' or sc_map_name is None:  # provide only if user has not provided already a scenario_map
                # Define a new scenario map with ns_to_update that is with all
                # sub_process namespaces
                new_scenario_map = {}
                new_scenario_map[self.INPUT_NAME] = None
                #new_scenario_map[INPUT_NS] = ''
                #new_scenario_map[OUTPUT_NAME] = ''
                #new_scenario_map[SCATTER_NS] = ''
                #new_scenario_map[GATHER_NS] = ''
                new_scenario_map[self.NS_TO_UPDATE] = self.ns_of_sub_proc
                driver_name = self.name
                self.dm.set_data(f'{self.ee.study_name}.{driver_name}.{self.SCENARIO_MAP}',
                                 'value', new_scenario_map, check_value=False)


##################### End: Sub methods for proc builder ##################

### Begin: Sub methods for proc builder to be specified in specific driver ####

    def get_cls_builder(self):
        return self.__cls_builder

    def set_cls_builder(self, value):
        self.__cls_builder = value
        # update also in mother class as attribute has other name
        # self.__builders
        self.set_builders(value)

    def set_ref_discipline_full_name(self):
        '''
            Specific function of the driver to define the full name of the reference disvcipline
            Function needed in _init_ of the driver
            Function to be specified per driver
        '''
        driver_name = self.name
        self.ref_discipline_full_name = f'{self.ee.study_name}.{driver_name}.{self.REFERENCE}'

        return

    def clean_driver_before_rebuilt(self):  # come from build_driver_subproc
        '''
            Specific function of the driver to clean all instances before rebuild and reset any needed parameter
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
            Function to be specified per driver
        '''
        # Create_nested builders from their nested process.
        self.clean_all_for_rebuild()
        # Also reset input mat in desc_in to default map
        driver_name = self.name
        self.dm.set_data(f'{self.ee.study_name}.{driver_name}.{self.SCENARIO_MAP}',
                         'value', self.default_scenario_map, check_value=False)
        self.previous_sc_map_dict = None
        self.previous_sc_map_name = None
        # We "clean" also all dynamic inputs to be reloaded by
        # the usecase
        self.add_inputs({})  # is it needed ?
        return

    # come from build_driver_subproc# come from build_driver_subproc
    def get_ns_of_driver(self):
        '''
            Specific function of the driver to get ns of driver
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
            Function to be specified per driver
        '''
        sc_map_dict = self.get_sosdisc_inputs(self.SCENARIO_MAP)
        if self.INPUT_NS in sc_map_dict.keys():
            ns_of_driver = self.get_ns_of_driver()
            sc_map_ns = sc_map_dict[self.INPUT_NS]
            if sc_map_ns is not None:
                ns_of_driver = [sc_map_ns]
            else:
                ns_of_driver = []
        else:
            ns_of_driver = []
        return ns_of_driver

    def add_reference_instance(self):
        """
            Add a 'reference' discipline instance (if not already existing) in data manager to
            allow to load data from usecase
            Function needed in manage_import_inputs_from_sub_process()
            Function to be specified per driver
        """
        sc_map_dict = self.get_sosdisc_inputs(self.SCENARIO_MAP)
        map_name = sc_map_dict[self.INPUT_NAME]

        current_scenario_list = self.get_sosdisc_inputs(map_name)
        new_scenario_list = current_scenario_list
        if 'reference' not in current_scenario_list:
            new_scenario_list = new_scenario_list + ['reference']
            driver_name = self.name
            self.dm.set_data(f'{self.ee.study_name}.{driver_name}.{map_name}',
                             'value', new_scenario_list, check_value=False)

    def update_proc_builder_status(self):
        """
            Functions to update proc builder status
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sub_process()
            Function to be specified per driver
        """
        # update status
        self.sub_proc_build_status = 'Unchanged_SP'
        self.sc_map_build_status = 'Unchanged'

    def check_if_ready_to_provide_driver_inputs(self):
        """
            Functions to check if ready for proc build setup
            Function needed in 
            Function to be specified per driver
        """
        ready_to_provide_driver_inputs = self.sc_map is not None
        return ready_to_provide_driver_inputs

    def check_if_ready_to_import_usecase(self):
        """
            Functions to check if ready to import usecase data
            Function needed in setup_sos_disciplines()
            Function to be specified per driver
        """
        ready_to_import_usecase = self.sc_map is not None and self.sub_proc_build_status != 'Empty_SP' and self.sc_map.get_input_name() in self._data_in
        return ready_to_import_usecase

    def check_if_need_for_cleaning(self):
        """
            Functions to check if need for cleaning
            Function needed in build_driver_subproc()
            Function to be specified per driver
        """
        need_for_cleaning = self.sub_proc_build_status == 'Replace_SP' and self.sc_map_build_status == 'Unchanged'
        return need_for_cleaning
#### End: Sub methods for proc builder to be specified in specific driver ####
