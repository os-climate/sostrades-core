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
from copy import copy
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.ns_manager import NS_SEP


class SoSDisciplineException(Exception):
    pass


class SoSDisciplineGather(SoSDiscipline):
    '''
    Class that gather output data from a scatter discipline
    '''

    # ontology information
    _ontology_data = {
        'label': 'Gather',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-outdent fa-fw',
        'version': '',
    }
    EE_PATH = 'sos_trades_core.execution_engine'

    def __init__(self, sos_name, ee, map_name, cls_builder):
        '''
        Constructor
        '''
        self.__factory = ee.factory
        self.__gather_data_map = []
        self.instance_list = []
        self.associated_disc = None
        self._maturity = ''
        self.var_to_gather = {}

        self.map_name = map_name
        self.input_map_value = None
        self.sc_map = ee.smaps_manager.get_build_map(self.map_name)
        self.builder = cls_builder

        SoSDiscipline.__init__(self, sos_name, ee)

        # add input_name to inst_desc_in
        self.build_inst_desc_in_with_map()

    @property
    def gather_data_map(self):
        return self.__gather_data_map

    def get_gather_variable(self):
        '''
        Variables to gather are the variable in the DESC_OUT of the instantiator which are shared 
        We suppose that local variable must remain local and consequently are not gathered
        '''
        var_to_gather_dict = {}
        for disc in self.builder.discipline_dict.values():
            for out_var, out_dict in disc._data_out.items():
                if out_var not in var_to_gather_dict:
                    # if the visibility is not defined it means that it is
                    # Local
                    if self.VISIBILITY in out_dict and out_dict[self.VISIBILITY] == self.SHARED_VISIBILITY:
                        var_to_gather_dict[out_var] = out_dict

        return var_to_gather_dict

    def build_inst_desc_in_with_map(self):
        '''
        Consult the associated scatter build map and complete the inst_desc_in
        '''
        input_name = self.sc_map.get_input_name()
        input_type = self.sc_map.get_input_type()
        input_ns = self.sc_map.get_input_ns()

        scatter_desc_in = {input_name: {
            SoSDiscipline.TYPE: input_type, SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY, SoSDiscipline.NAMESPACE: input_ns, SoSDiscipline.STRUCTURING: True}}
        self.inst_desc_in.update(scatter_desc_in)

    def build_dynamic_inst_desc_in_gather_variables(self):
        '''
        Complete inst_desc_in with scatter outputs to gather
        '''
        scatter_var_name = self.sc_map.get_input_name()  # ac_name_list

        if scatter_var_name in self._data_in:

            gather_ns_in = self.sc_map.get_gather_ns_in()
            sub_names = self.get_sosdisc_inputs(
                scatter_var_name)

            if sub_names is not None:
                self.input_map_value = copy(sub_names)
                # Cleaning step of the inst_desc_in
                self.clean_inst_desc_in_with_sub_names(
                    sub_names)
                # Check if we need to add new variables by cross checking
                # sub_names and var_to_gather
                new_variables = {
                    f'{key}.{var_name}': value for var_name, value in self.var_to_gather.items() for key in sub_names if f'{key}.{var_name}' not in self.inst_desc_in.keys()}

                if len(new_variables) != 0:
                    self.add_new_variables_in_inst_desc_in(
                        new_variables, gather_ns_in)

    def add_new_variables_in_inst_desc_in(self, new_variables, gather_ns_in):
        '''
        Add a variable in the inst_desc_in with its full name and the gather_ns_in defined in the map 
        '''
        for new_variable, value_dict in new_variables.items():
            full_key = self.ee.ns_manager.compose_ns(
                [self.ee.ns_manager.get_shared_namespace_value(self, gather_ns_in), new_variable])
            if full_key in self.ee.dm.data_id_map.keys():
                var_name_dict = {new_variable: {SoSDiscipline.TYPE: value_dict[SoSDiscipline.TYPE],
                                                SoSDiscipline.IO_TYPE: SoSDiscipline.IO_TYPE_IN,
                                                SoSDiscipline.VAR_NAME: new_variable,
                                                SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY,
                                                SoSDiscipline.NAMESPACE: gather_ns_in}}
                self.inst_desc_in.update(var_name_dict)

    def build_inst_desc_out(self):
        '''
        Build the inst_desc_out of the gather with the inst_desc_out of the instantiator of the scatter
        for now each variable is gathered automatically in a varname : f'{var_name}_dict'
        '''
        gather_ns_out = self.sc_map.get_gather_ns_out()

        # get gather builder
        mod_path = f'{self.EE_PATH}.sos_discipline_gather.SoSDisciplineGather'
        cls_gather = self.__factory.get_disc_class_from_module(mod_path)

        for var_name in self.var_to_gather:

            if self.builder.cls == cls_gather:
                var_name_dict = var_name
            else:
                var_name_dict = f'{var_name}_dict'

            if var_name_dict not in self.inst_desc_out:
                var_name_dict = {var_name_dict:
                                 {SoSDiscipline.TYPE: 'dict',
                                  SoSDiscipline.IO_TYPE: SoSDiscipline.IO_TYPE_OUT,
                                  SoSDiscipline.VISIBILITY: SoSDiscipline.SHARED_VISIBILITY,
                                  SoSDiscipline.NAMESPACE: gather_ns_out,
                                  SoSDiscipline.USER_LEVEL: 3}}
                self.inst_desc_out.update(var_name_dict)

    def configure(self):
        '''
        Configure the gather : 
        - build the inst_desc_in with gather variables (only shared variables) with the value of the scatter var_name
        - build the inst_desc_out (variable_dict)
        - configure the discipline with completed inst_desc_in and inst_desc_out
        '''

        if self.sc_map.get_input_name() not in self._data_in:
            # update data_in/data_out with new inputs/outputs
            SoSDiscipline.configure(self)
        else:
            if self.get_sosdisc_inputs(self.sc_map.get_input_name()) is not None:
                if self.check_builders_to_gather_are_configured():
                    # get variables to gather
                    self.var_to_gather = self.get_gather_variable()
                    # add scatter outputs to inst_desc_in
                    self.build_dynamic_inst_desc_in_gather_variables()
                    # add gather dict outputs to inst_desc_out
                    self.build_inst_desc_out()
                    # update data_in/data_out with new inputs/outputs
                    SoSDiscipline.configure(self)
                    # update data_io if namespace has changed
                    self.update_data_io_with_modified_inst_desc_io()
                    # update inputs user level
                    self.update_inputs_user_level()

    def check_builders_to_gather_are_configured(self):
        '''
        Check if all builders with outputs data to gather are configured
        Return False at least one builder need to be configured
        '''
        for disc in self.builder.discipline_dict.values():
            if not disc.is_configured():
                self.set_configure_status(False)
                return False
        return True

    def is_configured(self):
        '''
        Return False at least one builder with outputs data to gather need to be configured or structuring variables have changed, True if not
        '''
        return SoSDiscipline.is_configured(self) and self.check_builders_to_gather_are_configured()

    def update_inputs_user_level(self):
        '''
        Set user level of inputs to Expert
        '''
        for key, value in self._data_in.items():
            if key != self.sc_map.get_input_name():
                value[SoSDiscipline.USER_LEVEL] = 3

    def update_data_io_with_modified_inst_desc_io(self):
        '''
        Update data_in and data_out with inst_desc_in and inst_desc_out which have been modified during a configure
        '''

        modified_inputs = {}
        modified_outputs = {}

        for key, value in self.inst_desc_in.items():
            if key in self._data_in and self._data_in[key][self.NAMESPACE] != value[self.NAMESPACE]:
                modified_inputs[key] = value

        for key, value in self.inst_desc_out.items():
            if key in self._data_out and self._data_out[key][self.NAMESPACE] != value[self.NAMESPACE]:
                modified_outputs[key] = value

        if len(modified_inputs) > 0:
            completed_modified_inputs = self._prepare_data_dict(
                self.IO_TYPE_IN, modified_inputs)
            self._data_in.update(completed_modified_inputs)

        if len(modified_outputs) > 0:
            completed_modified_outputs = self._prepare_data_dict(
                self.IO_TYPE_OUT, modified_outputs)
            self._data_out.update(completed_modified_outputs)

    def clean_inst_desc_in_with_sub_names(self, sub_names):
        '''
        Clean the inst_desc_in with names that doesn't exist in the scatter anymore, 
        Update the gather function of scatter variables
        '''
        keys_to_delete = []
        for var_in in self.inst_desc_in:
            if NS_SEP in var_in:
                full_key = self.get_var_full_name(var_in, self._data_in)
                if var_in.split(NS_SEP)[0] not in sub_names or self.ee.dm.get_data(full_key, self.DISCIPLINES_DEPENDENCIES) == [self.disc_id]:
                    keys_to_delete.append(var_in)

        self.clean_variables(keys_to_delete, self.IO_TYPE_IN)

    def run(self):
        '''
        Run function of the SoSGather : Collect variables to gather in a dict 
        Assemble the output dictionary and store it in the DM
        '''
        # get gather builder
        mod_path = f'{self.EE_PATH}.sos_discipline_gather.SoSDisciplineGather'
        cls_gather = self.__factory.get_disc_class_from_module(mod_path)

        new_values_dict = {}

        input_name = self.sc_map.get_input_name()  # ac_name_list
        gather_inputs = self.get_sosdisc_inputs(in_dict=True)

        for var_gather in self.var_to_gather:
            gather_dict = {}
            if self.builder.cls == cls_gather:
                for name in gather_inputs[input_name]:
                    for sub_name, value in gather_inputs[f'{name}.{var_gather}'].items():
                        gather_dict[f'{name}.{sub_name}'] = value
                new_values_dict[f'{var_gather}'] = gather_dict
            else:
                for name in gather_inputs[input_name]:
                    if f'{name}.{var_gather}' in gather_inputs:
                        gather_dict[name] = gather_inputs[f'{name}.{var_gather}']
                new_values_dict[f'{var_gather}_dict'] = gather_dict

        self.store_sos_outputs_values(new_values_dict)
    #-- Configure handling

    def get_maturity(self):
        '''FIX: solve conflicts between commits
            709b4be "Modify the exec_engine for evaluator processes" VJ
        and fb91c7d "maturity fixing (WIP)" CG '''
        # maturity = {}
        # return maturity
        return ''
