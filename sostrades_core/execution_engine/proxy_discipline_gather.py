'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/17-2023/11/03 Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.ns_manager import NS_SEP


class SoSDisciplineException(Exception):
    pass


class ProxyDisciplineGather(ProxyDiscipline):
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

    def __init__(self, sos_name, ee, map_name, cls_builder, associated_namespaces=None):
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
        self.sc_map = ee.scattermap_manager.get_build_map(self.map_name)
        self.builder = cls_builder
        mod_path = f'{self.EE_PATH}.disciplines_wrappers.discipline_gather_wrapper.DisciplineGatherWrapper'
        cls = self.__factory.get_disc_class_from_module(mod_path)
        self.cls_gather = cls

        super().__init__(sos_name, ee, cls, associated_namespaces=associated_namespaces)

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
            for out_var, out_dict in disc.get_data_out().items():
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
        input_type = 'list'
        input_subtype_descriptor = {'list': 'string'}
        input_ns = self.sc_map.get_input_ns()

        scatter_desc_in = {input_name: {
            ProxyDiscipline.TYPE: input_type, ProxyDiscipline.SUBTYPE: input_subtype_descriptor,
            ProxyDiscipline.VISIBILITY: ProxyDiscipline.SHARED_VISIBILITY, ProxyDiscipline.NAMESPACE: input_ns,
            ProxyDiscipline.STRUCTURING: True}}
        self.inst_desc_in.update(scatter_desc_in)

    def build_dynamic_inst_desc_in_gather_variables(self):
        '''
        Complete inst_desc_in with scatter outputs to gather
        '''
        scatter_var_name = self.sc_map.get_input_name()  # ac_name_list

        if scatter_var_name in self.get_data_in():

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
                    f'{key}.{var_name}': value for var_name, value in self.var_to_gather.items() for key in sub_names if
                    f'{key}.{var_name}' not in self.inst_desc_in.keys()}

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
                var_name_dict = {new_variable: {ProxyDiscipline.TYPE: value_dict[ProxyDiscipline.TYPE],
                                                ProxyDiscipline.IO_TYPE: ProxyDiscipline.IO_TYPE_IN,
                                                ProxyDiscipline.VAR_NAME: new_variable,
                                                ProxyDiscipline.VISIBILITY: ProxyDiscipline.SHARED_VISIBILITY,
                                                ProxyDiscipline.NAMESPACE: gather_ns_in}}
                self.inst_desc_in.update(var_name_dict)

    def build_inst_desc_out(self):
        '''
        Build the inst_desc_out of the gather with the inst_desc_out of the instantiator of the scatter
        for now each variable is gathered automatically in a varname : f'{var_name}_dict'
        '''
        gather_ns_out = self.sc_map.get_gather_ns_out()

        # get gather builder
        mod_path = f'{self.EE_PATH}.proxy_discipline_gather.ProxyDisciplineGather'
        cls_gather = self.__factory.get_disc_class_from_module(mod_path)

        for var_name in self.var_to_gather:

            if self.builder.cls == cls_gather:
                var_name_dict = var_name
            else:
                var_name_dict = f'{var_name}_dict'

            if var_name_dict not in self.inst_desc_out:
                var_name_dict = {var_name_dict:
                                     {ProxyDiscipline.TYPE: 'dict',
                                      ProxyDiscipline.IO_TYPE: ProxyDiscipline.IO_TYPE_OUT,
                                      ProxyDiscipline.VISIBILITY: ProxyDiscipline.SHARED_VISIBILITY,
                                      ProxyDiscipline.NAMESPACE: gather_ns_out,
                                      ProxyDiscipline.USER_LEVEL: 3}}
                self.inst_desc_out.update(var_name_dict)

    def configure(self):
        '''
        Configure the gather : 
        - build the inst_desc_in with gather variables (only shared variables) with the value of the scatter var_name
        - build the inst_desc_out (variable_dict)
        - configure the discipline with completed inst_desc_in and inst_desc_out
        '''

        if self.sc_map.get_input_name() not in self.get_data_in():
            # update data_in/data_out with new inputs/outputs
            ProxyDiscipline.configure(self)
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
                    ProxyDiscipline.configure(self)
                    # update data_io if namespace has changed
                    self.update_data_io_with_modified_inst_desc_io()
                    # update inputs user level
                    self.update_inputs_user_level()
            else:
                # NB: without this else, the gather does not get reconfigured after its first configuration, which
                # involves a change in the structuring variables, not until its map is available. This spams the
                # configuration logs on the GUI with "100 iterations" warnings. This else clause needs to be reviewed
                # if the configuration loop is redesigned.
                ProxyDiscipline.configure(self)

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
        return ProxyDiscipline.is_configured(self) and self.check_builders_to_gather_are_configured()

    def update_inputs_user_level(self):
        '''
        Set user level of inputs to Expert
        '''
        for key, value in self.get_data_in().items():
            if key != self.sc_map.get_input_name():
                value[ProxyDiscipline.USER_LEVEL] = 3

    def update_data_io_with_modified_inst_desc_io(self):
        '''
        Update data_in and data_out with inst_desc_in and inst_desc_out which have been modified during a configure
        '''

        modified_inputs = {}
        modified_outputs = {}

        disc_in = self.get_data_in()
        disc_out = self.get_data_out()

        for key, value in self.inst_desc_in.items():
            if key in disc_in and disc_in[key][self.NAMESPACE] != value[self.NAMESPACE]:
                modified_inputs[key] = value

        for key, value in self.inst_desc_out.items():
            if key in disc_out and disc_out[key][self.NAMESPACE] != value[self.NAMESPACE]:
                modified_outputs[key] = value

        if len(modified_inputs) > 0:
            completed_modified_inputs = self._prepare_data_dict(
                self.IO_TYPE_IN, modified_inputs)
            inputs_var_ns_tuples = self._extract_var_ns_tuples(completed_modified_inputs)
            self._update_io_ns_map(inputs_var_ns_tuples, self.IO_TYPE_IN)
            self._update_data_io(zip(inputs_var_ns_tuples, completed_modified_inputs.values()), self.IO_TYPE_IN)
            self.build_simple_data_io(self.IO_TYPE_IN)
        if len(modified_outputs) > 0:
            completed_modified_outputs = self._prepare_data_dict(
                self.IO_TYPE_OUT, modified_outputs)
            outputs_var_ns_tuples = self._extract_var_ns_tuples(completed_modified_outputs)
            self._update_io_ns_map(outputs_var_ns_tuples, self.IO_TYPE_OUT)
            self._update_data_io(zip(outputs_var_ns_tuples, completed_modified_outputs.values()), self.IO_TYPE_OUT)
            self.build_simple_data_io(self.IO_TYPE_OUT)

    def clean_inst_desc_in_with_sub_names(self, sub_names):
        '''
        Clean the inst_desc_in with names that doesn't exist in the scatter anymore, 
        Update the gather function of scatter variables
        '''
        disc_in = self.get_data_in()
        keys_to_delete = []
        for var_in in self.inst_desc_in:
            if NS_SEP in var_in:
                full_key = self.get_var_full_name(var_in, disc_in)
                if var_in.split(NS_SEP)[0] not in sub_names or self.ee.dm.get_data(full_key,
                                                                                   self.DISCIPLINES_DEPENDENCIES) == [
                    self.disc_id]:
                    keys_to_delete.append(var_in)

        self.clean_variables(keys_to_delete, self.IO_TYPE_IN)

    def get_maturity(self):
        '''FIX: solve conflicts between commits
            709b4be "Modify the exec_engine for evaluator processes" VJ
        and fb91c7d "maturity fixing (WIP)" CG '''
        # maturity = {}
        # return maturity
        return ''

    def setup_sos_disciplines(self):
        """
        Method to be overloaded to add dynamic inputs/outputs using add_inputs/add_outputs methods.
        If the value of an input X determines dynamic inputs/outputs generation, then the input X is structuring and the item 'structuring':True is needed in the DESC_IN
        DESC_IN = {'X': {'structuring':True}}
        """
        pass

    def set_wrapper_attributes(self, wrapper):
        """ set the attribute attributes of wrapper
        """
        super().set_wrapper_attributes(wrapper)
        gather_attributes = {'input_name': self.get_var_full_name(self.sc_map.get_input_name(), self.get_data_in()),
                             'builder_cls': self.builder.cls,
                             'var_gather': self.var_to_gather,
                             'cls_gather': self.cls_gather,
                             'gather_ns': self.ee.ns_manager.get_shared_namespace_value(self,
                                                                                        self.sc_map.get_gather_ns())}
        wrapper.attributes.update(gather_attributes)
