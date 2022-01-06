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
from copy import deepcopy


class SoSGatherData(SoSDiscipline):
    '''
    Specification: GatherData discipline collects inputs and gathers them in outputs
    '''

    def __init__(self, sos_name, ee, map_name, parent=None):
        '''
        CLass to gather data
        '''
        self.name = sos_name
        self.sc_map = ee.smaps_manager.get_data_map(map_name)

        SoSDiscipline.__init__(self, sos_name, ee)
        # add scatter_var_name to inst_desc_in
        self.add_scatter_var_name()
        self.build_inst_desc_out()
        self.scatter_values = None

    def configure(self):
        '''
        Overloaded SoSDiscipline method
        '''
        if self.sc_map.get_scatter_var_name() not in self._data_in:
            # first call configure to add scatter var name in data_in
            SoSDiscipline.configure(self)
        else:
            if self.get_sosdisc_inputs(self.sc_map.get_scatter_var_name()) is not None:
                # add sub_varnames to inst_desc_in
                self.build_inst_desc_in()

                SoSDiscipline.configure(self)

                # update data_io if namespace has changed
                self.update_data_io_with_modified_inst_desc_io()
                # update inputs user level
                self.update_inputs_user_level()

    def update_inputs_user_level(self):
        '''
        Set user level of inputs to Expert
        '''
        for key, value in self._data_in.items():
            if key != self.sc_map.get_scatter_var_name():
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

    def clean_data_in(self, new_names):
        '''
        Remove keys built with old scatter names in inst_desc_in, data_in and data manager
        '''
        names_to_remove = []
        if self.scatter_values is not None and self.scatter_values != new_names:
            for name in self.scatter_values:
                if not name in new_names:
                    names_to_remove.append(name)

        self.scatter_values = deepcopy(new_names)

        inputs_to_remove = []
        for name in names_to_remove:
            for input_name in self.sc_map.get_input_name():
                if f'{name}.{input_name}' in self._data_in:
                    inputs_to_remove.append(f'{name}.{input_name}')

        self.clean_variables(inputs_to_remove, self.IO_TYPE_IN)

    def add_scatter_var_name(self):
        '''
        Add scatter_var_name to inst_desc_in using specified visibility
        '''
        scatter_var_name = self.sc_map.get_scatter_var_name()
        scatter_var_ns = self.ee.smaps_manager.get_input_ns_from_build_map(
            scatter_var_name)
        scatter_var_type = self.ee.smaps_manager.get_input_type_from_build_map(
            scatter_var_name)

        if scatter_var_name not in self._data_in:
            add_to_desc_in = {scatter_var_name: {
                self.TYPE: scatter_var_type, self.VISIBILITY: self.SHARED_VISIBILITY, self.NAMESPACE: scatter_var_ns, SoSDiscipline.STRUCTURING: True}}
            self.inst_desc_in.update(add_to_desc_in.copy())

    def build_inst_desc_in(self):
        '''
        Construct the inst_desc_in of the gather discipline
        with the value of the inputs per scatter value to scatter and the list to gather on 
        '''
        input_name_list = self.sc_map.get_input_name()
        input_type_list = self.sc_map.get_input_type()
        input_ns = self.sc_map.get_input_ns()
        scatter_var_name = self.sc_map.get_scatter_var_name()

        scatter_val_list = self.get_sosdisc_inputs(
            scatter_var_name)
        self.clean_data_in(scatter_val_list)

        # add variable to gather to inst_desc_in
        if scatter_val_list is not None:
            for scatter_val in scatter_val_list:
                for input_name, input_type in zip(input_name_list, input_type_list):
                    if f'{scatter_val}.{input_name}' not in self._data_in:
                        add_to_desc_in = {f'{scatter_val}.{input_name}': {
                            self.TYPE: input_type, self.VISIBILITY: self.SHARED_VISIBILITY, self.NAMESPACE: input_ns}}
                        self.inst_desc_in.update(add_to_desc_in.copy())

    def build_inst_desc_out(self):
        '''
        Construct the inst_desc_out of the gather discipline
        with dict of gathered inputs
        '''
        output_name_list = self.sc_map.get_output_name()
        output_type_list = self.sc_map.get_output_type()
        output_ns = self.sc_map.get_output_ns()

        for output_name, output_type in zip(output_name_list, output_type_list):
            if output_name not in self._data_in:
                add_to_desc_out = {output_name: {self.TYPE: output_type,
                                                 self.VISIBILITY: self.SHARED_VISIBILITY, self.NAMESPACE: output_ns, self.USER_LEVEL: 3}}
                self.inst_desc_out.update(add_to_desc_out.copy())

    def run(self):
        '''
        Overloaded SoSDiscipline method
        '''
        self.store_gather_outputs()

    def store_gather_outputs(self):
        '''
        Store outputs in dm 
        '''
        input_name_list = self.sc_map.get_input_name()
        output_name_list = self.sc_map.get_output_name()
        scatter_var_name = self.sc_map.get_scatter_var_name()

        scatter_list = self.get_sosdisc_inputs(
            scatter_var_name)

        dict_values = {}
        for input_name, output_name in zip(input_name_list, output_name_list):
            output_value = {}
            for scatter_name in scatter_list:
                output_value[scatter_name] = self.get_sosdisc_inputs(
                    f'{scatter_name}.{input_name}')

                dict_values[output_name] = output_value

        self.store_sos_outputs_values(dict_values)
