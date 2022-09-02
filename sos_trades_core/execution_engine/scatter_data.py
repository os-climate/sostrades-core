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


class ScatterDataException(Exception):
    pass


class SoSScatterData(SoSDiscipline):
    '''
    Specification: ScatterData discipline collects inputs and distributes them in outputs
    '''

    # ontology information
    _ontology_data = {
        'label': 'Scatter Data',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-project-diagram fa-fw',
        'version': '',
    }

    def __init__(self, sos_name, ee, map_name, parent=None):
        '''
        CLass to scatter data
        '''
        self.name = sos_name
        self.sc_map = ee.smaps_manager.get_data_map(map_name)

        SoSDiscipline.__init__(self, sos_name, ee)

        # add input_name and scatter_var_name to inst_desc_in
        self.build_inst_desc_in()
        self.scatter_values = None

        # check dataframe column info
        if 'dataframe' in self.sc_map.map[
            self.sc_map.INPUT_TYPE] and self.sc_map.SCATTER_COLUMN_NAME not in self.sc_map.map:
            raise ScatterDataException(
                f'At least one input type is a dataframe but the attribute "scatter_column_name" in not present in map: {self.sc_map.map}')

    def configure(self):
        '''
        Overloaded SoSDiscipline method
        '''
        if self.sc_map.get_scatter_var_name() not in self._data_in:
            # first call configure to add scatter var name in data_in
            SoSDiscipline.configure(self)
        else:
            new_scatter_value = self.get_sosdisc_inputs(
                self.sc_map.get_scatter_var_name())
            if new_scatter_value is not None and new_scatter_value != self.scatter_values:
                # add sub_varnames to inst_desc_out
                self.build_inst_desc_out()

            SoSDiscipline.configure(self)

            store_outputs = False
            # if var_to_scatter is an input, scatter_data need to be run during
            # configure step
            for input in self.sc_map.get_input_name():
                if self._data_in[input][self.IO_TYPE] == self.IO_TYPE_IN:
                    store_outputs = True
            # run scatter_data to store outputs in dm
            if store_outputs:
                self.store_scatter_outputs(store_in_dm=True)

    def clean_data_out(self, new_names):
        '''
        Remove keys built with old scatter names in inst_desc_out, data_out and data manager
        '''
        names_to_remove = []
        if self.scatter_values is not None and self.scatter_values != new_names:
            for name in self.scatter_values:
                if not name in new_names:
                    names_to_remove.append(name)

        self.scatter_values = deepcopy(new_names)

        outputs_to_remove = []
        for name in names_to_remove:
            for output_name in self.sc_map.get_output_name():
                outputs_to_remove.append(f'{name}.{output_name}')

        self.clean_variables(outputs_to_remove, self.IO_TYPE_OUT)

    def build_inst_desc_in(self):
        '''
        Construct the inst_desc_in of the scatter discipline (the one which only scatter data)
        with the dict to scatter and the list to scatter on 
        '''
        input_name_list = self.sc_map.get_input_name()
        input_type_list = self.sc_map.get_input_type()
        input_ns = self.sc_map.get_input_ns()
        scatter_var_name = self.sc_map.get_scatter_var_name()

        # add variable to scatter to inst_desc_in
        for input_name, input_type in zip(input_name_list, input_type_list):
            if input_name not in self.inst_desc_in:
                # store input_name as structuring variable to be able to detect changes during configure step
                # when scatter outputs are inputs of other disciplines, outputs
                # have to be stored during configure step for each input change
                add_to_desc_in = {input_name: {
                    self.TYPE: input_type, self.VISIBILITY: self.SHARED_VISIBILITY, self.NAMESPACE: input_ns,
                    self.STRUCTURING: True}}
                self.inst_desc_in.update(add_to_desc_in.copy())

        # add scatter_var_name to inst_desc_in using specified visibility
        if scatter_var_name not in input_name_list:
            scatter_var_ns = self.ee.smaps_manager.get_input_ns_from_build_map(
                scatter_var_name)
            #scatter_var_type = self.ee.smaps_manager.get_input_type_from_build_map(
                #scatter_var_name)
            scatter_var_type = 'list'
            scatter_var_subtype = {'list':'string'}
            add_to_desc_in = {f'{scatter_var_name}': {
                self.TYPE: scatter_var_type,self.SUBTYPE:scatter_var_subtype, self.VISIBILITY: self.SHARED_VISIBILITY, self.NAMESPACE: scatter_var_ns,
                self.STRUCTURING: True}}
            self.inst_desc_in.update(add_to_desc_in.copy())

    def build_inst_desc_out(self):
        '''
        Construct the inst_desc_out of the scatter discipline (the one which only scatter data)
        with the coupled keys defined in the setup  
        '''
        if self.sc_map.OUTPUT_NS in self.sc_map.map:
            output_ns = self.sc_map.get_output_ns()
        else:
            output_ns = self.sc_map.get_input_ns()
        output_name_list = self.sc_map.get_output_name()
        output_type_list = self.sc_map.get_output_type()
        scatter_var_name = self.sc_map.get_scatter_var_name()

        keys_list = self.get_sosdisc_inputs(scatter_var_name)
        self.clean_data_out(keys_list)

        if keys_list is not None:
            for build_name in keys_list:
                for output_name, output_type in zip(output_name_list, output_type_list):
                    if f'{build_name}.{output_name}' not in self._data_out:
                        add_to_desc_out = {self.TYPE: output_type,
                                           self.VISIBILITY: self.SHARED_VISIBILITY, self.NAMESPACE: output_ns}
                        self.inst_desc_out.update(
                            {f'{build_name}.{output_name}': add_to_desc_out.copy()})

    def fill_subtype_descriptor(self):
        """ Redefinition of sos_discipline's fill_subtype_descriptor method
        to fill the subtype_descriptors of input variables of a scatter data
        """
        input_name_list = self.sc_map.get_input_name()
        input_type_list = self.sc_map.get_input_type()
        output_ns = self.sc_map.get_output_ns()
        scatter_var_name = self.sc_map.get_scatter_var_name()
        new_scatter_inputs = self.get_sosdisc_inputs(scatter_var_name)
        output_name_list = self.sc_map.get_output_name()
        if len(new_scatter_inputs) > 0:
            first_scatter_node = new_scatter_inputs[0]
            i = 0
            for input_name, input_type in zip(input_name_list, input_type_list):
                output_ns_name = self.ee.ns_manager.disc_ns_dict[self]['others_ns'][output_ns].get_value()
                corresponding_output = f'{output_ns_name}.{first_scatter_node}.{output_name_list[i]}'
                type_of_output = self.ee.dm.get_data(corresponding_output, self.TYPE)
                subtype_descriptor = None

                if input_type == 'dict':
                    if type_of_output not in ['list', 'dict']:
                        subtype_descriptor = {'dict': type_of_output}
                    else:
                        subtype_descriptor = self.ee.dm.get_data(corresponding_output, self.SUBTYPE)
                        if subtype_descriptor is not None:
                            subtype_descriptor = {'dict': subtype_descriptor}

                if subtype_descriptor is not None:
                    self._data_in[input_name][self.SUBTYPE] = subtype_descriptor
                i += 1

    def run(self):

        self.store_scatter_outputs()

    def store_scatter_outputs(self, store_in_dm=False):
        '''
        Store outputs in dm 
        '''
        input_name_list = self.sc_map.get_input_name()
        output_name_list = self.sc_map.get_output_name()
        scatter_var_name = self.sc_map.get_scatter_var_name()
        scatter_column_name_list = self.sc_map.get_scatter_column_name()
        input_type_list = self.sc_map.get_input_type()

        scatter_list = self.get_sosdisc_inputs(
            scatter_var_name)

        dict_values = {}
        if scatter_list is not None:
            for i, (input_name, output_name, input_type) in enumerate(
                    zip(input_name_list, output_name_list, input_type_list)):
                to_scatter_variable = self.get_sosdisc_inputs(input_name)
                if to_scatter_variable is not None:
                    for scatter_name in scatter_list:
                        # check type, if it is a dataframe, use .loc
                        if input_type != 'dataframe':
                            if scatter_name in to_scatter_variable:
                                scatter_output_value = deepcopy(
                                    to_scatter_variable[scatter_name])
                            else:
                                self.logger.error(f'Value:"{scatter_name}" is missing in parameter {input_name} located at {self.get_var_full_name(input_name,self._data_in)}, update it before run!')
                        else:
                            col = scatter_column_name_list[i]
                            # check if column exists in dataframe
                            if col not in to_scatter_variable:
                                raise ScatterDataException(
                                    f'The column: {col} does not exist in the dataframe used in the scatter data defined in map: {self.sc_map.map}')
                            scatter_output_value = to_scatter_variable.loc[
                                to_scatter_variable[col] == scatter_name,].reset_index(drop=True
                                                                                       )
                        dict_values[f'{scatter_name}.{output_name}'] = scatter_output_value

        self.store_sos_outputs_values(dict_values, update_dm=store_in_dm)
