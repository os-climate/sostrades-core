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

from numpy import can_cast

STANDARD_LIST_TYPES = ['list', 'array']
TEMPORARY_LIST_TYPES = ['float_list', 'string_list', 'int_list']
POSSIBLE_VALUES_TYPES = ['int', 'float', 'string', 'bool']
RANGE_TYPES = ['int', 'float']


class CheckDataIntegrity():
    '''CheckDataIntegrity class is here to check the data integrity of a variable regarding its type or variable descriptor associated
    '''

    def __init__(self, sos_disc_class, new_check=True):
        '''
        Constructor
        '''
        self.new_check = new_check
        self.sos_disc_class = sos_disc_class
        self.TYPE = self.sos_disc_class.TYPE
        self.UNIT = self.sos_disc_class.UNIT
        self.VAR_TYPE_MAP = self.sos_disc_class.VAR_TYPE_MAP
        self.NO_UNIT_TYPES = self.sos_disc_class.NO_UNIT_TYPES
        self.IO_TYPE_IN = self.sos_disc_class.IO_TYPE_IN
        self.SUBTYPE = self.sos_disc_class.SUBTYPE
        self.IO_TYPE = self.sos_disc_class.IO_TYPE
        self.OPTIONAL = self.sos_disc_class.OPTIONAL
        self.RANGE = self.sos_disc_class.RANGE
        self.VALUE = self.sos_disc_class.VALUE
        self.POSSIBLE_VALUES = self.sos_disc_class.POSSIBLE_VALUES
        self.DATAFRAME_DESCRIPTOR = self.sos_disc_class.DATAFRAME_DESCRIPTOR
        self.DATAFRAME_EDITION_LOCKED = self.sos_disc_class.DATAFRAME_EDITION_LOCKED

    def check_variable_type_and_unit(self, var_name, var_data_dict):
        '''
        Check variable data except value
        1st : Check if the type specified in the DM is a SoSTrades type
        2nd : Check if variable has a unit (if a unit is needed)
        '''
        check_integrity_msg = ''
        variable_type = var_data_dict[self.TYPE]
        variable_unit = var_data_dict[self.UNIT]
        if variable_type not in self.VAR_TYPE_MAP.keys():
            check_integrity_msg = f'Variable: {var_name} of type {variable_type} not in allowed type {list(self.VAR_TYPE_MAP.keys())}'
        else:
                # check that the variable has a unit
            if variable_unit is None and variable_type not in self.NO_UNIT_TYPES:
                check_integrity_msg = f"The variable {var_name} unit is not defined"

        return check_integrity_msg

    def check_variable_value(self, var_name, var_data_dict):
        '''
        Check the value of a data
        1st : Check the type of the value vs the type specified in the dm 
        2nd : Check if the value is in the range 
        3rd : Check if the value is in possible values
        4th : For dataframe, check if the value is OK with the df_descriptor 
        5th : CHeck the subtype of the value if there is subtypes
        '''
        check_integrity_msg = ''

        self.variable_io_type = var_data_dict[self.IO_TYPE]
        self.variable_type = var_data_dict[self.TYPE]
        self.variable_optional = var_data_dict[self.OPTIONAL]
        self.variable_value = var_data_dict[self.VALUE]
        self.variable_range = var_data_dict[self.RANGE]
        self.variable_possible_values = var_data_dict[self.POSSIBLE_VALUES]
        # check if data is and input and is not optional
        if self.variable_io_type == self.IO_TYPE_IN:
            if self.variable_value is None and not self.variable_optional:
                check_integrity_msg = f'Variable: {var_name} value is not set!'

            elif self.variable_value is not None:
                # FIRST check type of the value

                if not isinstance(self.variable_value, self.VAR_TYPE_MAP[self.variable_type]) and self.new_check:
                    check_integrity_msg = f'Variable: {var_name} : the value {self.variable_value} has not the type specified in datamanager which is {self.variable_type}'
                else:
                    if self.variable_type in ['list', 'dict'] and self.new_check:
                        check_integrity_msg_subtypes = self.check_subtype_descriptor(
                            var_name, var_data_dict)
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_subtypes)
                    if self.variable_type == 'dataframe' and self.new_check:
                        check_integrity_msg_df_descriptor = self.check_dataframe_descriptor(
                            var_name, var_data_dict)
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_df_descriptor)

                    if self.variable_range is not None:
                        check_integrity_msg_range = self.check_variable_range(
                            var_name, var_data_dict)
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_range)
                    if self.variable_possible_values is not None:
                        check_integrity_msg_poss_values = self.check_possible_values(
                            var_name)
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_poss_values)

        return check_integrity_msg

    def check_variable_range(self, var_name, var_data_dict):
        '''
        CHeck the data range of the data_dict
        '''
        check_integrity_msg = ''

        if self.variable_type in RANGE_TYPES:

            # check type of range vs type of value
            check_integrity_msg_range_type = self.check_range_type_vs_value_type(
                var_name, self.variable_value, self.variable_range)
            check_integrity_msg = self.add_msg_to_check_integrity_msg(
                check_integrity_msg, check_integrity_msg_range_type)

            # if the type for the range is the same than the value
            if check_integrity_msg == '':
                if not self.variable_range[0] <= self.variable_value <= self.variable_range[1]:
                    check_integrity_msg = f'Variable: {var_name} : {self.variable_value} is not in range {self.variable_range}'

        elif self.variable_type in STANDARD_LIST_TYPES + TEMPORARY_LIST_TYPES:
            if self.SUBTYPE in var_data_dict:
                variable_subtype = var_data_dict[self.SUBTYPE]
                if variable_subtype['list'] in RANGE_TYPES:
                    for sub_value in self.variable_value:
                        check_integrity_msg += self.check_range_type_vs_value_type(
                            var_name, sub_value, self.variable_range)
                    if check_integrity_msg == '':
                        for i, sub_value in enumerate(self.variable_value):
                            if not self.variable_range[0] <= sub_value <= self.variable_range[1]:
                                check_integrity_msg_range = f'Variable: {var_name} : The value {self.variable_value} at index {i} is not in range {self.variable_range}'
                                check_integrity_msg = self.add_msg_to_check_integrity_msg(
                                    check_integrity_msg, check_integrity_msg_range)
                else:
                    check_integrity_msg = f'Variable: {var_name} type {self.variable_type} does not support *range*'
            else:
                pass
                # subtype should be declared in any way ?
        else:
            check_integrity_msg = f'Variable: {var_name} type {self.variable_type} does not support *range*'

        return check_integrity_msg

    def check_range_type_vs_value_type(self, var_name, value, variable_range):
        '''
        Check the type of the first value in the range vs the type of the value 
        '''
        check_integrity_msg = ''

        if not can_cast(type(value), type(variable_range[0])):
            check_integrity_msg_range_type = f'Variable: {var_name}: The type of {value} ({type(value)}) not the same as the type of {variable_range[0]} ({type(variable_range[0])}) in range list'
            check_integrity_msg = self.add_msg_to_check_integrity_msg(
                check_integrity_msg, check_integrity_msg_range_type)
        return check_integrity_msg

    def check_possible_values(self, var_name):
        '''
        Check the possible values of the data_dict
        '''
        check_integrity_msg = ''

        if self.variable_type in POSSIBLE_VALUES_TYPES:
            if self.variable_value not in self.variable_possible_values:
                check_integrity_msg = f'Variable: {var_name} : {self.variable_value} not in *possible values* {self.variable_possible_values}'
        elif self.variable_type in STANDARD_LIST_TYPES + TEMPORARY_LIST_TYPES:
            for sub_value in self.variable_value:
                if sub_value not in self.variable_possible_values:
                    check_integrity_msg_poss_values = f'Variable: {var_name} : {sub_value} in list {self.variable_value} not in *possible values* {self.variable_possible_values}'
                    check_integrity_msg = self.add_msg_to_check_integrity_msg(
                        check_integrity_msg, check_integrity_msg_poss_values)
        else:
            check_integrity_msg = f'Variable: {var_name}: type {self.variable_type} does not support *possible values*'

        return check_integrity_msg

    def check_dataframe_descriptor(self, var_name, var_data_dict):
        '''
        Check dataframe descriptor of the data_dict vs the value if the dataframe is unlocked
        '''
        check_integrity_msg = ''
        dataframe_descriptor = var_data_dict[self.DATAFRAME_DESCRIPTOR]
        dataframe_edition_locked = var_data_dict[self.DATAFRAME_EDITION_LOCKED]

        # Dataframe editable in GUI but no dataframe descriptor
        if dataframe_descriptor is None and not dataframe_edition_locked:
            check_integrity_msg = f'Variable: {var_name} has no dataframe descriptor set'

        elif not dataframe_edition_locked:

            for key in dataframe_descriptor:
                df_descriptor_well_defined = True
                # Check column data well described
                if len(dataframe_descriptor[key]) != 3:
                    check_integrity_msg_df_descriptor = f'Variable: {var_name} has a partial dataframe descriptor set up'
                    df_descriptor_well_defined = False
                # Check column type authorised
                elif dataframe_descriptor[key][0] not in self.VAR_TYPE_MAP.keys():
                    check_integrity_msg_df_descriptor = f'Variable: {var_name}, with dataframe descriptor has a column type ' \
                        f'{dataframe_descriptor[key][0]} not in allowed type {list(self.VAR_TYPE_MAP.keys())}'
                    df_descriptor_well_defined = False
                    check_integrity_msg = self.add_msg_to_check_integrity_msg(
                        check_integrity_msg, check_integrity_msg_df_descriptor)

            if df_descriptor_well_defined:
                for key in self.variable_value.columns:
                    if key not in dataframe_descriptor:
                        check_integrity_msg_df_descriptor = f'Variable: {var_name}, the dataframe value has a column {key} but the dataframe descriptor has not, df_descriptor keys : {dataframe_descriptor.keys()}'
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_df_descriptor)
                    else:
                        check_integrity_msg_column = self.check_dataframe_column_with_df_descriptor(
                            self.variable_value[key], dataframe_descriptor[key], var_name, key)
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_column)
        return check_integrity_msg

    def add_msg_to_check_integrity_msg(self, check_integrity_msg, new_msg):
        if new_msg != '':
            check_integrity_msg += new_msg
            if not new_msg.endswith('\n'):
                check_integrity_msg += '\n'
        return check_integrity_msg

    def check_dataframe_column_with_df_descriptor(self, column, column_descriptor, var_name, key):
        '''
        Check the tuple of the column in the dataframe descriptor with the dataframe column value
        ex : ('string', None, False)
        '''
        column_type = column_descriptor[0]
        column_range = column_descriptor[1]
        values_in_column = column.values
        check_integrity_msg = ''
        if not all(isinstance(item, self.VAR_TYPE_MAP[column_type]) for item in values_in_column):
            check_integrity_msg = f'Variable: {var_name}, all dataframe values in column {key} are not as type {column_type} requested in the dataframe descriptor'
        elif column_range is not None and len(column_range) == 2:
            if not all(item < column_range[1] for item in values_in_column) and all(column_range[0] < item for item in values_in_column):
                check_integrity_msg = f'Variable: {var_name}, all dataframe values in column {key} are not in the range {column_range} requested in the dataframe descriptor'

        return check_integrity_msg

    def check_subtype_descriptor(self, var_name, var_data_dict):
        '''
        Check subtype descriptor of the data_dict vs the value for list and dict
        '''
        check_integrity_msg = ''
        if self.SUBTYPE in var_data_dict:
            variable_subtype = var_data_dict[self.SUBTYPE]
            check_integrity_msg = self.check_subtype(var_name, variable_subtype, self.variable_type,
                                                     self.variable_value)

        return check_integrity_msg

    def check_subtype(self, var_name, subtype, type_to_check, variable_value):
        """This function checks that the subtype given to a list or dict is compliant
        with the defined standard for subtype and the value is compliant with the defined subtype descriptor
        """
        check_integrity_msg = ''
        if not isinstance(subtype, dict):
            check_integrity_msg = f'Variable: {var_name} :  The subtype descriptor must be a dictionnary'
        elif list(subtype.keys())[0] != type_to_check or len(subtype.keys()) != 1:
            check_integrity_msg = f'Variable: {var_name} : The subtype descriptor should have as unique key the keyword {type_to_check} because the variable type is {type_to_check}'
        elif isinstance(subtype[type_to_check], dict):
            if isinstance(variable_value, dict):
                for sub_value in variable_value.values():
                    check_integrity_msg_subtype = self.check_subtype(
                        var_name, subtype[type_to_check], 'dict', sub_value)
                    check_integrity_msg = self.add_msg_to_check_integrity_msg(
                        check_integrity_msg, check_integrity_msg_subtype)
            elif isinstance(variable_value, list):
                for value in variable_value:
                    check_integrity_msg_subtype = self.check_subtype(
                        var_name, subtype[type_to_check], 'dict', value)
                    check_integrity_msg = self.add_msg_to_check_integrity_msg(
                        check_integrity_msg, check_integrity_msg_subtype)
        else:
            if isinstance(variable_value, dict):
                for sub_value in variable_value.values():
                    if not isinstance(sub_value, self.VAR_TYPE_MAP[subtype[type_to_check]]):
                        check_integrity_msg_subtype = f'Variable: {var_name} : The value {sub_value} in {variable_value} should be a {subtype[type_to_check]} according to subtype descriptor {subtype}'
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_subtype)
            elif isinstance(variable_value, list):
                for sub_value in variable_value:
                    if not isinstance(sub_value, self.VAR_TYPE_MAP[subtype[type_to_check]]):
                        check_integrity_msg_subtype = f'Variable: {var_name} : The value {sub_value} in {variable_value} should be a {subtype[type_to_check]} according to subtype descriptor {subtype}'
                        check_integrity_msg = self.add_msg_to_check_integrity_msg(
                            check_integrity_msg, check_integrity_msg_subtype)
            else:
                if not isinstance(variable_value, self.VAR_TYPE_MAP[type_to_check]):
                    check_integrity_msg_subtype = f'Variable: {var_name} : The value {variable_value} should be a {type_to_check} according to subtype descriptor {subtype}'
                    check_integrity_msg = self.add_msg_to_check_integrity_msg(
                        check_integrity_msg, check_integrity_msg_subtype)
        return check_integrity_msg
