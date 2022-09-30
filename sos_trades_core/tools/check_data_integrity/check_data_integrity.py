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


def check_variable_type_and_unit(var_name, var_data_dict, sosdisc_class):
    '''
    Check variable data except value
    1st : Check if the type specified in the DM is a SoSTrades type
    2nd : Check if variable has a unit (if a unit is needed)
    '''
    check_integrity_msg = ''
    variable_type = var_data_dict[sosdisc_class.TYPE]
    variable_unit = var_data_dict[sosdisc_class.UNIT]
    if variable_type not in sosdisc_class.VAR_TYPE_MAP.keys():
        check_integrity_msg = f'Variable: {var_name} of type {variable_type} not in allowed type {list(sosdisc_class.VAR_TYPE_MAP.keys())}'
    else:
            # check that the variable has a unit
        if variable_unit is None and variable_type not in sosdisc_class.NO_UNIT_TYPES:
            check_integrity_msg = f"The variable {var_name} is used in {sosdisc_class} and unit is not defined"

    return check_integrity_msg


def check_variable_value(var_name, var_data_dict, sosdisc_class):
    '''
    Check the value of a data
    1st : Check the type of the value vs the type specified in the dm 
    2nd : Check if the value is in the range 
    3rd : Check if the value is in possible values
    4th : For dataframe, check if the value is OK with the df_descriptor 
    5th : CHeck the subtype of the value if there is subtypes
    '''
    check_integrity_msg = ''

    variable_io_type = var_data_dict[sosdisc_class.IO_TYPE]
    variable_type = var_data_dict[sosdisc_class.TYPE]
    variable_optional = var_data_dict[sosdisc_class.OPTIONAL]
    variable_value = var_data_dict[sosdisc_class.VALUE]
    variable_range = var_data_dict[sosdisc_class.RANGE]
    variable_possible_values = var_data_dict[sosdisc_class.POSSIBLE_VALUES]
    # check if data is and input and is not optional
    if variable_io_type == sosdisc_class.IO_TYPE_IN:
        if variable_value is None and not variable_optional:
            check_integrity_msg = f'Variable: {var_name} value is not set!'

        elif variable_value is not None:
            # FIRST check type of the value
            if not isinstance(variable_value, sosdisc_class.VAR_TYPE_MAP[variable_type]):
                pass
                #check_integrity_msg = f'Variable: {var_name} : the value {variable_value} has not the type specified in datamanager which is {variable_type}'
            else:
                if variable_range is not None:
                    check_integrity_msg = check_variable_range(
                        var_name, var_data_dict, sosdisc_class)
                if variable_possible_values is not None:
                    check_integrity_msg = check_possible_values(
                        var_name, var_data_dict, sosdisc_class)

                if variable_type == 'dataframe':
                    check_dataframe_descriptor(
                        var_name, var_data_dict, sosdisc_class)
    return check_integrity_msg


def check_variable_range(var_name, var_data_dict, sosdisc_class):
    '''
    CHeck the data range of the data_dict
    '''
    check_integrity_msg = ''
    variable_type = var_data_dict[sosdisc_class.TYPE]
    variable_value = var_data_dict[sosdisc_class.VALUE]
    variable_range = var_data_dict[sosdisc_class.RANGE]

    if variable_type in RANGE_TYPES:

        # check type of range vs type of value
        check_integrity_msg = check_range_type_vs_value_type(
            var_name, variable_value, variable_range)
        # if the type for the range is the same than the value
        if check_integrity_msg == '':
            if not variable_range[0] <= variable_value <= variable_range[1]:
                check_integrity_msg = f'Variable: {var_name} : {variable_value} is not in range {variable_range}'

    elif variable_type in STANDARD_LIST_TYPES + TEMPORARY_LIST_TYPES:
        if sosdisc_class.SUBTYPE in var_data_dict:
            variable_subtype = var_data_dict[sosdisc_class.SUBTYPE]
            if variable_subtype['list'] in RANGE_TYPES:
                for sub_value in variable_value:
                    check_integrity_msg += check_range_type_vs_value_type(
                        var_name, sub_value, variable_range)
                if check_integrity_msg == '':
                    for i, sub_value in enumerate(variable_value):
                        if not variable_range[0] <= sub_value <= variable_range[1]:
                            check_integrity_msg += f'Variable: {var_name} : The value {variable_value} at index {i} is not in range {variable_range}'
            else:
                check_integrity_msg = f'Variable: {var_name} type {variable_type} does not support *range*'
        else:
            pass
            # subtype should be declared in any way ?
    else:
        check_integrity_msg = f'Variable: {var_name} type {variable_type} does not support *range*'

    return check_integrity_msg


def check_range_type_vs_value_type(var_name, variable_value, variable_range):
    '''
    Check the type of each value in teh range vs the type of the value 
    '''
    check_integrity_msg = ''
    for var_range in variable_range:
        if not can_cast(type(variable_value), type(var_range)):
            check_integrity_msg = f'Variable: {var_name}: {variable_value} ({type(variable_value)}) not the same as {var_range} ({type(var_range)})'

    return check_integrity_msg


def check_possible_values(var_name, var_data_dict, sosdisc_class):
    '''
    Check the possible values of the data_dict
    '''
    check_integrity_msg = ''
    variable_type = var_data_dict[sosdisc_class.TYPE]
    variable_value = var_data_dict[sosdisc_class.VALUE]
    variable_possible_values = var_data_dict[sosdisc_class.POSSIBLE_VALUES]

    if variable_type in POSSIBLE_VALUES_TYPES:
        if variable_value not in variable_possible_values:
            check_integrity_msg = f'Variable: {var_name} : {variable_value} not in *possible values* {variable_possible_values}'
    elif variable_type in STANDARD_LIST_TYPES + TEMPORARY_LIST_TYPES:
        for sub_value in variable_value:
            if sub_value not in variable_possible_values:
                check_integrity_msg = f'Variable: {var_name} : {sub_value} in list {variable_value} not in *possible values* {variable_possible_values}'
    else:
        check_integrity_msg = f'Variable: {var_name}: type {variable_type} does not support *possible values*'

    return check_integrity_msg


def check_dataframe_descriptor(var_name, var_data_dict, sosdisc_class):
    '''
    Check dataframe descriptor of the data_dict vs the value
    '''
    check_integrity_msg = ''
    dataframe_descriptor = var_data_dict[sosdisc_class.DATAFRAME_DESCRIPTOR]
    dataframe_edition_locked = var_data_dict[sosdisc_class.DATAFRAME_EDITION_LOCKED]
    # Dataframe editable in GUI but no dataframe descriptor
    if dataframe_descriptor is None and not dataframe_edition_locked:
        check_integrity_msg = f'Variable: {var_name} has no dataframe descriptor set'

    elif not dataframe_edition_locked:
        for key in dataframe_descriptor:
            # Check column data well described
            if len(dataframe_descriptor[key]) != 3:
                check_integrity_msg = f'Variable: {var_name} has a partial dataframe descriptor set up'
            # Check column type authorised
            if dataframe_descriptor[key][0] not in sosdisc_class.VAR_TYPE_MAP.keys():
                check_integrity_msg = f'Variable: {var_name}, with dataframe descriptor has a column type ' \
                    f'{dataframe_descriptor[key][0]} not in allowed type {list(sosdisc_class.VAR_TYPE_MAP.keys())}'

    return check_integrity_msg
