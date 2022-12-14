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
from sostrades_core.tools.controllers.simpy_formula import SympyFormula
from copy import deepcopy


STANDARD_LIST_TYPES = ['list', 'array']
TEMPORARY_LIST_TYPES = ['float_list', 'string_list', 'int_list']
POSSIBLE_VALUES_TYPES = ['int', 'float', 'string', 'bool']
RANGE_TYPES = ['int', 'float']


class CheckDataIntegrity():
    '''CheckDataIntegrity class is here to check the data integrity of a variable regarding its type or variable descriptor associated
    '''

    def __init__(self, sos_disc_class, dm):
        '''
        Constructor
        '''
        self.dm = dm
        self.new_check = False
        self.check_integrity_msg_list = []
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
        self.IS_FORMULA = self.sos_disc_class.IS_FORMULA
        self.IS_EVAL = self.sos_disc_class.IS_EVAL
        self.FORMULA = self.sos_disc_class.FORMULA

        self.formula_dict = {}

    def check_variable_type_and_unit(self, var_data_dict):
        '''
        Check variable data except value
        1st : Check if the type specified in the DM is a SoSTrades type
        2nd : Check if variable has a unit (if a unit is needed)
        '''
        check_integrity_msg = ''
        variable_type = var_data_dict[self.TYPE]
        variable_unit = var_data_dict[self.UNIT]
        if variable_type not in self.VAR_TYPE_MAP.keys():
            check_integrity_msg = f'Type {variable_type} not in allowed type {list(self.VAR_TYPE_MAP.keys())}'
        else:
                # check that the variable has a unit
            if variable_unit is None and variable_type not in self.NO_UNIT_TYPES and self.new_check:
                check_integrity_msg = "Unit is not defined"

        return check_integrity_msg

    def check_variable_value(self, var_data_dict, new_check):
        '''
        Check the value of a data
        1st : Check the type of the value vs the type specified in the dm 
        2nd : Check if the value is in the range 
        3rd : Check if the value is in possible values
        4th : For dataframe, check if the value is OK with the df_descriptor 
        5th : CHeck the subtype of the value if there is subtypes
        '''
        self.check_integrity_msg_list = []
        self.new_check = new_check
        self.variable_io_type = var_data_dict[self.IO_TYPE]
        self.variable_type = var_data_dict[self.TYPE]
        self.variable_optional = var_data_dict[self.OPTIONAL]
        self.variable_value = var_data_dict[self.VALUE]
        self.variable_range = var_data_dict[self.RANGE]
        self.variable_possible_values = var_data_dict[self.POSSIBLE_VALUES]
        # check if data is and input and is not optional
        if self.variable_io_type == self.IO_TYPE_IN:
            if self.variable_value is None and not self.variable_optional:
                check_integrity_msg = f'Value is not set!'
                self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)
            elif self.variable_value is not None:
                # Do not check type of the value if the value is a formula
                if var_data_dict[self.IS_FORMULA]:
                    if var_data_dict[self.IS_EVAL]:
                        variable_formula = var_data_dict[self.FORMULA]
                    else:
                        variable_formula = self.variable_value
                    self.__check_formulas_in_variable(
                        var_data_dict, variable_formula)
                else:
                    # FIRST check type of the value
                    if not isinstance(self.variable_value, self.VAR_TYPE_MAP[self.variable_type]) and self.new_check:
                        check_integrity_msg = f'Value {self.variable_value} has not the type specified in datamanager which is {self.variable_type}'
                        self.__add_msg_to_check_integrity_msg_list(
                            check_integrity_msg)
                    else:
                        if self.variable_type in ['list', 'dict'] and self.new_check:
                            self.__check_subtype_descriptor(var_data_dict)
                        if self.variable_type == 'dataframe' and self.new_check:
                            self.__check_dataframe_descriptor(var_data_dict)
                        if self.variable_range is not None:
                            self.__check_variable_range(var_data_dict)
                        if self.variable_possible_values is not None:
                            self.__check_possible_values()

        return '\n'.join(self.check_integrity_msg_list)

    def __check_variable_range(self, var_data_dict):
        '''
        CHeck the data range of the data_dict
        '''

        if self.variable_type in RANGE_TYPES:

            # check type of range vs type of value
            self.__check_range_type_vs_value_type(
                self.variable_value, self.variable_range)
            # if the type for the range is the same than the value
            if self.check_integrity_msg_list == []:
                if not self.variable_range[0] <= self.variable_value <= self.variable_range[1]:
                    check_integrity_msg = f'Value {self.variable_value} is not in range {self.variable_range}'
                    self.__add_msg_to_check_integrity_msg_list(
                        check_integrity_msg)
        elif self.variable_type in STANDARD_LIST_TYPES + TEMPORARY_LIST_TYPES:
            if self.SUBTYPE in var_data_dict:
                variable_subtype = var_data_dict[self.SUBTYPE]
                if variable_subtype['list'] in RANGE_TYPES:
                    for sub_value in self.variable_value:
                        self.__check_range_type_vs_value_type(
                            sub_value, self.variable_range)
                    if self.check_integrity_msg_list == []:
                        for i, sub_value in enumerate(self.variable_value):
                            if not self.variable_range[0] <= sub_value <= self.variable_range[1]:
                                check_integrity_msg_range = f'Value {self.variable_value} at index {i} is not in range {self.variable_range}'
                                self.__add_msg_to_check_integrity_msg_list(
                                    check_integrity_msg_range)
                else:
                    check_integrity_msg = f'Type {self.variable_type} does not support *range*'
                    self.__add_msg_to_check_integrity_msg_list(
                        check_integrity_msg)
            else:
                pass
                # subtype should be declared in any way ?
        else:
            check_integrity_msg = f'Type {self.variable_type} does not support *range*'
            self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)

    def __check_range_type_vs_value_type(self, value, variable_range):
        '''
        Check the type of the first value in the range vs the type of the value 
        '''

        if not can_cast(type(value), type(variable_range[0])):
            check_integrity_msg_range_type = f'Type of {value} ({type(value)}) not the same as the type of {variable_range[0]} ({type(variable_range[0])}) in range list'
            self.__add_msg_to_check_integrity_msg_list(
                check_integrity_msg_range_type)

    def __check_possible_values(self):
        '''
        Check the possible values of the data_dict
        '''
        check_integrity_msg = ''

        if self.variable_type in POSSIBLE_VALUES_TYPES:
            if self.variable_value not in self.variable_possible_values:
                check_integrity_msg = f'Value {self.variable_value} not in *possible values* {self.variable_possible_values}'
                self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)
        elif self.variable_type in STANDARD_LIST_TYPES + TEMPORARY_LIST_TYPES:
            for sub_value in self.variable_value:
                if sub_value not in self.variable_possible_values:
                    check_integrity_msg_poss_values = f'Value {sub_value} in list {self.variable_value} not in *possible values* {self.variable_possible_values}'
                    self.__add_msg_to_check_integrity_msg_list(
                        check_integrity_msg_poss_values)
        else:
            check_integrity_msg = f'Type {self.variable_type} does not support *possible values*'
            self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)

    def __check_dataframe_descriptor(self, var_data_dict):
        '''
        Check dataframe descriptor of the data_dict vs the value if the dataframe is unlocked
        '''

        dataframe_descriptor = var_data_dict[self.DATAFRAME_DESCRIPTOR]
        dataframe_edition_locked = var_data_dict[self.DATAFRAME_EDITION_LOCKED]

        # Dataframe editable in GUI but no dataframe descriptor
        if dataframe_descriptor is None and not dataframe_edition_locked:
            check_integrity_msg = 'No dataframe descriptor set'
            self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)
        elif not dataframe_edition_locked:

            for key in dataframe_descriptor:
                df_descriptor_well_defined = True
                # Check column data well described
                if len(dataframe_descriptor[key]) != 3:
                    check_integrity_msg_df_descriptor = 'Partial dataframe descriptor set up'
                    self.__add_msg_to_check_integrity_msg_list(
                        check_integrity_msg_df_descriptor)
                    df_descriptor_well_defined = False
                # Check column type authorised
                elif dataframe_descriptor[key][0] not in self.VAR_TYPE_MAP.keys():
                    check_integrity_msg_df_descriptor = f'Dataframe descriptor has a column type ' \
                        f'{dataframe_descriptor[key][0]} not in allowed type {list(self.VAR_TYPE_MAP.keys())}'
                    df_descriptor_well_defined = False
                    self.__add_msg_to_check_integrity_msg_list(
                        check_integrity_msg_df_descriptor)

            if df_descriptor_well_defined:
                for key in self.variable_value.columns:
                    if key not in dataframe_descriptor:
                        check_integrity_msg_df_descriptor = f'Dataframe value has a column {key} but the dataframe descriptor has not, df_descriptor keys : {dataframe_descriptor.keys()}'
                        self.__add_msg_to_check_integrity_msg_list(
                            check_integrity_msg_df_descriptor)
                    else:
                        self.__check_dataframe_column_with_df_descriptor(
                            self.variable_value[key], dataframe_descriptor[key], key)

    def __add_msg_to_check_integrity_msg_list(self, new_msg):
        '''
        Add message in the message_list and join at the end of the function
        '''
        if new_msg != '':
            self.check_integrity_msg_list.append(new_msg)

    def __check_dataframe_column_with_df_descriptor(self, column, column_descriptor, key):
        '''
        Check the tuple of the column in the dataframe descriptor with the dataframe column value
        ex : ('string', None, False)
        '''
        column_type = column_descriptor[0]
        column_range = column_descriptor[1]
        values_in_column = column.values

        if not all(isinstance(item, self.VAR_TYPE_MAP[column_type]) for item in values_in_column):
            check_integrity_msg = f'Dataframe values in column {key} are not as type {column_type} requested in the dataframe descriptor'
            self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)
        elif column_range is not None and len(column_range) == 2:
            if not all(item < column_range[1] for item in values_in_column) and all(column_range[0] < item for item in values_in_column):
                check_integrity_msg = f'Dataframe values in column {key} are not in the range {column_range} requested in the dataframe descriptor'
                self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)

    def __check_subtype_descriptor(self, var_data_dict):
        '''
        Check subtype descriptor of the data_dict vs the value for list and dict
        '''

        if self.SUBTYPE in var_data_dict:
            variable_subtype = var_data_dict[self.SUBTYPE]
            self.__check_subtype(variable_subtype, self.variable_type,
                                 self.variable_value)

    def __check_subtype(self, subtype, type_to_check, variable_value):
        """This function checks that the subtype given to a list or dict is compliant
        with the defined standard for subtype and the value is compliant with the defined subtype descriptor
        """

        if not isinstance(subtype, dict):
            check_integrity_msg = 'Subtype descriptor must be a dictionnary'
            self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)
        elif list(subtype.keys())[0] != type_to_check or len(subtype.keys()) != 1:
            check_integrity_msg = f'Subtype descriptor should have as unique key the keyword {type_to_check} because the variable type is {type_to_check}'
            self.__add_msg_to_check_integrity_msg_list(check_integrity_msg)
        elif isinstance(subtype[type_to_check], dict):
            if isinstance(variable_value, dict):
                for sub_value in variable_value.values():
                    self.__check_subtype(
                        subtype[type_to_check], 'dict', sub_value)
            elif isinstance(variable_value, list):
                for value in variable_value:
                    self.__check_subtype(subtype[type_to_check], 'dict', value)
        else:
            if isinstance(variable_value, dict):
                for sub_value in variable_value.values():
                    if not isinstance(sub_value, self.VAR_TYPE_MAP[subtype[type_to_check]]):
                        check_integrity_msg_subtype = f'Value {sub_value} in {variable_value} should be a {subtype[type_to_check]} according to subtype descriptor {subtype}'
                        self.__add_msg_to_check_integrity_msg_list(
                            check_integrity_msg_subtype)
            elif isinstance(variable_value, list):
                for sub_value in variable_value:
                    if not isinstance(sub_value, self.VAR_TYPE_MAP[subtype[type_to_check]]):
                        check_integrity_msg_subtype = f'Value {sub_value} in {variable_value} should be a {subtype[type_to_check]} according to subtype descriptor {subtype}'
                        self.__add_msg_to_check_integrity_msg_list(
                            check_integrity_msg_subtype)
            else:
                if not isinstance(variable_value, self.VAR_TYPE_MAP[type_to_check]):
                    check_integrity_msg_subtype = f'Value {variable_value} should be a {type_to_check} according to subtype descriptor {subtype}'
                    self.__add_msg_to_check_integrity_msg_list(
                        check_integrity_msg_subtype)

    def __check_formulas_in_variable(self, var_data_dict, variable_formula):

        if self.variable_type == 'dataframe':
            for column in variable_formula.columns:
                # if string that should be a formula but error on the typo of formula cannot be raise as error
                # because maybe the string is not a formula but only a string
                if type(variable_formula[column][0]) == type('str') and variable_formula[column][0].startswith('formula:'):
                    formula = variable_formula[column].values[0].split(':')[
                        1]
                    self.__check_formula(formula)

        elif var_data_dict[self.TYPE] == 'dict':
            for key, value in variable_formula.items():
                if type(value) == type('str') and value.startswith('formula:'):
                    formula = variable_formula[key].split(':')[
                        1]
                    self.__check_formula(formula)
        else:
            if isinstance(variable_formula, str):
                variable_split_list = variable_formula.split(':')
                if len(variable_split_list) != 2:
                    formula_error_msg = 'Formula has to start with "formula:"'
                    self.__add_msg_to_check_integrity_msg_list(
                        formula_error_msg)
                else:
                    formula = variable_split_list[
                        1]
                    self.__check_formula(formula)
            else:
                formula_error_msg = f'Variable is referenced as formula, but no formula is given'
                self.__add_msg_to_check_integrity_msg_list(
                    formula_error_msg)

    def __check_formula(self, formula):
        '''
        Check a single formula 
        '''
        err_msg = None
        try:
            sympy_formula = SympyFormula(formula)
        except Exception as e:
            err_msg = str(e)
            self.__add_msg_to_check_integrity_msg_list(str(err_msg))
        if err_msg is None:
            self.__fill_formula_dict(
                sympy_formula)
            self.__check_formula_dict()

    def __fill_formula_dict(self, sympy_formula):
        """
        build dict with all formulas and parameters to evaluate :formula given
        """

        parameter_list = sympy_formula.get_token_list()
        parameter_list.sort()
        # look at each parameter of the formula
        for parameter in parameter_list:
            # if parameter is a variable in dm, check if it s a formula or not.
            # If formula, keep the exploitable part
            if parameter in self.dm.data_id_map:
                if self.dm.get_data(parameter, self.IS_FORMULA):
                    self.formula_dict[parameter] = self.dm.get_value(
                        parameter).split(':')[1]
                else:
                    self.formula_dict[parameter] = self.dm.get_value(
                        parameter)
            # if parameter not in dm, then it s a key of dict or df
            else:
                # first identify in which df/dict the parameter is
                splitted_parameter = parameter.split('.')
                el_key = splitted_parameter.pop()
                el_name_space = '.'.join(splitted_parameter)
                if el_name_space not in self.dm.data_id_map:
                    formula_error_msg = f'Parameter {parameter} does not exist in the formula'
                    self.__add_msg_to_check_integrity_msg_list(
                        formula_error_msg)
                # dataframe case
                else:
                    parameter_type = self.dm.get_data(el_name_space, self.TYPE)
                    parameter_value = self.dm.get_value(el_name_space)
                    if parameter_type == 'dataframe':
                        if el_key not in parameter_value.columns:
                            formula_error_msg = f'Column {el_key} does not exist in dataframe {el_name_space} as mentioned by {parameter}'
                            self.__add_msg_to_check_integrity_msg_list(
                                formula_error_msg)
                        else:
                            if isinstance(parameter_value[
                                    el_key].values[0], str):
                                self.formula_dict[parameter] = self.dm.get_value(
                                    el_name_space)[el_key].values[0].split(':')[1]
                            else:
                                self.formula_dict[parameter] = parameter_value[
                                    el_key].values[0]
                    # dict case
                    elif parameter_type == 'dict':
                        if el_key not in parameter_value.keys():
                            formula_error_msg = f'Key {el_key} does not exist in dict {el_name_space} as mentioned by {parameter}'
                            self.__add_msg_to_check_integrity_msg_list(
                                formula_error_msg)
                        else:
                            if isinstance(parameter_value[
                                    el_key], str):
                                self.formula_dict[parameter] = parameter_value[
                                    el_key].split(':')[1]
                            else:
                                self.formula_dict[parameter] = parameter_value[
                                    el_key]
                    else:
                        formula_error_msg = f'Type {parameter_type} for a parameter in a formula is not supported for formula at the moment'
                        self.__add_msg_to_check_integrity_msg_list(
                            formula_error_msg)

    def __check_formula_dict(self):
        """
        check if all parameter are in formula_dict. If not, fill in formula_dict
        twin_dict is created and updated. If twin_dict is different from formula_dict, formula_dict is updated and a new check is performed
        if twin_dict and formula_dict are the same, there is no more parameter to add.
        """
        twin_dict = deepcopy(self.formula_dict)
        for key in self.formula_dict.keys():
            # if it is a string, this is a formula to check
            if isinstance(self.formula_dict[key], str):
                sympy_formula = SympyFormula(
                    self.formula_dict[key])
                parameter_list = sympy_formula.get_token_list()
                # look at if each parameter are referenced
                for parameter in parameter_list:
                    if parameter not in self.formula_dict.keys():
                        sympy_formula = SympyFormula(self.formula_dict[key])
                        self.__fill_formula_dict(sympy_formula)
                        self.__update_dict(twin_dict, self.formula_dict)
        # if updates were made, a new check is performed. else, all parameter
        # needed are known
        if twin_dict != self.formula_dict:
            self.formula_dict = deepcopy(twin_dict)
            self.__check_formula_dict()

    def __update_dict(self, dict_to_update, filled_dict):
        """
        complete dict_to_update with the key : value of filled_dict if not possessed
        """
        for key, value in filled_dict.items():
            if key not in dict_to_update.keys():
                dict_to_update[key] = value
