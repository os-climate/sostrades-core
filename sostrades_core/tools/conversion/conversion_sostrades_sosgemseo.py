'''
Copyright 2022 Airbus SAS
Modifications on 2023/02/21-2024/05/16 Copyright 2023 Capgemini

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
from copy import deepcopy
from functools import reduce

import numpy as np
from numpy import append, arange, array, delete, ndarray
from numpy import complex128 as np_complex128
from numpy import float64 as np_float64
from numpy import int32 as np_int32
from numpy import int64 as np_int64
from pandas import DataFrame
from itertools import chain

BASE_TYPE_EXCLUDED = ['int', 'string']
DEFAULT_EXCLUDED_COLUMNS = ['year', 'years']
VAR_TYPE_ID = 'type'
VAR_SUBTYPE_ID = 'subtype_descriptor'
TYPE_METADATA = "type_metadata"
DF_EXCLUDED_COLUMNS = 'dataframe_excluded_columns'
INT_MAP = (int, np_int32, np_int64, np_complex128)
FLOAT_MAP = (float, np_float64, np_complex128)
VAR_TYPE_MAP = {
    # an integer cannot be a float
    'int': INT_MAP,
    # a designed float can be integer (with python 3 no problem)
    'float': FLOAT_MAP + INT_MAP,
    'string': str,
    'string_list': list,
    'string_list_list': list,
    'float_list': list,
    'int_list': list,
    'list': list,
    'dict_list': list,
    'array': ndarray,
    'df_dict': dict,
    'dict': dict,
    'dataframe': DataFrame,
    'bool': bool,
}
NEW_VAR_TYPE_MAP_INVERTED = {str: 'string', list: 'list', ndarray: 'array', dict: 'dict', DataFrame: 'dataframe',
                             bool: 'bool'}

VAR_TYPE_GEMS = ['int', 'array', 'float_list', 'int_list']
STANDARD_TYPES = [int, float, np_int32, np_int64, np_float64, bool]
#    VAR_TYPES_SINGLE_VALUES = ['int', 'float', 'string', 'bool', 'np_int32', 'np_float64', 'np_int64']
NEW_VAR_TYPE_STRING = ['dict', 'dataframe', 'string', 'list', 'bool']
NEW_VAR_TYPE = [VAR_TYPE_MAP[typ] for typ in NEW_VAR_TYPE_STRING]

NEW_VAR_TYPE = list(set(chain.from_iterable(elt if isinstance(elt, tuple) else [elt] for elt in NEW_VAR_TYPE)))


def is_value_type_handled(val):
    return isinstance(val, tuple(VAR_TYPE_MAP.values())
                      ) or isinstance(val, np_complex128)


def get_nested_val(dict_in, keys):
    ''' returns the value of a nested dictionary of depth len(keys)
    output : d[keys[0]][..][keys[n]]
    '''

    def func_dic(dict_in, key): return dict_in[key]

    nested_val = reduce(func_dic, keys, dict_in)

    return nested_val


def convert_array_into_dict_old_version(arr_to_convert, new_data, val_datalist):
    # convert list into dict using keys from dm.data_dict
    if len(val_datalist) == 0:
        # means the dictionary is empty or None
        return {}
    else:
        while len(val_datalist) != 0:
            metadata = val_datalist.pop(0)
            _type = metadata['__type__']
            _keys = metadata['__key__']

            nested_keys = _keys[:-1]
            to_update = get_nested_val(new_data, nested_keys)
            _key = _keys[-1]
            # dictionaries

            if _type == dict:
                to_update[_key] = {}
                convert_array_into_dict_old_version(
                    arr_to_convert, new_data, val_datalist)
            # DataFrames
            elif _type == DataFrame:
                _df = convert_array_into_df(arr_to_convert, metadata)
                to_update[_key] = _df
                _size = metadata['__size__']
                arr_to_convert = delete(arr_to_convert, arange(_size))

            # int, float, or complex
            elif _type in [int, float, np_int32, np_int64, np_float64, np_complex128, bool]:
                _val = arr_to_convert[0]
                arr_to_convert = delete(arr_to_convert, [0])
                to_update[_key] = _type(_val)

            # numpy array or list
            elif _type in [list, ndarray]:
                _shape = metadata['__shape__']
                _size = metadata['__size__']
                _arr = arr_to_convert[:_size]
                _arr = _arr.reshape(_shape)
                if _type == list:
                    _arr = _arr.tolist()
                if 'known_values' in metadata:
                    # Means that we have a string somewhere in the list or
                    # array
                    for index_arr, metadata_ind in metadata['known_values'].items():
                        int_value = int(_arr[index_arr])
                        _arr[index_arr] = next((strg for strg, int_to_convert in metadata_ind['known_values'].items(
                        ) if int_to_convert == int_value), None)

                arr_to_convert = delete(arr_to_convert, arange(_size))

                to_update[_key] = _arr

            elif _type == str:
                to_convert = arr_to_convert[0]
                arr_to_convert = delete(arr_to_convert, [0])
                _val = next((strg for strg, int_to_convert in metadata['known_values'].items(
                ) if int_to_convert == to_convert), None)
                to_update[_key] = _type(_val)

            else:
                raise Exception(
                    f'The type {_type} in the dict {arr_to_convert} is not taken into account')
        return to_update


def convert_array_into_df(arr_to_convert, metadata, excluded_columns=DEFAULT_EXCLUDED_COLUMNS):
    # convert list into dataframe using columns from dm.data_dict
    _shape = metadata['__shape__']
    _size = metadata['__size__']
    _col = metadata['__columns__'].copy()
    _dtypes = metadata['__dtypes__'].copy()
    _arr = arr_to_convert[:_size]
    # to flatten by lines erase the option 'F' or put the 'C' option
    _arr = _arr.reshape(_shape, order='F')
    # create multi index columns if tuples in columns

    # Use the 2Darrays init which is 4 times faster than the dict initialization
    # if indices are stored we use them to reconstruct the dataframe
    if '__indices__' in metadata:
        df = DataFrame(data=_arr, columns=_col,
                       index=metadata['__indices__'])
    else:
        df = DataFrame(data=_arr, columns=_col)

    df_dtypes = df._mgr.get_dtypes()
    # if _mgr attribute does not exist in further versions switch to the following line (less efficient)
    # the following line create a series to visualize dtypes
    # df_dtypes = df.dtypes.values

    # if one types is different from metadata
    if not list(df_dtypes) == _dtypes:
        # build a dict of different types to loop only on needed columns
        diff_dtypes = {_col[i]: _dtypes[i] for i in range(
            len(_dtypes)) if df_dtypes[i] != _dtypes[i]}
        # Do not revert complex values if there is because it comes from
        # complex step
        for col, dtype in diff_dtypes.items():
            if len(df[col].values) > 0:
                if type(df[col].values[0]).__name__ != 'complex' and type(df[col].values[0]).__name__ != 'complex128':
                    df[col] = df[col].astype(dtype)

    # Insert excluded columns at the beginning of the dataframe
    # It is faster to add them before the init BUT the type of the column must be the same with a 2D arrays init
    # Then we need to switch to dict initialization and it becomes slower
    # than the insert method
    for column_excl in excluded_columns:
        if column_excl in metadata:
            df.insert(loc=0, column=column_excl,
                      value=metadata[column_excl])
    return df


def convert_array_into_new_type(name, var_array, reduced_dm={}):
    ''' convert list in local_data into correct type in data_in
        returns an updated copy of local_data
    '''

    _type = reduced_dm.get(VAR_TYPE_ID)
    metadata_list = reduced_dm.get(TYPE_METADATA)
    if array is None:
        var_new_type = None
    else:
        # check dict type in data_to_update and visibility
        if _type == 'dict':

            subtype = reduced_dm.get(VAR_SUBTYPE_ID)

            # TODO : remove this when we get rid of df_dict type
            if metadata_list is None and subtype is None:
                raise ValueError(
                    f' Variable {name} cannot be converted since no metadata is available')
            new_data = {}
            if subtype is None:

                var_new_type = convert_array_into_dict_old_version(
                    var_array, new_data, deepcopy(metadata_list))

            else:
                check_subtype(name, subtype, 'dict')
                var_new_type = convert_array_into_dict(
                    var_array, deepcopy(metadata_list), subtype)
        # check list type in data_to_update and visibility
        elif _type == 'list':

            subtype = reduced_dm.get(VAR_SUBTYPE_ID)

            if subtype is not None:
                if metadata_list is None and subtype['list'] not in ['int', 'string', 'float'] and check_subtype(
                        name, subtype, 'list') not in BASE_TYPE_EXCLUDED:
                    raise ValueError(
                        f' Variable {name} cannot be converted since no metadata is available')

                # check_subtype(key, subtype, 'list')
                var_new_type = convert_array_into_list(
                    var_array, deepcopy(metadata_list), subtype)
            else:
                var_new_type = list(var_array)

        # check dataframe type in data_in and visibility
        elif _type == 'dataframe':
            if metadata_list is None:
                raise ValueError(
                    f'Variable {name} cannot be converted since no metadata is available')
            metadata = metadata_list[0]
            excluded_columns = reduced_dm.get(DF_EXCLUDED_COLUMNS, DEFAULT_EXCLUDED_COLUMNS)

            var_new_type = convert_array_into_df(
                var_array, metadata, excluded_columns)
        # elif _type == 'string':
        #     metadata = metadata_list[0]
        #
        #     local_data_updt[key] = next((strg for strg, int_to_convert in metadata['known_values'].items(
        #     ) if int_to_convert == to_convert), None)
        # elif _type == 'string_list':
        #     local_data_updt[key] = []
        #     for i, val in enumerate(to_convert):
        #         metadata = metadata_list[i]
        #         local_data_updt[key].append(
        #             next((strg for strg, int_to_convert in metadata['known_values'].items(
        #             ) if int_to_convert == val), None))
        # elif _type in ['float', 'int']:

        else:
            var_new_type = var_array
    return var_new_type


def convert_dict_into_array_old_version(var_dict, values_list, metadata, prev_keys, prev_metadata):
    """
        Convert a nested var_dict into a numpy array, and stores metadata
        useful to build the dictionary afterwards
        """

    for key, val in var_dict.items():
        # for each value in the dictionary to convert
        nested_keys = prev_keys + [key]
        _type = type(val)
        # Previous metadata is used to get back the previous known values
        # for string to int conversion
        if prev_metadata is None:
            prev_metadata_key = None
        else:
            if len(prev_metadata) != 0.:
                prev_metadata_key = prev_metadata.pop(0)
            else:
                prev_metadata_key = None
        val_data = {}
        val_data['__key__'] = nested_keys
        val_data['__type__'] = _type
        if _type == dict:
            # if value is a nested dict
            metadata.append(val_data)
            values_list, metadata = convert_dict_into_array_old_version(
                val, values_list, metadata, val_data['__key__'], prev_metadata)
        elif _type == DataFrame:
            # if value is a dataframe
            values_list, metadata = convert_df_into_array(
                val, values_list, metadata, nested_keys)
        elif _type in STANDARD_TYPES:
            # if value is a int or float
            values_list = append(values_list, [val])
            metadata.append(val_data)
        elif _type == np_complex128:
            # for gradient analysis
            values_list = append(values_list, [val])
            val_data['__type__'] = np_float64
            metadata.append(val_data)
        elif _type in [list, ndarray]:
            # if val contains strings :
            if any(isinstance(elem, str) for elem in val):
                val_data['known_values'] = {}
                # We look for strings inside the list
                for i_elem, elem in enumerate(val):
                    if isinstance(elem, str):
                        val_data_ielem = {}
                        val_data_ielem['known_values'] = {}
                        # when string is found we look for its known values
                        if prev_metadata_key is not None:
                            if i_elem < len(prev_metadata_key['known_values']) and 'known_values' in \
                                    prev_metadata_key['known_values'][i_elem]:
                                val_data_ielem['known_values'] = prev_metadata_key['known_values'][i_elem][
                                    'known_values']
                        # convert the string into int and replace the
                        # string by this int in the list
                        int_val, val_data_ielem = convert_string_to_int(
                            elem, val_data_ielem)

                        val[i_elem] = int_val
                        val_data['known_values'][i_elem] = val_data_ielem

            if isinstance(val, list):
                size = len(val)
                val_data['__shape__'] = (size,)
                val_data['__size__'] = size
                values_list = append(values_list, val)
            else:
                val_data['__shape__'] = val.shape
                val_data['__size__'] = val.size
                values_list = append(values_list, val.flatten())
            metadata.append(val_data)
        elif _type == str:
            # if value is a string look for is prev_metadata to find known
            # values

            if prev_metadata_key is not None and 'known_values' in prev_metadata_key:
                val_data['known_values'] = prev_metadata_key['known_values']
            else:
                val_data['known_values'] = {}
            # convert the string into int
            int_val, val_data = convert_string_to_int(
                val, val_data)
            values_list = append(values_list, int_val)

            metadata.append(val_data)
        else:
            raise Exception(
                f'The type {_type} in the dict {var_dict} is not taken into account')
    return values_list, metadata


def convert_dict_into_array(var_dict, subtype):
    '''
    Convert a nested var_dict into a numpy array, and stores metadata
    useful to build the dictionary afterwards
    '''

    if (check_subtype('', subtype, 'dict') in BASE_TYPE_EXCLUDED) or (len(var_dict) == 0):
        return var_dict, None
    if type(subtype['dict']).__name__ != 'dict':
        type_inside_the_dict = subtype['dict']
        if type_inside_the_dict in ['int', 'string']:
            # raise ValueError(
            #     f'conversion of dict  of int or string is not supported')
            return var_dict, None

        elif type_inside_the_dict == 'float':
            return array(list(var_dict.values())), {'length': len(var_dict.keys()), 'size': len(var_dict.keys()),
                                                    'value': [key for key in var_dict.keys()]}

        elif type_inside_the_dict == 'array':
            return np.concatenate([single_element for single_element in list(var_dict.values())]), {
                'length': len(var_dict.keys()),
                'value': {key: len(value) for (key, value) in var_dict.items()},
                'size': sum([len(var_element) for
                             var_element in var_dict.values()])}
        elif type_inside_the_dict == 'dataframe':

            dict_metadata = {'length': len(var_dict.keys()), 'value': {}}
            converted_values = []
            size = 0
            for key, element_to_convert in var_dict.items():
                values_list = []
                metadata = []
                prev_key = []

                values_list, metadata = convert_df_into_array(
                    element_to_convert, values_list, metadata, prev_key)

                dict_metadata['value'][key] = (len(values_list), metadata)
                converted_values.append(values_list)
                size += len(values_list)

            converted_values_to_return = np.concatenate([single_element for single_element in converted_values])
            dict_metadata['size'] = size
            return converted_values_to_return, dict_metadata

        else:
            raise Exception(
                f'The type {type_inside_the_dict} in the dict {var_dict} is not yet handled in dict conversion')

    else:
        # We have a recursive dict
        recursive_subtype = (list(subtype['dict'].keys()))[0]
        dict_metadata = {'length': len(var_dict.keys()), 'value': {}}
        converted_values = []
        size = 0
        for key, element_to_convert in var_dict.items():
            if recursive_subtype == 'dict':
                converted_subdict, converted_submetadata = convert_dict_into_array(element_to_convert, subtype['dict'])
            else:
                converted_subdict, converted_submetadata = convert_list_into_array(element_to_convert, subtype['dict'])
            converted_values.append(converted_subdict)
            dict_metadata['value'][key] = converted_submetadata
            size += converted_submetadata['size']

        dict_metadata['size'] = size
        return np.concatenate([single_element for single_element in converted_values]), dict_metadata


def convert_array_into_dict(to_convert, metadata, subtype):
    """ convert an array into initial dict
    """
    if (check_subtype('', subtype, 'dict') in BASE_TYPE_EXCLUDED) or (len(to_convert) == 0):
        return to_convert
    # Here we deal with a dict of only one level eg {'dict':'array'}, {'dict':'float'} , {'dict':'dataframe'}
    if type(subtype['dict']).__name__ != 'dict':
        type_inside_the_dict = subtype['dict']
        if type_inside_the_dict in ['float']:
            return {key: value for (key, value) in zip(metadata['value'], to_convert.tolist())}
        elif type_inside_the_dict in ['int', 'string']:
            return to_convert

        elif type_inside_the_dict == 'array':

            dict_keys = list(metadata['value'].keys())
            converted_dict = {}
            idx = 0
            for key in dict_keys:
                converted_dict[key] = to_convert[idx: idx + metadata['value'][key]]
                idx += metadata['value'][key]

            return converted_dict

        elif type_inside_the_dict == 'dataframe':

            dict_keys = list(metadata['value'].keys())
            converted_dict = {}
            idx = 0
            for key in dict_keys:
                converted_dict[key] = convert_array_into_df(to_convert[idx: idx + metadata['value'][key][0]],
                                                            metadata['value'][key][1][0])
                idx += metadata['value'][key][0]

            return converted_dict

        else:
            raise ValueError(
                f' subtype  {type_inside_the_dict} is not yet handled in dict conversion')

    else:
        # case of a recursive dict
        recursive_subtype = (list(subtype['dict'].keys()))[0]
        dict_keys = list(metadata['value'].keys())
        converted_dict = {}
        idx = 0
        for key in dict_keys:
            if recursive_subtype == 'dict':
                converted_dict[key] = convert_array_into_dict(to_convert[idx: idx + metadata['value'][key]['size']],
                                                              metadata['value'][key], subtype['dict'])
            else:
                converted_dict[key] = convert_array_into_list(to_convert[idx: idx + metadata['value'][key]['size']],
                                                              metadata['value'][key], subtype['dict'])
            idx += metadata['value'][key]['size']

        return converted_dict


def convert_df_into_array(var_df, values_list, metadata, keys, excluded_columns=DEFAULT_EXCLUDED_COLUMNS):
    '''
    Converts dataframe into array, and stores metada
    useful to build the dataframe afterwards
    '''
    # gather df data including index column
    #         data = var_df.to_numpy()

    val_data = {column: list(var_df[column].values)
                for column in excluded_columns if column in var_df}

    new_var_df = var_df[[column for column in var_df.columns if column not in excluded_columns]]

    data = new_var_df.to_numpy()

    # indices = var_df.index.to_numpy()
    columns = new_var_df.columns
    # To delete indices in convert delete the line below
    # data = hstack((atleast_2d(indices).T, values))

    val_data['__key__'] = keys
    val_data['__type__'] = DataFrame
    val_data['__columns__'] = columns
    val_data['__shape__'] = data.shape
    val_data['__size__'] = data.size
    val_data['__dtypes__'] = [new_var_df[col].dtype for col in columns]
    # to flatten by lines erase the option 'F' or put the 'C' option

    if not (new_var_df.index == arange(0, data.shape[0])).all():
        val_data['__indices__'] = new_var_df.index

    values_list = append(values_list, data.flatten(order='F'))
    metadata.append(val_data)
    return values_list, metadata


def convert_float_into_array(var_dict):
    '''
    Check element type in var_dict, convert float or int into numpy array
        in order to deal with linearize issues in GEMS
    '''
    for key, var in var_dict.items():
        if isinstance(var, (float, int, complex)):
            var_dict[key] = array([var])

    return var_dict


def convert_new_type_into_array(
        key, var, reduced_dm={}):
    '''
    Check element type in var_dict, convert new type into numpy array
        and stores metadata into DM for after reconversion
    '''

    var_type = type(var)
    metadata = []
    new_reduced_dm = {}
    if var_type in NEW_VAR_TYPE:

        if var is None:
            var_converted = None
        else:
            var_converted = []
            prev_key = []
            # if type needs to be converted
            if var_type == dict:
                # if value is a dictionary
                all_values = list(var.values())
                if all([is_value_type_handled(val)
                        for val in all_values]):
                    # convert if all values are handled by
                    # SoSTrades

                    subtype = reduced_dm.get(VAR_SUBTYPE_ID)

                    if subtype is None:
                        prev_metadata = reduced_dm.get(TYPE_METADATA, [])
                        var_converted, metadata = convert_dict_into_array_old_version(
                            var, var_converted, metadata, prev_key, deepcopy(prev_metadata))
                    else:

                        check_subtype(key, subtype, 'dict')
                        var_converted, metadata = convert_dict_into_array(
                            var, subtype)
                else:
                    msg = f"\n Dictionary values must be among {list(VAR_TYPE_MAP.keys())}"
                    raise ValueError(msg)
            elif var_type == DataFrame:
                # if value is a DataFrame
                # TO DO: Create a function to get the dtypes of columns without the "big" function 'convert_df_into_array'
                init_var_converted, init_metadata = deepcopy(var_converted), deepcopy(metadata)

                excluded_columns = reduced_dm.get(DF_EXCLUDED_COLUMNS, DEFAULT_EXCLUDED_COLUMNS)
                var_converted, metadata = convert_df_into_array(
                    var, var_converted, metadata, prev_key, excluded_columns)
                # Define a list of dtypes to exclude and perform convert again if there are new cols to exclude
                dtypes_to_exclude = [object, bool, str]
                new_excluded_columns = deepcopy(excluded_columns)
                for col, metadata_dtype in zip(metadata[0]['__columns__'], metadata[0]['__dtypes__']):
                    if metadata_dtype in dtypes_to_exclude:
                        new_excluded_columns += [col]
                if new_excluded_columns != excluded_columns:
                    var_converted, metadata = convert_df_into_array(
                        var, init_var_converted, init_metadata, prev_key, new_excluded_columns)
                    new_reduced_dm[DF_EXCLUDED_COLUMNS] = new_excluded_columns

            elif var_type == list:

                subtype = reduced_dm.get(VAR_SUBTYPE_ID)

                if subtype is not None:
                    check_subtype(key, subtype, 'list')
                    var_converted, metadata = convert_list_into_array(var, subtype)
                else:
                    var_converted, metadata = var, None

                var_converted = array(var_converted)
            else:
                var_converted = var
    else:
        var_converted = var
    # default in value_to_array of BaseDataConverter we return value
    if VAR_TYPE_ID not in reduced_dm and var_type in NEW_VAR_TYPE_MAP_INVERTED:
        new_reduced_dm[VAR_TYPE_ID] = NEW_VAR_TYPE_MAP_INVERTED[var_type]
    new_reduced_dm[TYPE_METADATA] = metadata

    return var_converted, new_reduced_dm


def convert_string_to_int(val, val_data):
    '''
    Small function to convert a string into an int following the metadata known_values
    if the value is new, the int will be the len of the known values + 1
    '''
    if val not in val_data['known_values']:
        int_val = len(val_data['known_values']) + 1
        val_data['known_values'][val] = int_val
    else:
        int_val = val_data['known_values'][val]

    return int_val, val_data


def convert_list_into_array(var, subtype):
    """This function converts a list into an array
    It also creates the metadata needed to reconvert the array into the initial list
    """
    if (check_subtype('', subtype, 'list') in BASE_TYPE_EXCLUDED) or (len(var) == 0):
        return var, None
    # Here we deal with a list of only one level eg {'list':'string'}, {'list':'float'} , {'list':'df'}
    if type(subtype['list']).__name__ != 'dict':
        type_inside_the_list = subtype['list']
        if type_inside_the_list in ['int', 'string']:
            # raise ValueError(
            #     f'conversion of list of ints or string is not supported')
            return var, "no_metadata"

        elif type_inside_the_list == 'float':
            return array(var), {'length': len(var), 'value': None, 'size': len(var)}

        elif type_inside_the_list == 'array':
            return np.concatenate([single_element for single_element in var]), {'length': len(var),
                                                                                'value': [len(var_element) for
                                                                                          var_element in var],

                                                                                'size': sum([len(var_element) for
                                                                                             var_element in var])}

        elif type_inside_the_list == 'dataframe':

            list_metadata = {'length': len(var), 'value': [], 'type': type_inside_the_list}
            converted_values = []
            size = 0
            for element_to_convert in var:
                values_list = []
                metadata = []
                prev_key = []

                values_list, metadata = convert_df_into_array(
                    element_to_convert, values_list, metadata, prev_key)

                list_metadata['value'].append((len(values_list), metadata))
                converted_values.append(values_list)
                size += len(values_list)

            converted_values_to_return = np.concatenate([single_element for single_element in converted_values])
            list_metadata['size'] = size
            return converted_values_to_return, list_metadata

        else:
            raise ValueError(
                f' subtype  {type_inside_the_list} is not yet handled in list conversion')

    else:
        # We have a recursive list
        recursive_subtype = (list(subtype['list'].keys()))[0]
        list_metadata = {'length': len(var), 'value': [], 'type': 'list'}
        converted_list = []
        size = 0
        for element in var:
            if recursive_subtype == 'list':
                converted_sublist, converted_submetadata = convert_list_into_array(element, subtype['list'])
            else:
                converted_sublist, converted_submetadata = convert_dict_into_array(element, subtype['list'])
            converted_list.append(converted_sublist)
            list_metadata['value'].append(converted_submetadata)
            size += converted_submetadata['size']

        list_metadata['size'] = size
        return np.concatenate([single_element for single_element in converted_list]), list_metadata


def convert_array_into_list(to_convert, metadata, subtype):
    """ convert an array into initial list
    """
    if (check_subtype('', subtype, 'list') in BASE_TYPE_EXCLUDED) or (len(to_convert)) == 0:
        return to_convert
    # Here we deal with a list of only one level eg {'list':'string'}, {'list':'float'} , {'list':'df'}
    if type(subtype['list']).__name__ != 'dict':
        type_inside_the_list = subtype['list']
        if type_inside_the_list in ['float', 'int', 'string']:
            if type_inside_the_list in ['int', 'string']:
                return to_convert
            else:
                return to_convert.tolist()

        elif type_inside_the_list == 'array':
            initial_list_length = metadata['length']
            converted_list = []
            idx = 0
            for i in range(initial_list_length):
                arr_to_convert = to_convert[idx: idx + metadata['value'][i]]
                converted_list.append(arr_to_convert)
                idx += metadata['value'][i]

            return converted_list

        elif type_inside_the_list == 'dataframe':

            initial_list_length = metadata['length']
            converted_list = []
            idx = 0
            for i in range(initial_list_length):
                arr_to_convert = to_convert[idx: idx + metadata['value'][i][0]]
                converted_list.append(convert_array_into_df(arr_to_convert, metadata['value'][i][1][0]))

                idx += metadata['value'][i][0]

            return converted_list

        else:
            raise ValueError(
                f' subtype  {type_inside_the_list} is not yet handled in list conversion')

    else:
        # case of a recursive list
        recursive_subtype = (list(subtype['list'].keys()))[0]
        initial_list_length = metadata['length']
        converted_list = []
        idx = 0
        for i in range(initial_list_length):
            arr_to_convert = to_convert[idx: idx + metadata['value'][i]['size']]
            if recursive_subtype == 'list':
                converted_list.append(convert_array_into_list(arr_to_convert, metadata['value'][i], subtype['list']))
            else:
                converted_list.append(convert_array_into_dict(arr_to_convert, metadata['value'][i], subtype['list']))
            idx += metadata['value'][i]['size']

        return converted_list


def check_subtype(var_full_name, subtype, type_to_check):
    """This function checks that the subtype given to a list is compliant
    with the defined standard for subtype
    """
    if type(subtype).__name__ != 'dict':
        raise ValueError(
            f' subtype of variable {var_full_name} must be a dictionnary')
    elif list(subtype.keys())[0] != type_to_check or len(list(subtype.keys())) != 1:
        raise ValueError(
            f' subtype of variable {var_full_name} should have as unique key the keyword {type_to_check}')
    elif type(subtype[type_to_check]).__name__ != 'dict':
        if subtype[type_to_check] == type_to_check:
            raise ValueError(
                f' subtype of variable {var_full_name} should indicate the type inside the {type_to_check}')
        else:
            return subtype[type_to_check]
    else:

        return check_subtype(var_full_name, subtype[type_to_check], list(subtype[type_to_check].keys())[0])
