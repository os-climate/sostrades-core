from copy import deepcopy
from functools import reduce

from numpy import int32 as np_int32, float64 as np_float64, complex128 as np_complex128, int64 as np_int64
from numpy import ndarray, append, arange, delete, array
from pandas import DataFrame

DEFAULT_EXCLUDED_COLUMNS = ['year', 'years']
VAR_TYPE_ID = 'type'
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
    'dict_list': list,
    'array': ndarray,
    'df_dict': dict,
    'dict': dict,
    'dataframe': DataFrame,
    'bool': bool,
}
VAR_TYPE_GEMS = ['int', 'array', 'float_list', 'int_list']
STANDARD_TYPES = [int, float, np_int32, np_int64, np_float64, bool]
#    VAR_TYPES_SINGLE_VALUES = ['int', 'float', 'string', 'bool', 'np_int32', 'np_float64', 'np_int64']
NEW_VAR_TYPE = ['dict', 'dataframe',
                'string_list', 'string', 'float', 'int']


def _is_value_type_handled(val):
    return isinstance(val, tuple(VAR_TYPE_MAP.values())
                      ) or isinstance(val, np_complex128)


def _get_dataframe_excluded_columns(key, dm_reduced_to_type_and_metadata):
    """ returns the excluded columns of a dataframe
        output : list of excluded columns
    """
    if not isinstance(dm_reduced_to_type_and_metadata, dict):
        excluded_columns = dm_reduced_to_type_and_metadata.get_data(
            key, DF_EXCLUDED_COLUMNS)
    else:
        excluded_columns = dm_reduced_to_type_and_metadata[key][DF_EXCLUDED_COLUMNS]
    if excluded_columns is None:
        return []
    return excluded_columns


def __get_nested_val(dict_in, keys):
    ''' returns the value of a nested dictionary of depth len(keys)
    output : d[keys[0]][..][keys[n]]
    '''

    def func_dic(dict_in, key): return dict_in[key]

    nested_val = reduce(func_dic, keys, dict_in)

    return nested_val


def _convert_array_into_dict(arr_to_convert, new_data, val_datalist):
    # convert list into dict using keys from dm.data_dict
    if len(val_datalist) == 0:
        # means the dictionary is empty or None
        return {}
    else:
        while len(val_datalist) != 0:
            metadata = val_datalist.pop(0)
            _type = metadata['type']
            _keys = metadata['key']

            nested_keys = _keys[:-1]
            to_update = __get_nested_val(new_data, nested_keys)
            _key = _keys[-1]
            # dictionaries

            if _type == dict:
                to_update[_key] = {}
                _convert_array_into_dict(
                    arr_to_convert, new_data, val_datalist)
            # DataFrames
            elif _type == DataFrame:
                _df = _convert_array_into_df(arr_to_convert, metadata)
                to_update[_key] = _df
                _size = metadata['size']
                arr_to_convert = delete(arr_to_convert, arange(_size))

            # int, float, or complex
            elif _type in [int, float, np_int32, np_int64, np_float64, np_complex128, bool]:
                _val = arr_to_convert[0]
                arr_to_convert = delete(arr_to_convert, [0])
                to_update[_key] = _type(_val)

            # numpy array or list
            elif _type in [list, ndarray]:
                _shape = metadata['shape']
                _size = metadata['size']
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


def _convert_array_into_df(arr_to_convert, metadata, excluded_columns=DEFAULT_EXCLUDED_COLUMNS):
    # convert list into dataframe using columns from dm.data_dict
    _shape = metadata['shape']
    _size = metadata['size']
    _col = metadata['columns'].copy()
    _dtypes = metadata['dtypes'].copy()
    _arr = arr_to_convert[:_size]
    # to flatten by lines erase the option 'F' or put the 'C' option
    _arr = _arr.reshape(_shape, order='F')
    # create multi index columns if tuples in columns

    # Use the 2Darrays init which is 4 times faster than the dict initialization
    # if indices are stored we use them to reconstruct the dataframe
    if 'indices' in metadata:
        df = DataFrame(data=_arr, columns=_col,
                       index=metadata['indices'])
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


def convert_array_into_new_type(local_data, dm_reduced_to_type_and_metadata):
    ''' convert list in local_data into correct type in data_in
        returns an updated copy of local_data
    '''
    local_data_updt = deepcopy(local_data)

    for key, to_convert in local_data_updt.items():
        # get value in DataManager
        if not isinstance(dm_reduced_to_type_and_metadata, dict):
            _type = dm_reduced_to_type_and_metadata.get_data(key, VAR_TYPE_ID)
            metadata_list = dm_reduced_to_type_and_metadata.get_data(
                key, TYPE_METADATA)
        else:
            _type = dm_reduced_to_type_and_metadata[key][VAR_TYPE_ID]
            metadata_list = dm_reduced_to_type_and_metadata[key][TYPE_METADATA]
        if to_convert is None:
            local_data_updt[key] = None
        else:
            # check dict type in data_to_update and visibility
            if _type == 'dict' or _type == 'df_dict':
                if metadata_list is None:
                    raise ValueError(
                        f' Variable {key} cannot be converted since no metadata is available')
                new_data = {}

                local_data_updt[key] = _convert_array_into_dict(
                    to_convert, new_data, deepcopy(metadata_list))

            # check dataframe type in data_in and visibility
            elif _type == 'dataframe':
                if metadata_list is None:
                    raise ValueError(
                        f'Variable {key} cannot be converted since no metadata is available')
                metadata = metadata_list[0]
                excluded_columns = _get_dataframe_excluded_columns(
                    key, dm_reduced_to_type_and_metadata)
                local_data_updt[key] = _convert_array_into_df(
                    to_convert, metadata, excluded_columns)
            elif _type == 'string':
                metadata = metadata_list[0]

                local_data_updt[key] = next((strg for strg, int_to_convert in metadata['known_values'].items(
                ) if int_to_convert == to_convert), None)
            elif _type == 'string_list':
                local_data_updt[key] = []
                for i, val in enumerate(to_convert):
                    metadata = metadata_list[i]
                    local_data_updt[key].append(
                        next((strg for strg, int_to_convert in metadata['known_values'].items(
                        ) if int_to_convert == val), None))
            elif _type in ['float', 'int']:
                if isinstance(to_convert, ndarray):
                    # Check if metadata has been created
                    # Check if the value is complex that means that we are
                    # in a complex step method do not kill the complex part
                    if metadata_list is not None and not isinstance(to_convert[0], complex):
                        # if both conditions are OK reuse the float type in
                        # the metadata for the value
                        local_data_updt[key] = metadata_list['var_type'](
                            to_convert[0])

                    else:
                        local_data_updt[key] = to_convert[0]
    return local_data_updt


def _convert_dict_into_array(var_dict, values_list, metadata, prev_keys, prev_metadata):
    '''
    Convert a nested var_dict into a numpy array, and stores metadata
    useful to build the dictionary afterwards
    '''

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
        val_data['key'] = nested_keys
        val_data['type'] = _type
        if _type == dict:
            # if value is a nested dict
            metadata.append(val_data)
            values_list, metadata = _convert_dict_into_array(
                val, values_list, metadata, val_data['key'], prev_metadata)
        elif _type == DataFrame:
            # if value is a dataframe
            values_list, metadata = _convert_df_into_array(
                val, values_list, metadata, nested_keys)
        elif _type in STANDARD_TYPES:
            # if value is a int or float
            values_list = append(values_list, [val])
            metadata.append(val_data)
        elif _type == np_complex128:
            # for gradient analysis
            values_list = append(values_list, [val])
            val_data['type'] = np_float64
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
                val_data['shape'] = (size,)
                val_data['size'] = size
                values_list = append(values_list, val)
            else:
                val_data['shape'] = val.shape
                val_data['size'] = val.size
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


def _convert_df_into_array(var_df, values_list, metadata, keys, excluded_columns=DEFAULT_EXCLUDED_COLUMNS):
    '''
    Converts dataframe into array, and stores metada
    useful to build the dataframe afterwards
    '''
    # gather df data including index column
    #         data = var_df.to_numpy()

    val_data = {column: list(var_df[column].values)
                for column in excluded_columns if column in var_df}

    new_var_df = var_df.drop(
        columns=[column for column in excluded_columns if column in var_df])

    data = new_var_df.to_numpy()

    # indices = var_df.index.to_numpy()
    columns = new_var_df.columns
    # To delete indices in convert delete the line below
    # data = hstack((atleast_2d(indices).T, values))

    val_data['key'] = keys
    val_data['type'] = DataFrame
    val_data['columns'] = columns
    val_data['shape'] = data.shape
    val_data['size'] = data.size
    val_data['dtypes'] = [new_var_df[col].dtype for col in columns]
    # to flatten by lines erase the option 'F' or put the 'C' option

    if not (new_var_df.index == arange(0, data.shape[0])).all():
        val_data['indices'] = new_var_df.index

    values_list = append(values_list, data.flatten(order='F'))
    metadata.append(val_data)
    return values_list, metadata


def _convert_float_into_array(var_dict):
    '''
    Check element type in var_dict, convert float or int into numpy array
        in order to deal with linearize issues in GEMS
    '''
    for key, var in var_dict.items():
        if isinstance(var, (float, int, complex)):
            var_dict[key] = array([var])

    return var_dict


def convert_new_type_into_array(
        var_dict, dm_reduced_to_type_and_metadata):
    '''
    Check element type in var_dict, convert new type into numpy array
        and stores metadata into DM for afterwards reconversion
    '''

    dict_to_update_dm = {}
    for key, var in var_dict.items():
        if not isinstance(dm_reduced_to_type_and_metadata, dict):
            var_type = dm_reduced_to_type_and_metadata.get_data(
                key, VAR_TYPE_ID)
        else:
            var_type = dm_reduced_to_type_and_metadata[key][VAR_TYPE_ID]
        if var_type in NEW_VAR_TYPE:
            if not isinstance(
                    var, VAR_TYPE_MAP[var_type]) and var is not None:
                msg = f"Variable {key} has type {type(var)}, "
                msg += f"however type {VAR_TYPE_MAP[var_type]} was expected."
                # msg += f'before run of discipline {self} with name {self.get_disc_full_name()} '
                raise ValueError(msg)
            else:
                if var is None:
                    var_dict[key] = None
                else:
                    values_list = []
                    metadata = []
                    prev_key = []
                    if var_type in ['dict', 'string', 'string_list']:
                        if not isinstance(dm_reduced_to_type_and_metadata, dict):
                            prev_metadata = dm_reduced_to_type_and_metadata.get_data(
                                key, TYPE_METADATA)
                        else:
                            prev_metadata = dm_reduced_to_type_and_metadata[key][TYPE_METADATA]
                    # if type needs to be converted
                    if var_type == 'dict':
                        # if value is a dictionary
                        all_values = list(var.values())
                        if all([_is_value_type_handled(val)
                                for val in all_values]):
                            # convert if all values are handled by
                            # SoSTrades
                            values_list, metadata = _convert_dict_into_array(
                                var, values_list, metadata, prev_key, deepcopy(prev_metadata))
                        else:
                            evaluated_types = [type(val)
                                               for val in all_values]
                            # msg = f"\n Invalid type of parameter {key}: {var}/'{evaluated_types}' in discipline {self.sos_name}."
                            msg = f"\n Dictionary values must be among {list(VAR_TYPE_MAP.keys())}"
                            raise ValueError(msg)
                    elif var_type == 'dataframe':
                        # if value is a DataFrame

                        if not isinstance(dm_reduced_to_type_and_metadata, dict):
                            excluded_columns = dm_reduced_to_type_and_metadata.get_data(
                                key, DF_EXCLUDED_COLUMNS)
                        else:
                            excluded_columns = dm_reduced_to_type_and_metadata[key][DF_EXCLUDED_COLUMNS]
                        values_list, metadata = _convert_df_into_array(
                            var, values_list, metadata, prev_key, excluded_columns)
                    elif var_type == 'string':
                        # if value is a string
                        metadata_dict = {}
                        metadata_dict['known_values'] = {}
                        if prev_metadata is not None and 'known_values' in prev_metadata[0]:
                            metadata_dict['known_values'] = prev_metadata[0]['known_values']

                        values_list, metadata_dict = convert_string_to_int(
                            var, metadata_dict)

                        metadata.append(metadata_dict)

                    elif var_type == 'string_list':
                        # if value is a list of strings
                        for i_elem, elem in enumerate(var):
                            metadata_dict_elem = {}
                            metadata_dict_elem['known_values'] = {}
                            if prev_metadata is not None and i_elem < len(prev_metadata) and 'known_values' in \
                                    prev_metadata[i_elem]:
                                metadata_dict_elem['known_values'] = prev_metadata[i_elem]['known_values']

                            value_elem, metadata_dict_elem = convert_string_to_int(
                                elem, metadata_dict_elem)
                            values_list.append(value_elem)
                            metadata.append(metadata_dict_elem)
                    elif var_type in ['float', 'int']:
                        # store float into array for gems
                        metadata = {'var_type': type(var)}
                        values_list = array([var])

                    # update current dictionary value
                    var_dict[key] = values_list
                    # Update metadata
                    # self.dm.set_data(key, self.TYPE_METADATA,
                    #                 metadata, check_value=False)
                    dict_to_update_dm[key] = metadata

    return var_dict, dict_to_update_dm


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
