VAR_TYPE_ID = 'type'
VAR_SUBTYPE_ID = 'subtype_descriptor'
VAR_NUMERICAL = 'numerical'
BASE_TYPE_TO_CONVERT = ['dataframe', 'float', 'array']


def check_subtype(var_full_name, subtype, type_to_check):
    """This function checks that the subtype given to a list or dictionnary is compliant
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

        return check_subtype(var_full_name, subtype[type_to_check],
                             list(subtype[type_to_check].keys())[0])


def filter_variables_to_convert(reduced_dm, list_to_filter, write_logs=False, logger=None):
    """  filter variables to convert
    """
    filtered_keys = []

    for variable in list_to_filter:
        will_be_converted = False
        variable_local_data = reduced_dm[variable]
        is_numerical = variable_local_data[VAR_NUMERICAL]
        if not is_numerical:
            type = variable_local_data[VAR_TYPE_ID]
            if type in BASE_TYPE_TO_CONVERT:
                filtered_keys.append(variable)
                will_be_converted = True
            elif type not in ['string', 'string_list', 'string_list_list', 'int_list', 'float_list', 'bool',
                              'dict_list', 'df_dict']:
                subtype = variable_local_data.get(VAR_SUBTYPE_ID)
                if subtype is not None:
                    final_type = check_subtype(
                        variable, subtype, type)
                    if final_type in BASE_TYPE_TO_CONVERT:
                        filtered_keys.append(variable)
                        will_be_converted = True

        if not will_be_converted and write_logs:
            logger.info(
                f'variable {variable} in strong couplings wont be taken into consideration in residual computation')
    return filtered_keys
