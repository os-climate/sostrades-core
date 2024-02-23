'''
Copyright 2024 Capgemini

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
from gemseo.algos.design_space import DesignSpace
from numpy import array, ndarray, delete, nonzero, logical_not

VARIABLES = "variable"
VALUES = "value"
UPPER_BOUND = "upper_bnd"
LOWER_BOUND = "lower_bnd"
ENABLE_VARIABLE_BOOL = "enable_variable"
LIST_ACTIVATED_ELEM = "activated_elem"
VARIABLE_TYPE = "variable_type"  # TODO: to discuss

def create_gemseo_dspace_from_dspace_df(dspace_df):
    """
    Create gemseo dspace from sostrades updated dspace_df
    It parses the dspace_df DataFrame to create the gemseo DesignSpace

    Arguments:
        dspace_df (dataframe): updated dspace_df

    Returns:
        design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
    """
    names = list(dspace_df[VARIABLES])
    values = list(dspace_df[VALUES])
    l_bounds = list(dspace_df[LOWER_BOUND])
    u_bounds = list(dspace_df[UPPER_BOUND])
    enabled_variable = list(dspace_df[ENABLE_VARIABLE_BOOL])
    list_activated_elem = list(dspace_df[LIST_ACTIVATED_ELEM])


    # looking for the optionnal variable type in the design space
    if VARIABLE_TYPE in dspace_df:
        var_types = dspace_df[VARIABLE_TYPE]
    else:
        # set to None for all variables if not exists
        var_types = [None] * len(names)

    design_space = DesignSpace()
    dict_desactivated_elem = {}
    for dv, val, lb, ub, l_activated, enable_var, vtype in zip(names, values, l_bounds, u_bounds,
                                                               list_activated_elem, enabled_variable, var_types):

        # check if variable is enabled to add it or not in the design var
        if enable_var:
            dict_desactivated_elem[dv] = {}

            if type(val) != list and type(val) != ndarray:
                size = 1
                var_type = ['float']
                l_b = array([lb])
                u_b = array([ub])
                value = array([val])
            else:
                # check if there is any False in l_activated
                if not all(l_activated):
                    index_false = nonzero(logical_not(l_activated))[0]  # NB: assumption is that array is 1D
                    dict_desactivated_elem[dv] = {
                        'value': array(val)[index_false], 'position': index_false}

                    val = delete(val, index_false)
                    lb = delete(lb, index_false)
                    ub = delete(ub, index_false)

                size = len(val)
                var_type = ['float'] * size
                l_b = array(lb)
                u_b = array(ub)
                value = array(val)

            # 'automatic' var_type values are overwritten if filled by the user
            if vtype is not None:
                var_type = vtype

            design_space.add_variable(
                dv, size, var_type, l_b, u_b, value)
    return design_space, dict_desactivated_elem

def check_design_space_data_integrity(design_space, possible_variables_types):
    design_space_integrity_msg = []
    if design_space.empty or not design_space[VARIABLES].tolist():
        design_space_integrity_msg.append("The design space should contain at least one variable.")
    else:
        if possible_variables_types:
            # possible value checks (with current implementation should be OK by construction)
            vars_not_possible = design_space[VARIABLES][
                ~design_space[VARIABLES].apply(lambda _var: _var in possible_variables_types)].to_list()
            for var_not_possible in vars_not_possible:
                design_space_integrity_msg.append(
                    f'Variable {var_not_possible} is not among the possible input values.'
                )
        # check of dimensions coherences
        wrong_dim_vars = design_space[VARIABLES][
            ~design_space.apply(_check_design_space_dimensions_for_one_variable, axis=1)].to_list()
        for wrong_dim_var in wrong_dim_vars:
            design_space_integrity_msg.append(
                f'Columns {LOWER_BOUND}, {UPPER_BOUND} and {VALUES} should be of type '
                f'{possible_variables_types[wrong_dim_var]} for variable {wrong_dim_var} '
                f'and should have coherent shapes.')

    return design_space_integrity_msg

def _check_design_space_dimensions_for_one_variable(design_space_row):
    """
    Utility method that checks that values in the columns 'lower_bnd', 'upper_bnd', 'value' of the design space do
    have the same shape for a same variable.

    Arguments:
        eval_inputs_row (pd.Series): row of the design space dataframe to check
    """
    lb = design_space_row[LOWER_BOUND] if LOWER_BOUND in design_space_row.index else None
    ub = design_space_row[UPPER_BOUND] if UPPER_BOUND in design_space_row.index else None
    val = design_space_row[VALUES] if VALUES in design_space_row.index else None
    lb_shape = array(lb).shape
    ub_shape = array(ub).shape
    val_shape = array(val).shape
    lb_ub_dim_mismatch = lb is not None and ub is not None and lb_shape != ub_shape
    lb_val_dim_mismatch = lb is not None and val is not None and lb_shape != val_shape
    val_ub_dim_mismatch = val is not None and ub is not None and val_shape != ub_shape
    return not (lb_ub_dim_mismatch or lb_val_dim_mismatch or val_ub_dim_mismatch)

## TODO: looks unused
# def read_from_dict(self, dp_dict):
#     """Parses a dictionary to read the DesignSpace
#
#     :param dp_dict : design space dictionary
#     :returns:  the design space
#     """
#     design_space = DesignSpace()
#     for key in dp_dict:
#         print(key)
#         if type(dp_dict[key]['value']) != list and type(dp_dict[key]['value']) != ndarray:
#             name = key
#             var_type = ['float']
#
#             size = 1
#             l_b = array([dp_dict[key]['lower_bnd']])
#             u_b = array([dp_dict[key]['upper_bnd']])
#             value = array([dp_dict[key]['value']])
#         else:
#             size = len(dp_dict[key]['value'])
#             var_type = ['float'] * size
#
#             name = key
#             l_b = array(dp_dict[key]['lower_bnd'])
#             u_b = array(dp_dict[key]['upper_bnd'])
#             value = array(dp_dict[key]['value'])
#
#         design_space.add_variable(name, size, var_type, l_b, u_b, value)
#     return design_space