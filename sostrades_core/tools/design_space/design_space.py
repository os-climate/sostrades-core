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
from numpy import array, ndarray, delete

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
#     names = list(dspace_df[VARIABLES])
#     values = list(dspace_df[VALUES])
#     l_bounds = list(dspace_df[LOWER_BOUND])
#     u_bounds = list(dspace_df[UPPER_BOUND])
#     enabled_variable = list(dspace_df[ENABLE_VARIABLE_BOOL])
#     list_activated_elem = list(dspace_df[LIST_ACTIVATED_ELEM])
#     design_space = DesignSpace()
#     for dv, val, lb, ub, l_activated, enable_var in zip(names, values, l_bounds, u_bounds, list_activated_elem,
#                                                         enabled_variable):
#
#         # check if variable is enabled to add it or not in the design var
#         if enable_var:
#
#             # self.sample_generator.dict_desactivated_elem[dv] = {}
#             name = dv
#             if type(val) != list and type(val) != ndarray:
#                 size = 1
#                 var_type = ['float']
#                 l_b = array([lb])
#                 u_b = array([ub])
#                 value = array([val])
#             else:
#                 # check if there is any False in l_activated
#                 if not all(l_activated):
#                     # FIXME: implementation doesn't look good for >1 deactivated elem
#                     index_false = l_activated.index(False)
#                     # self.sample_generator.dict_desactivated_elem[dv] = {
#                     #     'value': val[index_false], 'position': index_false}
#
#                     val = delete(val, index_false)
#                     lb = delete(lb, index_false)
#                     ub = delete(ub, index_false)
#
#                 size = len(val)
#                 var_type = ['float'] * size
#                 l_b = array(lb)
#                 u_b = array(ub)
#                 value = array(val)
#             design_space.add_variable(
#                 name, size, var_type, l_b, u_b, value)
#     return design_space
# def read_from_dataframe(self, df):
#     """Parses a DataFrame to read the DesignSpace
#
#     :param df : design space df
#     :returns:  the design space
#     """
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
                    index_false = l_activated.index(False)
                    dict_desactivated_elem[dv] = {
                        'value': val[index_false], 'position': index_false}

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