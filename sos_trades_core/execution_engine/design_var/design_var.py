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
from numpy import arange
from pandas import DataFrame, Series
from sos_trades_core.tools.bspline.bspline import BSpline
from copy import deepcopy

import numpy as np


class DesignVar(object):
    """
    Class Design variable
    """
    ACTIVATED_ELEM_LIST = "activated_elem"
    VARIABLES = "variable"
    VALUE = "value"

    def __init__(self, inputs_dict):
        self.design_var_descriptor = inputs_dict['design_var_descriptor']
        self.output_dict = {}
        self.bspline_dict = {}
        self.dspace = inputs_dict['design_space']

    def configure(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.output_dict = {}
        list_ctrl = self.design_var_descriptor.keys()

        for elem in list_ctrl:

            # checks activated elements
            l_activated = self.dspace.loc[self.dspace[self.VARIABLES]
                                          == elem, self.ACTIVATED_ELEM_LIST].to_list()[0]
            value_dv = self.dspace.loc[self.dspace[self.VARIABLES]
                                       == elem, self.VALUE].to_list()[0]
            elem_val = inputs_dict[elem]
            index_false = None
            if not all(l_activated):
                index_false = l_activated.index(False)
                elem_val = list(elem_val)
                elem_val.insert(index_false, value_dv[index_false])
                elem_val = np.asarray(elem_val)

            # check output length and compute BSpline only if necessary
            # remark: float do not require any BSpline usage
            output_length = len(self.design_var_descriptor[elem]['index'])

            if len(inputs_dict[elem]) == output_length:
                self.bspline_dict[elem] = {
                    'bspline': None, 'eval_t': inputs_dict[elem], 'b_array': np.identity(output_length)}
            else:
                list_t = np.linspace(0.0, 1.0, output_length)
                bspline = BSpline(n_poles=len(elem_val))
                bspline.set_ctrl_pts(elem_val)
                eval_t, b_array = bspline.eval_list_t(list_t)
                b_array = bspline.update_b_array(b_array, index_false)

                self.bspline_dict[elem] = {
                    'bspline': bspline, 'eval_t': eval_t, 'b_array': b_array}

        # loop over design_var_descriptor to build output
        for key in self.design_var_descriptor.keys():
            out_name = self.design_var_descriptor[key]['out_name']
            out_type = self.design_var_descriptor[key]['out_type']

            if out_type == 'float':
                if inputs_dict[key].size != 1:
                    raise ValueError(" The input must be of size 1 for a float output")
                self.output_dict[out_name] = inputs_dict[key][0]
            elif out_type == 'array':
                self.output_dict[out_name] = self.bspline_dict[key]['eval_t']
            elif out_type == 'dataframe':
                if self.design_var_descriptor[key]['out_name'] not in self.output_dict.keys():
                    # init output dataframes with index
                    index = self.design_var_descriptor[key]['index']
                    index_name = self.design_var_descriptor[key]['index_name']
                    self.output_dict[out_name] = DataFrame({index_name: index})

                col_name = self.design_var_descriptor[key]['key']
                self.output_dict[out_name][col_name] = self.bspline_dict[key]['eval_t']
            else:
                raise (ValueError('Output type not yet supported'))


    # def update_design_space_out(self):
    #     """
    #     Method to update design space with opt value
    #     """
    #     design_space = deepcopy(self.design_var_descriptor)
    #     l_variables = design_space[self.VARIABLES]
    #     for var in l_variables:
    #         full_name_var = self.get_full_names([var])[0]
    #         if full_name_var in self.activated_variables:
    #             value_x_opt = list(self.formulation.design_space._current_x.get(
    #                 full_name_var))
    #             if self.dict_desactivated_elem[full_name_var] != {}:
    #                 # insert a desactivated element
    #                 value_x_opt.insert(
    #                     self.dict_desactivated_elem[full_name_var]['position'],
    #                     self.dict_desactivated_elem[full_name_var]['value'])
    #
    #             design_space.loc[design_space[self.VARIABLES] == var, self.VALUE] = pd.Series(
    #                 [value_x_opt] * len(design_space))


