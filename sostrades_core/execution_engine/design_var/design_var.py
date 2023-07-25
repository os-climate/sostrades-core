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
from pandas import DataFrame, Series, concat
from sostrades_core.tools.bspline.bspline import BSpline
from copy import deepcopy

import numpy as np


class DesignVar(object):
    """
    Class Design variable
    """
    ACTIVATED_ELEM_LIST = "activated_elem"
    VARIABLES = "variable"
    VALUE = "value"
    DATAFRAME_FILL = 'dataframe_fill'
    COLUMNS_NAMES = 'columns_names'
    ONE_COLUMN_PER_KEY = 'one column per key'
    ONE_COLUMN_FOR_KEY = 'one column for key, one for value'
    DATAFRAME_FILL_POSSIBLE_VALUES = [ONE_COLUMN_PER_KEY, ONE_COLUMN_FOR_KEY]
    DESIGN_SPACE = 'design_space'
    DESIGN_VAR_DESCRIPTOR = 'design_var_descriptor'
    INDEX = 'index'
    INDEX_NAME = 'index_name'
    OUT_TYPE = 'out_type'
    OUT_NAME = 'out_name'

    def __init__(self, inputs_dict):
        '''
        Constructor
        '''
        self.design_var_descriptor = inputs_dict[self.DESIGN_VAR_DESCRIPTOR]
        self.output_dict = {}
        self.bspline_dict = {}
        self.dspace = inputs_dict[self.DESIGN_SPACE]

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
            output_length = len(self.design_var_descriptor[elem][self.INDEX])

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
            out_name = self.design_var_descriptor[key][self.OUT_NAME]
            out_type = self.design_var_descriptor[key][self.OUT_TYPE]

            if out_type == 'float':
                if inputs_dict[key].size != 1:
                    raise ValueError(" The input must be of size 1 for a float output")
                self.output_dict[out_name] = inputs_dict[key][0]
            elif out_type == 'array':
                self.output_dict[out_name] = self.bspline_dict[key]['eval_t']
            elif out_type == 'dataframe':
                # dataframe fill is optional ,by default we fill the dataframe with one column per key
                if self.DATAFRAME_FILL in self.design_var_descriptor[key]:
                    dataframe_fill = self.design_var_descriptor[key][self.DATAFRAME_FILL]
                else:
                    dataframe_fill = self.ONE_COLUMN_PER_KEY
                index = self.design_var_descriptor[key][self.INDEX]
                index_name = self.design_var_descriptor[key][self.INDEX_NAME]
                if dataframe_fill == self.ONE_COLUMN_PER_KEY:
                    # for the method one column per key we create a dataframe if it does not exists
                    if self.design_var_descriptor[key][self.OUT_NAME] not in self.output_dict.keys():
                        # init output dataframes with index

                        self.output_dict[out_name] = DataFrame({index_name: index})
                    # we use the key 'key' in the design_var_descriptor for the name of the column and the column to the dataframe
                    col_name = self.design_var_descriptor[key]['key']
                    self.output_dict[out_name][col_name] = self.bspline_dict[key]['eval_t']
                elif dataframe_fill == self.ONE_COLUMN_FOR_KEY:

                    column_names = self.design_var_descriptor[key][self.COLUMNS_NAMES]
                    # # create a dataframe using column_names, in this method the dataframe will ALWAYS have 2 columns
                    # first column will store the key
                    # second column the value

                    df_to_merge = DataFrame(
                        {index_name: index,
                         column_names[0]: self.design_var_descriptor[key]['key'],
                         column_names[1]: self.bspline_dict[key]['eval_t']})
                    # if the dataframe still not exists werite it
                    if self.design_var_descriptor[key][self.OUT_NAME] not in self.output_dict.keys():
                        self.output_dict[out_name] = df_to_merge
                        # if it exists, concatenate it in order to have multiple lines in the dataframe for each key
                    else:
                        self.output_dict[out_name] = concat([self.output_dict[out_name], df_to_merge],
                                                            ignore_index=True)
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
