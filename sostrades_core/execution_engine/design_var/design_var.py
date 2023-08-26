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
    FILL_ACTIVATED_ELEMENTS = 'fill_activated_elements'
    LAST_ELEMENT_ACTIVATED = 'last element activated'
    INITIAL_VALUE = 'initial_value'
    FILL_ACTIVATED_ELEMENTS_POSSIBLE_VALUES = [LAST_ELEMENT_ACTIVATED, INITIAL_VALUE]

    def __init__(self, inputs_dict, logger):
        '''
        Constructor
        '''
        self.design_var_descriptor = inputs_dict[self.DESIGN_VAR_DESCRIPTOR]
        self.output_dict = {}
        self.bspline_dict = {}
        self.dspace = inputs_dict[self.DESIGN_SPACE]
        self.logger = logger

    def configure(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.output_dict = {}
        list_ctrl = self.design_var_descriptor.keys()

        for elem in list_ctrl:
            l_activated = self.dspace.loc[self.dspace[self.VARIABLES]
                                          == elem, self.ACTIVATED_ELEM_LIST].to_list()[0]
            # check output length and compute BSpline only if necessary
            # remark: float do not require any BSpline usage
            output_length = len(self.design_var_descriptor[elem][self.INDEX])

            if not all(l_activated):
                final_value, gradient = self.rebuild_input_array_with_activated_elements(inputs_dict, elem)
            else:
                final_value = inputs_dict[elem]
                gradient = np.identity(output_length)

            if len(final_value) == output_length:
                self.bspline_dict[elem] = {
                    'bspline': None, 'eval_t': final_value, 'b_array': gradient}
            else:
                self.create_array_with_bspline(elem, final_value, output_length)

            # loop over design_var_descriptor to build output
            self.build_output_with_design_var_descriptor(elem, final_value)

    def create_array_with_bspline(self, elem, final_value, output_length):

        list_t = np.linspace(0.0, 1.0, output_length)
        bspline = BSpline(n_poles=len(final_value))
        bspline.set_ctrl_pts(final_value)
        eval_t, b_array = bspline.eval_list_t(list_t)
        b_array = bspline.update_b_array(b_array)

        self.bspline_dict[elem] = {
            'bspline': bspline, 'eval_t': eval_t, 'b_array': b_array}

    def rebuild_input_array_with_activated_elements(self, inputs_dict, elem):
        '''

        If some elements are desactivated wit the option activated_elem in design space, the function rebuild the correct array depending on the method

        '''
        # checks that dspace and activated elements are coherent with input element size
        l_activated = self.dspace.loc[self.dspace[self.VARIABLES]
                                      == elem, self.ACTIVATED_ELEM_LIST].to_list()[0]

        elem_input_value = list(inputs_dict[elem])
        if sum(l_activated) != len(elem_input_value):
            self.logger.error(
                f'The size of the input element {elem} is not coherent with the design space and its activated elements : {sum(l_activated)} activated elements and elem of length {len(elem_input_value)}')

        final_value = []
        # TODO compute the gradient for each case
        gradient = np.identity(len(l_activated))
        # We fill deactivated elements with the last element activated in the array
        if self.FILL_ACTIVATED_ELEMENTS in self.design_var_descriptor[elem] and self.design_var_descriptor[elem][
            self.FILL_ACTIVATED_ELEMENTS] == self.LAST_ELEMENT_ACTIVATED:

            for activated_bool in l_activated:
                if activated_bool:
                    final_value.append(elem_input_value.pop(0))
                else:
                    final_value.append(final_value[-1])
        # by default we use initial value to fill the deactivated elements
        else:
            initial_value = self.dspace.loc[self.dspace[self.VARIABLES]
                                            == elem, self.VALUE].to_list()[0]

            for i, activated_bool in enumerate(l_activated):
                if activated_bool:
                    final_value.append(elem_input_value.pop(0))
                else:
                    final_value.append(initial_value[i])
        return np.array(final_value), gradient

    def build_output_with_design_var_descriptor(self, elem, final_value):

        out_name = self.design_var_descriptor[elem][self.OUT_NAME]
        out_type = self.design_var_descriptor[elem][self.OUT_TYPE]

        if out_type == 'float':
            if final_value.size != 1:
                raise ValueError(" The input must be of size 1 for a float output")
            self.output_dict[out_name] = final_value[0]
        elif out_type == 'array':
            self.output_dict[out_name] = self.bspline_dict[elem]['eval_t']
        elif out_type == 'dataframe':
            # dataframe fill is optional ,by default we fill the dataframe with one column per key
            if self.DATAFRAME_FILL in self.design_var_descriptor[elem]:
                dataframe_fill = self.design_var_descriptor[elem][self.DATAFRAME_FILL]
            else:
                dataframe_fill = self.ONE_COLUMN_PER_KEY
            index = self.design_var_descriptor[elem][self.INDEX]
            index_name = self.design_var_descriptor[elem][self.INDEX_NAME]
            if dataframe_fill == self.ONE_COLUMN_PER_KEY:
                # for the method one column per key we create a dataframe if it does not exists
                if self.design_var_descriptor[elem][self.OUT_NAME] not in self.output_dict.keys():
                    # init output dataframes with index

                    self.output_dict[out_name] = DataFrame({index_name: index})
                # we use the key 'key' in the design_var_descriptor for the name of the column and the column to the dataframe
                col_name = self.design_var_descriptor[elem]['key']
                self.output_dict[out_name][col_name] = self.bspline_dict[elem]['eval_t']
            elif dataframe_fill == self.ONE_COLUMN_FOR_KEY:

                column_names = self.design_var_descriptor[elem][self.COLUMNS_NAMES]
                # # create a dataframe using column_names, in this method the dataframe will ALWAYS have 2 columns
                # first column will store the key
                # second column the value

                df_to_merge = DataFrame(
                    {index_name: index,
                     column_names[0]: self.design_var_descriptor[elem]['key'],
                     column_names[1]: self.bspline_dict[elem]['eval_t']})
                # if the dataframe still not exists werite it
                if self.design_var_descriptor[elem][self.OUT_NAME] not in self.output_dict.keys():
                    self.output_dict[out_name] = df_to_merge
                    # if it exists, concatenate it in order to have multiple lines in the dataframe for each key
                else:
                    self.output_dict[out_name] = concat([self.output_dict[out_name], df_to_merge],
                                                        ignore_index=True)
        else:
            raise (ValueError('Output type not yet supported'))
