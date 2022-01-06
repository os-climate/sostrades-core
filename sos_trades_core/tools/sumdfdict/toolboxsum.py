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

import numpy as np
import pandas as pd
from copy import deepcopy

from _functools import reduce


class toolboxsum(object):
    '''
    Tool box with sum methods
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.sum_df = None
        self.sum_dict_dict = None

    def compute_sum_df(self, list_df, not_sum=None):
        """
        Method to compute sum of dict of dataframes
        not_sum : column name to not sum
        """
        # infer not summable columns in dataframe
        not_summable = list_df[0].convert_dtypes().select_dtypes(
            exclude=[np.number, 'datetime']).columns.to_list()
        if len(not_summable):
            if not_sum is None:
                not_sum = not_summable
            else:
                not_sum.extend(not_summable)
                not_sum = list(set(not_sum))

        if not_sum is not None:

            # existing columns in dataframe
            in_columns = [
                col for col in not_sum if col in list(list_df[0].columns)]
            restored_df = list_df[0][in_columns]
            list_df_copy = deepcopy(list_df)
            list_df_wo_columns = [df.drop(columns=in_columns)
                                  for df in list_df_copy]
        sum_df = reduce(lambda x, y: x.add(
            y, fill_value=0), list_df_wo_columns)
        if not_sum is not None:
            sum_df = sum_df.join(restored_df)

        sum_abs_df = reduce(lambda x, y: abs(x).add(
            abs(y), fill_value=0), list_df_wo_columns)

        resource_percent = pd.DataFrame(columns=['years'])

        if 'cash_in' in sum_df:
            resource_percent['years'] = sum_df['years']
            for i in range(len(list_df)):
                resource_percent['resource_' +
                                 str(i)] = (abs(list_df[i]['cash_in']) / sum_abs_df['cash_in'] * 100.).fillna(0)

        return sum_df, resource_percent

    def compute_sum_dict_dict_float(self, dict_to_sum):
        """
        Method to compute sum of dict of dict of float
        """

        # init the dict
        self.sum_dict_dict = {}
        for dict_1 in dict_to_sum.values():
            for dict_key, dict_dict in dict_1.items():
                self.sum_dict_dict[dict_key] = {}
                for dict_dict_key, dict_dict_value in dict_dict.items():
                    self.sum_dict_dict[dict_key][dict_dict_key] = 0

        for dict_1 in dict_to_sum.values():
            for dict_key, dict_dict in dict_1.items():
                for dict_dict_key, dict_dict_value in dict_dict.items():
                    self.sum_dict_dict[dict_key][dict_dict_key] += dict_dict_value
