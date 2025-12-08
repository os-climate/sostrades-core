'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16-2024/06/07 Copyright 2024 Capgemini

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

from __future__ import annotations

from copy import deepcopy
from functools import reduce
from typing import Dict, Optional

import numpy as np
import pandas as pd


class toolboxsum:
    """Toolbox with sum methods for dataframes and dictionaries."""

    def __init__(self) -> None:
        """Initialize the toolbox with sum result storage."""
        self.sum_df: Optional[pd.DataFrame] = None
        self.sum_dict_dict: Optional[Dict[str, Dict[str, float]]] = None

    def compute_sum_df(self, list_df: list[pd.DataFrame], not_sum: Optional[list[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute sum of a list of dataframes.

        Args:
            list_df: List of dataframes to sum.
            not_sum: Column names to exclude from summation.

        Returns:
            Tuple of (sum_df, resource_percent) where:
            - sum_df: Sum of all dataframes
            - resource_percent: Resource percentage breakdown

        """
        # initializations
        list_df_wo_columns = None
        restored_df = None
        # infer not summable columns in dataframe
        not_summable = list_df[0].convert_dtypes().select_dtypes(
            exclude=[np.number, 'datetime']).columns.to_list()
        not_sum_new=[]
        if len(not_summable):
            not_sum_new.extend(not_summable)

        if not_sum is not None:
            not_sum_new.extend(not_sum)
        not_sum_new = list(set(not_sum_new))

            # existing columns in dataframe
        in_columns = [
            col for col in not_sum_new if col in list(list_df[0].columns)]
        restored_df = list_df[0][in_columns]
        list_df_copy = deepcopy(list_df)
        list_df_wo_columns = [df.drop(columns=in_columns)
                              for df in list_df_copy]
        sum_df = reduce(lambda x, y: x.add(
            y, fill_value=0), list_df_wo_columns)
        if not_sum_new is not None:
            # sum_df = sum_df.join(restored_df)
            sum_df = restored_df.join(sum_df)

        sum_abs_df = reduce(lambda x, y: abs(x).add(
            abs(y), fill_value=0), list_df_wo_columns)

        resource_percent = pd.DataFrame(columns=['years'])

        if 'cash_in' in sum_df:
            resource_percent['years'] = sum_df['years']
            for i in range(len(list_df)):
                resource_percent['resource_' +
                                 str(i)] = (abs(list_df[i]['cash_in']) / sum_abs_df['cash_in'] * 100.).fillna(0)

        return sum_df, resource_percent

    def compute_sum_dict_dict_float(self, dict_to_sum: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """
        Compute sum of nested dictionaries containing float values.

        Args:
            dict_to_sum: Dictionary of dictionaries containing float values to sum.

        Note:
            Updates the instance's sum_dict_dict attribute with the computed sum.

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
