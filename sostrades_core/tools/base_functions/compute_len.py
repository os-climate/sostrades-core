'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/27-2024/05/16 Copyright 2023 Capgemini

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

from numpy import complex128, float32, float64, int32, int64, ndarray, number
from pandas.core.frame import DataFrame

'''
Exp_min function minimize an array with a min_value with a smooth decreasing exponential
The gradient of this function can also be used
'''

DEFAULT_EXCLUDED_COLUMNS = ['year', 'years']


def compute_len(obj, excluded_columns=DEFAULT_EXCLUDED_COLUMNS):
    '''
    Return len of any python object type
    '''
    if obj is None:
        return 0
    elif isinstance(obj, (int, float, bool, int32, int64, float32, float64, complex128, str)):
        return 1
    elif isinstance(obj, ndarray):
        return obj.size
    elif isinstance(obj, DataFrame):
        num_cols = len(set(obj.select_dtypes(include=[number]).columns) - set(excluded_columns))
        df_without_strings = obj.select_dtypes(exclude=['object'])
        # df_without_nans = df_without_strings.dropna()
        num_rows = len(df_without_strings)
        return num_rows * num_cols
    elif isinstance(obj, (tuple, list)):
        computed_len = 0
        for val in obj:
            computed_len += compute_len(val)
        return computed_len
    elif isinstance(obj, dict):
        computed_len = 0
        for val in obj.values():
            computed_len += compute_len(val)
        return computed_len
