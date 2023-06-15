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
import unittest
from sostrades_core.tools.sumdfdict.toolboxsum import toolboxsum

import pandas as pd
from pandas.testing import assert_frame_equal


class TestSumDF(unittest.TestCase):
    """
    Sum dataframe tool test class
    """

    def test_01_sum_df(self):
        df1 = pd.DataFrame(
            data={'column1': ['row1', 'row2'], 'column2': [1, 1]})

        df_ref = pd.DataFrame(
            data={'column1': ['row1', 'row2'], 'column2': [2, 2]})

        list_to_sum = [df1, df1]
        toolboxsumtest = toolboxsum()
        df_result, percent = toolboxsumtest.compute_sum_df(
            list_to_sum, ['column1'])

        df_result = df_result.reindex(columns=df_ref.columns)
        assert_frame_equal(df_ref, df_result)

    def test_02_sum_dict(self):
        dict_1 = {'key1': {'key2': {'key3': 5}}, 'key2': {'key2': {'key3': 8}}}
        dict_1_sum_ref = {'key2': {'key3': 13}}

        toolboxsumtest = toolboxsum()
        toolboxsumtest.compute_sum_dict_dict_float(dict_1)

        dict_sum = toolboxsumtest.sum_dict_dict
        assert dict_1_sum_ref == dict_sum


if __name__ == "__main__":
    unittest.main()
