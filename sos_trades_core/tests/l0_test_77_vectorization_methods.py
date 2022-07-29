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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import numpy as np
import pandas as pd
from sos_trades_core.tools.vectorization.vectorization_methods import *


class TestVectorizationMethods(unittest.TestCase):
    """
    Base Function test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        pass

    def test_01_merge_df_dict_with_path_function(self):
        '''
        Test the function
        '''

        input_df_dict = {
            'AC1': pd.DataFrame(
                {
                    'EIS': 2020,
                    'year_service': [1, 2, 3],
                    'composite_fraction': [85.0, 80.0, 70.0],
                    'max_npax': 66.0,
                    'load_factor': 0.0,
                }
            ),
            'AC2': pd.DataFrame(
                {
                    'EIS': 2020,
                    'year_service': [1, 2, 3],
                    'composite_fraction': [55.0, 50.0, 40.0],
                    'max_npax': 66.0,
                    'load_factor': 0.0,
                }
            ),
            'AC50': pd.DataFrame(
                {
                    'EIS': 2020,
                    'year_service': [1, 2, 3],
                    'composite_fraction': [35.0, 30.0, 20.0],
                    'max_npax': 66.0,
                    'load_factor': 0.0,
                }
            ),
        }

        df_with_path_default = pd.DataFrame(
            {
                'EIS': 2020,
                'year_service': [1, 2, 3, 1, 2, 3, 1, 2, 3],
                'composite_fraction': [
                    85.0,
                    80.0,
                    70.0,
                    55.0,
                    50.0,
                    40.0,
                    35.0,
                    30.0,
                    20.0,
                ],
                'max_npax': 66.0,
                'load_factor': 0.0,
                'PATH': [
                    'AC1',
                    'AC1',
                    'AC1',
                    'AC2',
                    'AC2',
                    'AC2',
                    'AC50',
                    'AC50',
                    'AC50',
                ],
            }
        )
        df_with_path = merge_df_dict_with_path(input_df_dict)
        if not df_with_path.equals(df_with_path_default):
            raise Exception('Dataframes not equal!')

    def test_02_get_inputs_for_path_function(self):
        df_with_path_default = pd.DataFrame(
            {
                'EIS': 2020,
                'year_service': [1, 2, 3, 4, 5, 6, 5, 7, 9, 1, 2, 3, 4, 5, 6, 1, 2, 3],
                'composite_fraction': [
                    85.0,
                    80.0,
                    70.0,
                    85.0,
                    80.0,
                    70.0,
                    85.0,
                    80.0,
                    70.0,
                    55.0,
                    50.0,
                    40.0,
                    55.0,
                    50.0,
                    40.0,
                    35.0,
                    30.0,
                    20.0,
                ],
                'max_npax': 66.0,
                'load_factor': 0.0,
                'PATH': [
                    'AC1.component1',
                    'AC1.component1',
                    'AC1.component1',
                    'AC1.component2',
                    'AC1.component2',
                    'AC1.component2',
                    'AC1',
                    'AC1',
                    'AC1',
                    'AC2.component1',
                    'AC2.component1',
                    'AC2.component1',
                    'AC2.component2',
                    'AC2.component2',
                    'AC2.component2',
                    'AC50',
                    'AC50',
                    'AC50',
                ],
            }
        )

        df_with_AC1_component1_path = pd.DataFrame(
            {
                'EIS': 2020,
                'year_service': [1, 2, 3],
                'composite_fraction': [
                    85.0,
                    80.0,
                    70.0,
                ],
                'max_npax': 66.0,
                'load_factor': 0.0,
                'PATH': [
                    'AC1.component1',
                    'AC1.component1',
                    'AC1.component1',
                ],
            }
        )
        df_with_AC1_path = pd.DataFrame(
            {
                'EIS': 2020,
                'year_service': [5, 7, 9],
                'composite_fraction': [
                    85.0,
                    80.0,
                    70.0,
                ],
                'max_npax': 66.0,
                'load_factor': 0.0,
                'PATH': [
                    'AC1',
                    'AC1',
                    'AC1',
                ],
            }
        )
        filtered_input_parameter = get_inputs_for_path(
            input_parameter=df_with_path_default,
            PATH='AC1.component1',
            parent_path_admissible=False,
        )
        if not filtered_input_parameter.equals(df_with_AC1_component1_path):
            raise Exception('Dataframes not equal!')

        filtered_input_parameter = get_inputs_for_path(
            input_parameter=df_with_path_default,
            PATH='AC1.component3',
            parent_path_admissible=True,
        )
        if not filtered_input_parameter.equals(df_with_AC1_path):
            raise Exception('Dataframes not equal!')

    def test_03_compute_parent_path_sum_function(self):
        df_with_path_default = pd.DataFrame(
            {
                'EIS': [
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                ],
                'year_service': [1, 2, 3, 4, 5, 6, 5, 7, 9, 1, 2, 3, 4, 5, 6, 1, 2, 3],
                'composite_fraction': [
                    85.0,
                    80.0,
                    70.0,
                    85.0,
                    80.0,
                    70.0,
                    85.0,
                    80.0,
                    70.0,
                    55.0,
                    50.0,
                    40.0,
                    55.0,
                    50.0,
                    40.0,
                    35.0,
                    30.0,
                    20.0,
                ],
                'max_npax': 66.0,
                'load_factor': 0.0,
                'PATH': [
                    'AC1.component1',
                    'AC1.component1',
                    'AC1.component1',
                    'AC1.component2',
                    'AC1.component2',
                    'AC1.component2',
                    'AC1',
                    'AC1',
                    'AC1',
                    'AC2.component1',
                    'AC2.component1',
                    'AC2.component1',
                    'AC2.component2',
                    'AC2.component2',
                    'AC2.component2',
                    'AC50',
                    'AC50',
                    'AC50',
                ],
            }
        )

        df_with_all_paths_parent_included = pd.DataFrame(
            {
                'EIS': [
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                    2020,
                    2021,
                    2022,
                ],
                'year_service': [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    5,
                    7,
                    9,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    5,
                    7,
                    9,
                    1,
                    2,
                    3,
                ],
                'composite_fraction': [
                    85.0,
                    80.0,
                    70.0,
                    85.0,
                    80.0,
                    70.0,
                    85.0,
                    80.0,
                    70.0,
                    55.0,
                    50.0,
                    40.0,
                    55.0,
                    50.0,
                    40.0,
                    55.0,
                    50.0,
                    40.0,
                    35.0,
                    30.0,
                    20.0,
                ],
                'max_npax': 66.0,
                'load_factor': 0.0,
                'PATH': [
                    'AC1.component1',
                    'AC1.component1',
                    'AC1.component1',
                    'AC1.component2',
                    'AC1.component2',
                    'AC1.component2',
                    'AC1',
                    'AC1',
                    'AC1',
                    'AC2.component1',
                    'AC2.component1',
                    'AC2.component1',
                    'AC2.component2',
                    'AC2.component2',
                    'AC2.component2',
                    'AC2',
                    'AC2',
                    'AC2',
                    'AC50',
                    'AC50',
                    'AC50',
                ],
            }
        )
        df_with_parent_path_included = check_compute_parent_path_sum(
            df_with_path_default,
            columns_not_to_sum=['EIS', 'composite_fraction', 'max_npax'],
            based_on='EIS',
        )
        df_with_all_paths_parent_included = (
            df_with_all_paths_parent_included.sort_values(["PATH", "EIS"]).reset_index(
                drop=True
            )
        )
        if not df_with_parent_path_included.equals(df_with_all_paths_parent_included):
            raise Exception('Dataframes not equal!')


if __name__ == "__main__":
    unittest.main()
