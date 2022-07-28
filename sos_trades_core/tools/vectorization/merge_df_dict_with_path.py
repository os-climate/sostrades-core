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
Created on 27 july 2022

@author: NG9430A
'''
import numpy as np
import pandas as pd


def merge_df_dict_with_path(df_dict):
    df_with_path = pd.DataFrame({})
    for key, val in df_dict.items():
        val['PATH'] = key
        df_with_path = df_with_path.append(val, ignore_index=True)

    return df_with_path
