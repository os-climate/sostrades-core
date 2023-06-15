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


def specific_check_years(dm):
    # check all elements of data dict
    data_dict = dm.data_dict

    for var_id in data_dict.keys():
        value = data_dict[var_id]['value']
        vtype = data_dict[var_id]['type']
        var_f_name = dm.get_var_full_name(var_id)
        io_type = data_dict[var_id]['io_type']
        optional = data_dict[var_id]['optional']
        # check if variable is a dataframe and if years is a column of this
        # dataframe
        if io_type == 'in' and not optional:
            if value is None:
                raise Exception(f'Value is None for variable : {var_f_name}')
            if vtype == 'dataframe' and 'years' in value:
                # get year start and year end
                year_start_name = dm.get_all_namespaces_from_var_name(
                    'year_start')[0]
                year_end_name = dm.get_all_namespaces_from_var_name('year_end')[
                    0]
                year_start = dm.get_value(year_start_name)
                year_end = dm.get_value(year_end_name)
                if len(dm.get_all_namespaces_from_var_name('time_step')) > 0:

                    time_step_name = dm.get_all_namespaces_from_var_name('time_step')[
                        0]
                    time_step = dm.get_value(time_step_name)
                else:
                    time_step = 1

                if not all(np.isin(np.arange(year_start, year_end + 1, time_step), value['years'].values)):
                    errors_in_dm_msg = f"Variable: The column years of '{var_f_name}' dataframe is not coherent with year start {year_start} and year end {year_end} \n years in dataframe : {min(value['years'].values)} to {max(value['years'].values)} !"
                    dm.logger.error(errors_in_dm_msg)

                    raise ValueError(
                        f'DataManager contains *value errors*: {errors_in_dm_msg}')
