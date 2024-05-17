'''
Copyright 2022 Airbus SAS
Modifications on {} Copyright 2024 Capgemini

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
import importlib
import json
import platform
import subprocess
from os import makedirs
from os.path import dirname, exists, join

import numpy as np
import pandas as pd

"""
    Script to generate a usecase.py with associated input data from a dm.pkl file

    Usage: 
        1- import class UsecaseCreator
        2- instantiate UsecaseCreator class with pickle path and options
        3- call pickle_file_to_usecase() method to generate usecase.py and input data
        4- move the generated usecase.py and data folder in the relevant process folder
        5- update the self.data_dir on the generated usecase.py file with the correct path

    Example:
        from sostrades_core.tools.pkl_converter.pickle_file_to_usecase import UsecaseCreator

        path = join(dirname(__file__), 'data','2187', 'dm.pkl')
        uc_creator = UsecaseCreator(
            pkl_path=path,
            usecase_name="usecase_reference_v1",
            write_default_value=True,
            write_outputs=False,
        )
        uc_creator.pickle_file_to_usecase()
"""


class NpEncoder(json.JSONEncoder):
    """Class to encode Numpy object to JSON"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class UsecaseCreator:
    def __init__(
        self,
        pkl_path: str,
        usecase_name: str,
        write_default_value: bool = False,
        write_outputs: bool = False,
        inputs_from_usecase = None,
    ) -> None:
        """Class to generate usecase.py file and related input and output data from a pickle file

        Args:
            pkl_path (str): path of the pickle file to convert. The usecase.py and data folder will also be created in the same folder
            usecase_name (str): name of the generated usecase
            write_default_value (bool, optional): Option to write the parameter value into the usecase even if it is the default value. Defaults to False.
            write_outputs (bool, optional): Option to write csv of selected outputs parameters. Defaults to False.
        """
        self.pkl_path = pkl_path
        self.usecase_name = usecase_name
        self.write_default_value = write_default_value
        self.write_outputs = write_outputs
        self.study_to_match=None
        if inputs_from_usecase:
            spec = importlib.util.spec_from_file_location("usecase", inputs_from_usecase)
            # creates a new module based on spec
            foo = importlib.util.module_from_spec(spec)
            # executes the module in its own namespace
            # when a module is imported or reloaded.
            spec.loader.exec_module(foo)
            self.study_to_match=foo.Study()
            # self.study_to_match.load_data()
            # self.study_to_match.ee.configure()
        self.dm_data_dict = pd.read_pickle(self.pkl_path)
        self.dump_dir = dirname(self.pkl_path)
        self.conversion_full_short = {}

        # list of all ignore parameter. most of them are numerical parameters
        self.ignore_list = [
            'cache_type',
            'cache_file_path',
            'n_processes',
            'chain_linearize',
            'use_lu_fact',
            'warm_start',
            'acceleration',
            'linear_solver_MDA',
            'linear_solver_MDO',
            'reset_history_each_run',
            'warm_start_threshold',
            'n_subcouplings_parallel',
            'linearization_mode',
            'residuals_history',
            'group_mda_disciplines',
            'linear_solver_MDA_preconditioner',
            'linear_solver_MDO_preconditioner',
            'max_mda_iter_gs',
            'relax_factor',
        ]

        # list of output parameter to write as csv is option is selected
        self.output_list = [
            'production_capacity_df',
            'production_rate_df',
            'rc',
            'rc_total',
            'nrc',
            'nrc_weights_final',
            'sale_price',
            'ac_infos_range_dict',
            'coc_airline_cat_df_dict',
            'coc_df_dict',
            'ref_deliveries',
            'sales_qty_df_dict',
            'cashflow_infos',
            'cashflow_infos_dollars',
            'cashflow_product',
            'cashflow_product_dollars',
            'hypothesis_summary',
            'pnl_product',
            'pnl_product_dollars',
            'cashflow_info_dollars_scenario_df',
            'cashflow_info_scenario_df',
            'cashflow_product_dollars_scenario_df',
            'cashflow_product_scenario_df',
            'hypothesis_summary_scenario_df',
            'pnl_dollars_scenario_df',
            'pnl_scenario_df',
            'sales_qty_scenario_df',
        ]

        # beginning of the usecase.py file
        self.begin = '''
from os.path import join, dirname
import numpy as np
import pandas as pd
import json
from sostrades_core.study_manager.study_manager import StudyManager

class Study(StudyManager):

\tdef __init__(self, run_usecase=True, execution_engine=None):
\t\tsuper().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

\tdef setup_usecase(self):
\t\tsty = self.study_name
\t\tinputs = {}\n
'''
        # end of the usecase.py file
        self.end = '''
\t\treturn [inputs]


if '__main__' == __name__:
\tuc_cls = Study(run_usecase=True)
\tuc_cls.load_data()
\tuc_cls.run(logger_level='DEBUG', for_test=False)
    '''

        self.dm_dict_to_write = {}
        self.outputs_to_write = {}
        self.same_value_dict = {}
        self.values_to_file_dict = {}
        self.short_names_list = []

    def get_short_name(self, ns: str) -> str:
        """Method to generate a short name to store the parameter
        This is useful to avoid long name and path that can be a problem on windows with path limit

        Args:
            ns (str): parameter key to shorten

        Returns:
            str: short parameter name generated
        """
        ns_clean = ns.replace('<study_ph>.', '')
        for i in range(1, len(ns.split('.')) + 1):
            short_name = '_'.join(ns_clean.split('.')[-i:])
            if short_name not in self.short_names_list:
                self.short_names_list.append(short_name)
                return short_name

    def convert_to_string(
        self, key: str, value, dump_dir: str, short_name: str = None
    ) -> str:
        """function to convert any parameter value to string.
        if the value is too big as a string, it will generate a file (csv or json depending on the value type)
        and return the path to the file

        Args:
            key (str): key of the parameter in the datamanager
            value (_type_): value of the parameter in the datamanager
            dump_dir (str): path of the directory to dump the created files
            short_name (str, optional): short name of the parameter file if necessary. Defaults to None.

        Returns:
            str: String to include in the usecase.py setting the value of the parameter
        """
        result_string = ""
        if isinstance(value, dict):
            result_string += "{"
            for sub_key in value:
                result_string += (
                    "'"
                    + str(sub_key)
                    + "' : "
                    + self.convert_to_string(sub_key, value[sub_key], dump_dir)
                    + ",\n"
                )
            result_string += "}"
            if len(result_string) > 1000:
                # the result will be very big in the python file so we convert it into JSON
                if short_name is None:
                    json_name = self.get_short_name(key)
                else:
                    json_name = short_name
                result_string = self.load_from_file(
                    dirName=join(dump_dir, 'data'),
                    fileName=f'{json_name}.json',
                    value=value,
                    key=key,
                )
        elif isinstance(value, pd.DataFrame):
            if value.size <= 50:
                result_string += "pd.DataFrame({\n"
                for col in value.columns:
                    result_string += (
                        "'"
                        + str(col)
                        + "' : "
                        + str(value[col].fillna('').values.tolist())
                        + ",\n"
                    )
                result_string += "})"
            else:
                if short_name is None:
                    csv_name = self.get_short_name(key)
                else:
                    csv_name = short_name
                result_string += self.load_from_file(
                    dirName=join(dump_dir, 'data'),
                    fileName=f'{csv_name}.csv',
                    value=value,
                    key=key,
                )

        elif isinstance(value, (int, float, list)):
            result_string += str(value)
        elif isinstance(value, (str)):
            result_string += "'" + str(value) + "'"
        elif isinstance(value, np.ndarray):
            result_string += f'np.asarray({value.tolist()})'
        else:

            print(type(value))

        return result_string

    def isEqual(self, value1, value2) -> bool:
        """Equality between 2 values able to handle dataframe, dict or list

        Raises:
            Exception: Type error if the values given are not dataframe, dict or list

        Returns:
            bool: True if the 2 values are identical, False otherwise
        """
        if isinstance(value1, pd.DataFrame):
            if value1.equals(value2):
                return True
        elif isinstance(value1, (dict, list)):
            return value1 == value2
        else:
            raise Exception(f'Unkown type to compare: {type(value1)}')

    def write_file(self, dirName: str, fileName: str, value) -> None:
        """Method to write a dataframe or dict as a file with a specific directory and filename

        Args:
            dirName (str): directory path to write the file to
            fileName (str): file name
            value (_type_): dataframe or dict to write as a file

        Raises:
            Exception: Type error is the value given is neither a dataframe or a dict
        """
        filePath = join(dirName, fileName)
        if isinstance(value, pd.DataFrame):
            value.to_csv(filePath, sep=',', index=False)

        elif isinstance(value, dict):
            try:
                with open(filePath, "w+") as outfile:
                    json.dump(value, outfile, cls=NpEncoder)
            except:
                print(f'Impossible to write {filePath}')
        else:
            raise Exception(f'Unkown type to write: {type(value)}')

    def get_same_values(self, key: str) -> list:
        """Method to retrieve parameter keys from the parameter to write that have the same exact value
        All the parameter keys which share the same values are stored in the result list same_value_ns_list

        Args:
            key (str): Parameter key to use as reference for comparing values

        Returns:
            list: Return same_value_ns_list representing all the parameters keys that have the same exact values as the input parameter key
        """
        same_value_ns_list = []
        compare_value = self.dm_dict_to_write[key]
        for ns, value in self.dm_dict_to_write.items():
            if ns != key and type(value) == type(compare_value):
                if self.isEqual(compare_value, value):
                    same_value_ns_list.append(ns)
        return same_value_ns_list

    def generate_same_values_dict(self) -> None:
        """Method to look for identical values used several times across the dm
        This is useful to avoid huge generated usecase.py or duplicated csv or json files
        This check is only happening for parameter types Dataframe, dict and list
        The results is stored in the dict self.same_value_dict.
        self.same_value_dict = {
            <parameter_key>: {
                'write':True / False --> only the first instance of the parameter will be written
                'variable_name': str --> generated short name of the variable, common to all instances
                'string_setup': str --> string result to include in usecase.py when write == True to set the variable
            }
        }
        """
        already_checked = []
        for ns, value in self.dm_dict_to_write.items():
            if (
                isinstance(value, (pd.DataFrame, dict, list))
                and ns not in already_checked
            ):
                same_value_ns_list = self.get_same_values(ns)
                if len(same_value_ns_list) > 0:
                    variable_name = self.get_short_name(ns)

                    variable_value_string = self.convert_to_string(
                        variable_name, value, self.dump_dir, short_name=variable_name
                    )
                    string_setup = (
                        '\t\t' + variable_name + '=' + variable_value_string + '\n'
                    )

                    self.same_value_dict[ns] = {
                        'write': True,
                        'variable_name': variable_name,
                        'string_setup': string_setup,
                    }
                    for ns_same_value in same_value_ns_list:
                        self.same_value_dict[ns_same_value] = {
                            'write': False,
                            'variable_name': variable_name,
                        }
                        already_checked.append(ns_same_value)

    def load_from_file(self, dirName: str, fileName: str, value, key: str) -> str:
        """Method to save a Dataframe or dict as a file in the dirName directory
        with the name fileName and return the string to put into the usecase.py to
        load this file in the usecase

        Args:
            dirName (str): directory path of the file
            fileName (str): filename
            value (_type_): Dataframe or dict to write
            key (str): parameter key of the value to write

        Raises:
            OSError: raison an error if the path is too long for Windows

        Returns:
            str: string to load the relevant file to add in the usecase.py
        """
        already_exists = False
        filePath = join(dirName, fileName)
        paramName = key.split('.')[-1]
        # check if value already saved into file
        if paramName in self.values_to_file_dict:
            # check if values are identical
            for file_value_dict in self.values_to_file_dict[paramName]:
                if self.isEqual(file_value_dict['value'], value):
                    already_exists = True
                    fileName = file_value_dict['fileName']
                    break
        else:
            self.values_to_file_dict[paramName] = []

        if not already_exists:
            # verify that data folder exists
            if not exists(dirName):
                makedirs(dirName)
            if platform.system() == 'Windows' and len(filePath) > 257:
                raise OSError(
                    f'Impossible to save {fileName}, path too long for windows'
                )

            # save value to file
            self.write_file(dirName, fileName, value)

            # save value into self.values_to_file_dict for future use
            self.values_to_file_dict[paramName].append(
                {'value': value, 'fileName': fileName}
            )

        # return string to load file
        if isinstance(value, pd.DataFrame):
            str_eval=self.get_converter_string(value)
            return f"pd.read_csv(join(self.data_dir,'{fileName}'){str_eval})"

        elif isinstance(value, dict):
            return f"json.load(open(join(self.data_dir,'{fileName}')))"

    def get_converter_string(self, df):
        """Method to build a string with the appropriate converter dict
        For some dataframe with elements that are lists, the read_csv function
        returns a string instead of a list if a converter is not specified.
        This method checks if such a conversion is needed for a given df and
        returns a string with the complement for 'pd.read_csv({csv}, {complement})'
        """
        col_to_eval = list(df.dtypes[(df.dtypes == object)].index) #list of col with object type
        str_eval, i_eval = '', 0 # initialize return string and counter
        if col_to_eval:
            str_eval = ', converters={'
            for col in col_to_eval:
                try: # test if the eval (str -> list) is achievable
                    assert isinstance(eval(str(df[col].values[0])), list)
                    str_eval = str_eval + f"'{col}': eval, "
                    i_eval += 1
                except:
                    pass
            str_eval = str_eval + '}'
        if i_eval == 0: # if there is no successful eval str to list, return empty string
            str_eval = ''
        return str_eval

    def filter_dm(self) -> None:
        """Method to fill out the dict self.dm_dict_to_write from the DataManager with only the input parameters values
        that are not in the ignore_list and the dict self.outputs_to_write with theselected output parameters
        if the option to write outputs is selected
        """

        for key in sorted(self.dm_data_dict.keys()):
            data_dict = self.dm_data_dict[key]
            if (
                key.split('.')[-1] not in self.ignore_list
                and data_dict['io_type'] == 'in'
            ):
                write = True
                if not self.write_default_value:
                    # check if value is different from default value:
                    if 'default' in data_dict and data_dict['default'] is not None:
                        if isinstance(data_dict['value'], pd.DataFrame) and isinstance(
                            data_dict['default'], pd.DataFrame
                        ):
                            if data_dict['value'].equals(data_dict['default']):
                                write = False
                        else:
                            if data_dict['value'] == data_dict['default']:
                                write = False
                if write:
                    self.dm_dict_to_write[key] = data_dict['value']

            if (
                key.split('.')[-1] in self.output_list
                and data_dict['io_type'] == 'out'
                and self.write_outputs
            ):
                self.outputs_to_write[key] = data_dict['value']

    def filter_dm_from_other_usecase(self) -> None:
        """Method to fill out the dict self.dm_dict_to_write from the DataManager using
        a list of inputs from another process
        """
        merged_setup_dict = {}
        for setup_dict in self.study_to_match.setup_usecase():
            merged_setup_dict.update(setup_dict)
        self.study_to_match.ee.load_study_from_input_dict(merged_setup_dict)
        for key, value in sorted(merged_setup_dict.items()):
            abstracted_key = key.replace(self.study_to_match.study_name, '<study_ph>')
            try:
                data_dict=self.study_to_match.ee.dm.get_data(key)
            except:
                continue
            if data_dict['io_type']=='in' and key.split('.')[-1] not in self.ignore_list:
                write = True
                if not self.write_default_value:
                    # check if value is different from default value:
                    if 'default' in data_dict and data_dict['default'] is not None:
                        if isinstance(data_dict['value'], pd.DataFrame) and isinstance(
                                data_dict['default'], pd.DataFrame
                        ):
                            if data_dict['value'].equals(data_dict['default']):
                                write = False
                        else:
                            if data_dict['value'] == data_dict['default']:
                                write = False
                if write:
                    simple_matches = [pkl_key for pkl_key in self.dm_data_dict.keys() if
                                      pkl_key.split('.')[-1] == key.split('.')[-1]]
                    if len(simple_matches) == 1:
                        self.dm_dict_to_write[abstracted_key] = self.dm_data_dict[simple_matches[0]]['value']
                    elif len(simple_matches)>1:
                        middle_key = '.'.join(abstracted_key.split('.')[-2:])
                        new_matches = [pkl_key for pkl_key in self.dm_data_dict.keys() if
                                       '.'.join(pkl_key.split('.')[-2:]) == middle_key]
                        if len(new_matches)==1:
                            self.dm_dict_to_write[abstracted_key] = self.dm_data_dict[new_matches[0]]['value']
                        elif len(new_matches)==0:
                            if type(value)==pd.DataFrame:
                                self.dm_dict_to_write[abstracted_key] = value.applymap(str)
                            elif type(value) in  [dict, list]:
                                self.dm_dict_to_write[abstracted_key] = [str(val) for val in value]
                            else:
                                if value is not None:
                                    self.dm_dict_to_write[abstracted_key] = value
                                    print(
                                        f'WARNING : input value {key} not found in pkl dm, value from usecase taken')
                                else:
                                    print(f'WARNING : input value {key} not added to usecase')
                        else:
                            raise ValueError(f'Too many matches found for {key}')
                    else:
                        self.dm_dict_to_write[abstracted_key] = value
                        print(f'WARNING : Input value {key} not found in pkl dm, value from usecase taken')

    def write_strings_to_file(self) -> None:
        """Method to write the usecase.py from the self.same_value_dict and the self.dm_dict_to_write"""
        with open(join(self.dump_dir, f'usecase_{self.usecase_name}.py'), 'w') as file:
            file.write(self.begin)

            # write data_dir
            file.write("\t\tself.data_dir = join(dirname(__file__), 'data')\n")

            # write all variables used several times
            for variable_info_dict in self.same_value_dict.values():
                if variable_info_dict['write']:
                    file.write(variable_info_dict['string_setup'])

            file.write('\n\n')
            for key, value in self.dm_dict_to_write.items():
                key_value_as_string = (
                    "\t\tinputs[f'{sty}." + '.'.join(key.split('.')[1:]) + "'] = "
                )
                if key in self.same_value_dict:
                    key_value_as_string += (
                        self.same_value_dict[key]['variable_name'] + "\n"
                    )
                else:
                    key_value_as_string += (
                        self.convert_to_string(key, value, self.dump_dir) + "\n"
                    )
                file.write(key_value_as_string)

            file.write(self.end)

    def pickle_file_to_usecase(self) -> None:
        """Main method to generate the usecase.py file and the associated data"""

        if not self.study_to_match:
            self.filter_dm()
        else:
            self.filter_dm_from_other_usecase()
        self.generate_same_values_dict()
        self.write_strings_to_file()

        # run black Python Code formatter to have a nice result in the usecase.py
        subprocess.run(
            ['black', join(self.dump_dir, f'usecase_{self.usecase_name}.py')]
        )
        print(
            f'✨✨ Usecase usecase_{self.usecase_name}.py has been written into {self.dump_dir} ✨✨'
        )

        if self.write_outputs:
            print('Write outputs')
            for key, value in self.outputs_to_write.items():
                fileName = key.replace('<study_ph>.', '').replace('.', '_')
                self.write_file(
                    dirName=join(self.dump_dir, 'outputs'),
                    fileName=fileName,
                    value=value,
                )

            print(
                f'{len(self.outputs_to_write.keys())} outputs files have been written'
            )
