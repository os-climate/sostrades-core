'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/23-2023/11/03 Copyright 2023 Capgemini

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
Data manager pickle (de)serializer
'''
from os.path import join, dirname, basename
from pathlib import Path
from os import makedirs, remove
from time import sleep
from tempfile import gettempdir
from io import BytesIO, StringIO
from shutil import rmtree, make_archive
import warnings

from pandas import DataFrame, read_pickle, concat
from numpy import ndarray

from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.execution_engine.ns_manager import NS_SEP

CSV_SEP = ','
FILE_URL = 'file:///'

# def anonymize_dict(dict_to_convert, anonymize_fct=None):
#     if anonymize_fct is not None:
#         converted_dict = {}
#         for key in dict_to_convert.keys():
#             new_key = anonymize_fct(key)
#             converted_dict[new_key] = dict_to_convert[key]
#     else:
#         converted_dict = deepcopy(dict_to_convert)
#     return converted_dict


def strip_study_from_namespace(var):
    st_n = get_study_from_namespace(var)
    return var.replace(st_n + NS_SEP, '')


def get_study_from_namespace(var):
    return var.split(NS_SEP)[0]


def generate_unique_data_csv(data_dict, csv_file_path):
    # just convert data to dataframe regarding its type
    df_data = DataSerializer.convert_data_to_dataframe(data_dict)
    # export data as a DataFrame using buffered I/O streams
    df_data.to_csv(csv_file_path, sep=CSV_SEP, header=True, index=False)


class DataSerializer:
    """
    Data serializer class
    """

    pkl_filename = 'dm.pkl'
    val_filename = 'dm_values.csv'
    disc_status_filename = 'disciplines_status.pkl'
    cache_filename = 'cache.pkl'

    default_persistance_strategy = DirectLoadDump

    def __init__(self, root_dir=None, rw_object=None, study_filename=None):
        if root_dir is not None:
            self.dm_db_root_dir = root_dir
        else:
            self.dm_db_root_dir = join(gettempdir(), 'DM_dB')

        self.study_filename = study_filename
        self.dm_val_file = None
        self.dm_pkl_file = None
        self.cache_file = None
        # set load/dump strategy
        self.direct_rw_strategy = DirectLoadDump()
        self.encryption_strategy = None
        self.set_strategy_method(rw_object)
        self.check_db_dir()

    def set_strategy_method(self, rw_strat):
        ''' method that set load/dump strategy method '''
        if rw_strat is None:
            rw_strat = self.default_persistance_strategy()
        elif not isinstance(rw_strat, object):
            raise Exception('object %s type %s should be a strategy class' % (
                str(rw_strat), type(rw_strat)))
        self.encryption_strategy = rw_strat

    def check_db_dir(self):
        ''' prepare folder that will store local study DM files '''
        db_dir = self.dm_db_root_dir
        if not Path(db_dir).is_dir():
            if Path(db_dir).is_file():
                # sometimes the folder is a file (!) so we remove it
                try:
                    remove(db_dir)
                except OSError as e:
                    print("Error DM_db should not be a file and could not be deleted: %s : %s" % (db_dir, e.strerror))
            # we set the option exists_ok=True so that if the folder already exists it doen't raise an error
            makedirs(db_dir, exist_ok=True)
        

    def is_structured_data_type(self, data):
        return isinstance(data, ndarray) \
            or isinstance(data, list) \
            or isinstance(data, DataFrame) \
            or isinstance(data, dict)

#     def dump_data_dict(self, data_dict, anonymize_function=None):
#         '''
#         :params: anonymize_function, a function that map a given key of the data
#         dictionary using rule for the saving process
#         :type: function
#         '''
#         dm_dir = join(self.dm_db_root_dir, self.study_filename)
#         if not Path(dm_dir).is_dir():
#             makedirs(dm_dir)
#             sleep(0.1)
#         converted_dict = anonymize_dict(data_dict,
#                                         anonymize_fct=anonymize_function)
#         # export full DM data_dict to unique pickle file
#         study_dir = join(self.dm_db_root_dir, self.study_filename)
#         self.dm_pkl_file = join(study_dir, self.pkl_filename)
#         # serialise raw tree_node.data dict with pickle
#         self.encryption_strategy.dump(converted_dict, self.dm_pkl_file)

    @staticmethod
    def study_data_manager_file_path(study_to_load):
        """
        Get the file path to the dm.pkl file
        :param study_to_load: folder to look fo study file data
        :return: str
        """

        return join(study_to_load, DataSerializer.pkl_filename)

    @staticmethod
    def study_disciplines_status_file_path(study_to_load):
        """
        Get the file path to the disciplines_status.pkl file
        :param study_to_load: folder to look fo study file data
        :return: str
        """

        return join(study_to_load, DataSerializer.disc_status_filename)

    @staticmethod
    def study_cache_file_path(study_to_load):
        """
        Get the file path to the disciplines_status.pkl file
        :param study_to_load: folder to look fo study file data
        :return: str
        """

        return join(study_to_load, DataSerializer.cache_filename)

    def dump_disc_status_dict(self, study_to_load, rw_strategy, status_dict):
        ''' export disciplines status into binary file (containing disc/status info into dictionary) '''

        status_dict_f = join(study_to_load, self.disc_status_filename)

        rw_strategy.dump(status_dict, status_dict_f)
        
    def load_cache_dict(self, study_to_load, rw_strategy):
        ''' load disciplines status from binary file (containing disc/status info into dictionary) '''

        cache_dict_f = self.get_dm_file(study_to_load=study_to_load,
                                         file_type=self.cache_filename)
        
        if cache_dict_f is not None:
            return rw_strategy.load(cache_dict_f)

    def load_disc_status_dict(self, study_to_load, rw_strategy):
        ''' load disciplines status from binary file (containing disc/status info into dictionary) '''

        status_dict_f = self.get_dm_file(study_to_load=study_to_load,
                                         file_type=self.disc_status_filename)

        return rw_strategy.load(status_dict_f)

    def export_data_dict_to_csv(self, origin_dict, export_dir=None):
        ''' export values and units of the whole DM data_dict to csv file '''
        data_df = DataFrame(columns=['unit', 'value'])
        for key, val in origin_dict.items():
            val_to_display = val['value']
            if self.is_structured_data_type(val_to_display):
                csv_f = join(export_dir, key + '.csv')
                generate_unique_data_csv(val_to_display, csv_f)
                val_to_display = FILE_URL + basename(csv_f)
            data_df.loc[key] = {'unit': val['unit'],
                                'value': val_to_display}
        # force null cells to None instead of NaN (avoiding SQL issue)
        data_df = data_df.where(data_df.notnull(), None)
        return data_df

    def export_data_dict_and_zip(self, origin_dict, export_dir=None):
        ''' export values and units of the whole DM data_dict to csv file and zip '''
        if not Path(export_dir).is_dir():
            makedirs(export_dir)
            sleep(0.1)
        data_df = self.export_data_dict_to_csv(origin_dict,
                                               export_dir=export_dir)
        self.dm_val_file = join(export_dir, self.val_filename)
        data_df.to_csv(self.dm_val_file, sep=CSV_SEP,
                       columns=data_df.columns)
        # zip folder and return zip filepath
        export_dir_zip = make_archive(export_dir,
                                      'zip',
                                      dirname(export_dir),
                                      basename(export_dir))
        sleep(0.1)
        rmtree(export_dir)
        return export_dir_zip

    def load_from_pickle(self,
                         data_dict,
                         rw_strategy,
                         just_return_data_dict=False,):
        '''
        load a pickled file to a dict
        according to serialisation strategy (pickled dataframe or pickled raw dict data)
        and update data_dict with this data
        loop on the items of this dict
            if key matches with given data_dict key
                restore data from raw data (type is kept)
                and update data_dict with this data
        '''

        loaded_dict = rw_strategy.load(self.dm_pkl_file)

        def is_variable_wo_studyname_exists(var, a_d):
            ''' check if variable exists in data dict
            by ignoring the study name, i.e. the first element splitted by .'''
            key_wo_st = strip_study_from_namespace(var)
            no_study_keys = [strip_study_from_namespace(k) for k in a_d.keys()]
            return key_wo_st in no_study_keys

        current_st_n = None
        loaded_st_n = None
        if not just_return_data_dict:
            current_st_n = get_study_from_namespace(list(data_dict.keys())[0])
            loaded_st_n = get_study_from_namespace(list(loaded_dict.keys())[0])
        for param_id, param_dict in loaded_dict.items():
            if just_return_data_dict or is_variable_wo_studyname_exists(
                    param_id, data_dict):
                if current_st_n != loaded_st_n:
                    param_id = param_id.replace(loaded_st_n, current_st_n)
                if just_return_data_dict:
                    data_dict[param_id] = {}
                data_dict[param_id].update(param_dict)
            else:
                sp_var = strip_study_from_namespace(param_id)
                raise KeyError(
                    f'Variable {sp_var} does not exist into {data_dict.keys()}')

    def get_dm_file(self, study_to_load, file_type=None):
        ''' return  paths of files containing data for a given study  '''
        if file_type is None:
            file_type = self.pkl_filename

        if not Path(study_to_load).is_dir():
            # if just name of a study was given, compose the path with root_dir
            study_to_load = join(self.dm_db_root_dir, study_to_load)
            if not Path(study_to_load).is_dir():
                raise IOError('DM file %s does not exist' % study_to_load)
        dm_files = [f for f in Path(study_to_load).rglob(file_type)]
        if len(dm_files) != 1:
            f_t = 'values csv' if file_type == self.val_filename else 'pickle'
            if len(dm_files) == 0:
                # instead of of data.pkl and disciplines_status.pkl, cache.pkl file is optional
                if file_type == self.cache_filename:
                    return None
                else:
                    raise IOError(
                        f'There is no DM {f_t} file from {study_to_load}')
            else:
                _d = dirname(dm_files[0])
                _f = ', '.join([basename(f) for f in dm_files])
                raise IOError(
                    f'Too many DM {f_t} files from {study_to_load}, from {_d}: {_f}')
        return dm_files[0]

    def set_dm_pkl_files(self, study_to_load=None):
        self.dm_pkl_file = self.get_dm_file(study_to_load=study_to_load)

    def set_dm_val_files(self, study_to_load):
        self.dm_val_file = self.get_dm_file(study_to_load=study_to_load,
                                            file_type=self.val_filename)

    def put_dict_from_study(self, study_to_load: str, rw_strategy, data_dict):
        '''
        :params: anonymize_function, a function that map a given key of the data
        dictionary using rule for the saving process
        :type: function
        '''

        if not Path(study_to_load).is_dir():
            makedirs(study_to_load, exist_ok=True)
            sleep(0.1)

        # export full data_dict to unique pickle file
        self.dm_pkl_file = join(study_to_load, self.pkl_filename)

        # serialise raw tree_node.data dict with pickle
        rw_strategy.dump(data_dict, self.dm_pkl_file)
        
    def put_cache_from_study(self, study_to_load, rw_strategy, cache_map):
        '''
        :params: anonymize_function, a function that map a given key of the data
        dictionary using rule for the saving process
        :type: function
        '''

        if not Path(study_to_load).is_dir():
            makedirs(study_to_load, exist_ok=True)
            sleep(0.1)

        # export full cache_map to unique pickle file
        self.cache_file = join(study_to_load, self.cache_filename)

        # serialise raw cache_map with pickle
        rw_strategy.dump(cache_map, self.cache_file)

    def get_dict_from_study(self, study_to_load, rw_strategy):
        ''' function that load every pickle files into a location
        and return a dictionary updated with loaded info '''
        self.set_dm_pkl_files(study_to_load)

        return_data_dict = {}

        self.load_from_pickle(data_dict=return_data_dict,
                              just_return_data_dict=True,
                              rw_strategy=rw_strategy)

        if not return_data_dict:
            raise Exception(f'nothing to load from {study_to_load}')
        return return_data_dict

    def get_data_dict_from_pickle(self):
        ''' extract data from pickled file '''
        # no need to get the unique pickle dm file path, it should exist yet
        # read pickle as a dict
        return read_pickle(self.dm_pkl_file)

    @staticmethod
    def convert_data_to_dataframe(param_data):
        # convert data to DataFrame regarding to its type
        if isinstance(param_data, DataFrame):
            # if data is already a dataframe, nothing to do
            df_data = param_data
        elif isinstance(param_data, ndarray) or isinstance(param_data, list):
            # add header 'value'
            if len(param_data) == 0:
                df_data = DataFrame(columns=['value'])
            else:
                first_el = param_data[0]
                if isinstance(first_el, dict):
                    # get the keys of sub dict to make columns of dataframe
                    df_sub_col = []
                    for param_dict in param_data:
                        df_sub_col.extend(list(param_dict.keys()))
                    df_sub_col = list(set(df_sub_col))
                    df_col = ['variable'] + list(df_sub_col)
                    df_data = DataFrame(columns=df_col)
                    # iterate on keys of this dict and concatenate value
                    for k, a_d in enumerate(param_data):
                        # append built dataframe to global one
                        df_sub_col = a_d.keys()
                        a_df = DataFrame([a_d.values()], columns=df_sub_col)
                        df_data = concat([df_data, a_df.assign(variable=k)],
                                         sort=False)
                    df_data = df_data[df_col]
                else:
                    df_data = DataFrame(param_data, columns=['value'])
        elif isinstance(param_data, dict):

            if len(param_data.keys()) == 0:
                df_data = DataFrame(columns=['variable', 'value'])
            else:
                # if data is a dict, look at the content type
                first_el = param_data[list(param_data.keys())[0]]

                if isinstance(first_el, DataFrame):
                    # use the columns of sub df as columns of dataframe
                    df_col = first_el.columns
                    df_col = df_col.insert(0, 'variable')
                    df_data = DataFrame(columns=df_col)
                    # iterate on keys of this dict and concatenate value
                    for k, a_df in param_data.items():
                        # append built dataframe to global one
                        df_data = concat([df_data, a_df.assign(variable=k)],
                                         sort=False)
                    df_data = df_data[df_col]
                else:
                    # dict of values, so just add header 'value'
                    df_data = DataFrame(param_data.items(),
                                        columns=['variable', 'value'])
        else:
            # # single value as scalar, compose a dataframe anyway adding header 'value'
            df_data = DataFrame([param_data], columns=['value'])
        return df_data

    def get_parameter_data(self, var_key):
        # get data_dict from pickle file
        self.set_dm_pkl_files()
        return self.get_data_dict_from_pickle()[var_key]

    def convert_to_dataframe_and_bytes_io(self, param_value, param_key):
        # and convert to dataframe
        df_data = self.convert_data_to_dataframe(param_value)
        # export data as a DataFrame using buffered I/O streams
        return self.convert_to_bytes_io(df_data, param_key)

    def convert_model_table_to_df_and_bytes_io(self, model_table, param_key):
        ''' convert to dataframe '''
        df_data = DataFrame(model_table[1:], columns=model_table[0])
        return self.convert_to_bytes_io(df_data, param_key)

    def convert_model_table_to_df(self, model_table):
        ''' convert to dataframe '''
        df_data = DataFrame(model_table[1:], columns=model_table[0])
        return df_data

    def convert_to_bytes_io(self, df_data, data_key):
        ''' export data as a DataFrame using buffered I/O streams '''
        try:
            df_stream = StringIO()
            df_data.to_csv(df_stream, sep=CSV_SEP, header=True, index=False)
            df_bytes = df_stream.getvalue().encode()
            # just return the in-memory bytes buffer
            return BytesIO(df_bytes)
        except Exception as error:
            raise Exception(
                f'ERROR converting {data_key} value to bytes: ' + str(error))
