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

import os
import pickle

from .logger_data import LoggerData

class TimeLogger(object):
    '''
    Logger is an object to get the log of the calculation process.
    Each object defines the variables to save through the dictionary "OUT_DICT"
    and gives access to these variables through the method "get_output"
    '''

    def __init__(self, tag, main_var='time', main_unit='(s)', out_save_dir='Outputs'):
        '''
        Constructor
        '''
        self.base_save_dir = out_save_dir
        self.__tag = tag
        self.__LoggerData = LoggerData(
            tag, main_var=main_var, main_unit=main_unit, out_save_dir=out_save_dir)
        self.__object_dict = dict()
        self.__index = 0

    #-- Setters
    def set_complex_mode(self, complex_mode):
        '''
        Set the complex mode in the Logger database
        '''
        self.__LoggerData.set_complex_mode(complex_mode)

    #-- Accessors
    def get_full_save_dir(self, save_dir=None):
        '''
        Save directory from the Logger Database
        '''
        return self.__LoggerData.get_full_save_dir(save_dir)

    def get_units(self, var_id):
        '''
        Get units from the Logger Database
        '''
        return self.__LoggerData.get_units(var_id)

    def get_var_ids(self):
        '''
        Getvariable names from the Logger Database
        '''
        return self.__LoggerData.get_var_ids()

    def get_data(self, var_id):
        '''
        Get a specific data named var_id from the Logger Database
        '''
        return self.__LoggerData.get_data(var_id)

    def get_index(self):
        '''
        Get the index of the logger : ?
        '''
        return self.__index

    #-- Methods
    def add_object_to_log(self, obj):
        '''
        Add an object to the logger database in object_dict
        '''
        if obj not in self.__object_dict:
            if hasattr(obj, "OUT_DICT"):
                self.__object_dict[obj] = obj.OUT_DICT
                for var_id in self.__object_dict[obj]:
                    database_id = obj.get_tag() + "." + var_id
                    self.__LoggerData.add_unit(
                        database_id, self.__object_dict[obj][var_id])
                    self.__LoggerData.add_to_database(database_id, None)
            else:
                print("Warning: ", obj.__class__,
                      "has no variable \"OUT_DICT\" ")

    def update_object_in_log(self, obj):
        '''
        Update an object to the logger database in object_dict
        '''
        if obj in self.__object_dict:
            if hasattr(obj, "OUT_DICT"):
                self.__object_dict[obj] = obj.OUT_DICT
                for var_id in self.__object_dict[obj]:
                    if obj.get_tag() + "." + var_id not in self.get_var_ids():
                        database_id = obj.get_tag() + "." + var_id
                        self.__LoggerData.add_unit(
                            database_id, self.__object_dict[obj][var_id])
                        self.__LoggerData.add_to_database(database_id, None)
                        self.__LoggerData.init_single_var_data(database_id)
            else:
                print("Warning: ", obj.__class__,
                      "has no variable \"OUT_DICT\" ")

    def add_diagram(self, name, x_axis, y_axis,
                    save_dir=None, sub_save_dir=None, y_log_scale=False):
        '''
        Add a diagram plot from the database into the specific directory
        '''
        self.__LoggerData.add_diagram(
            name, x_axis, y_axis, save_dir, sub_save_dir, y_log_scale)

    def resize(self, new_dim=None):
        '''
        Resize the logger database
        '''
        self.__LoggerData.resize(new_dim)

    def reset_index(self):
        '''
        Reset the index of the logger
        '''
        self.__index = 0

    def initialize(self, size=0):
        '''
        Initialize the logger :
            -index = 0
            -init the database with the keys from OUT_DICT object in object_dict
        '''
        self.__index = 0

        list_keys = []
        for obj in self.__object_dict:
            if hasattr(obj, "OUT_DICT"):
                self.__object_dict[obj] = obj.OUT_DICT
                for var_id in self.__object_dict[obj]:
                    database_id = obj.get_tag() + "." + var_id
                    if database_id not in self.__LoggerData.get_var_ids():
                        self.__LoggerData.add_unit(
                            database_id, self.__object_dict[obj][var_id])
                        self.__LoggerData.add_to_database(database_id, None)
                    list_keys.append(database_id)

        #-- Init all arrays for database_id in list_keys
        self.__LoggerData.init_database(list_keys, size)

    def run(self, main_var):
        '''
        Update the database with variables from the current index
        and move the index to the next time
        '''
        self.__LoggerData.add_to_main_var_index(self.__index, main_var)
        for obj in self.__object_dict:
            for var_id in self.__object_dict[obj]:
                database_id = obj.get_tag() + "." + var_id
                self.__LoggerData.add_to_database_index(
                    database_id, self.__index, obj.get_output(var_id))
        self.__index += 1

    def export_to_file(self):
        '''
        Export the Logger database to a file
        '''
        self.__LoggerData.export_to_file()

    def plot(self):
        '''
        Plot the Logger database
        '''
        self.__LoggerData.plot()

    def dump_logger_data(self):
        '''
        Dump the Logger database
        '''
        save_dir = self.get_full_save_dir()
        filename = os.path.join(save_dir, self.__tag + '_logger_data.pkl')
        fid = open(filename, 'wb')
        pickle.dump(self.__LoggerData, fid)
        fid.close()
