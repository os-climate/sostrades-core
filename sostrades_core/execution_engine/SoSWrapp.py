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
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class SoSWrappException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class SoSWrapp(object):
    '''**SoSWrapp** is the class from which inherits our model Wrapper
    It contains necessary information for the discipline configuration
    Its methods setup_sos_disciplines, run,... are overloaded by the discipline Wrapper

    '''
    # -- Disciplinary attributes
    DESC_IN = {}
    DESC_OUT = {}
    TYPE = 'type'
    SUBTYPE = 'subtype_descriptor'
    COUPLING = 'coupling'
    VISIBILITY = 'visibility'
    LOCAL_VISIBILITY = 'Local'
    INTERNAL_VISIBILITY = 'Internal'
    SHARED_VISIBILITY = 'Shared'
    NAMESPACE = 'namespace'
    VALUE = 'value'
    DEFAULT = 'default'
    EDITABLE = 'editable'
    USER_LEVEL = 'user_level'
    STRUCTURING = 'structuring'
    POSSIBLE_VALUES = 'possible_values'
    RANGE = 'range'
    UNIT = 'unit'
    NUMERICAL = 'numerical'
    DESCRIPTION = 'description'
    VISIBLE = 'visible'
    CONNECTOR_DATA = 'connector_data'
    VAR_NAME = 'var_name'
    # Dict  ex: {'ColumnName': (column_data_type, column_data_range,
    # column_editable)}
    DATAFRAME_DESCRIPTOR = 'dataframe_descriptor'
    DATAFRAME_EDITION_LOCKED = 'dataframe_edition_locked'
    IO_TYPE_IN = 'in'
    IO_TYPE_OUT = 'out'

    def __init__(self, sos_name):
        '''
        Constructor
        '''
        self.sos_name = sos_name
        self.local_data_short_name = {}
        self.run_output = {}

    def setup_sos_disciplines(self, proxy):  # type: (...) -> None
        """Define the set_up_sos_discipline of its proxy

        To be overloaded by subclasses.
        """
        pass

    def run(self):  # type: (...) -> None
        """Define the run of the discipline

        To be overloaded by subclasses.
        """
        raise NotImplementedError()
    
    def get_sosdisc_inputs(self, keys=None, in_dict=False):
        """Accessor for the inputs values as a list or dict

        :param keys: the input short names list
        :param in_dict: if output format is dict
        :returns: the inputs values list or dict
        """

        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            keys = list(self.local_data_short_name.keys())
            in_dict = True
        inputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_IN)
        if in_dict:
            # return inputs in an dictionary
            return inputs
        else:
            # return inputs in an ordered tuple (default)
            if len(inputs) > 1:
                return list(inputs.values())
            else:
                return list(inputs.values())[0]

    def _get_sosdisc_io(self, keys, io_type):
        """ Generic method to retrieve sos inputs and outputs

        :param keys: the data names list
        :param io_type: 'in' or 'out'
        :param full_name: if keys in returned dict are full names
        :returns: dict of keys values
        """

        # convert local key names to namespaced ones
        if isinstance(keys, str):
            keys = [keys]

        values_dict = {}

        for key in keys:
            values_dict[key] = self.local_data_short_name[key]

        return values_dict
    
    def _run(self):
        
        self.run()
        return self.run_output
    
    def store_sos_outputs_values(self, dict_values):
        
        self.run_output = dict_values
