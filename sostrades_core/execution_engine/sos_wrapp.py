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
import logging

LOGGER = logging.getLogger(__name__)

class SoSWrappException(Exception):
    pass

class SoSWrapp(object):
    '''**SoSWrapp** is the class from which inherits our model wrapper when using 'SoSTrades' wrapping mode.

    It contains necessary information for the discipline configuration. It is owned by both the MDODisciplineWrapp and
    the SoSMDODiscipline.

    Its methods setup_sos_disciplines, run,... are overloaded by the user-provided Wrapper.

    N.B.: setup_sos_disciplines needs take as argument the proxy and call proxy.add_inputs() and/or proxy.add_outputs().

    Attributes:
        sos_name (string): name of the discipline
        local_data_short_name (Dict[Dict]): short name version of the local data for model input and output
        local_data (Dict[Any]): output of the model last run
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
        Constructor.

        Arguments:
            sos_name (string): name of the discipline
        '''
        self.sos_name = sos_name
        self.input_full_name_map = {}
        self.output_full_name_map = {}
        self.input_data_names = []
        self.output_data_names = []
        self.attributes = {}
        self.local_data = {}

    def setup_sos_disciplines(self, proxy):  # type: (...) -> None
        """
        Define the set_up_sos_discipline of its proxy

        To be overloaded by subclasses.

        Arguments:
            proxy (ProxyDiscipline): the proxy discipline for dynamic i/o configuration
        """
        pass

    def run(self):  # type: (...) -> None
        """
        Define the run of the discipline

        To be overloaded by subclasses.
        """
        raise NotImplementedError()
    
    def get_sosdisc_inputs(self, keys=None, in_dict=False, full_name_keys=False):
        """
        Accessor for the inputs values as a list or dict.

        Arguments:
            keys (List): the input short or full names list (depending on value of full_name_keys)
            in_dict (bool): if output format is dict
            full_name_keys (bool): if keys in args AND returned dictionary are full names or short names. Note that True
                                   allows several variables to have same short name, whereas False gives spurious behaviour
                                   for doubled short names in discipline, as a short to full name conversion is needed.
        Returns:
            The inputs values list or dict
        """
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = self.input_data_names # discipline and subprocess
            else:
                keys = list(self.attributes['input_full_name_map'].keys()) # discipline only
            in_dict = True
        inputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_IN, full_name_keys = full_name_keys)
        if in_dict:
            # return inputs in an dictionary
            return inputs
        else:
            # return inputs in an ordered tuple (default)
            if len(inputs) > 1:
                return list(inputs.values())
            else:
                return list(inputs.values())[0]

    def get_sosdisc_outputs(self, keys=None, in_dict=False, full_name_keys = False):
        """
        Accessor for the outputs values as a list or dict.

        Arguments:
            keys (List): the output short or full names list (depending on value of full_name_keys)
            in_dict (bool): if output format is dict
            full_name_keys (bool): if keys in args AND returned dictionary are full names or short names. Note that True
                                   allows several variables to have same short name, whereas False gives spurious behaviour
                                   for doubled short names in discipline, as a short to full name conversion is needed.
        Returns:
            The outputs values list or dict
        """
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = self.output_data_names # discipline and subprocess
            else:
                keys = list(self.attributes['output_full_name_map'].keys()) # discipline only
            in_dict = True
        outputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_OUT, full_name_keys=full_name_keys)
        if in_dict:
            # return outputs in an dictionary
            return outputs
        else:
            # return outputs in an ordered tuple (default)
            if len(outputs) > 1:
                return list(outputs.values())
            else:
                return list(outputs.values())[0]

    def _get_sosdisc_io(self, keys, io_type, full_name_keys):
        """
        Generic method to retrieve sos inputs and outputs

        Arguments:
            keys (List[String]): the data names list in short or full names (depending on value of full_name_keys)
            io_type (string): 'in' or 'out' [NOT USED]
            full_name_keys: if keys in args and returned dict are full names
        Returns:
            dict of keys values
        Raises:
            ValueError if i_o type is not IO_TYPE_IN or IO_TYPE_OUT
            KeyError if asked for an output key when self.local_data is not initialized
        """

        # convert local key names to namespaced ones
        if isinstance(keys, str):
            keys = [keys]


        if full_name_keys:
            query_keys = keys
        else:
            if io_type == self.IO_TYPE_IN:
                query_keys = [self.attributes['input_full_name_map'][key] for key in keys]
            elif io_type == self.IO_TYPE_OUT:
                query_keys = [self.attributes['output_full_name_map'][key] for key in keys]
            else:
                raise ValueError("Unknown io_type :" +
                                 str(io_type))

        values_dict = dict(zip(keys, map(self.local_data.get, query_keys)))
        return values_dict
    
    def _run(self):
        """
        Run user-defined model.

        Returns:
            local_data (Dict): outputs of the model run
        """
        self.run()
        return self.local_data
    
    def store_sos_outputs_values(self, dict_values, full_name_keys=False):
        """"
        Store run outputs in the local_data attribute.

        NB: permits coherence with EEV3 wrapper run definition.

        Arguments:
            dict_values (Dict): variables' values to store
        """
        if full_name_keys:
            self.local_data.update(dict_values) 
        else:
            outputs = dict(zip(map(self.attributes['output_full_name_map'].get, dict_values.keys()), dict_values.values()))
            self.local_data.update(outputs)