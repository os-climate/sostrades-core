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

from gemseo.core.discipline import MDODiscipline
from sostrades_core.tools.filter.filter import filter_variables_to_convert
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.data_connector.data_connector_factory import ConnectorFactory
import logging
# debug mode
from copy import deepcopy
from pandas import DataFrame
from numpy import ndarray, floating

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

class SoSMDODisciplineException(Exception):
    pass

# get module logger not sos logger
LOGGER = logging.getLogger(__name__)

class SoSMDODiscipline(MDODiscipline):
    """**SoSMDODiscipline** is the class that overloads MDODiscipline when using SoSTrades wrapping mode. It handles the
    execution of the user-provided wrapper of the discipline (on the GEMSEO side)

    It is instantiated by the MDODisciplineWrapp during the prepare_execution step, and it is in one-to-one aggregation
    with the user-defined wrapper (Specialization of SoSWrapp). The _run() method is overloaded by the user-defined wrapper.

    NB: overloading of MDODiscipline has been limited in EEV4 namely wrt EEV3's SoSDiscipline implementation

    Attributes:
        sos_wrapp (SoSWrapp): the user-defined wrapper of the discipline
        reduced_dm (Dict[Dict]): reduced data manager for i/o handling (NB: there is only one reduced_dm per process)
        output_full_name_map (Dict[string]): map from short names to full names of model output variables
   """

    _NEW_ATTR_TO_SERIALIZE = ['reduced_dm', 'sos_wrapp']
    DEBUG_MODE = 'debug_mode'

    def __init__(self, full_name, grammar_type, cache_type, cache_file_path, sos_wrapp, reduced_dm):
        '''
        Constructor

        Args:
            full_name (string): full name of the discipline
            grammar_type (string): type of GEMSEO grammar
            cache_type (string): type of cache to be passed to the MDODiscipline
            cache_file_path (string): file path for the cache pickle
            sos_wrapp (SoSWrapp): user-defined wrapper of the discipline
            reduced_dm (Dict[Dict]): reduced version of datamanager for i/o handling
        '''
        self.sos_wrapp = sos_wrapp
        self.reduced_dm = reduced_dm
        self.output_full_name_map = None
        MDODiscipline.__init__(self, name=full_name,
                               grammar_type=grammar_type,
                               cache_type=cache_type,
                               cache_file_path=cache_file_path)
        self.is_sos_coupling = False

    def _run(self):
        """
        Call user-defined wrapper run.
        """

        # local data with short names for the wrapper
        self.sos_wrapp.local_data_short_name = self.create_local_data_short_name()

        # debug mode: input change
        if self.sos_wrapp.get_sosdisc_inputs(self.DEBUG_MODE) in ['input_change', 'all']:
            disc_inputs_before_execution = {key: {'value': value} for key, value in deepcopy(
                self.local_data).items() if key in self.input_grammar.data_names}

        # SoSWrapp run
        run_output = self.sos_wrapp._run()
        self.store_local_data(map_short_to_full_names = True, **run_output)

        # get output from data connector
        self.fill_output_value_connector()

        # debug modes
        if self.sos_wrapp.get_sosdisc_inputs(self.DEBUG_MODE) in ['nan', 'all']:
            self.__check_nan_in_data(self.local_data)

        if self.sos_wrapp.get_sosdisc_inputs(self.DEBUG_MODE) in ['input_change', 'all']:
            disc_inputs_after_execution = {key: {'value': value} for key, value in deepcopy(
                self.local_data).items() if key in self.input_grammar.data_names}
            output_error = self.check_discipline_data_integrity(disc_inputs_before_execution,
                                                                disc_inputs_after_execution,
                                                                'Discipline inputs integrity through run',
                                                                is_output_error=True)
            if output_error != '':
                raise ValueError(output_error)

        if self.sos_wrapp.get_sosdisc_inputs(self.DEBUG_MODE) in ['min_max_couplings', 'all']:
            self.display_min_max_couplings()

    def create_local_data_short_name(self):
        """
        Create a local_data_short_name (every run) and initialise the attribute output_full_name_map (the first run)

        Return:
            local_data_short_name (Dict[Dict])
        """

        if type(self.sos_wrapp).__name__ == 'DisciplineGatherWrapper':
            local_data_short_name = self.local_data
        else :
            full_name_input_keys = self.get_input_data_names()
            local_data_input_values = self.get_local_data_by_name(full_name_input_keys)
            local_data_short_name = dict(zip(map(self.io_full_name_to_short, full_name_input_keys), local_data_input_values))


        if self.output_full_name_map is None:
            full_name_output_keys = self.get_output_data_names()
            self.output_full_name_map = dict(zip(map(self.io_full_name_to_short, full_name_output_keys), full_name_output_keys))
        return local_data_short_name

    def io_full_name_to_short(self, full_name_key):
        return self.reduced_dm[full_name_key][SoSWrapp.VAR_NAME]

    def io_short_name_to_full(self, short_name_key):
        return self.output_full_name_map[short_name_key]

    def store_local_data(self, map_short_to_full_names = False, **kwargs):
        """
        Update local_data[full_name] using the run_output[short_name].

        Arguments:
            map_short_to_full_names (bool) : whether to map short to full names in kwargs.keys(), only available for
                                             output variables.
            **kwargs : unpacked dict to update the local_data with

        Raises:
            KeyError if map_short_to_full_names is True and the variables to update include anything other than outputs.
        """
        if map_short_to_full_names:
            short_name_keys = kwargs.keys()
            full_name_keys = map(self.io_short_name_to_full, short_name_keys)
            to_store = dict(zip(full_name_keys, kwargs.values()))
            super().store_local_data(**to_store)
        else:
            super().store_local_data(**kwargs)

    def fill_output_value_connector(self):
        """
        Get value of output variables with data connectors and update local_data.
        """
        updated_values = {}
        for key in self.get_output_data_names():
            # if data connector is needed, use it
            if self.reduced_dm[key][SoSWrapp.CONNECTOR_DATA] is not None:
                updated_values[key] = ConnectorFactory.use_data_connector(
                    self.reduced_dm[key][SoSWrapp.CONNECTOR_DATA],
                    LOGGER)

        self.store_local_data(map_short_to_full_names=False, **updated_values)

    def get_input_data_names(self, filtered_inputs=False):  # type: (...) -> List[str]
        """
        Retrieve the names of the input variables from the input_grammar.

        Arguments:
            filtered_inputs (bool): flag whether to filter variables

        Return:
            List[string] The names of the input variables.
        """
        if not filtered_inputs:
            return self.input_grammar.get_data_names()
        else:
            return filter_variables_to_convert(self.reduced_dm, self.input_grammar.get_data_names(),
                                                    logger=LOGGER)

    def get_output_data_names(self, filtered_outputs=False):  # type: (...) -> List[str]
        """
        Retrieve the names of the output variables from the output_grammar

        Arguments:
            filtered_outputs (bool): flag whether to filter variables

        Return:
            List[string] The names of the output variables.
        """
        if not filtered_outputs:
            return self.output_grammar.get_data_names()
        else:
            return filter_variables_to_convert(self.reduced_dm, self.output_grammar.get_data_names(),
                                               logger=self.LOGGER)

    def get_attributes_to_serialize(self):  # pylint: disable=R0201
        """
        Define the names of the attributes to be serialized.

        overload of gemseo's method.

        Return:
            List[string] the names of the attributes to be serialized.
        """
        # pylint warning ==> method could be a function but when overridden,
        # it is a function==> self is required

        return super().get_attributes_to_serialize() + [self._NEW_ATTR_TO_SERIALIZE]

    # ----------------------------------------------------
    # ----------------------------------------------------
    #  METHODS TO DEBUG DISCIPLINE
    # ----------------------------------------------------
    # ----------------------------------------------------

    def __check_nan_in_data(self, data):
        """ Using entry data, check if nan value exist in data's

        :params: data
        :type: composite data

        """
        has_nan = self.__check_nan_in_data_rec(data, "")
        if has_nan:
            raise ValueError(f'NaN values found in {self.name}')

    def __check_nan_in_data_rec(self, data, parent_key):
        """
        Using entry data, check if nan value exist in data's as recursive method

        :params: data
        :type: composite data

        :params: parent_key, on composite type (dict), reference parent key
        :type: str

        """
        has_nan = False
        import pandas as pd
        for data_key, data_value in data.items():

            nan_found = False
            if isinstance(data_value, DataFrame):
                if data_value.isnull().any():
                    nan_found = True
            elif isinstance(data_value, ndarray):
                # None value in list throw an exception when used with isnan
                if sum(1 for _ in filter(None.__ne__, data_value)) != len(data_value):
                    nan_found = True
                elif pd.isnull(list(data_value)).any():
                    nan_found = True
            elif isinstance(data_value, list):
                # None value in list throw an exception when used with isnan
                if sum(1 for _ in filter(None.__ne__, data_value)) != len(data_value):
                    nan_found = True
                elif pd.isnull(data_value).any():
                    nan_found = True
            elif isinstance(data_value, dict):
                self.__check_nan_in_data_rec(
                    data_value, f'{parent_key}/{data_key}')
            elif isinstance(data_value, floating):
                if pd.isnull(data_value).any():
                    nan_found = True

            if nan_found:
                full_key = data_key
                if len(parent_key) > 0:
                    full_key = f'{parent_key}/{data_key}'
                LOGGER.debug(f'NaN values found in {full_key}')
                LOGGER.debug(data_value)
                has_nan = True
        return has_nan


    def check_discipline_data_integrity(self, left_dict, right_dict, test_subject, is_output_error=False):
        """
        Compare data is equal in left_dict and right_dict and print a warning otherwise.

        Arguments:
            left_dict (dict): data dict to compare
            right_dict (dict): data dict to compare
            test_subject (string): to identify the executor of the check
            is_output_error (bool): whether to return a dict of errors

        Return:
            output_error (dict): dict with mismatches spotted in comparison
        """
        from gemseo.utils.compare_data_manager_tooling import compare_dict

        dict_error = {}
        compare_dict(left_dict, right_dict, '', dict_error)
        output_error = ''
        if dict_error != {}:
            for error in dict_error:
                output_error = '\n'
                output_error += f'Error while test {test_subject} on sos discipline {self.name} :\n'
                output_error += f'Mismatch in {error}: {dict_error.get(error)}'
                output_error += '\n---------------------------------------------------------'
                print(output_error)

        if is_output_error:
            return output_error

    def display_min_max_couplings(self):
        '''
        Method to display the minimum and maximum values among a discipline's couplings
        '''
        min_coupling_dict, max_coupling_dict = {}, {}
        for key, value in self.local_data.items():
            is_coupling = self.reduced_dm[key]['coupling']
            if is_coupling:
                min_coupling_dict[key] = min(abs(value))
                max_coupling_dict[key] = max(abs(value))
        min_coupling = min(min_coupling_dict, key=min_coupling_dict.get)
        max_coupling = max(max_coupling_dict, key=max_coupling_dict.get)
        LOGGER.info(
            "in discipline <%s> : <%s> has the minimum coupling value <%s>" % (
                self.name, min_coupling, min_coupling_dict[min_coupling]))
        LOGGER.info(
            "in discipline <%s> : <%s> has the maximum coupling value <%s>" % (
                self.name, max_coupling, max_coupling_dict[max_coupling]))