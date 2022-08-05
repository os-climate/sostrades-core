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
from sostrades_core.execution_engine.SoSWrapp import SoSWrapp
from sostrades_core.execution_engine.data_connector.data_connector_factory import ConnectorFactory
import logging

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

_NEW_ATTR_TO_SERIALIZE = ['reduced_dm', 'sos_wrapp']


class SoSMDODisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'
LOGGER = logging.getLogger(__name__)


class SoSMDODiscipline(MDODiscipline):
    """**SoSMDODiscipline** is the class that overloads MDODiscipline when using SoSTrades wrapping mode. It handles the
    execution of the user-provided wrapper of the discipline (on the GEMSEO side)

    It is instantiated by the MDODisciplineWrapp during the prepare_execution phase, and it is in one-to-one aggregation
    with the user-defined wrapper (inheriting from SoSWrapp). The _run() method is overloaded by the user-defined wrapper.

    NB: overloading of MDODiscipline has been limited in EEV4 namely wrt EEV3's SoSDiscipline implementation

    Attributes:
        sos_wrapp (SoSWrapp): the user-defined wrapper of the discipline
        reduced_dm (Dict[Dict]): reduced data manager for i/o handling (NB: there is only one reduced_dm per process)
        output_full_name_map (Dict[string]): map from short names to full names of model output variables ???
   """

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

    def _run(self):
        """
        Call user-defined wrapper run.
        """
        # SoSWrapp run
        self.sos_wrapp.local_data_short_name = self.create_local_data_short_name()
        run_output = self.sos_wrapp._run()
        self.update_local_data(run_output)
        
        # get output from data connector
        self.fill_output_value_connector()
        
    
    def create_local_data_short_name(self):
        """
        Create a local_data_short_name (every run) and initialise the attribute output_full_name_map (the first run)

        Return:
            local_data_short_name (Dict[Dict])
        """

        local_data_short_name = {}
        for key in self.get_input_data_names():
            #FIXME: why there is a generic try-pass in every model run ?
            try:
                local_data_short_name[self.reduced_dm[key][SoSWrapp.VAR_NAME]] = self.local_data.get(key)
            except:
                pass

        if self.output_full_name_map is None:
            self.output_full_name_map = {}
            for key in self.get_output_data_names():
                self.output_full_name_map[self.reduced_dm[key][SoSWrapp.VAR_NAME]] = key
            
        return local_data_short_name
    
    def update_local_data(self, run_output):
        """
        Update local_data[full_name] using the run_output[short_name].

        Arguments:
            run_output (Dict): the values to update the local_data with
        """
        for key, value in run_output.items():
            self.local_data[self.output_full_name_map[key]] = value


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

        self.local_data.update(updated_values)
            

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
            return self.filter_variables_to_convert(self.reduced_dm, self.input_grammar.get_data_names(),
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

        Shall be overloaded by disciplines.

        Return:
            List[string] the names of the attributes to be serialized.
        """
        # pylint warning ==> method could be a function but when overridden,
        # it is a function==> self is required

        return super().get_attributes_to_serialize() + _NEW_ATTR_TO_SERIALIZE
