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

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

_NEW_ATTR_TO_SERIALIZE = ['reduced_dm', 'sos_wrapp']


class SoSMDODisciplineException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class SoSMDODiscipline(MDODiscipline):
    """**SoSMDODiscipline** is the class which overloads MDODiscipline
    The _run method is overloaded and new methods ( formerly from SoSDiscipline) are added

   """

    def __init__(self, full_name, grammar_type, cache_type, sos_wrapp, reduced_dm):
        '''
        Constructor
        '''
        self.sos_wrapp = sos_wrapp
        self.reduced_dm = reduced_dm
        self.output_full_name_map = None
        MDODiscipline.__init__(self, name=full_name,
                               grammar_type=grammar_type,
                               cache_type=cache_type)

    def _run(self):
        self.sos_wrapp.local_data_short_name = self.create_local_data_short_name()
        run_output = self.sos_wrapp._run()
        self.update_local_data(run_output)
        # post_execute ? status ?
    
    def create_local_data_short_name(self):
        
        local_data_short_name = {}
        for key in self.get_input_data_names():
            local_data_short_name[self.reduced_dm[key][SoSWrapp.VAR_NAME]] = self.local_data.get(key)
            
        if self.output_full_name_map is None:
            self.output_full_name_map = {}
            for key in self.get_output_data_names():
                self.output_full_name_map[self.reduced_dm[key][SoSWrapp.VAR_NAME]] = key
            
        return local_data_short_name
    
    def update_local_data(self, run_output):
        
        for key, value in run_output.items():
            self.local_data[self.output_full_name_map[key]] = value
            

    def get_input_data_names(self, filtered_inputs=False):  # type: (...) -> List[str]
        """Return the names of the input variables.

        Returns:
            The names of the input variables.
        """
        if not filtered_inputs:
            return self.input_grammar.get_data_names()
        else:
            return self.filter_variables_to_convert(self.reduced_dm, self.input_grammar.get_data_names(),
                                                    logger=self.LOGGER)

    def get_output_data_names(self, filtered_outputs=False):  # type: (...) -> List[str]
        """Return the names of the output variables.

        Returns:
            The names of the output variables.
        """
        if not filtered_outputs:
            return self.output_grammar.get_data_names()
        else:
            return filter_variables_to_convert(self.reduced_dm, self.output_grammar.get_data_names(),
                                               logger=self.LOGGER)

    def get_attributes_to_serialize(self):  # pylint: disable=R0201
        """Define the names of the attributes to be serialized.

        Shall be overloaded by disciplines

        Returns:
            The names of the attributes to be serialized.
        """
        # pylint warning ==> method could be a function but when overridden,
        # it is a function==> self is required

        return super().get_attributes_to_serialize() + _NEW_ATTR_TO_SERIALIZE
