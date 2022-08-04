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
from sostrades_core.execution_engine.SoSMDODiscipline import SoSMDODiscipline
from gemseo.mda.mda_chain import MDAChain

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class SoSWrappException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class MDODisciplineWrapp(object):
    '''**MDODisciplineWrapp** is the interface to create MDODiscipline from sostrades or gemseo objects


    '''

    def __init__(self, name, wrapper=None, wrapping_mode='SoSTrades'):
        '''
        Constructor
        '''
        self.name = name
        self.wrapping_mode = wrapping_mode
        self.mdo_discipline = None
        if wrapper is not None:
            self.wrapper = wrapper(name)

    def get_input_data_names(self, filtered_inputs=False):  # type: (...) -> List[str]
        """Return the names of the input variables.

        Returns:
            The names of the input variables.
        """
        return self.mdo_discipline.get_input_data_names(filtered_inputs)

    def get_output_data_names(self, filtered_outputs=False):  # type: (...) -> List[str]
        """Return the names of the output variables.

        Returns:
            The names of the input variables.
        """
        return self.mdo_discipline.get_output_data_names(filtered_outputs)

    def setup_sos_disciplines(self, proxy):  # type: (...) -> None
        """Define setup

        """
        if self.wrapper is not None:
            self.wrapper.setup_sos_disciplines(proxy)

    def create_gemseo_discipline(self, proxy=None, input_data=None, reduced_dm=None, cache_type=None, cache_file_path=None):  # type: (...) -> None
        """ MDODiscipline instanciation

        """
        if self.wrapping_mode == 'SoSTrades':
            self.mdo_discipline = SoSMDODiscipline(full_name=proxy.get_disc_full_name(),
                                                   grammar_type=proxy.SOS_GRAMMAR_TYPE,
                                                   cache_type=cache_type,
                                                   cache_file_path=cache_file_path,
                                                   sos_wrapp=self.wrapper,
                                                   reduced_dm=reduced_dm)
            self._init_grammar_with_keys(proxy)
            self._update_default_values(input_data)

        elif self.wrapping_mode == 'GEMSEO':
            pass

        proxy.status = self.mdo_discipline.status

    def _init_grammar_with_keys(self, proxy):
        ''' initialize GEMS grammar with names and type None
        '''
        input_names = proxy.get_input_data_names()
        grammar = self.mdo_discipline.input_grammar
        grammar.clear()
        grammar.initialize_from_base_dict({input: None for input in input_names})

        output_names = proxy.get_output_data_names()
        grammar = self.mdo_discipline.output_grammar
        grammar.clear()
        grammar.initialize_from_base_dict({output: None for output in output_names})
        
    def _update_default_values(self, input_data):
        ''' store input_data in default_inputs of mdo_discipline
        '''
        if input_data is not None:
            for key in self.mdo_discipline.input_grammar.get_data_names():
                self.mdo_discipline._default_inputs[key] = input_data.get(key)
        
    def create_mda_chain(self, sub_mdo_disciplines, proxy=None, input_data=None):  # type: (...) -> None
        """ MDAChain instanciation

        """
        self.mdo_discipline = MDAChain(
                                      disciplines=sub_mdo_disciplines,
                                      name=proxy.get_disc_full_name(),
                                      grammar_type=proxy.SOS_GRAMMAR_TYPE,
                                      ** proxy._get_numerical_inputs())
        
        self._init_grammar_with_keys(proxy)
        self._update_default_values(input_data)
        proxy.status = self.mdo_discipline.status

    def create_wrapp(self):  # type: (...) -> None
        """ SoSWrapp instanciation

        """
        if self.wrapping_mode == 'SoSTrades':
            # self.wrapper = SoSMDODiscipline(self.sos_name,self.wrapper)
            pass
        else:
            # self.mdo_discipline = create_discipline(self.sos_name)
            pass

    def execute(self, input_data):
        """ Discipline Execution
	    """

        return self.mdo_discipline.execute(input_data)
