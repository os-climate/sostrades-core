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
from gemseo.mda.mda_chain import MDAChain
from sostrades_core.execution_engine.MDODisciplineWrapp import MDODisciplineWrapp

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

class MDAChainWrappWrappException(Exception):
    pass


# to avoid circular redundancy with nsmanager
NS_SEP = '.'


class MDAChainWrapp(MDODisciplineWrapp):
    '''**MDAChainWrappWrapp** is the interface to create MDODiscipline from sostrades or gemseo objects


    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

    def create_gemseo_discipline(self):  # type: (...) -> None
        """ MDAChain instanciation

        """
        self.mdo_discipline = MDAChain(self.name, self.wrapper)

