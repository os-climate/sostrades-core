'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

import multiprocessing as mp

from gemseo.core.chain import MDOParallelChain
from gemseo.core.discipline import MDODiscipline

from sostrades_core.execution_engine.parallel_execution.sos_parallel_execution import (
    SoSDiscParallelExecution,
    SoSDiscParallelLinearization,
)


class SoSParallelChain(MDOParallelChain):
    """
    Class that inherits from GEMSEO's MDO parallel chain
    with specific DiscParallelExecution and DiscParallelLinearization SoSTrades classes
    """
    N_CPUS = mp.cpu_count()

    def __init__(self, disciplines, name=None,
                 grammar_type=MDODiscipline.GrammarType.JSON,
                 use_threading=True, n_processes=N_CPUS):
        '''
        Constructor
        '''
        super(SoSParallelChain, self).__init__(disciplines, name=name,
                                               grammar_type=grammar_type,
                                               use_threading=use_threading)
        # replace DiscParallelExecution and DiscParallelLinearization GEMSEO's
        # attributes
        dpe = SoSDiscParallelExecution(self._isciplines,
                                       n_processes=n_processes,
                                       use_threading=use_threading)
        self.parallel_execution = dpe
        dpl = SoSDiscParallelLinearization(self._disciplines,
                                           n_processes=n_processes,
                                           use_threading=use_threading)
        self.parallel_lin = dpl
