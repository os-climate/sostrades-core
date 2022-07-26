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

    def create_gemseo_discipline(self, sub_mdo_disciplines, proxy=None):  # type: (...) -> None
        """ MDAChain instanciation

        """
        self.mdo_discipline = MDAChain(
                                      disciplines=sub_mdo_disciplines,
                                      name=proxy.get_disc_full_name(),
                                      grammar_type=proxy.SOS_GRAMMAR_TYPE,
                                      ** proxy._get_numerical_inputs())
        
        self._init_grammar_with_keys(proxy)
        
    def execute(self, input_data):
        
        self.pre_run_mda(input_data)
        
        return self.mdo_discipline.execute(input_data)
    
    
    def pre_run_mda(self, input_data):
        '''
        Pre run needed if one of the strong coupling variables is None in a MDA 
        No need of prerun otherwise 
        '''
        strong_couplings_values = [input_data[key] for key in self.mdo_discipline.strong_couplings]
        if any(elem is None for elem in strong_couplings_values):
            self.logger.info(
                f'Execute a pre-run for the coupling ' + self.get_disc_full_name())
            self.recreate_order_for_first_execution()
            self.logger.info(
                f'End of pre-run execution for the coupling ' + self.get_disc_full_name())
            
    def recreate_order_for_first_execution(self):
        '''
        For each sub mda defined in the GEMS execution sequence, 
        we run disciplines by disciplines when they are ready to fill all values not initialized in the DM 
        until all disciplines have been run. 
        While loop cannot be an infinite loop because raise an exception
        if no disciplines are ready while some disciplines are missing in the list 
        '''
        for parallel_tasks in self.mdo_discipline.coupling_structure.sequence:
            # to parallelize, check if 1 < len(parallel_tasks)
            # for now, parallel tasks are run sequentially
            for coupled_proxy_disciplines in parallel_tasks:
                # several disciplines coupled
                coupled_mdo_disciplines = [discipline.mdo_discipline for discipline in coupled_proxy_disciplines]
                first_disc = coupled_proxy_disciplines[0]
                if len(coupled_proxy_disciplines) > 1 or (
                        len(coupled_proxy_disciplines) == 1
                        and self.coupling_structure.is_self_coupled(first_disc)
                        and not coupled_proxy_disciplines[0].is_sos_coupling
                ):
                    # several disciplines coupled

                    # get the disciplines from self.disciplines
                    # order the MDA disciplines the same way as the
                    # original disciplines
                    sub_mda_disciplines = []
                    for disc in self.mdo_discipline.disciplines:
                        if disc in coupled_mdo_disciplines:
                            sub_mda_disciplines.append(disc)
                    # submda disciplines are not ordered in a correct exec
                    # sequence...
                    # Need to execute ready disciplines one by one until all
                    # sub disciplines have been run
                    while sub_mda_disciplines != []:
                        ready_disciplines = self.get_first_discs_to_execute(
                            sub_mda_disciplines)

                        for discipline in ready_disciplines:
                            # Execute ready disciplines and update local_data
                            if discipline.proxy_discipline.is_sos_coupling:
                                # recursive call if subdisc is a SoSCoupling
                                # TODO: check if it will work for cases like
                                # Coupling1 > Driver > Coupling2
                                discipline.pre_run_mda()
                                self.mdo_discipline.local_data.update(discipline.local_data)
                            else:
                                temp_local_data = discipline.execute(
                                    self.mdo_discipline.local_data)
                                self.mdo_discipline.local_data.update(temp_local_data)

                        sub_mda_disciplines = [
                            disc for disc in sub_mda_disciplines if disc not in ready_disciplines]
                else:
                    discipline = coupled_mdo_disciplines[0]
                    if discipline.proxy_discipline.is_sos_coupling:
                        # recursive call if subdisc is a SoSCoupling
                        discipline.pre_run_mda()
                        self.mdo_discipline.local_data.update(discipline.local_data)
                    else:
                        temp_local_data = discipline.execute(self.mdo_discipline.local_data)
                        self.mdo_discipline.local_data.update(temp_local_data)

        self.mdo_discipline.default_inputs.update(self.mdo_discipline.local_data)
        
    def get_first_discs_to_execute(self, disciplines):

        ready_disciplines = []
        disc_vs_keys_none = {}
        for disc in disciplines:
#             # get inputs values of disc with full_name
#             inputs_values = disc.get_inputs_by_name(
#                 in_dict=True, full_name=True)
            # update inputs values with SoSCoupling local_data
            inputs_values = {}
            inputs_values.update(disc._filter_inputs(self.local_data()))
            keys_none = [key for key, value in inputs_values.items()
                         if value is None and not any([key.endswith(num_key) for num_key in self.NUM_DESC_IN])]
            if keys_none == []:
                ready_disciplines.append(disc)
            else:
                disc_vs_keys_none[disc.sos_name] = keys_none
        if ready_disciplines == []:
            message = '\n'.join(' : '.join([disc, str(keys_none)])
                                for disc, keys_none in disc_vs_keys_none.items())
            raise Exception(
                f'The MDA cannot be pre-runned, some input values are missing to run the MDA \n{message}')
        else:
            return ready_disciplines
                
        
        
        
        
        
        
        
        
        
        
        
        
        

