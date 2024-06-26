'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/06/24 Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.mdo_discipline_wrapp import MDODisciplineWrapp
from sostrades_core.execution_engine.sos_mdo_discipline_driver import (
    SoSMDODisciplineDriver,
)


class MDODisciplineDriverWrappException(Exception):
    pass


class MDODisciplineDriverWrapp(MDODisciplineWrapp):
    '''**MDODisciplineWrapp** is the interface to create MDODiscipline from SoSTrades wrappers, GEMSEO objects, etc.

    An instance of MDODisciplineWrapp is in one-to-one aggregation with an instance inheriting from MDODiscipline and
    might or might not have a SoSWrapp to supply the user-defined model run. All GEMSEO objects are instantiated during
    the prepare_execution phase.

    Attributes:
        name (string): name of the discipline/node
        wrapping_mode (string): mode of supply of model run by user ('SoSTrades'/'GEMSEO')
        mdo_discipline (MDODiscipline): aggregated GEMSEO object used for execution eventually with model run
        wrapper (SoSWrapp/???): wrapper instance used to supply the model run to the MDODiscipline (or None)
    '''

    def create_gemseo_discipline(self, proxy=None, reduced_dm=None, cache_type=None,
                                 cache_file_path=None):  # type: (...) -> None
        """
        SoSMDODiscipline instanciation.

        Arguments:
            proxy (ProxyDiscipline): proxy discipline grammar initialisation
            input_data (dict): input values to update default values of the MDODiscipline with
            reduced_dm (Dict[Dict]): reduced data manager without values for i/o configuration
            cache_type (string): type of cache to be passed to the MDODiscipline
            cache_file_path (string): file path of the pickle file to dump/load the cache [???]
        """
        # get all executable sub disciplines
        sub_mdo_disciplines = self.get_sub_mdo_disciplines(proxy)

        # create the SoSMDODisciplineDriver
        if self.wrapping_mode == 'SoSTrades':
            self.mdo_discipline = SoSMDODisciplineDriver(full_name=proxy.get_disc_full_name(),
                                                         grammar_type=proxy.SOS_GRAMMAR_TYPE,
                                                         cache_type=cache_type,
                                                         cache_file_path=cache_file_path,
                                                         sos_wrapp=self.wrapper,
                                                         reduced_dm=reduced_dm,
                                                         disciplines=sub_mdo_disciplines,
                                                         logger=self.logger.getChild("SoSMDODisciplineDriver"),
                                                         )
            self._init_grammar_with_keys(proxy)
            # self._update_all_default_values(input_data) #TODO: numerical inputs etc?
            self._set_wrapper_attributes(proxy, self.wrapper)
            # self._set_discipline_attributes(proxy, self.mdo_discipline)

        elif self.wrapping_mode == 'GEMSEO':
            pass

    def get_sub_mdo_disciplines(self, proxy):

        sub_mdo_disciplines = [pdisc.mdo_discipline_wrapp.mdo_discipline
                               for pdisc in proxy.proxy_disciplines
                               if pdisc.mdo_discipline_wrapp is not None]
        return sub_mdo_disciplines

    def reset_subdisciplines(self, proxy):

        sub_mdo_disciplines = self.get_sub_mdo_disciplines(proxy)

        if self.mdo_discipline is not None:
            self.mdo_discipline.disciplines = sub_mdo_disciplines
