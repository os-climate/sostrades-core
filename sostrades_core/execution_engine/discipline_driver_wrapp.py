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

from __future__ import annotations

from gemseo.core.discipline import Discipline

from sostrades_core.execution_engine.discipline_wrapp import DisciplineWrapp
from sostrades_core.execution_engine.sos_discipline_driver import (
    SoSDisciplineDriver,
)


class DisciplineDriverWrappException(Exception):
    pass


class DisciplineDriverWrapp(DisciplineWrapp):
    """**DisciplineWrapp** is the interface to create Discipline from SoSTrades wrappers, GEMSEO objects, etc.

    An instance of DisciplineWrapp is in one-to-one aggregation with an instance inheriting from Discipline and
    might or might not have a SoSWrapp to supply the user-defined model run. All GEMSEO objects are instantiated during
    the prepare_execution phase.

    Attributes:
        name (string): name of the discipline/node
        wrapping_mode (string): mode of supply of model run by user ('SoSTrades'/'GEMSEO')
        discipline (Discipline): aggregated GEMSEO object used for execution eventually with model run
        wrapper (SoSWrapp/???): wrapper instance used to supply the model run to the Discipline (or None)
    """

    def create_gemseo_discipline(
        self, proxy=None, reduced_dm=None, cache_type=Discipline.CacheType.NONE, cache_file_path=None
    ):  # type: (...) -> None
        """
        SoSDiscipline instanciation.

        Arguments:
            proxy (ProxyDiscipline): proxy discipline grammar initialisation
            input_data (dict): input values to update default values of the Discipline with
            reduced_dm (Dict[Dict]): reduced data manager without values for i/o configuration
            cache_type (string): type of cache to be passed to the Discipline
            cache_file_path (string): file path of the pickle file to dump/load the cache [???]
        """
        # get all executable sub disciplines
        sub_disciplines = self.get_sub_disciplines(proxy)

        # create the SoSDisciplineDriver
        if self.wrapping_mode == 'SoSTrades':
            self.discipline = SoSDisciplineDriver(
                full_name=proxy.get_disc_full_name(),
                grammar_type=proxy.SOS_GRAMMAR_TYPE,
                cache_type=cache_type,
                sos_wrapp=self.wrapper,
                reduced_dm=reduced_dm,
                disciplines=sub_disciplines,
                logger=self.logger.getChild("SoSDisciplineDriver"),
            )
            self._init_grammar_with_keys(proxy)
            # self._update_all_default_values(input_data) #TODO: numerical inputs etc?
            self._set_wrapper_attributes(proxy, self.wrapper)

        elif self.wrapping_mode == 'GEMSEO':
            raise NotImplementedError("GEMSEO native wrapping mode not yet available.")

    def get_sub_disciplines(self, proxy):
        return [
            pdisc.discipline_wrapp.discipline for pdisc in proxy.proxy_disciplines if pdisc.discipline_wrapp is not None
        ]

    def reset_subdisciplines(self, proxy):
        sub_disciplines = self.get_sub_disciplines(proxy)

        if self.discipline is not None:
            self.discipline._disciplines = sub_disciplines
