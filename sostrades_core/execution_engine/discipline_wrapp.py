'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/06-2024/06/24 Copyright 2023 Capgemini

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

from typing import TYPE_CHECKING

from gemseo.core.discipline import Discipline

from sostrades_core.execution_engine.sos_discipline import SoSDiscipline
from sostrades_core.execution_engine.sos_mda_chain import SoSMDAChain
from sostrades_core.execution_engine.sos_mdo_scenario_adapter import SoSMDOScenarioAdapter

if TYPE_CHECKING:
    import logging

    from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling


class DisciplineWrappException(Exception):
    pass


class DisciplineWrapp:
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

    def __init__(self, name: str, logger: logging.Logger, wrapper=None, wrapping_mode: str = 'SoSTrades'):
        """
        Constructor.

        Arguments:
            name (string): name of the discipline/node
            wrapper (Class): class constructor of the user-defined wrapper (or None)
            wrapping_mode (string): mode of supply of model run by user ('SoSTrades'/'GEMSEO')
        """
        self.logger = logger
        self.name = name
        self.wrapping_mode = wrapping_mode
        self.discipline: SoSDiscipline | SoSMDOScenarioAdapter | SoSMDAChain = None
        if wrapper is not None:
            self.wrapper = wrapper(name, self.logger.getChild(wrapper.__name__))
        else:
            self.wrapper = None

    def get_input_data_names(self, filtered_inputs=False):
        """
        Return the names of the input variables.

        Arguments:
            filtered_inputs (bool): flag whether to filter the input names.

        Returns:
            The names of the input variables.
        """
        return self.discipline.get_input_data_names(filtered_inputs)

    def get_output_data_names(self, filtered_outputs=False):
        """Return the names of the output variables.

        Arguments:
            filtered_outputs (bool): flag whether to filter the output names.

        Returns:
            The names of the input variables.
        """
        return self.discipline.get_output_data_names(filtered_outputs)

    def setup_sos_disciplines(self):  # type: (...) -> None
        """
        Dynamic setup delegated to the wrapper using the proxy for i/o configuration.

        Arguments:
            proxy (ProxyDiscipline): corresponding proxy discipline
        """
        if self.wrapper is not None:
            self.wrapper.setup_sos_disciplines()

    def check_data_integrity(self):  # type: (...) -> None
        """
        check_data_integrity delegated to the wrapper using the proxy for i/o configuration.

        Arguments:
            proxy (ProxyDiscipline): corresponding proxy discipline
        """
        if self.wrapper is not None:
            self.wrapper.check_data_integrity()

    def create_gemseo_discipline(self, proxy=None, reduced_dm=None, cache_type=Discipline.CacheType.NONE, cache_file_path=None):
        """
        SoSDiscipline instanciation.

        Arguments:
            proxy (ProxyDiscipline): proxy discipline grammar initialisation
            input_data (dict): input values to update default values of the Discipline with
            reduced_dm (Dict[Dict]): reduced data manager without values for i/o configuration
            cache_type (string): type of cache to be passed to the Discipline
            cache_file_path (string): file path of the pickle file to dump/load the cache [???]
        """
        if self.wrapping_mode == 'SoSTrades':
            self.discipline = SoSDiscipline(
                full_name=proxy.get_disc_full_name(),
                grammar_type=proxy.SOS_GRAMMAR_TYPE,
                cache_type=cache_type,
                sos_wrapp=self.wrapper,
                reduced_dm=reduced_dm,
                logger=self.logger.getChild("SoSDiscipline"),
                debug_mode=proxy.get_sosdisc_inputs(SoSDiscipline.DEBUG_MODE),
            )
            self._init_grammar_with_keys(proxy)
            self._set_wrapper_attributes(proxy, self.wrapper)
            self._update_all_default_values(proxy)
            self.discipline.linearization_mode = proxy.get_sosdisc_inputs(SoSDiscipline.LINEARIZATION_MODE)

        elif self.wrapping_mode == 'GEMSEO':
            raise NotImplementedError("GEMSEO native wrapping mode not yet available.")

    def _set_wrapper_attributes(self, proxy, wrapper):
        proxy.set_wrapper_attributes(wrapper)

    def _init_grammar_with_keys(self, proxy):
        """
        initialize GEMS grammar with names and type None

        Arguments:
            proxy (ProxyDiscipline): the proxy discipline to get input and output full names from
        """
        input_names_and_defaults = proxy.get_input_data_names_and_defaults(numerical_inputs=False)
        grammar = self.discipline.input_grammar
        grammar.clear()
        grammar.update_from_names(list(input_names_and_defaults.keys()))
        grammar.update_defaults({key: value for key, value in input_names_and_defaults.items() if value is not None})

        output_names = proxy.get_output_data_names(numerical_inputs=False)
        grammar = self.discipline.output_grammar
        grammar.clear()
        grammar.update_from_names(output_names)

    def update_default_from_dict(self, input_dict, check_input=True):
        """
        Store values from input_dict in default values of discipline (when keys are present in input grammar data
        names or input is not checked)

        Arguments:
            input_dict (dict): values to store
            check_input (bool): flag to specify if inputs are checked or not to exist in input grammar
        """
        if input_dict is not None:
            to_update = [
                (key, value)
                for (key, value) in input_dict.items()
                if not check_input or key in self.discipline.input_grammar.names
            ]
            self.discipline.default_input_data.update(to_update)

    def create_mda_chain(self, sub_disciplines, proxy: ProxyCoupling | None = None, input_data=None, reduced_dm=None):  # type: (...) -> None
        """
        MDAChain instantiation when owned by a ProxyCoupling.

        Arguments:
            sub_disciplines (List[Discipline]): list of sub-Disciplines of the MDAChain
            proxy: proxy coupling for grammar initialisation and numericla inputs.
            input_data (dict): input data to update default values of the MDAChain with
        """
        if reduced_dm is None:
            reduced_dm = {}
        if self.wrapping_mode == 'SoSTrades':
            discipline = SoSMDAChain(
                disciplines=sub_disciplines,
                reduced_dm=reduced_dm,
                name=proxy.get_disc_full_name(),
                grammar_type=proxy.SOS_GRAMMAR_TYPE,
                **proxy._get_numerical_inputs(),
                # authorize_self_coupled_disciplines=proxy.get_sosdisc_inputs(proxy.AUTHORIZE_SELF_COUPLED_DISCIPLINES),
                logger=self.logger.getChild("SoSMDAChain"),
            )

            self.discipline = discipline

            self.__update_gemseo_grammar(proxy, discipline)

            # set linear solver options (todo after call to _get_numerical_inputs() )
            # TODO: check with IRT how to handle it
            discipline.linear_solver_MDA = proxy.linear_solver_MDA
            discipline.linear_solver_settings_MDA = proxy.linear_solver_settings_MDA
            discipline.linear_solver_tolerance_MDA = proxy.linear_solver_tolerance_MDA
            discipline.linear_solver_MDO = proxy.linear_solver_MDO
            discipline.linear_solver_settings_MDO = proxy.linear_solver_settings_MDO
            discipline.linear_solver_tolerance_MDO = proxy.linear_solver_tolerance_MDO
            discipline.linearization_mode = proxy.linearization_mode

            # # set other additional options (SoSTrades)
            # discipline.authorize_self_coupled_disciplines = proxy.get_sosdisc_inputs(
            #     'authorize_self_coupled_disciplines')

            #             self._init_grammar_with_keys(proxy)
            # self._update_all_default_values(input_data) # TODO: check why/if it is really needed
            proxy.status = self.discipline.execution_status.value

        elif self.wrapping_mode == 'GEMSEO':
            self.discipline = proxy.cls_builder
            # NEED TO UPDATE DEFAULTS OF self.discipline WITH get_sos_disc_inputs of proxy, HOW TO DO IT ?

    def create_mdo_scenario(self, sub_disciplines, proxy=None, reduced_dm=None):  # type: (...) -> None
        """
        SoSMDOScenario instantiation when owned by a ProxyOptim.

        Arguments:
            sub_disciplines (List[Discipline]): list of sub-Disciplines of the MDAChain
            proxy (ProxyDiscipline): proxy discipline for grammar initialisation
            input_data (dict): input data to update default values of the MDAChain with
        """
        if self.wrapping_mode == 'SoSTrades':
            # Pass as arguments to __init__ parameters needed for MDOScenario
            # creation

            mdo_options = {
                'algo_name': proxy.algo_name,
                'max_iter': proxy.max_iter,
                'eval_mode': proxy.eval_mode,
                'eval_jac': proxy.eval_jac,
                'dict_desactivated_elem': proxy.dict_desactivated_elem,
                'input_design_space': proxy.get_sosdisc_inputs('design_space'),
                # retrieve the option to desactivate the storage of the design space outputs for post processings
                'desactivate_optim_out_storage': proxy.get_sosdisc_inputs(proxy.DESACTIVATE_OPTIM_OUT_STORAGE),
            }

            discipline = SoSMDOScenarioAdapter(
                sub_disciplines,
                proxy.sos_name,
                proxy.formulation,
                proxy.objective_name,
                proxy.design_space,
                proxy.maximize_objective,
                proxy.get_input_data_names(numerical_inputs=False),
                proxy.get_output_data_names(numerical_inputs=False),
                logger=self.logger.getChild("SoSMDOScenarioAdapter"),
                reduced_dm=reduced_dm,
                mdo_options=mdo_options,
            )
            # Set parameters for SoSMDOScenarioAdapter

            self.discipline = discipline

            # self.__update_gemseo_grammar(proxy, discipline, mdoscenario=True)
            proxy.status = self.discipline.execution_status.value

        elif self.wrapping_mode == 'GEMSEO':
            raise NotImplementedError("GEMSEO native wrapping mode not yet available.")

    def _update_all_default_values(self, proxy):
        """Store all input grammar data names' values from input data in default values of discipline"""
        for key, value in proxy.get_data_in().items():
            if value[proxy.VALUE] is not None and not value[proxy.NUMERICAL]:
                full_key = proxy.get_var_full_name(key, proxy.get_data_in())
                self.discipline.default_input_data.update({full_key: value[proxy.VALUE]})

    def __update_gemseo_grammar(self, proxy, coupling, mdoscenario=False):
        """
        update GEMSEO grammar with sostrades
        # NOTE: this introduces a gap between the MDAChain i/o grammar and those of the MDOChain, as attribute of MDAChain
        """
        # - retrieve all the i/o of the ProxyCoupling that are not in the GEMSEO grammar of the MDAChain
        # (e.g., numerical inputs mainly)
        # TODO: [to discuss] ensure that/if all the SoSTrades added i/o ProxyCoupling are flagged as numerical, we can use this flag instead of performing set operations.
        #       -> need to check that outputs can be numerical (to cover the case of residuals for example, that is an output)

        soscoupling_inputs_and_defaults = proxy.get_run_needed_input()
        soscoupling_inputs = set(soscoupling_inputs_and_defaults.keys())

        coupling_inputs = set(coupling.io.input_grammar.names)

        missing_inputs = soscoupling_inputs | coupling_inputs

        soscoupling_outputs = set(proxy.get_output_data_names(numerical_inputs=False))
        coupling_outputs = set(coupling.io.output_grammar.names)
        missing_outputs = soscoupling_outputs | coupling_outputs

        # if this a mdoscenario then we add design space inputs to the outputs :
        if mdoscenario:
            design_space_inputs = coupling.scenario.design_space.variable_names
            missing_outputs.update(design_space_inputs)

        # i/o grammars update with SoSTrades i/o
        old_defaults = coupling.input_grammar.defaults
        coupling.input_grammar.clear()
        coupling.input_grammar.update_from_names(missing_inputs)
        missing_inputs_defaults = {
            key: value
            for key, value in soscoupling_inputs_and_defaults.items()
            if value is not None and key in missing_inputs
        }
        missing_inputs_defaults.update({key: value for key, value in old_defaults.items() if key in missing_inputs})
        coupling.input_grammar.update_defaults(missing_inputs_defaults)

        coupling.output_grammar.clear()
        coupling.output_grammar.update_from_names(missing_outputs)

    def execute(self, input_data):
        """Discipline execution delegated to the GEMSEO objects."""
        return self.discipline.execute(input_data)
