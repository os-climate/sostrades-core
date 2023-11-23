'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/07-2023/11/02 Copyright 2023 Capgemini

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
from scipy.sparse.lil import lil_matrix
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

class SoSMDODisciplineException(Exception):
    pass


class SoSMDODiscipline(MDODiscipline):
    """**SoSMDODiscipline** is the class that overloads MDODiscipline when using SoSTrades wrapping mode. It handles the
    execution of the user-provided wrapper of the discipline (on the GEMSEO side)

    It is instantiated by the MDODisciplineWrapp during the prepare_execution step, and it is in one-to-one aggregation
    with the user-defined wrapper (Specialization of SoSWrapp). The _run() method is overloaded by the user-defined wrapper.

    NB: overloading of MDODiscipline has been limited in EEV4 namely wrt EEV3's SoSDiscipline implementation

    Attributes:
        sos_wrapp (SoSWrapp): the user-defined wrapper of the discipline
        reduced_dm (Dict[Dict]): reduced data manager for i/o handling (NB: there is only one reduced_dm per process)
        io_full_name_map (Dict[string]): map from short names to full names of model output variables
   """

    _NEW_ATTR_TO_SERIALIZE = ['reduced_dm', 'sos_wrapp']
    DEBUG_MODE = 'debug_mode'
    LINEARIZATION_MODE = 'linearization_mode'
    NUM_DESC_IN = {LINEARIZATION_MODE,'cache_type','cache_file_path','debug_mode'}

    def __init__(self,
                 full_name :str, grammar_type: str, cache_type: str,
                 cache_file_path: str, sos_wrapp: SoSWrapp, reduced_dm: dict, logger:logging.Logger):
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
        # self.disciplines = [] # TODO: remove and leave in driver
        self.sos_wrapp = sos_wrapp
        self.reduced_dm = reduced_dm
        self.input_full_name_map = None
        self.output_full_name_map = None
        self.logger = logger
        super().__init__(name=full_name, grammar_type=grammar_type, cache_type=cache_type, cache_file_path=cache_file_path)
        self.is_sos_coupling = False

    def _run(self):
        """
        Call user-defined wrapper run.
        """
        # TODO: [discuss] is this to be done at the prepare execution? (with set_wrapper_attributes)?
        # send local data to the wrapper for i/o
        # self.sos_wrapp.local_data = self.local_data
        # self.sos_wrapp.input_data_names = self.get_input_data_names()
        # self.sos_wrapp.output_data_names = self.get_output_data_names()
        # self.sos_wrapp.input_full_name_map, self.sos_wrapp.output_full_name_map = self.create_io_full_name_map()

        # debug mode: input change
        if self.sos_wrapp.get_sosdisc_inputs(self.DEBUG_MODE) in ['input_change', 'all']:
            disc_inputs_before_execution = {key: {'value': value} for key, value in deepcopy(
                self.local_data).items() if key in self.input_grammar.data_names}

        # SoSWrapp run
        local_data = self.sos_wrapp._run()
        # local data update

        self.store_local_data(**local_data)

        # get output from data connector
        self.fill_output_value_connector()

        # debug modes
        if self.sos_wrapp.get_sosdisc_inputs(self.DEBUG_MODE) in ['nan', 'all']:
            self._check_nan_in_data(self.local_data)

        if self.sos_wrapp.get_sosdisc_inputs(self.DEBUG_MODE) in ['linearize_data_change']:
            self.check_linearize_data_changes=True

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

    def execute(
            self,
            input_data,  # type:Optional[Dict[str, Any]]
    ):  # type: (...) -> Dict[str, Any]
        '''
        Overload method in order to catch exception through a try/except
        '''

        try:
            self.local_data = super().execute(input_data)
        except Exception as error:
            # Update data manager status (status 'FAILED' is not propagate correctly due to exception
            # so we have to force data manager status update in this case
            self.status = self.STATUS_FAILED
            raise error
        return self.local_data

    def linearize(self, input_data=None, force_all=False, force_no_exec=False,
                  exec_before_linearize=True):
        """overloads GEMS linearize function
        """

        self.default_inputs = self._default_inputs
        if input_data is not None:
            self.default_inputs.update(input_data)

        if self.linearization_mode == self.COMPLEX_STEP:
            # is complex_step, switch type of inputs variables
            # perturbed to complex
            inputs, _ = self._retreive_diff_inouts(force_all)
            def_inputs = self.default_inputs
            for name in inputs:
                def_inputs[name] = def_inputs[name].astype('complex128')
        else:
            pass

        # need execution before the linearize
        if not force_no_exec and exec_before_linearize:
            self.reset_statuses_for_run()
            self.exec_for_lin = True
            self.execute(self.default_inputs)
            self.exec_for_lin = False
            force_no_exec = True
            need_execution_after_lin = False

        # need execution but after linearize, in the NR GEMSEO case an
        # execution is done bfore the while loop which udates the local_data of
        # each discipline
        elif not force_no_exec and not exec_before_linearize:
            force_no_exec = True
            need_execution_after_lin = True

        # no need of any execution
        else:
            need_execution_after_lin = False
            # maybe no exec before the first linearize, GEMSEO needs a
            # local_data with inputs and outputs for the jacobian computation
            # if the local_data is empty
            if self.local_data == {}:
                own_data = {
                    k: v for k, v in self.default_inputs.items() if self.is_input_existing(k) or self.is_output_existing(k)}
                self.local_data = own_data

        if self.check_linearize_data_changes and not self.is_sos_coupling:
            disc_data_before_linearize = {key: {'value': value} for key, value in deepcopy(
                self.default_inputs).items() if key in self.input_grammar.data_names}

        # Set STATUS to LINEARIZE for GUI visualization
        self.status = self.STATUS_LINEARIZE
        result = MDODiscipline.linearize(
            self, self.default_inputs, force_all, force_no_exec)
        self.status = self.STATUS_DONE

        self._check_nan_in_data(result)
        if self.check_linearize_data_changes and not self.is_sos_coupling:
            disc_data_after_linearize = {key: {'value': value} for key, value in deepcopy(
                self.default_inputs).items() if key in disc_data_before_linearize.keys()}
            is_output_error = True
            output_error = self.check_discipline_data_integrity(disc_data_before_linearize,
                                                                disc_data_after_linearize,
                                                                'Discipline data integrity through linearize',
                                                                is_output_error=is_output_error)
            if output_error != '':
                raise ValueError(output_error)

        if need_execution_after_lin:
            self.reset_statuses_for_run()
            self.execute(self.default_inputs)

        return result

    def check_jacobian(self, input_data=None, derr_approx=MDODiscipline.FINITE_DIFFERENCES,
                       step=1e-7, threshold=1e-8, linearization_mode='auto',
                       inputs=None, outputs=None, parallel=False,
                       n_processes=MDODiscipline.N_CPUS,
                       use_threading=False, wait_time_between_fork=0,
                       auto_set_step=False, plot_result=False,
                       file_path="jacobian_errors.pdf",
                       show=False, figsize_x=10, figsize_y=10, input_column=None, output_column=None,
                       dump_jac_path=None, load_jac_path=None):
        """
        Overload check jacobian to execute the init_execution
        """

        # The init execution allows to check jacobian without an execute before the check
        # however if an execute was done, we do not want to restart the model
        # and potentially loose informations to compute gradients (some
        # gradients are computed with the model)
        if self.status != self.STATUS_DONE:
            self.init_execution()

        # if dump_jac_path is provided, we trigger GEMSEO dump
        if dump_jac_path is not None:
            reference_jacobian_path = dump_jac_path
            save_reference_jacobian = True
        # if dump_jac_path is provided, we trigger GEMSEO dump
        elif load_jac_path is not None:
            reference_jacobian_path = load_jac_path
            save_reference_jacobian = False
        else:
            reference_jacobian_path = None
            save_reference_jacobian = False

        approx = DisciplineJacApprox(
            self,
            derr_approx,
            step,
            parallel,
            n_processes,
            use_threading,
            wait_time_between_fork,
        )
        if inputs is None:
            inputs = self.get_input_data_names(filtered_inputs=True)
        if outputs is None:
            outputs = self.get_output_data_names(filtered_outputs=True)

        if auto_set_step:
            approx.auto_set_step(outputs, inputs, print_errors=True)

        # Differentiate analytically
        self.add_differentiated_inputs(inputs)
        self.add_differentiated_outputs(outputs)
        self.linearization_mode = linearization_mode
        self.reset_statuses_for_run()
        # Linearize performs execute() if needed
        self.linearize(input_data)

        if input_column is None and output_column is None:
            indices = None
        else:
            indices = self._get_columns_indices(
                inputs, outputs, input_column, output_column)

        jac_arrays = {
            key_out: {key_in: value.toarray() if not isinstance(value, ndarray) else value for key_in, value in
                      subdict.items()}
            for key_out, subdict in self.jac.items()}
        o_k = approx.check_jacobian(
            jac_arrays,
            outputs,
            inputs,
            self,
            threshold,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            figsize_x=figsize_x,
            figsize_y=figsize_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
        )
        return o_k

    def compute_sos_jacobian(self):
        """
        Overload compute_sos_jacobian of MDODiscipline to call the function in the discipline wrapp
        Then retrieves the 'jac_dict' attribute of the wrapp to update the self.jac
        """
        self.sos_wrapp.compute_sos_jacobian()
        for y_key, x_key_dict in self.sos_wrapp.jac_dict.items():
            for x_key, value in x_key_dict.items():
                self.set_partial_derivative(y_key, x_key, value)

    def set_partial_derivative(self, y_key, x_key, value):
        '''
        Set the derivative of y_key by x_key inside the jacobian of GEMS self.jac
        '''

        if x_key in self.jac[y_key]:
            if isinstance(value, ndarray):
                value = lil_matrix(value)
            self.jac[y_key][x_key] = value

    # def create_io_full_name_map(self):
    #     """
    #     Create an io_full_name_map ainsi que des input_full_name_map and output_full_name_map for its sos_wrapp
    #
    #     Return:
    #         input_full_name_map (Dict[Str]): dict whose keys are input short names and values are input full names
    #         output_full_name_map (Dict[Str]): dict whose keys are output short names and values are output full names
    #     Sets attribute:
    #         self.io_full_name_map (Dict[Str]): union of the two above used for local data update
    #     """
    #
    #     if self.output_full_name_map is None:
    #         self.output_full_name_map = {}
    #         for key in self.get_output_data_names():
    #             short_name_key = self.io_full_name_to_short(key)
    #             #FIXME: quick fix
    #             if short_name_key not in self.output_full_name_map:
    #                 self.output_full_name_map[short_name_key] = key
    #
    #     if self.input_full_name_map is None:
    #         self.input_full_name_map = {}
    #         for key in self.get_input_data_names():
    #             short_name_key = self.io_full_name_to_short(key)
    #             # FIXME: quick fix
    #             if short_name_key not in self.input_full_name_map:
    #                 self.input_full_name_map[short_name_key] = key
    #
    #     return self.input_full_name_map, self.output_full_name_map

    def io_full_name_to_short(self, full_name_key):
        return self.reduced_dm[full_name_key][SoSWrapp.VAR_NAME]

    # def io_short_name_to_full(self, short_name_key):
    #     return self.io_full_name_map[short_name_key]

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
                    self.logger)

        self.store_local_data(**updated_values)

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
                                                    logger=self.logger)

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
            return filter_variables_to_convert(self.reduced_dm, self.output_grammar.get_data_names())

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

    def _get_columns_indices(self, inputs, outputs, input_column, output_column):
        """
        returns indices of input_columns and output_columns
        """
        # Get boundaries of the jacobian to compare
        if inputs is None:
            inputs = self.get_input_data_names()
        if outputs is None:
            outputs = self.get_output_data_names()

        indices = None
        if input_column is not None or output_column is not None:
            if len(inputs) == 1 and len(outputs) == 1:

                if hasattr(self, 'disciplines') and self.disciplines is not None:
                    for discipline in self.disciplines:
                        self.sos_wrapp.jac_boundaries.update(
                            discipline.jac_boundaries)

                indices = {}
                if output_column is not None:
                    jac_bnd = self.sos_wrapp.jac_boundaries[f'{outputs[0]},{output_column}']
                    tup = [jac_bnd['start'], jac_bnd['end']]
                    indices[outputs[0]] = [i for i in range(*tup)]

                if input_column is not None:
                    jac_bnd = self.sos_wrapp.jac_boundaries[f'{inputs[0]},{input_column}']
                    tup = [jac_bnd['start'], jac_bnd['end']]
                    indices[inputs[0]] = [i for i in range(*tup)]

            else:
                raise Exception(
                    'Not possible to use input_column and output_column options when \
                    there is more than one input and output')

        return indices

    @MDODiscipline.local_data.setter
    def local_data(
            self, data  # type: MutableMapping[str, Any]
    ):  # type: (...) -> None
        super(SoSMDODiscipline, type(self)).local_data.fset(self, data)
        self.sos_wrapp.local_data = data
        self.sos_wrapp.input_data_names = self.get_input_data_names()
        self.sos_wrapp.output_data_names = self.get_output_data_names()

    # ----------------------------------------------------
    # ----------------------------------------------------
    #  METHODS TO DEBUG DISCIPLINE
    # ----------------------------------------------------
    # ----------------------------------------------------

    def _check_nan_in_data(self, data):
        """ Using entry data, check if nan value exist in data's

        :params: data
        :type: composite data

        """
        has_nan = self._check_nan_in_data_rec(data, "")
        if has_nan:
            raise ValueError(f'NaN values found in {self.name}')

    def _check_nan_in_data_rec(self, data, parent_key):
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
                self._check_nan_in_data_rec(
                    data_value, f'{parent_key}/{data_key}')
            elif isinstance(data_value, floating):
                if pd.isnull(data_value).any():
                    nan_found = True

            if nan_found:
                full_key = data_key
                if len(parent_key) > 0:
                    full_key = f'{parent_key}/{data_key}'
                self.logger.debug(f'NaN values found in {full_key}')
                self.logger.debug(data_value)
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
        self.logger.info(
            "in discipline <%s> : <%s> has the minimum coupling value <%s>" % (
                self.name, min_coupling, min_coupling_dict[min_coupling]))
        self.logger.info(
            "in discipline <%s> : <%s> has the maximum coupling value <%s>" % (
                self.name, max_coupling, max_coupling_dict[max_coupling]))
