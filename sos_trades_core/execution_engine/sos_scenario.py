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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from copy import deepcopy
from numpy import array, ndarray, delete
from multiprocessing import cpu_count
import pandas as pd

from gemseo.algos.design_space import DesignSpace
from gemseo.core.scenario import Scenario
from gemseo.formulations.formulations_factory import MDOFormulationsFactory

from sos_trades_core.execution_engine.sos_discipline_builder import SoSDisciplineBuilder
from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.ns_manager import NS_SEP, NamespaceManager
from sos_trades_core.execution_engine.data_manager import POSSIBLE_VALUES


class SoSScenario(SoSDisciplineBuilder, Scenario):
    '''
    Generic implementation of Scenario
    '''
    # Default values of algorithms
    default_algo_options = {}
    default_parallel_options = {'parallel': False,
                                'n_processes': cpu_count(),
                                'use_threading': False,
                                'wait_time_between_fork': 0}
    USER_GRAD = 'user'
    # Design space dataframe headers
    VARIABLES = "variable"
    VALUES = "value"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    TYPE = "type"
    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"
    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    algo_dict = {}

    DESC_IN = {'algo': {'type': 'string', 'structuring': True},
               'design_space': {'type': 'dataframe', 'structuring': True},
               'formulation': {'type': 'string', 'structuring': True},
               'objective_name': {'type': 'string', 'structuring': True},
               'differentiation_method': {'type': 'string', 'default': Scenario.FINITE_DIFFERENCES,
                                          'possible_values': [USER_GRAD, Scenario.FINITE_DIFFERENCES,
                                                              Scenario.COMPLEX_STEP],
                                          'structuring': True},
               'algo_options': {'type': 'dict',  'dataframe_descriptor': {VARIABLES: ('string', None, False),
                                                                          VALUES: ('string', None, True)},
                                'dataframe_edition_locked': False,
                                'structuring': True},
               PARALLEL_OPTIONS: {'type': 'dict',  # SoSDisciplineBuilder.OPTIONAL: True,
                                  'dataframe_descriptor': {VARIABLES: ('string', None, False),  # bool
                                                           VALUES: ('string', None, True)},
                                  #                                   'dataframe_descriptor': {'parallel': ('int', None, True), #bool
                                  #                                                          'n_processes': ('int', None, True),
                                  #                                                          'use_threading': ('int', None, True),#bool
                                  #                                                          'wait_time_between_fork': ('int', None, True)},
                                  'dataframe_edition_locked': False,
                                  'default': default_parallel_options,
                                  'structuring': True},
               'eval_mode': {'type': 'bool', 'default': False, POSSIBLE_VALUES: [True, False], 'structuring': True},
               'eval_jac': {'type': 'bool', 'default': False, POSSIBLE_VALUES: [True, False]},
               'execute_at_xopt': {'type': 'bool', 'default': True}}

    DESC_OUT = {'design_space_out': {'type': 'dataframe'}
                }

    def __init__(self, sos_name, ee, cls_builder):
        """
        Constructor
        """
        self.__factory = ee.factory
        self.cls_builder = cls_builder
        self.formulation = None
        self.opt_problem = None
        self._maturity = None

        self._reload(sos_name, ee)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.SoSScenario')

        self.DESIGN_SPACE = 'design_space'
        self.FORMULATION = 'formulation'
        self.OBJECTIVE_NAME = 'objective_name'
        self.FORMULATION_OPTIONS = 'formulation_options'

#        self.SEARCH_PATHS = 'search_paths'
        self.SCENARIO_MANDATORY_FIELDS = [
            self.DESIGN_SPACE,
            self.FORMULATION,
            self.OBJECTIVE_NAME]
#            self.SEARCH_PATHS]
        self.OPTIMAL_OBJNAME_SUFFIX = "opt"
        self.dict_desactivated_elem = {}
        self.activated_variables = []

    def _reload(self, sos_name, ee):
        """
        reload object
        """
        SoSDisciplineBuilder._reload(self, sos_name, ee)

    def build(self):
        """
        build of subdisciplines
        """
        # build and set sos_disciplines (if any)
        if len(self.cls_builder) != 0:
            old_current_discipline = self.ee.factory.current_discipline
            self.ee.factory.current_discipline = self
            # get the list of builders
            builder_list = self.cls_builder
            if not isinstance(self.cls_builder, list):
                builder_list = [self.cls_builder]
            # build the disciplines if not already built
            for builder in builder_list:
                disc = builder.build()
                if disc not in self.sos_disciplines:
                    self.ee.factory.add_discipline(disc)

            self.ee.factory.current_discipline = old_current_discipline

    def configure(self):
        """
        Configuration of SoSScenario, call to super Class and
        """
        self.configure_io()

        self.configure_execution()

        # Extract variables for eval analysis
        if self.sos_disciplines is not None and len(self.sos_disciplines) > 0:
            self.set_eval_possible_values()

        # update MDA flag to flush residuals between each mda run
        self._set_flush_submdas_to_true()

    def is_configured(self):
        """
        Return False if at least one sub discipline needs to be configured, True if not
        """
        return self.get_configure_status() and not self.check_structuring_variables_changes() and (self.get_disciplines_to_configure() == [])

    def setup_sos_disciplines(self):
        """
        Overload setup_sos_disciplines to create a dynamic desc_in
        """
        if self.ALGO_OPTIONS in self._data_in:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            if algo_name is not None:
                default_dict = self.get_algo_options(algo_name)
                self._data_in[self.ALGO_OPTIONS][self.DEFAULT] = default_dict
                if algo_options is not None:
                    values_dict = deepcopy(default_dict)

                    for k in algo_options.keys():

                        if k not in values_dict.keys():
                            self.logger.warning(
                                f'option {k} is not in option list of the algorithm')

                        else:
                            values_dict.update({k: algo_options[k]})

                    self._data_in[self.ALGO_OPTIONS][self.VALUE] = values_dict

        self.set_edition_inputs_if_eval_mode()

    def set_edition_inputs_if_eval_mode(self):
        '''
        if eval mode then algo and algo options will turn to not editable
        '''

        if 'eval_mode' in self._data_in:
            eval_mode = self.get_sosdisc_inputs('eval_mode')
            if eval_mode:
                self._data_in[self.ALGO][self.EDITABLE] = False
                self._data_in[self.ALGO_OPTIONS][self.EDITABLE] = False
                self._data_in[self.FORMULATION][self.EDITABLE] = False
                self._data_in[self.PARALLEL_OPTIONS][self.EDITABLE] = False

                self._data_in[self.ALGO][self.OPTIONAL] = True
                self._data_in[self.ALGO_OPTIONS][self.OPTIONAL] = True
                self._data_in[self.FORMULATION][self.OPTIONAL] = True
                self._data_in[self.PARALLEL_OPTIONS][self.OPTIONAL] = True
            else:
                self._data_in['eval_jac'][self.VALUE] = False

    def pre_set_scenario(self):
        """
        prepare the set GEMS set_scenario method
        """
        design_space = None
        formulation = None
        obj_full_name = None

        dspace = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        if dspace is not None:
            # build design space
            design_space = self.set_design_space()
            if design_space.variables_names:

                _, formulation, obj_name = self.get_sosdisc_inputs(
                    self.SCENARIO_MANDATORY_FIELDS)

                # get full objective ids
                obj_name = self.get_sosdisc_inputs(self.OBJECTIVE_NAME)
                obj_full_name = self._update_names([obj_name])[0]

        return design_space, formulation, obj_full_name

    def set_scenario(self):
        """
        set GEMS scenario, to be overloaded with each type of scenario (MDO, DOE, ...)
        """
        pass

    def set_design_space_for_complex_step(self):
        '''
        Set design space values to complex if the differentiation method is complex_step
        '''
        diff_method = self.get_sosdisc_inputs('differentiation_method')
        if diff_method == self.COMPLEX_STEP:
            dspace = deepcopy(self.opt_problem.design_space)
            curr_x = dspace._current_x
            for var in curr_x:
                curr_x[var] = curr_x[var].astype('complex128')
            self.opt_problem.design_space = dspace

    def update_default_coupling_inputs(self):
        '''
        Update default inputs of the couplings
        '''
        for disc in self.sos_disciplines:
            if disc.is_sos_coupling:
                self._set_default_inputs_from_dm(disc)

    def get_algo_options(self, algo_name):
        """
        Create default dict for algo options
        :param algo_name: the name of the algorithm
        :returns: dictionary with algo options default values
        """
        # TODO : add warning and log algo options

        default_dict = {}
        driver_lib = self._algo_factory.create(algo_name)
        driver_lib.init_options_grammar(algo_name)
        schema_dict = driver_lib.opt_grammar.schema.to_dict()
        properties = schema_dict.get(driver_lib.opt_grammar.PROPERTIES_FIELD)
        algo_options_keys = list(properties.keys())

        found_algo_names = [
            key for key in self.algo_dict.keys() if key in algo_name]
        if len(found_algo_names) == 1:
            key = found_algo_names[0]
            for algo_option in algo_options_keys:
                default_val = self.algo_dict[key][algo_option]
                if default_val is not None:
                    default_dict[algo_option] = default_val
        else:
            for algo_option in algo_options_keys:
                if algo_option in self.default_algo_options:
                    algo_default_val = self.default_algo_options[algo_option]
                    if algo_default_val is not None:
                        default_dict[algo_option] = algo_default_val

        return default_dict

    def run(self):
        '''
        Run method
        '''
        # TODO: to delete when MDA initialization is improved
        self.update_default_coupling_inputs()

        self.set_design_space_for_complex_step()

        eval_mode = self.get_sosdisc_inputs('eval_mode')
        if eval_mode:
            self.run_eval_mode()

        else:
            self.run_scenario()

        # convert local_data into new types and store values in data manager
        local_data_sos = self._convert_array_into_new_type(self.local_data)
        self.dm.set_values_from_dict(local_data_sos)

    def run_scenario(self):
        '''
        Run the scenario and store last design_space
        '''
        pass

    def run_eval_mode(self):
        '''
        Run evaluate functions with the initial x 
        '''
        eval_jac = self.get_sosdisc_inputs('eval_jac')
        design_space = self.get_sosdisc_inputs('design_space')

        self.opt_problem.evaluate_functions(
            eval_jac=eval_jac, normalize=False)
        # if eval mode design space was not modified
        self.store_sos_outputs_values(
            {'design_space_out': design_space}, update_dm=True)

    def _run_algorithm(self):
        """
        Runs the algo
        """
        pass

    def _set_flush_submdas_to_true(self):
        # update MDA flag to flush residuals between each mda run
        for disc in self.sos_disciplines:
            if disc.is_sos_coupling:
                if len(disc.sub_mda_list) > 0:
                    for sub_mda in disc.sub_mda_list:
                        sub_mda.reset_history_each_run = True

                for subdisc in disc.sos_disciplines:
                    subdisc._cache_type = disc._cache_type

    def _set_default_inputs_from_dm(self, disc):
        """
        Based on dm values, default_inputs are set to mdachains,
        and default_inputs dtype is set to complex in case of complex_step gradient computation.
        """
        input_data = {}
        input_data_names = disc.get_input_data_names()
        for data_name in input_data_names:
            val = self.ee.dm.get_value(data_name)
            # for cases of early configure steps
            if val is not None:
                input_data[data_name] = val

        # convert sostrades types into numpy arrays
        # no need to update DM since call by SoSTrades
        input_data = disc._convert_new_type_into_array(var_dict=input_data)
        disc.mdo_chain.default_inputs.update(input_data)

        for disc in disc.sos_disciplines:
            if disc.is_sos_coupling:
                self._set_default_inputs_from_dm(disc)

    def configure_io(self):
        """
        Configure discipline  and all sub-disciplines
        """
        if self._data_in == {} or self.check_structuring_variables_changes():
            super().configure()

        disc_to_configure = self.get_disciplines_to_configure()

        if len(disc_to_configure) > 0:
            self.set_configure_status(False)
        else:
            self.set_configure_status(True)

        for disc in disc_to_configure:
            disc.configure()

    def get_disciplines_to_configure(self):
        """
        Get sub disciplines list to configure
        """
        disc_to_configure = []
        for disc in self.sos_disciplines:
            if not disc.is_configured():
                disc_to_configure.append(disc)
        return disc_to_configure

    def configure_execution(self):
        """
        - configure GEMS grammar
        - set scenario
        """
        for disc in self.sos_disciplines:
            disc.update_gems_grammar_with_data_io()
        self.set_scenario()
        self.set_parallel_options()

    def _update_names(self, names):
        """
        if no dot in the name, it looks for the full name in the dm
        else we suppose that this is a full name that needs to be updated with current
        study name
        |!| it will NOT work for names with a dot in data_io...
        """
        local_names = []
        full_names = []
        for name in names:
            if NamespaceManager.NS_SEP not in name:
                local_names.append(name)
            else:
                full_names.append(name)
        return self.get_full_names(local_names) + \
            self._update_study_ns_in_varname(full_names)

    def set_diff_method(self):
        """
        Set differentiation method and send a WARNING
        if some linearization_mode are not coherent with diff_method
        """
        diff_method = self.get_sosdisc_inputs('differentiation_method')

        if diff_method in self.APPROX_MODES:
            for disc in self.sos_disciplines:
                if disc.linearization_mode != diff_method:
                    self.logger.warning(
                        f'The differentiation method "{diff_method}" will overload the linearization mode "{disc.linearization_mode}" ')

        Scenario.set_differentiation_method(
            self, diff_method, 1e-6)

    def set_parallel_options(self):
        """
        sets parallel options for jacobian approximation
        """

        # update default options with user options
        user_options = self.get_sosdisc_inputs(self.PARALLEL_OPTIONS)
        if user_options is None:
            user_options = {}
        options = deepcopy(self.default_parallel_options)
        options.update(user_options)
        parallel = options.pop("parallel")
        # update problem options
        if self.formulation is not None:
            self.opt_problem = self.formulation.opt_problem
            self.opt_problem.parallel_differentiation = parallel
            self.opt_problem.parallel_differentiation_options = options

    def set_design_space(self):
        """
        reads design space (set_design_space)
        """

        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        # update design space dv with full names
        dvs = list(dspace_df[self.VARIABLES])
        full_dvs = []
        dspace_dict_updated = {}

        for key in dvs:

            full_key_l = self.get_full_names([key])
            if len(full_key_l) > 0:
                full_key = full_key_l[0]
                full_dvs.append(full_key)
                # dspace_dict_updated[full_key] = dspace_df[key]
            else:
                self.logger.warning(f" missing design variable in dm : {key}")
        if len(full_dvs) == len(dvs):
            dspace_dict_updated = dspace_df.copy()
            dspace_dict_updated[self.VARIABLES] = full_dvs

            design_space = self.read_from_dataframe(dspace_dict_updated)

        else:

            design_space = DesignSpace()
        return design_space

    def read_from_dict(self, dp_dict):
        """Parses a dictionary to read the DesignSpace

        :param dp_dict : design space dictionary
        :returns:  the design space
        """
        design_space = DesignSpace()
        for key in dp_dict:
            print(key)
            if type(dp_dict[key]['value']) != list and type(dp_dict[key]['value']) != ndarray:
                name = key
                var_type = ['float']

                size = 1
                l_b = array([dp_dict[key]['lower_bnd']])
                u_b = array([dp_dict[key]['upper_bnd']])
                value = array([dp_dict[key]['value']])
            else:
                size = len(dp_dict[key]['value'])
                var_type = ['float'] * size

                name = key
                l_b = array(dp_dict[key]['lower_bnd'])
                u_b = array(dp_dict[key]['upper_bnd'])
                value = array(dp_dict[key]['value'])

            design_space.add_variable(name, size, var_type, l_b, u_b, value)
        return design_space

    def read_from_dataframe(self, df):
        """Parses a DataFrame to read the DesignSpace

        :param df : design space df
        :returns:  the design space
        """
        names = list(df[self.VARIABLES])
        values = list(df[self.VALUES])
        l_bounds = list(df[self.LOWER_BOUND])
        u_bounds = list(df[self.UPPER_BOUND])
        enabled_variable = list(df[self.ENABLE_VARIABLE_BOOL])
        list_activated_elem = list(df[self.LIST_ACTIVATED_ELEM])
        design_space = DesignSpace()
        for dv, val, lb, ub, l_activated, enable_var in zip(names, values, l_bounds, u_bounds, list_activated_elem, enabled_variable):

            # check if variable is enabled to add it or not in the design var
            if enable_var:
                self.dict_desactivated_elem[dv] = {}

                if [type(val), type(lb), type(ub)] == [str] * 3:
                    val = val
                    lb = lb
                    ub = ub
                name = dv
                if type(val) != list and type(val) != ndarray:
                    size = 1
                    var_type = ['float']
                    l_b = array([lb])
                    u_b = array([ub])
                    value = array([val])
                else:
                    # check if there is any False in l_activated
                    if not all(l_activated):
                        index_false = l_activated.index(False)
                        self.dict_desactivated_elem[dv] = {
                            'value': val[index_false], 'position': index_false}

                        val = delete(val, index_false)
                        lb = delete(lb, index_false)
                        ub = delete(ub, index_false)

                    size = len(val)
                    var_type = ['float'] * size
                    l_b = array(lb)
                    u_b = array(ub)
                    value = array(val)
                design_space.add_variable(
                    name, size, var_type, l_b, u_b, value)
        return design_space

    def read_from_dataframe_new(self, df):
        """Parses a DataFrame to read the DesignSpace

        :param df : design space df
        :returns:  the design space
        """
        names = df[self.VARIABLES]
        values = df[self.VALUES]
        l_bounds = df[self.LOWER_BOUND]
        u_bounds = df[self.UPPER_BOUND]

        design_space = DesignSpace()
        for dv, val, lb, ub in zip(names, values, l_bounds, u_bounds):
            #             if [type(val), type(lb), type(ub)] == [str] * 3:
            #                 val = eval(val)
            #                 lb = eval(lb)
            #                 ub = eval(ub)
            name = dv
            if type(val) != list and type(val) != ndarray:
                size = 1
                var_type = ['float']
                l_b = array([lb])
                u_b = array([ub])
                value = array([val])
            else:
                size = len(val)
                var_type = ['float'] * size
                l_b = array(lb)
                u_b = array(ub)
                value = array(val)
            design_space.add_variable(name, size, var_type, l_b, u_b, value)
        return design_space

    # -- GEMS overload
    def _update_input_grammar(self):
        """
        Updates input grammar from algo names
        """
#         # TODO: rename all the scenario grammar properly
#         ## updates input_grammar of MDOScenario with namespaced inputs
#         # get input namespaced names
#         ns_algo, = self._convert_list_of_keys_to_namespace_name(
#             self.ALGO, self.IO_TYPE_IN)
#         ns_maxiter, = self._convert_list_of_keys_to_namespace_name(
#             self.MAX_ITER, self.IO_TYPE_IN)
#         # build a grammar and initialize it from mandatory fields
#         gram = JSONGrammar("opt_gram")
#         gram.initialize_from_data_names([ns_algo, ns_maxiter])
#         self.input_grammar.update_from(gram)
#
#         # fill in the namespaced fields
#         available_algos = self.get_available_driver_names()
#         algo_grammar = {"type": "string", "enum": available_algos}
#         self.input_grammar.set_item_value(ns_algo, algo_grammar)
#
#         max_iter_grammar = {"type" : "integer", "minimum":1}
#         self.input_grammar.set_item_value(ns_maxiter, max_iter_grammar)
        algo, = self._convert_list_of_keys_to_namespace_name(
            self.ALGO, self.IO_TYPE_IN)
        available_algos = self.get_available_driver_names()
        # change type from string to int in GEMs grammar since SoSTrades
        # converts strings to int
        algo_grammar = {"type": "integer"}  # "enum": available_algos
        self.input_grammar.set_item_value(algo, algo_grammar)

    def update_design_space_out(self):
        """
        Method to update design space with opt value
        """
        design_space = deepcopy(self.get_sosdisc_inputs(self.DESIGN_SPACE))
        l_variables = design_space[self.VARIABLES]
        for var in l_variables:
            full_name_var = self.get_full_names([var])[0]
            if full_name_var in self.activated_variables:
                value_x_opt = list(self.formulation.design_space._current_x.get(
                    full_name_var))
                if self.dict_desactivated_elem[full_name_var] != {}:
                    # insert a desactivated element
                    value_x_opt.insert(
                        self.dict_desactivated_elem[full_name_var]['position'], self.dict_desactivated_elem[full_name_var]['value'])

                design_space.loc[design_space[self.VARIABLES] == var, self.VALUE] = pd.Series(
                    [value_x_opt] * len(design_space))

        self.store_sos_outputs_values(
            {'design_space_out': design_space}, update_dm=True)

    def _init_base_grammar(self, name):
        """ *** GEMS overload ***
        Initializes the base grammars from MDO scenario inputs and outputs
        This ensures that subclasses have base scenario inputs and outputs
        Can be overloaded by subclasses if this is not desired.

        :param name: name of the scenario, used as base name for the json
            schema to import: name_input.json and name_output.json
        """
#         gems_in_keys = list(self._data_in.keys())
#         gems_out_keys = list(self._data_out.keys())
#         self._init_grammar_with_keys(gems_in_keys, self.IO_TYPE_IN)
#         self._init_grammar_with_keys(gems_out_keys, self.IO_TYPE_OUT)
        self.update_gems_grammar_with_data_io()

#     def _convert_new_type_into_array(self, var_dict):
#         input_data = SoSDiscipline._convert_new_type_into_array(self,var_dict=var_dict)
#         # replace integer value by corresponding string for algo name
#         ns_algo, = self._convert_list_of_keys_to_namespace_name(self.ALGO, self.IO_TYPE_IN)
#         if ns_algo in input_data:
#             input_data[ns_algo] = self.get_sosdisc_inputs(self.ALGO)
#         return input_data

#     def check_input_data(self, input_data, raise_exception=True):
#         """Check the input data validity.
#
#         :param input_data: the input data dict
#         :param raise_exception: Default value = True)
#         """
#         # replace integer value by corresponding string for algo name
#         ns_algo, = self._convert_list_of_keys_to_namespace_name(self.ALGO, self.IO_TYPE_IN)
#         if ns_algo in input_data:
#             input_data[ns_algo] = self.get_sosdisc_inputs(self.ALGO)
#         try:
#             self.input_grammar.load_data(input_data, raise_exception)
#         except InvalidDataException:
#             raise InvalidDataException("Invalid input data for: " + self.name)

    # -- maturities
    def get_maturity(self):
        ref_dict_maturity = deepcopy(self.dict_maturity_ref)
        for discipline in self.sos_disciplines:
            disc_maturity = discipline.get_maturity()

            if isinstance(disc_maturity, dict):
                for m_k, m_v in ref_dict_maturity.items():
                    if m_v != disc_maturity[m_k]:
                        ref_dict_maturity[m_k] += disc_maturity[m_k]
            elif disc_maturity in ref_dict_maturity:
                ref_dict_maturity[disc_maturity] += 1

        self._maturity = ref_dict_maturity
        return self._maturity

    def set_eval_possible_values(self):

        analyzed_disc = self.sos_disciplines
        possible_out_values = self.fill_possible_values(
            analyzed_disc)  # possible_in_values

        possible_out_values = self.find_possible_values(
            analyzed_disc, possible_out_values)  # possible_in_values

        # Take only unique values in the list
        possible_out_values = list(set(possible_out_values))

        # Fill the possible_values of obj and constraints
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.OBJECTIVE_NAME}',
                         self.POSSIBLE_VALUES, possible_out_values)

        if self.is_constraints:
            self.dm.set_data(f'{self.get_disc_full_name()}.{self.INEQ_CONSTRAINTS}',
                             self.POSSIBLE_VALUES, possible_out_values)
            self.dm.set_data(f'{self.get_disc_full_name()}.{self.EQ_CONSTRAINTS}',
                             self.POSSIBLE_VALUES, possible_out_values)
        # fill the possible values of algos
        self._init_algo_factory()
        avail_algos = self._algo_factory.algorithms
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.ALGO}',
                         self.POSSIBLE_VALUES, avail_algos)
        # fill the possible values of formulations
        self._form_factory = MDOFormulationsFactory()
        avail_formulations = self._form_factory.formulations
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.FORMULATION}',
                         self.POSSIBLE_VALUES, avail_formulations)

    # -- Set possible design variables and objevtives
    # adapted from soseval
    # TODO: find a better way to select constraints and objectives

    def find_possible_values(
            self, sos_disciplines, possible_out_values):  # possible_in_values
        """
            Run through all disciplines and sublevels
            to find possible values for eval_inputs and eval_outputs
        """
        if len(sos_disciplines) != 0:
            for disc in sos_disciplines:
                sub_out_values = self.fill_possible_values(
                    [disc])  # sub_in_values
#                 possible_in_values.extend(sub_in_values)
                possible_out_values.extend(sub_out_values)
                self.find_possible_values(
                    disc.sos_disciplines, possible_out_values)  # possible_in_values

        return possible_out_values  # possible_in_values

    def fill_possible_values(self, sos_disciplines):
        """
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        """
        # poss_in_values = []
        poss_out_values = []
        for disc in sos_disciplines:
            #             for data_in_key in disc.get_input_data_names(): #disc._data_in.keys():
            #                 is_float = disc._data_in[data_in_key.split(NS_SEP)[-1]][self.TYPE] == 'float'
            #                 in_coupling_numerical = data_in_key in SoSCoupling.DEFAULT_NUMERICAL_PARAM
            #                 if not in_coupling_numerical: #is_float and
            #                     # Caution ! This won't work for variables with points in name
            #                     # as for ac_model
            #                     poss_in_values.append(data_in_key)
            for data_out_key in disc.get_output_data_names():  # disc._data_out.keys():
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                data_out_key = data_out_key.split(NS_SEP)[-1]
                poss_out_values.append(data_out_key)

        return poss_out_values  # poss_in_values

    def __str__(self):
        """
        Summarize results for display

        :returns: string summarizing results
        """
        msg = ""
        if hasattr(self, "disciplines"):
            msg = self.__class__.__name__ + ":\nDisciplines: "
            disc_names = [disc.name
                          for disc in self.disciplines]  # pylint: disable=E1101
            msg += " ".join(disc_names)
            msg += "\nMDOFormulation: "  # We keep MDO here has is done in gemseo
            msg += self.formulation.__class__.__name__
            msg += "\nAlgorithm: "
            msg += str(self.get_sosdisc_inputs(self.ALGO)) + "\n"

        return msg

    def get_full_names(self, names):
        '''
        get full names of ineq_names and obj_names
        '''
        full_names = []
        for i_name in names:
            full_id_l = self.dm.get_all_namespaces_from_var_name(i_name)
            if full_id_l != []:
                if len(full_id_l) > 1:
                    #full_id = full_id_l[0]
                    full_id = self.get_scenario_lagr(full_id_l)
                else:
                    full_id = full_id_l[0]
                full_names.append(full_id)

        return full_names

    def get_algo_options_dict(self):
        """
        Method to get the dictionnary of algo options from dataframe so that it is in GEMS format
        """
        algo_options_df = self.get_sosdisc_inputs('algo_options')
        return algo_options_df.to_dict('records')[0]
