'''
Copyright 2023 Capgemini

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
from copy import copy, deepcopy
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.execution_engine.sample_generators.simple_sample_generator import SimpleSampleGenerator
from sostrades_core.execution_engine.sample_generators.grid_search_sample_generator import GridSearchSampleGenerator
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import \
    CartesianProductSampleGenerator
from sostrades_core.tools.gather.gather_tool import check_eval_io
import pandas as pd
from numpy import array


class ProxySampleGeneratorException(Exception):
    pass


class ProxySampleGenerator(ProxyDiscipline):
    '''
    Class that gather output data from a scatter discipline
    '''

    # ontology information
    _ontology_data = {
        'label': 'Sample Generator',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-outdent fa-fw',
        'version': '',
    }

    INPUT_TYPE = ['float', 'array', 'int', 'string']

    MULTIPLIER_PARTICULE = "__MULTIPLIER__" # todo: to delete

    SAMPLES_DF = SampleGeneratorWrapper.SAMPLES_DF
    SELECTED_SCENARIO = SampleGeneratorWrapper.SELECTED_SCENARIO
    SCENARIO_NAME = SampleGeneratorWrapper.SCENARIO_NAME
    REFERENCE_SCENARIO_NAME = 'Reference Scenario'

    # TODO: move to tools ?
    N_SAMPLES = "n_samples"
    ALGO = SampleGeneratorWrapper.ALGO
    ALGO_OPTIONS = SampleGeneratorWrapper.ALGO_OPTIONS
    DESIGN_SPACE = SampleGeneratorWrapper.DESIGN_SPACE
    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    LOWER_BOUND = DoeSampleGenerator.LOWER_BOUND
    UPPER_BOUND = DoeSampleGenerator.UPPER_BOUND
    VARIABLES = DoeSampleGenerator.VARIABLES
    VALUES = DoeSampleGenerator.VALUES
    NB_POINTS = DoeSampleGenerator.NB_POINTS

    NS_SAMPLING = 'ns_sampling'

    SAMPLES_DF_DEFAULT = pd.DataFrame({SELECTED_SCENARIO: [True],
                                       SCENARIO_NAME: REFERENCE_SCENARIO_NAME})
    SAMPLES_DF_DESC = {
        ProxyDiscipline.TYPE: 'dataframe',
        ProxyDiscipline.DEFAULT: SAMPLES_DF_DEFAULT.copy(),
        ProxyDiscipline.DATAFRAME_DESCRIPTOR: {SELECTED_SCENARIO: ('bool', None, True),
                                               SCENARIO_NAME: ('string', None, True)},
        ProxyDiscipline.DYNAMIC_DATAFRAME_COLUMNS: True,
        ProxyDiscipline.DATAFRAME_EDITION_LOCKED: False,
        ProxyDiscipline.EDITABLE: True,
        ProxyDiscipline.STRUCTURING: False
    }
    SAMPLES_DF_DESC_SHARED = SAMPLES_DF_DESC.copy()
    SAMPLES_DF_DESC_SHARED[ProxyDiscipline.NAMESPACE] = NS_SAMPLING
    SAMPLES_DF_DESC_SHARED[ProxyDiscipline.VISIBILITY] = ProxyDiscipline.SHARED_VISIBILITY

    EVAL_INPUTS = SampleGeneratorWrapper.EVAL_INPUTS
    SELECTED_INPUT = SampleGeneratorWrapper.SELECTED_INPUT
    FULL_NAME = SampleGeneratorWrapper.FULL_NAME
    EVAL_INPUTS_DF_DESC = {SELECTED_INPUT: ('bool', None, True),
                           FULL_NAME: ('string', None, True)}
    EVAL_INPUTS_DESC = {ProxyDiscipline.TYPE: 'dataframe',
                        ProxyDiscipline.DATAFRAME_DESCRIPTOR: EVAL_INPUTS_DF_DESC.copy(),
                        ProxyDiscipline.DATAFRAME_EDITION_LOCKED: False,
                        ProxyDiscipline.STRUCTURING: True,
                        ProxyDiscipline.DEFAULT: pd.DataFrame(columns=[SELECTED_INPUT, FULL_NAME]),
                        #ProxyDiscipline.VISIBILITY: ProxyDiscipline.SHARED_VISIBILITY,
                        #ProxyDiscipline.NAMESPACE: NS_SAMPLING,
                        }
    EVAL_INPUTS_CP_DF_DESC = EVAL_INPUTS_DF_DESC.copy()
    LIST_OF_VALUES = SampleGeneratorWrapper.LIST_OF_VALUES
    EVAL_INPUTS_CP_DF_DESC.update({LIST_OF_VALUES: ('list', None, True)})

    SAMPLING_METHOD = 'sampling_method'
    SIMPLE_SAMPLING_METHOD = 'simple'
    DOE_ALGO = 'doe_algo'
    CARTESIAN_PRODUCT = 'cartesian_product'
    GRID_SEARCH = 'grid_search'
    AVAILABLE_SAMPLING_METHODS = [SIMPLE_SAMPLING_METHOD, DOE_ALGO, CARTESIAN_PRODUCT, GRID_SEARCH]
    # classes of the sample generator tools associated to each method in AVAILABLE_SAMPLING_METHODS
    SAMPLE_GENERATOR_CLS = {
        SIMPLE_SAMPLING_METHOD: SimpleSampleGenerator,
        DOE_ALGO: DoeSampleGenerator,
        CARTESIAN_PRODUCT: CartesianProductSampleGenerator,
        GRID_SEARCH: GridSearchSampleGenerator
    }

    SAMPLING_GENERATION_MODE = 'sampling_generation_mode'
    AT_CONFIGURATION_TIME = 'at_configuration_time'
    AT_RUN_TIME = 'at_run_time'
    available_sampling_generation_modes = [AT_CONFIGURATION_TIME, AT_RUN_TIME]

    OVERWRITE_SAMPLES_DF = 'overwrite_samples_df'
    OVERWRITE_SAMPLES_DF_DESC = {
        ProxyDiscipline.TYPE: 'bool',
        ProxyDiscipline.STRUCTURING: True,
        ProxyDiscipline.DEFAULT: False,  # TODO: think about
    }
    MAX_AUTO_SELECT_SCENARIOS = 1024  # maximum number of scenarios to be auto-selected after a sampling at config. time

    DESC_IN = {SAMPLING_METHOD: {'type': 'string',
                                 'structuring': True,
                                 'possible_values': AVAILABLE_SAMPLING_METHODS,
                                 'default': SIMPLE_SAMPLING_METHOD},
               SAMPLING_GENERATION_MODE: {'type': 'string',
                                          'structuring': True,
                                          'possible_values': available_sampling_generation_modes,
                                          'default': AT_CONFIGURATION_TIME,
                                          'editable': True},
               EVAL_INPUTS: EVAL_INPUTS_DESC.copy()
               }

    def __init__(self, sos_name, ee, cls_builder=None, associated_namespaces=None):
        super().__init__(sos_name=sos_name,
                         ee=ee,
                         cls_builder=cls_builder,
                         associated_namespaces=associated_namespaces)

        self.check_integrity_msg_list = []
        self.sg_data_integrity = True

        self.sampling_method = None
        self.sampling_generation_mode = None

        self.sample_pending = False
        self.samples_gene_df = None   # sample generated at configuration-time

        self.force_sampling_at_configuration_time = False
        # TODO: actually no need for two variables as the type dict could be sorted and its keys be the possible_values
        self.eval_in_possible_values = []
        self.eval_in_possible_types = {}
        self.samples_df_f_name = None
        #FIXME: using samples_df_f_name to sample means configuration-time sampling needs to be banned on standalone sample gen.

    def set_eval_in_possible_values(self,
                                    possible_values: list[str],
                                    possible_types: dict[str]):
        """
        Method used by a driver in composition with a sample generator to pass the set of inputs of the subprocess
        that can be selected in eval_inputs.

        Arguments:
            possible_values (list(string)): possible values of the eval_inputs variable names
        Returns:
             driver_is_configured (bool): flag to detect whether driver could ask sample generator for necessary
                configuration actions
        """
        if self.eval_in_possible_types.keys() != possible_types.keys():
            self.eval_in_possible_values = possible_values
            self.eval_in_possible_types = possible_types
            self.set_configure_status(False)

    def _check_eval_inputs_types_for_one_variable(self, eval_inputs_row):
        """
        Utility method that checks type and first subtype integrity in the 'list_of_values' column of evaluated inputs,
        which contains the factors of the product in case of a cartesian product sampling.

        Arguments:
            eval_inputs_row (pd.Series): row of the evaluated inputs dataframe to check
        """
        selected = eval_inputs_row[self.SELECTED_INPUT]
        var_name = eval_inputs_row[self.FULL_NAME]
        var_type = self.eval_in_possible_types[var_name]
        list_of_values = eval_inputs_row[self.LIST_OF_VALUES]
        return not selected or (
                isinstance(list_of_values, list) and
                all(map(lambda _val: isinstance(_val, self.VAR_TYPE_MAP[var_type]),
                        list_of_values)))

    def _check_design_space_dimensions_for_one_variable(self, design_space_row):
        """
        Utility method that checks that values in the columns 'lower_bnd', 'upper_bnd', 'value' of the design space do
        have the same shape for a same variable.

        Arguments:
            eval_inputs_row (pd.Series): row of the design space dataframe to check
        """
        lb = design_space_row[self.LOWER_BOUND] if self.LOWER_BOUND in design_space_row.index else None
        ub = design_space_row[self.UPPER_BOUND] if self.UPPER_BOUND in design_space_row.index else None
        val = design_space_row[self.VALUES] if self.VALUES in design_space_row.index else None
        lb_shape = array(lb).shape
        ub_shape = array(ub).shape
        val_shape = array(val).shape
        lb_ub_dim_mismatch = lb is not None and ub is not None and lb_shape != ub_shape
        lb_val_dim_mismatch = lb is not None and val is not None and lb_shape != val_shape
        val_ub_dim_mismatch = val is not None and ub is not None and val_shape != ub_shape
        return not (lb_ub_dim_mismatch or lb_val_dim_mismatch or val_ub_dim_mismatch)

    def check_data_integrity(self):
        """
        Data integrity checks of the Sample Generator including:
        - evaluated inputs (if working with driver):
            - check that the variables selected make sense with the subprocess
            - check that the type and subtype of column 'list_of_values' are correct
        - design space:
            - check that the variables selected make sense with the subprocess (if working with driver)
            - check that the ['lower_bnd', 'upper_bnd', 'value'] columns have coherent shapes with each other
        """
        super().check_data_integrity()
        self.sg_data_integrity = True
        disc_in = self.get_data_in()

        # check integrity for eval_inputs # TODO: move to cartesian product sample generator ?
        eval_inputs_integrity_msg = []
        if self.configurator and self.EVAL_INPUTS in disc_in:
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            if eval_inputs is not None:
                # possible value checks (with current implementation should be OK by construction)
                vars_not_possible = eval_inputs[self.FULL_NAME][
                        ~eval_inputs[self.FULL_NAME].apply(lambda _var: _var in self.eval_in_possible_types)].to_list()
                for var_not_possible in vars_not_possible:
                    eval_inputs_integrity_msg.append(
                        f'Variable {var_not_possible} is not among the possible input values.'
                    )
                # for cartesian product check that factors' values have the right type. NB: this also checks that
                # list_of_values is a list (redundantly with dataframe descriptor) as to avoid spurious sampling
                if not vars_not_possible and self.LIST_OF_VALUES in eval_inputs.columns:
                    wrong_type_vars = eval_inputs[self.FULL_NAME][
                        ~eval_inputs.apply(self._check_eval_inputs_types_for_one_variable, axis=1)].to_list()
                    for wrong_type_var in wrong_type_vars:
                        eval_inputs_integrity_msg.append(
                            f'Column {self.LIST_OF_VALUES} should be a list of '
                            f'{self.eval_in_possible_types[wrong_type_var]} for selected variable {wrong_type_var}.')
        if eval_inputs_integrity_msg:
            self.sg_data_integrity = False
            self.ee.dm.set_data(self.get_var_full_name(self.EVAL_INPUTS, disc_in),
                                self.CHECK_INTEGRITY_MSG, '\n'.join(eval_inputs_integrity_msg))

        # check integrity for design space # todo: move to doe and gridsearch generators ?
        design_space_integrity_msg = []
        if self.DESIGN_SPACE in disc_in:
            design_space = self.get_sosdisc_inputs(self.DESIGN_SPACE)
            if design_space is not None and not design_space.empty:
                if self.configurator:
                    # possible value checks (with current implementation should be OK by construction)
                    vars_not_possible = design_space[self.VARIABLES][
                            ~design_space[self.VARIABLES].apply(lambda _var: _var in self.eval_in_possible_types)].to_list()
                    for var_not_possible in vars_not_possible:
                        design_space_integrity_msg.append(
                            f'Variable {var_not_possible} is not among the possible input values.'
                        )
                # check of dimensions coherences
                wrong_dim_vars = design_space[self.VARIABLES][
                    ~design_space.apply(self._check_design_space_dimensions_for_one_variable, axis=1)].to_list()
                for wrong_dim_var in wrong_dim_vars:
                    design_space_integrity_msg.append(
                        f'Columns {self.LOWER_BOUND}, {self.UPPER_BOUND} and {self.VALUES} should be of type '
                        f'{self.eval_in_possible_types[wrong_dim_var]} for variable {wrong_dim_var} '
                        f'and should have coherent shapes.')
                if self.NB_POINTS in design_space.columns:
                    wrong_nb_points_vars = design_space[self.VARIABLES][~design_space[self.NB_POINTS].apply(
                        lambda _nb_pts: isinstance(_nb_pts, self.VAR_TYPE_MAP['int']) and _nb_pts >= 0)
                    ].to_list()
                    if wrong_nb_points_vars:
                        design_space_integrity_msg.append(
                            f'Column {self.NB_POINTS} should contain non-negative integers only.')
        if design_space_integrity_msg:
            self.sg_data_integrity = False
            self.ee.dm.set_data(self.get_var_full_name(self.DESIGN_SPACE, disc_in),
                                self.CHECK_INTEGRITY_MSG, '\n'.join(design_space_integrity_msg))

    def is_configured(self):
        """
        Configuration criterion including whether a configuration-time sample is pending.
        """
        return super().is_configured() and not self.sample_pending

    def setup_sos_disciplines(self):
        """
        Dynamic i/o of the sample generator.
        """
        disc_in = self.get_data_in()
        if disc_in:
            self.sampling_method = self.get_sosdisc_inputs(self.SAMPLING_METHOD)
            self.sampling_generation_mode = self.configure_generation_mode(disc_in)
            self.instantiate_sampling_tool()
            self.update_eval_inputs(disc_in)
            dynamic_inputs, dynamic_outputs = self.mdo_discipline_wrapp.wrapper.sample_generator.setup(self)
            
            self.check_data_integrity()
            if self.sampling_generation_mode == self.AT_RUN_TIME:
                # if sampling at run-time add the corresponding output
                dynamic_outputs[self.SAMPLES_DF] = self.SAMPLES_DF_DESC_SHARED.copy()
                self.all_input_structuring = False
            elif self.sampling_generation_mode == self.AT_CONFIGURATION_TIME:
                dynamic_inputs.update({
                    self.OVERWRITE_SAMPLES_DF: self.OVERWRITE_SAMPLES_DF_DESC.copy(),
                    # self.SAMPLES_DF: self.SAMPLES_DF_DESC_SHARED.copy(),
                })
                # if sampling is at config-time, set all input structuring and add samples_df input
                self.all_input_structuring = True
                self.sample_at_configuration_time(disc_in)

            # TODO: manage config-time sample for grid search and test for DoE as well as coupled run-time sampling for CP
            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def instantiate_sampling_tool(self):
        """
           Instantiate a new SampleGenerator only if needed
        """
        if self.sampling_method is not None:
            if self.sampling_method in self.AVAILABLE_SAMPLING_METHODS:
                sample_generator_cls = self.SAMPLE_GENERATOR_CLS[self.sampling_method]
                if self.mdo_discipline_wrapp.wrapper.sample_generator.__class__ != sample_generator_cls:
                    self.mdo_discipline_wrapp.wrapper.sample_generator = sample_generator_cls(logger=self.logger.getChild(sample_generator_cls.__name__))

    def configure_generation_mode(self, disc_in):
        """
        Return a generation mode at_configuration_time/at_run_time taking into account editability constraints and user
        input. Set the editability of the corresponding variables.
        """
        sampling_generation_mode = self.get_sosdisc_inputs(self.SAMPLING_GENERATION_MODE)
        # variable needs to be made non-editable for special cases namely: simple_sample_generator => at config. time,
        # self.force_sampling_at_config_time (i.e. working with multi-instance driver) => at config. time
        forced_methods_modes = {
            self.SIMPLE_SAMPLING_METHOD: self.AT_CONFIGURATION_TIME
        } if not self.force_sampling_at_configuration_time else {
            k: self.AT_CONFIGURATION_TIME for k in self.AVAILABLE_SAMPLING_METHODS
        }
        if self.sampling_method in forced_methods_modes:
            disc_in[self.SAMPLING_GENERATION_MODE][self.EDITABLE] = False
            expected_mode = forced_methods_modes[self.sampling_method]
            if sampling_generation_mode != expected_mode:
                # TODO: discuss warnings and exception handlings
                # warn and force config time sampling
                self.logger.warning(f'Forcing {self.SAMPLING_GENERATION_MODE} to {expected_mode} for '
                                    f'{self.sampling_method} {self.SAMPLING_METHOD}.')
                disc_in[self.SAMPLING_GENERATION_MODE][self.VALUE] = sampling_generation_mode = expected_mode
        else:
            disc_in[self.SAMPLING_GENERATION_MODE][self.EDITABLE] = True
        return sampling_generation_mode

    def update_eval_inputs(self, disc_in):
        """
        Method to update the dataframe descriptor and columns of eval_inputs depending on the sampling method and handle
        whether the column with variable names is editable (i.e. not when these are set by the driver as configurator).

        Arguments:
            disc_in (dict): input data dict of the discipline obtained via self.get_data_in()
        """
        _df_desc = None
        # build right dataframe descriptor
        if self.sampling_method == self.CARTESIAN_PRODUCT:
            _df_desc = self.EVAL_INPUTS_CP_DF_DESC.copy()
        elif self.sampling_method in self.AVAILABLE_SAMPLING_METHODS:
            _df_desc = self.EVAL_INPUTS_DF_DESC.copy()
        if _df_desc:
            # handle editability of the dataframe column with variable names when these are set by the driver
            if self.configurator:
                _df_desc[self.FULL_NAME] = ('string', None, False)
            self._update_eval_inputs_columns(_df_desc, disc_in)
        self.mdo_discipline_wrapp.wrapper.sample_generator.filter_inputs(self)
        self._update_eval_inputs_with_possible_values(disc_in)

    def _update_eval_inputs_columns(self, eval_inputs_df_desc, disc_in=None):
        """
        Method to update eval_inputs dataframe descriptor and variable columns in accordance when the first changes
        (i.e. when changing sampling_method).

        Arguments:
            eval_inputs_df_desc (dict): dataframe descriptor to impose
            disc_in (dict): the discipline inputs dict (to avoid an extra call to self.get_data_in())
        """
        # get the data_in only if not provided
        d_in = disc_in or self.get_data_in()
        if self.EVAL_INPUTS in d_in:
            eval_inputs_f_name = self.get_var_full_name(self.EVAL_INPUTS, d_in)
            # update dataframe descriptor
            self.dm.set_data(eval_inputs_f_name,
                             self.DATAFRAME_DESCRIPTOR,
                             eval_inputs_df_desc,
                             check_value=False)
            # update variable value with corresponding columns
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            if eval_inputs is not None:
                eval_inputs = eval_inputs.reindex(columns=eval_inputs_df_desc.keys(),
                                                  fill_value=[])  # hardcoded compliance with 'list_of_values' column default
                self.dm.set_data(eval_inputs_f_name,
                                 self.VALUE,
                                 eval_inputs,
                                 check_value=False)

    def _update_eval_inputs_with_possible_values(self, disc_in):
        """
        Method to update eval_inputs['full_name'] column with the subprocess possible inputs as set by the configurator
        using set_eval_in_possible_values method.

        Arguments:
            disc_in (dict): the discipline inputs dict (to avoid an extra call to self.get_data_in())
        """
        if self.eval_in_possible_values and self.EVAL_INPUTS in disc_in:
            default_in_dataframe = pd.DataFrame({self.SELECTED_INPUT: [False for _ in self.eval_in_possible_values],
                                                 self.FULL_NAME: self.eval_in_possible_values})
            eval_input_new_dm = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            eval_inputs_f_name = self.get_var_full_name(self.EVAL_INPUTS, disc_in)
            if eval_input_new_dm is None:
                self.dm.set_data(eval_inputs_f_name,
                                 'value', default_in_dataframe, check_value=False)
            # check if the eval_inputs need to be updated after a subprocess input change
            elif set(eval_input_new_dm[self.FULL_NAME].tolist()) != (set(default_in_dataframe[self.FULL_NAME].tolist())):
                # reindex eval_inputs to the possible values keeping other values and columns of the df
                eval_input_new_dm = eval_input_new_dm.\
                    drop_duplicates(self.FULL_NAME).set_index(self.FULL_NAME).reindex(self.eval_in_possible_values).\
                    reset_index().reindex(columns=eval_input_new_dm.columns)
                eval_input_new_dm[self.SELECTED_INPUT] = eval_input_new_dm[self.SELECTED_INPUT].fillna(False).astype('bool')
                # manage the empty lists on column list_of_values (as df.fillna([]) will not work)
                if self.LIST_OF_VALUES in eval_input_new_dm.columns:
                    new_in = eval_input_new_dm[self.LIST_OF_VALUES].isna()
                    eval_input_new_dm.loc[new_in, self.LIST_OF_VALUES] = pd.Series([[]] * new_in.sum()).values
                self.dm.set_data(eval_inputs_f_name,
                                 'value', eval_input_new_dm, check_value=False)

    def prepare_execution(self):
        """
        Overload of the prepare_execution allowing to instantiate a gemseo object only if the sampling generation is at
        run-time.
        """
        if self.sampling_generation_mode == self.AT_RUN_TIME:
            super().prepare_execution()
        else:
            # Here self.mdo_discipline_wrapp.wrapper exists during configuration, but it is not associated to any gemseo
            # object during execution (self.mdo_discipline_wrapp.mdo_discipline is None).
            self._update_status_dm(self.STATUS_DONE)

    def _get_non_structuring_variables_keys(self):
        # Configuration-time sampling uses the flag self.all_variables_are_structuring.
        # Here we exclude samples_df from the non-structuring variables that are made structuring when sampling at
        # configuration-time. This avoids resampling when some scenarios are edited on the driver after a 1st sampling
        return super()._get_non_structuring_variables_keys() - {self.SAMPLES_DF}

    def sample_at_configuration_time(self, disc_in):
        """
        Method used to ask the sample generator to sample and push the samples_df into the data manager when a sampling
        at configuration time is performed.
        """
        # is_ready_to_sample is similar to data_integrity except that no error is logged (mainly for incomplete config.)
        if self.sg_data_integrity and self.mdo_discipline_wrapp.wrapper.sample_generator.is_ready_to_sample(self):
            if self.OVERWRITE_SAMPLES_DF in disc_in:
                # if self.samples_df_f_name:
                samples_df_dm = self.dm.get_value(self.samples_df_f_name)
                # avoid sampling and pushing the generated samples_df into the dm, UNLESS:
                # - the current scenario names are the default (i.e. no previous modification or sampling made), or
                # - the user asked to force re-sampling on reconfiguration using input flag overwrite_samples_df
                overwrite_samples_df =\
                    samples_df_dm[self.SCENARIO_NAME].equals(self.SAMPLES_DF_DEFAULT[self.SCENARIO_NAME]) or \
                    self.get_sosdisc_inputs(self.OVERWRITE_SAMPLES_DF)
                if overwrite_samples_df:
                    try:
                        self.samples_gene_df = self.mdo_discipline_wrapp.wrapper.sample()
                    except ValueError as cm:  # TODO: larger clause ?
                        self.samples_gene_df = None
                        self.logger.error('Failed to sample due to ' + str(cm))
                    if self.samples_gene_df is not None and not self.samples_gene_df.empty:
                        self.max_auto_select_scenarios_warning()
                        self.dm.set_data(self.samples_df_f_name,
                                         self.VALUE, self.samples_gene_df, check_value=False)
                        self.dm.set_data(self.get_var_full_name(self.OVERWRITE_SAMPLES_DF, disc_in),
                                         self.VALUE, False, check_value=False)
                self.sample_pending = False
            else:
                self.sample_pending = True

    def max_auto_select_scenarios_warning(self):
        """
        In the scope of a sampling at configuration time, before pushing to dm a generated sample too big, potentially
        engaging in a long configuration or run, warn user and ask to manually select which scenarios are to be built
        and/or evaluated. Note that otherwise after a sampling all scenarios are selected automatically, which needs to
        be the case for run-time sampling to work normally.
        """
        if self.MAX_AUTO_SELECT_SCENARIOS is not None and len(self.samples_gene_df) > self.MAX_AUTO_SELECT_SCENARIOS:
            self.samples_gene_df[self.SELECTED_SCENARIO] = False
            self.logger.warning(
                f'Sampled over {self.MAX_AUTO_SELECT_SCENARIOS} scenarios, please select manually which ones are'
                f'to be built and/or evaluated using input samples dataframe ({self.SELECTED_SCENARIO} column). '
                f'For a large number of scenarios, a mono-instance driver with sample generation at run-time is advised.')
