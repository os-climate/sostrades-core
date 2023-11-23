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

    NS_SAMPLING = 'ns_sampling'
    SAMPLES_DF_DESC = {
        ProxyDiscipline.TYPE: 'dataframe',
        ProxyDiscipline.DEFAULT: pd.DataFrame({SELECTED_SCENARIO: [True],
                                               SCENARIO_NAME: REFERENCE_SCENARIO_NAME}),
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

    # TODO: 'full_name', 'selected_input' etc. class variables which probably belong in the tools rather than here.
    EVAL_INPUTS = 'eval_inputs'
    EVAL_INPUTS_DF_DESC = {'selected_input': ('bool', None, True),
                           'full_name': ('string', None, True)}
    EVAL_INPUTS_DESC = {ProxyDiscipline.TYPE: 'dataframe',
                        ProxyDiscipline.DATAFRAME_DESCRIPTOR: EVAL_INPUTS_DF_DESC.copy(),
                        ProxyDiscipline.DATAFRAME_EDITION_LOCKED: False,
                        ProxyDiscipline.STRUCTURING: True,
                        ProxyDiscipline.DEFAULT: pd.DataFrame(columns=['selected_input', 'full_name']),
                        ProxyDiscipline.VISIBILITY: ProxyDiscipline.SHARED_VISIBILITY,
                        ProxyDiscipline.NAMESPACE: NS_SAMPLING,
                        }
    EVAL_INPUTS_CP_DF_DESC = EVAL_INPUTS_DF_DESC.copy()
    EVAL_INPUTS_CP_DF_DESC.update({'list_of_values': ('list', None, True)})

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

        self.sampling_method = None
        self.sampling_generation_mode = None
        # TODO: generalise management of self.sample_pending when decoupling sampling from setup
        self.sample_pending = False
        # sample generated at configuration-time
        self.samples_gene_df = None


    def set_eval_in_possible_values(self, possible_values: list[str]) -> bool:
        """
        Method used by a driver in composition with a sample generator to pass the set of inputs of the subprocess
        that can be selected in eval_inputs.

        Arguments:
            possible_values (list(string)): possible values of the eval_inputs variable names
        Returns:
             driver_is_configured (bool): flag to detect whether driver could ask sample generator for necessary
                configuration actions
        """
        driver_is_configured = True
        # TODO: might want to refactor this eventually. If so, take into account that this "driver_is_configured" flag
        #  is a quick fix. The proper way is probably as follows: in this method just set the attribute eval_in_possible_values
        #  and handle SampleGenerator configuration status if it has changed. Then in SampleGenerator configuration do the
        #  remaining actions in the code below (set eval_inputs and handle corresponding samples_df columns update).
        if possible_values:
            driver_is_configured = False
            disc_in = self.get_data_in()
            if self.EVAL_INPUTS in disc_in:
                driver_is_configured = True
                default_in_dataframe = pd.DataFrame({'selected_input': [False for _ in possible_values],
                                                     'full_name': possible_values})
                eval_input_new_dm = self.get_sosdisc_inputs(self.EVAL_INPUTS)
                eval_inputs_f_name = self.get_var_full_name(self.EVAL_INPUTS, disc_in)

                if eval_input_new_dm is None:
                    self.dm.set_data(eval_inputs_f_name,
                                     'value', default_in_dataframe, check_value=False)
                # check if the eval_inputs need to be updated after a subprocess
                # configureon s'ap
                elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
                    error_msg = check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
                                       is_eval_input=True)
                    for msg in error_msg:
                        self.logger.warning(msg)

                    # reindex eval_inputs to the possible values keeping other values and columns of the df
                    eval_input_new_dm = eval_input_new_dm.\
                        drop_duplicates('full_name').set_index('full_name').reindex(possible_values).\
                        reset_index().reindex(columns=eval_input_new_dm.columns)
                    eval_input_new_dm['selected_input'] = eval_input_new_dm['selected_input'].fillna(False).astype('bool')
                    # manage the empty lists on column list_of_values (as df.fillna([]) will not work)
                    if 'list_of_values' in eval_input_new_dm.columns:
                        new_in = eval_input_new_dm['list_of_values'].isna()
                        eval_input_new_dm.loc[new_in, 'list_of_values'] = pd.Series([[]] * new_in.sum()).values

                    self.dm.set_data(eval_inputs_f_name,
                                     'value', eval_input_new_dm, check_value=False)

                selected_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
                selected_inputs = selected_inputs[selected_inputs['selected_input'] == True]['full_name'].tolist()
                all_columns = [self.SELECTED_SCENARIO,
                               self.SCENARIO_NAME] + selected_inputs
                default_custom_dataframe = pd.DataFrame(
                    [[None for _ in range(len(all_columns))]], columns=all_columns)
                dataframe_descriptor = self.SAMPLES_DF_DESC_SHARED['dataframe_descriptor'].copy()
                # This reflects 'samples_df' dynamic input has been configured and that
                # eval_inputs have changed
                if self.SAMPLES_DF in disc_in:
                    final_dataframe = pd.DataFrame(None, columns=all_columns)
                    samples_df = self.get_sosdisc_inputs(self.SAMPLES_DF)
                    if samples_df is not None:
                        from_samples = list(samples_df.keys())
                        from_eval_inputs = list(default_custom_dataframe.keys())
                        len_df = 1
                        for element in from_eval_inputs:
                            if element in from_samples:
                                len_df = len(samples_df)

                        for element in from_eval_inputs:
                            if element in from_samples:
                                final_dataframe[element] = samples_df[element]
                                dataframe_descriptor[element] = ('multiple', None, True)
                                # TODO: dataframe descriptor should be corrected by driver based on samples_df so that
                                #  it can properly work in standalone driver. Currently multi-instance driver does not
                                #  have the mechanism..
                            else:
                                final_dataframe[element] = [None for _ in range(len_df)]
                                dataframe_descriptor[element] = ('multiple', None, True)
                    samples_df_f_name = self.get_var_full_name(self.SAMPLES_DF, disc_in)
                    self.dm.set_data(samples_df_f_name, self.VALUE, final_dataframe, check_value=False)
                    self.dm.set_data(samples_df_f_name, self.DATAFRAME_DESCRIPTOR, dataframe_descriptor, check_value=False)
                elif self.get_sosdisc_inputs(self.SAMPLING_GENERATION_MODE) == self.AT_CONFIGURATION_TIME:
                    driver_is_configured = False
        return driver_is_configured

    # TODO: refactor all below to assure the attributes are in good place (proxy/wrapper/tool)
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
            self.update_eval_inputs_columns(disc_in)
            dynamic_inputs, dynamic_outputs = self.mdo_discipline_wrapp.wrapper.sample_generator.setup(
                self)  # TODO: separate the sample generation from setup

            # 4. if sampling at run-time add the corresponding output
            if self.sampling_generation_mode == self.AT_RUN_TIME:
                dynamic_outputs[self.SAMPLES_DF] = self.SAMPLES_DF_DESC_SHARED.copy()
                self.all_input_structuring = False
            elif self.sampling_generation_mode == self.AT_CONFIGURATION_TIME:
                self.all_input_structuring = True
                # self.sample_at_config_time()  # TODO: separate the sample generation from setup
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
        sampling_generation_mode = self.get_sosdisc_inputs(self.SAMPLING_GENERATION_MODE)
        # variable needs to be made non-editable for special cases (namely simple_sample_generator => at config. time)
        forced_methods_modes = {
            self.SIMPLE_SAMPLING_METHOD: self.AT_CONFIGURATION_TIME
        }
        if self.sampling_method in forced_methods_modes:
            disc_in[self.SAMPLING_GENERATION_MODE][self.EDITABLE] = False
            expected_mode = forced_methods_modes[self.sampling_method]
            if sampling_generation_mode != expected_mode:
                # TODO: discuss and review exception handlings
                # warn and force config time sampling
                self.logger.warning(f'Setting {self.SAMPLING_GENERATION_MODE} to {expected_mode} for '
                                    f'{self.sampling_method} {self.SAMPLING_METHOD}.')
                disc_in[self.SAMPLING_GENERATION_MODE][self.VALUE] = sampling_generation_mode = expected_mode
        else:
            disc_in[self.SAMPLING_GENERATION_MODE][self.EDITABLE] = True
        return sampling_generation_mode

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

    def update_eval_inputs_columns(self, disc_in):
        """
        Method to update the dataframe descriptor and columns of eval_inputs depending on the sampling method and handle
        whether the column with variable names is editable (i.e. not when these are set by the driver as configurator).

        Arguments:
            disc_in (dict): input data dict of the discipline obtained via self.get_data_in()
        """
        _df_desc = None
        if self.sampling_method == self.CARTESIAN_PRODUCT:
            _df_desc = self.EVAL_INPUTS_CP_DF_DESC.copy()
        elif self.sampling_method in self.AVAILABLE_SAMPLING_METHODS:
            _df_desc = self.EVAL_INPUTS_DF_DESC.copy()
        if _df_desc:
            # handle editability of the dataframe column with variable names when these are set by the driver
            if self.configurator:
                _df_desc['full_name'] = ('string', None, False)
            self._update_eval_inputs_columns(_df_desc, disc_in)

    def prepare_execution(self):
        """
        Overload of the prepare_execution allowing to instantiate a gemseo object only if the sampling generation is at
        run-time.
        """
        if self.sampling_generation_mode == self.AT_RUN_TIME:
            super().prepare_execution()
        else:
            # Note that self.mdo_discipline_wrapp.wrapper exists but self.mdo_discipline_wrapp.mdo_discipline is None
            self._update_status_dm(self.STATUS_DONE)

    # TODO: DISCUSS IMPLEMENTATION
    def _get_non_structuring_variables_keys(self):
        # need to exclude samples_df to avoid config-time resampling when scenarios are edited on driver after sampling
        return super()._get_non_structuring_variables_keys() - {self.SAMPLES_DF}

    def set_sample(self):  # TODO: check implementation when splitting sampling at config-time from setup
        self.samples_gene_df = self.mdo_discipline_wrapp.wrapper.sample()
        # self.samples_gene_df = self.mdo_discipline_wrapp.wrapper.set_scenario_columns(
        #     self.mdo_discipline_wrapp.wrapper.sample(self))


