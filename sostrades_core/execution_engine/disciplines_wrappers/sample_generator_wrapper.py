'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/08 Copyright 2023 Capgemini

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
import logging

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.sample_generators.simple_sample_generator import SimpleSampleGenerator
from sostrades_core.execution_engine.sample_generators.grid_search_sample_generator import GridSearchSampleGenerator
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import \
    CartesianProductSampleGenerator
import pandas as pd


class SampleGeneratorWrapper(SoSWrapp):
    # TODO: docstring is not up to date
    '''
    Generic SampleGenerator class
    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SAMPLING_METHOD (structuring)
                |_ EVAL_INPUTS (namespace: 'ns_sampling', structuring, dynamic : SAMPLING_METHOD == self.DOE_ALGO)
                        |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO != None) NB: default DESIGN_SPACE depends on EVAL_INPUTS (Has to be "Not empty")
                |_ SAMPLING_ALGO (structuring, dynamic : SAMPLING_METHOD == self.DOE_ALGO)
                        |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
                            |_ GENERATED_SAMPLES (namespace: 'ns_sampling', structuring,
                                                 dynamic: EVAL_INPUTS_CP != None and ALGO_OPTIONS set
                                                     and SAMPLING_GENERATION_MODE == 'at_configuration_time')
                |_ EVAL_INPUTS_CP (namespace: 'ns_sampling', structuring, dynamic : SAMPLING_METHOD == self.CARTESIAN_PRODUCT)
                        |_ GENERATED_SAMPLES (namespace: 'ns_sampling', structuring,
                                               dynamic: EVAL_INPUTS_CP != None and SAMPLING_GENERATION_MODE == 'at_configuration_time')
            |_ SAMPLING_GENERATION_MODE ('editable': False)
        |_ DESC_OUT
            |_ SAMPLES_DF (namespace: 'ns_sampling')

    2) Description of DESC parameters:
        |_ DESC_IN
            |_ SAMPLING_METHOD
                |_ EVAL_INPUTS
                        |_ DESIGN_SPACE
                |_ SAMPLING_ALGO
                        |_ ALGO_OPTIONS
                            |_ GENERATED_SAMPLES
                |_ EVAL_INPUTS_CP
                        |_ GENERATED_SAMPLES
        |_ DESC_OUT
            |_ SAMPLES_DF

    '''

    _ontology_data = {
        'label': 'Sample_Generator wrapper',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'Sample_Generator wrapper that implements the genearotion of a samples from a DoE (Design of Experiment) algorithm or from a cross product.',
        # icon for sample generator from
        # https://fontawesome.com/icons/grid-4?s=regular&f=classic
        'icon': 'fas fa-grid-4 fa-fw',
        'version': ''
    }

    POSSIBLE_VALUES = 'possible_values'
    TYPE = "type"

    NS_SEP = '.'
    INPUT_TYPE = ['float', 'array', 'int', 'string']

    DESIGN_SPACE = "design_space"
    DIMENSION = "dimension"

    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"


    N_SAMPLES = "n_samples"
    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    USER_GRAD = 'user'

    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    SAMPLING_METHOD = 'sampling_method'
    SIMPLE_SAMPLING_METHOD = 'simple'
    DOE_ALGO = 'doe_algo'
    CARTESIAN_PRODUCT = 'cartesian_product'
    GRID_SEARCH = 'grid_search'
    FULLFACT = 'fullfact'
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

    SAMPLES_DF = 'samples_df'
    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'
    NS_SAMPLING = 'ns_sampling'
    REFERENCE_SCENARIO_NAME = 'Reference Scenario'
    SAMPLES_DF_DESC = {
        SoSWrapp.TYPE: 'dataframe',
        SoSWrapp.DEFAULT: pd.DataFrame({SELECTED_SCENARIO: [True],
                                        SCENARIO_NAME: REFERENCE_SCENARIO_NAME}),
        SoSWrapp.DATAFRAME_DESCRIPTOR: {SELECTED_SCENARIO: ('bool', None, True),
                                        SCENARIO_NAME: ('string', None, True)},
        SoSWrapp.DYNAMIC_DATAFRAME_COLUMNS: True,
        SoSWrapp.DATAFRAME_EDITION_LOCKED: False,
        SoSWrapp.EDITABLE: True,
        SoSWrapp.STRUCTURING: False
    }
    SAMPLES_DF_DESC_SHARED = SAMPLES_DF_DESC.copy()
    SAMPLES_DF_DESC_SHARED[SoSWrapp.NAMESPACE]=NS_SAMPLING
    SAMPLES_DF_DESC_SHARED[SoSWrapp.VISIBILITY]=SoSWrapp.SHARED_VISIBILITY

    EVAL_INPUTS = 'eval_inputs'
    EVAL_INPUTS_DF_DESC = {'selected_input': ('bool', None, True),
                           'full_name': ('string', None, False)} # FIXME: should not be non-editable in standalone sample
    EVAL_INPUTS_DESC = {SoSWrapp.TYPE: 'dataframe',
                        SoSWrapp.DATAFRAME_DESCRIPTOR: EVAL_INPUTS_DF_DESC.copy(),
                        SoSWrapp.DATAFRAME_EDITION_LOCKED: False,
                        SoSWrapp.STRUCTURING: True,
                        SoSWrapp.DEFAULT: pd.DataFrame(columns=['selected_input', 'full_name']),
                        SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                        SoSWrapp.NAMESPACE: NS_SAMPLING,
                        }

    EVAL_INPUTS_CP_DF_DESC = EVAL_INPUTS_DF_DESC.copy()
    EVAL_INPUTS_CP_DF_DESC.update({'list_of_values': ('list', None, True)})

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

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name=sos_name, logger=logger)
        self.sampling_method = None
        self.sampling_generation_mode = None
        self.sample_generator = None  # sample generator tool

        # todo KEEP?
        self.samples_gene_df = None
        # todo: generalise management of sample_pending when decoupling sampling from setup
        self.sample_pending = False

    def setup_sos_disciplines(self):
        '''
        Overload of setup_sos_disciplines to specify the specific dynamic inputs of sample generator
        '''

        disc_in = self.get_data_in()
        if disc_in:
            self.sampling_method = self.get_sosdisc_inputs(self.SAMPLING_METHOD)
            self.sampling_generation_mode = self.configure_generation_mode(disc_in)
            self.instantiate_sampling_tool()
            self.update_eval_inputs_columns(disc_in)
            dynamic_inputs, dynamic_outputs = self.sample_generator.setup(self) # TODO: separate the sample generation from setup

            # 4. if sampling at run-time add the corresponding output
            if self.sampling_generation_mode == self.AT_RUN_TIME:
                dynamic_outputs[self.SAMPLES_DF] = self.SAMPLES_DF_DESC_SHARED.copy()
            # elif self.sampling_generation_mode == self.AT_CONFIGURATION_TIME: # TODO: separate the sample generation from setup
            #     self.sample_at_config_time()
            # TODO: manage config-time sample for grid search and test for DoE

            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

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

    def instantiate_sampling_tool(self):
        """
           Instantiate a new SampleGenerator only if needed
        """
        if self.sampling_method is not None:
            if self.sampling_method in self.AVAILABLE_SAMPLING_METHODS:
                sample_generator_cls = self.SAMPLE_GENERATOR_CLS[self.sampling_method]
                if self.sample_generator.__class__ != sample_generator_cls:
                    self.sample_generator = sample_generator_cls(logger=self.logger.getChild(sample_generator_cls.__name__))

    def run(self):
        if self.sampling_generation_mode == self.AT_RUN_TIME: # TODO: soon to be (non-)instantiation of the GEMSEO object
            samples_df = self.sample_generator.sample(self)
            # TODO: rethink management of this
            samples_df = self.set_scenario_columns(samples_df)
            # TODO: is this the place for this exception ?
            if isinstance(samples_df, pd.DataFrame):
                pass
            else:
                raise Exception(
                    f"Sampling has not been made")
            self.store_sos_outputs_values({self.SAMPLES_DF: samples_df})

    # TODO: maybe move to AbstractSampleGenerator ?
    def set_scenario_columns(self, samples_df, scenario_names=None):
        '''
        Add the columns SELECTED_SCENARIO and SCENARIO_NAME to the samples_df dataframe
        '''
        if self.SELECTED_SCENARIO not in samples_df:
            ordered_columns = [self.SELECTED_SCENARIO, self.SCENARIO_NAME] + samples_df.columns.tolist()
            if scenario_names is None:
                samples_df[self.SCENARIO_NAME] = [f'scenario_{i}' for i in range(1, len(samples_df) + 1)]
            else:
                samples_df[self.SCENARIO_NAME] = scenario_names
            samples_df[self.SELECTED_SCENARIO] = [True] * len(samples_df)
            samples_df = samples_df[ordered_columns]
        return samples_df

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
            # TODO: when moving to proxy implement -> if self.configurator: df_desc['full_name'] non-editable else editable
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
        if self.sampling_method == self.CARTESIAN_PRODUCT:
            self._update_eval_inputs_columns(self.EVAL_INPUTS_CP_DF_DESC.copy(), disc_in)
        elif self.sampling_method in self.AVAILABLE_SAMPLING_METHODS:
            self._update_eval_inputs_columns(self.EVAL_INPUTS_DF_DESC.copy(), disc_in)

    def is_configured(self):
        return not self.sample_pending
