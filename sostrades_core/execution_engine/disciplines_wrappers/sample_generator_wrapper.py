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
    SAMPLES_DF = 'samples_df'
    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'
    # TODO: move to doe tool ?
    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    DESIGN_SPACE = "design_space"

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name=sos_name, logger=logger)
        # self.sampling_method = None
        # self.sampling_generation_mode = None
        self.sample_generator = None  # sample generator tool

        # todo KEEP?
        self.samples_gene_df = None

    def run(self):
        samples_df = self.sample_generator.sample(self)
        # TODO: rethink management of this below (see todo set_scenario_columns)
        samples_df = self.set_scenario_columns(samples_df)
        # TODO: is this the place for this exception ?
        if isinstance(samples_df, pd.DataFrame):
            pass
        else:
            raise Exception(
                f"Sampling has not been made")
        self.store_sos_outputs_values({self.SAMPLES_DF: samples_df})

    # TODO: maybe move to AbstractSampleGenerator ? or render private then create a SampleGeneratorWrapper.sample() method calling this one?
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
