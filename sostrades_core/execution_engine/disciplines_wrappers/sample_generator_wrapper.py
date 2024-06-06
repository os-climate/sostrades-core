'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/05/16 Copyright 2023 Capgemini

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

import pandas as pd

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.design_space import design_space as dspace_tool


class SampleGeneratorWrapper(SoSWrapp):
    # TODO: docstring is not up to date
    '''
    SampleGeneratorWrapper for ProxySampleGenerator discipline sampling at run-time.
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
    # eval_inputs and columns
    EVAL_INPUTS = 'eval_inputs'
    SELECTED_INPUT = 'selected_input'
    FULL_NAME = 'full_name'
    LIST_OF_VALUES = 'list_of_values'
    # samples_df and columns
    SAMPLES_DF = 'samples_df'
    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'
    # TODO: move to doe tool ?
    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    DESIGN_SPACE = dspace_tool.DESIGN_SPACE

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name=sos_name, logger=logger)
        self.sample_generator = None  # sample generator tool

    def run(self):
        samples_df = self.sample()
        # TODO [to discuss]: is this the place for this exception ?
        if isinstance(samples_df, pd.DataFrame):
            pass
        else:
            raise Exception( "Sampling has not been made")
        self.store_sos_outputs_values({self.SAMPLES_DF: samples_df})

    def sample(self):
        """
        Ask sample generator to sample using wrapper object to retrieve inputs.
        """
        return self.set_scenario_columns(self.sample_generator.sample(self))

    def set_scenario_columns(self, samples_df, scenario_names=None):
        # TODO: [to discuss] move to AbstractSampleGenerator ?
        '''
        Add the columns SELECTED_SCENARIO and SCENARIO_NAME to the samples_df, by default selecting all scenarios.
        '''
        if self.SELECTED_SCENARIO not in samples_df:
            ordered_columns = [self.SELECTED_SCENARIO, self.SCENARIO_NAME] + samples_df.columns.tolist()
            if samples_df.empty:
                # this return avoids adding a spurious scenario_1 in case of empty sample at run-time
                return pd.DataFrame(columns=ordered_columns)
            elif scenario_names is None:
                samples_df[self.SCENARIO_NAME] = [f'scenario_{i}' for i in range(1, len(samples_df) + 1)]
            else:
                samples_df[self.SCENARIO_NAME] = scenario_names
            samples_df[self.SELECTED_SCENARIO] = [True] * len(samples_df)
            samples_df = samples_df[ordered_columns]
        return samples_df
