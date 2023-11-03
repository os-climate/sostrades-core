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
from builtins import NotImplementedError

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import AbstractSampleGenerator,\
    SampleTypeError

import pandas as pd
import numpy as np

import itertools

import logging
LOGGER = logging.getLogger(__name__)

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class SimpleSampleGeneratorTypeError(SampleTypeError):
    pass


class SimpleSampleGenerator(AbstractSampleGenerator):
    '''
    Caresian Product class that generates sampling
    '''
    GENERATOR_NAME = "SIMPLE_SAMPLE_GENERATOR"

    def __init__(self, logger: logging.Logger):
        '''
        Constructor
        '''
        super().__init__(self.GENERATOR_NAME, logger=logger)

    def _generate_samples(self, samples_df, var_names):
        '''
        Method that only modifies the columns of samples_df based on var_names, removing the necessary columns and adding
        empty columns as to have var_names as trade variables.

        Arguments:
            samples_df (dataframe): input samples_df to modify
            var_names (list[string]): trade variables to become column name list
        Returns:
            samples_df (dataframe) : generated samples with appropriate columns
        '''
        return samples_df.reindex(columns=samples_df.columns[:2].tolist() + var_names)

    def _check_samples(self, samples_df):
        '''
        Method that checks the sample output type
        Arguments:
            samples_df (dataframe) : generated samples
        Raises:
            Exception if samples_df is not a dataframe
        '''
        if not(isinstance(samples_df, pd.DataFrame)):
            msg = "Expected sampling output type should be pandas.core.frame.DataFrame"
            msg += "however sampling type of sampling generator <%s> " % str(
                self.__class__.__name__)
            msg += "is <%s> " % str(type(samples_df))
            raise SimpleSampleGeneratorTypeError()
