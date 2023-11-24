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

    # def setup(self, proxy):
    #     # 1. handle dynamic inputs of the mode
    #     dynamic_inputs, dynamic_outputs = {}, {}
    #     disc_in = proxy.get_data_in()
    #
    #     dynamic_inputs.update({proxy.SAMPLES_DF: proxy.SAMPLES_DF_DESC_SHARED.copy()})
    #
    #     # FIXME: refacto with a call to self.sample (when modifying setup_sos_disciplines)
    #     # 2. retrieve input that configures the sampling tool
    #     if proxy.EVAL_INPUTS in disc_in and proxy.SAMPLES_DF in disc_in:
    #         samples_df = proxy.get_sosdisc_inputs(proxy.SAMPLES_DF)
    #         eval_inputs = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
    #         if eval_inputs is not None and samples_df is not None:
    #             selected_inputs = eval_inputs[eval_inputs['selected_input'] == True]['full_name'].tolist()
    #             if selected_inputs:
    #                 # 3. if sampling at config.time set the generated samples
    #                 proxy.samples_gene_df = self.generate_samples(samples_df, selected_inputs)
    #                 proxy.dm.set_data(proxy.get_var_full_name(proxy.SAMPLES_DF, disc_in),
    #                                   proxy.VALUE, proxy.samples_gene_df, check_value=False)
    #     return dynamic_inputs, dynamic_outputs

    def is_ready_to_sample(self, proxy):
        disc_in = proxy.get_data_in()
        return proxy.EVAL_INPUTS in disc_in and \
            proxy.SAMPLES_DF in disc_in and \
            proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS) is not None and \
            proxy.get_sosdisc_inputs(proxy.SAMPLES_DF) is not None

    def get_arguments(self, proxy):
        eval_inputs = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
        samples_df = proxy.get_sosdisc_inputs(proxy.SAMPLES_DF)
        selected_inputs = eval_inputs[eval_inputs['selected_input'] == True]['full_name'].tolist()
        simple_kwargs = {'samples_df': samples_df, 'var_names': selected_inputs}
        return [], simple_kwargs