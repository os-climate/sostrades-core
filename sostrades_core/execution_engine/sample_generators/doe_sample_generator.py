'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/17-2024/07/04 Copyright 2023 Capgemini

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

import logging
from collections import ChainMap
from typing import Any

import pandas as pd
from gemseo import get_available_doe_algorithms
from gemseo.algos.base_driver_settings import BaseDriverSettings
from gemseo.algos.doe.factory import DOELibraryFactory

from sostrades_core.execution_engine.sample_generators.abstract_sample_generator import (
    AbstractSampleGenerator,
    SampleTypeError,
)
from sostrades_core.tools.design_space import design_space as dspace_tool


class DoeSampleGenerator(AbstractSampleGenerator):
    """Abstract class that generates sampling"""

    GENERATOR_NAME = "DOE_GENERATOR"
    VARIABLES = dspace_tool.VARIABLES
    VALUES = dspace_tool.VALUES
    UPPER_BOUND = dspace_tool.UPPER_BOUND
    LOWER_BOUND = dspace_tool.LOWER_BOUND
    ENABLE_VARIABLE_BOOL = dspace_tool.ENABLE_VARIABLE_BOOL
    LIST_ACTIVATED_ELEM = dspace_tool.LIST_ACTIVATED_ELEM
    NB_POINTS = "nb_points"

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    N_PROCESSES = 'n_processes'
    WAIT_TIME_BETWEEN_SAMPLES = 'wait_time_between_samples'

    N_SAMPLES = "n_samples"

    UNSUPPORTED_GEMSEO_ALGORITHMS: tuple[str] = ("CustomDOE", "DiagonalDOE")
    """The list of unsupported algorithms.

    GEMSEO's custom DOE: https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#customdoe
    GEMSEO's DiagonalDOE: https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#diagonaldoe
    """
    # The DiagonalDOE algorithm is special: it has parameters "reverse" that can have name of variable
    # Do we want it in SoSTrades. Does it works also or not ?

    TYPE_PERMISSIVE_ALGORITHMS: tuple[str] = ("PYDOE_FULLFACT", "OT_FULLFACT", "PYDOE_PBDESIGN", "PYDOE_FF2N")
    """The list of algorithms that accept inputs other than floats or arrays.

    Algorithms not listed below will have input constrained to floats and arrays.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Constructor"""
        logger_aux = logger
        if logger_aux is None:
            logger_aux = logging.getLogger(__name__)
        # - inits super class
        super().__init__(self.GENERATOR_NAME, logger=logger_aux)
        # - create attributes
        self.doe_factory = None
        self.__available_algo_names = None
        # - set attribute values
        self._reload()

        self.selected_inputs = []
        self.selected_inputs_types = {}

    def _reload(self):
        """
        Reloads all attributes of the class
        - creates the DOEFactory
        - creates the DOEFactory from GEMSEO
        """
        # DOEFactory is instantiated once here
        self.doe_factory = DOELibraryFactory()
        # all the DOE algorithms in GEMSEO that are available in current environment
        all_names = self.doe_factory.algorithms
        # filter with the unsupported GEMSEO algorithms.
        self.__available_algo_names = list(set(all_names) - set(self.UNSUPPORTED_GEMSEO_ALGORITHMS))

    def get_available_algo_names(self):
        """
        Method that provides the already created list of available algo_names, filtered by the unsupported ones

        Returns:
             algo_names_list (list): list of available algo names
        """
        return self.__available_algo_names

    def _check_algo_name(self, algo_name):
        """
        Check provided algo name before getting its algo_options
        Arguments:
            algo_name (string): name of the numerical algorithm
        Raises:
            Exception if sampling_algo_name is not in the list of available algorithms
        """
        algo_names_list = self.get_available_algo_names()
        if algo_name not in algo_names_list:
            if algo_name in self.UNSUPPORTED_GEMSEO_ALGORITHMS:
                msg = f"The provided algorithm name {algo_name} is not allowed in doe sample generator"
                raise ValueError(msg)
            msg = f"The provided algorithm name {algo_name} is not in the available algorithm list : {algo_names_list}"
            raise ValueError(msg)

    def get_options_and_default_values(self, sampling_algo_name: str) -> tuple[dict[str, Any]]:
        """Method that provides the list of options of an algorithm with there default values (if any) and description.

        Arguments:
            sampling_algo_name: The name of the numerical algorithm

        Returns:
            the Sample Generator expected inputs (as DESC_IN format)
                                (to be provided to proxy i/o grammars)
            More precisely:
             algo_options (dict): dict of algo options with default values (if any). It is in algo_options desci_in format
             algo_options_descr_dict (dict): dict of description of algo options
             e.g. https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html#fullfact

        """
        # check sampling_algo_name
        self._check_algo_name(sampling_algo_name)

        # create the algo library
        algo_lib = self.doe_factory.create(sampling_algo_name)

        all_options = algo_lib.ALGORITHM_INFOS[sampling_algo_name].Settings.model_fields
        # Keep only the DOE-related options
        algo_options = {
            key: value for key, value in all_options.items() if key not in BaseDriverSettings.model_fields
        }
        algo_options_default = {
            option_name: option.default if not option.is_required() else None
            for option_name, option in algo_options.items()
        }

        algo_options_descr_dict = {option_name: option.description for option_name, option in algo_options.items()}

        return algo_options_default, algo_options_descr_dict

    def _check_options(self, sampling_algo_name, algo_options, design_space):
        """
        Check provided options before sample generation
        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
        """
        if self.N_SAMPLES not in algo_options:
            self.logger.warning(
                "N_samples is not defined; pay attention you use fullfact algo and that levels are well defined"
            )

    def _check_samples(self, samples_df):
        """
        Method that checks the sample output type
        Arguments:
            samples_df (dataframe) : generated samples
        Raises:
            Exception if samples_df is not a dataframe
        """
        if not (isinstance(samples_df, pd.DataFrame)):
            msg = "Expected sampling output type should be pandas.core.frame.DataFrame"
            msg += f"however sampling type of sampling generator <{self.__class__.__name__!s}> "
            msg += f"is <{type(samples_df)!s}> "
            raise SampleTypeError(msg)

    def generate_samples(self, sampling_algo_name, algo_options, design_space):
        """
        Method that generate samples

        Arguments:
            sampling_algo_name (string): name of the numerical algorithm
            algo_options (dict): provides the selected value of each option of the algorithm
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs

        Returns:
            samples_df (dataframe) : generated samples
        """
        algo = self.doe_factory.create(sampling_algo_name)
        samples = algo.compute_doe(design_space, **algo_options)
        samples = self._reformat_samples_from_design_space(samples, design_space)
        return self._put_samples_in_df_format(samples, design_space)

    def _reformat_samples_from_design_space(self, samples, design_space):
        """
        Reformat samples based on the design space to take into account variables with dim >1
        It uses methods from gemseo Design Space
        For instance in case of variables of x of dim 1 and z of dim 2
        [0.0,-10.0, 0.0] becomes [[0.0]   [-10.0, 0.0]]

        Arguments:
            samples (numpy matrix of floats) : matrix of n raws  (each raw is an input point to be evaluated)
                                     any variable of dim m will be an array of dim m in a single column of the matrix
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs

        Returns:
            reformated_samples (numpy matrix of arrays) : Reformated samples that takes into account variables with dim >1
                                    matrix of n raws  (each raw is an input point to be evaluated)
                                    any variable of dim m is an array of dim m in a single column of the matrix
        """
        selected_inputs = design_space.variable_names

        reformated_samples = []
        for current_point in samples:  # To be vectorized
            # Current point  is an array with variables ordered as in selected_inputs
            # Find the dictionary version of the current point sample
            current_point_dict = design_space.convert_array_to_dict(current_point)

            reformated_current_point = []
            for in_variable in selected_inputs:
                # convert array into data when needed
                if in_variable in self.selected_inputs_types and (
                    self.selected_inputs_types[in_variable] in ['float', 'int', 'string']
                ):
                    current_point_dict[in_variable] = current_point_dict[in_variable][0]
                if in_variable in self.selected_inputs_types and (self.selected_inputs_types[in_variable] == 'list'):
                    current_point_dict[in_variable] = list(current_point_dict[in_variable])

                reformated_current_point.append(current_point_dict[in_variable])
            reformated_samples.append(reformated_current_point)

        return reformated_samples

    def _put_samples_in_df_format(self, samples, design_space):
        """
        construction of a dataframe of the generated samples
        # To be vectorized

        Arguments:
            samples (numpy matrix of arrays) : matrix of n raws  (each raw is an input point to be evaluated)
                                               any variable of dim m will be an array of dim m in a single column of the matrix
            design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
        Returns:
            samples_df (data_frame) : dataframe of a matrix of n raws  (each raw is an input point to be evaluated)
                                      any variable of dim m is an array of dim m in a single column of the matrix
        """
        selected_inputs = design_space.variable_names

        return pd.DataFrame(data=samples, columns=selected_inputs)

    def setup(self, proxy):
        """Method that setup the doe_algo method"""
        dynamic_inputs = {}
        dynamic_outputs = {}

        # Setup dynamic inputs in case of DOE_ALGO selection:
        # i.e. EVAL_INPUTS and SAMPLING_ALGO
        self.setup_dynamic_inputs_for_doe_generator_method(dynamic_inputs, proxy)
        # Setup dynamic inputs when EVAL_INPUTS/SAMPLING_ALGO are already set
        self.setup_dynamic_inputs_algo_options_design_space(dynamic_inputs, proxy)

        return dynamic_inputs, dynamic_outputs

    def setup_dynamic_inputs_for_doe_generator_method(self, dynamic_inputs, proxy):
        """
        Method that setup dynamic inputs in case of DOE_ALGO selection: i.e. EVAL_INPUTS and SAMPLING_ALGO
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated

        """
        if proxy.sampling_method == proxy.DOE_ALGO:
            # Get possible values for sampling algorithm name
            available_doe_algorithms = self.get_available_algo_names()
            dynamic_inputs.update({
                'sampling_algo': {
                    proxy.TYPE: 'string',
                    proxy.STRUCTURING: True,
                    proxy.POSSIBLE_VALUES: available_doe_algorithms,
                }
            })

    def setup_dynamic_inputs_algo_options_design_space(self, dynamic_inputs, proxy):
        """
        Setup dynamic inputs when EVAL_INPUTS/SAMPLING_ALGO are already set
        Create or update DESIGN_SPACE
        Create or update ALGO_OPTIONS
        """
        self.setup_design_space(dynamic_inputs, proxy)
        self.setup_algo_options(dynamic_inputs, proxy)

    def setup_design_space(self, dynamic_inputs, proxy):
        """
        Method that setup 'design_space'
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        disc_in = proxy.get_data_in()
        # Dynamic input of default design space
        if proxy.EVAL_INPUTS in disc_in:
            # save possible types in sample generator
            if proxy.eval_in_possible_types is not None:
                self.selected_inputs_types = proxy.eval_in_possible_types.copy()

            eval_inputs = proxy.get_sosdisc_inputs(proxy.EVAL_INPUTS)
            if eval_inputs is not None:
                selected_inputs = eval_inputs.loc[eval_inputs['selected_input']]['full_name'].tolist()

                if set(selected_inputs) != set(self.selected_inputs):
                    self.selected_inputs = selected_inputs

                default_design_space = pd.DataFrame()
                design_space_dataframe_descriptor = {
                    self.VARIABLES: ('string', None, False),
                    self.LOWER_BOUND: ('multiple', None, True),
                    self.UPPER_BOUND: ('multiple', None, True),
                    self.VALUES: ('multiple', None, True),
                    self.ENABLE_VARIABLE_BOOL: ('bool', None, True),
                    self.LIST_ACTIVATED_ELEM: ('list', None, True),
                }

                if proxy.sampling_method == proxy.DOE_ALGO:
                    default_design_space = pd.DataFrame({
                        self.VARIABLES: self.selected_inputs,
                        self.LOWER_BOUND: [None] * len(self.selected_inputs),
                        self.UPPER_BOUND: [None] * len(self.selected_inputs),
                        self.LIST_ACTIVATED_ELEM: [[]] * len(self.selected_inputs),
                        self.ENABLE_VARIABLE_BOOL: [False] * len(self.selected_inputs),
                        self.VALUES: [None] * len(self.selected_inputs),
                    })
                    default_design_space[self.ENABLE_VARIABLE_BOOL] = default_design_space[
                        self.ENABLE_VARIABLE_BOOL
                    ].astype(bool)
                elif proxy.sampling_method == proxy.GRID_SEARCH:
                    default_design_space = pd.DataFrame({
                        self.VARIABLES: self.selected_inputs,
                        self.LOWER_BOUND: [0.0] * len(self.selected_inputs),
                        self.UPPER_BOUND: [100.0] * len(self.selected_inputs),
                        self.NB_POINTS: [2] * len(self.selected_inputs),
                        self.LIST_ACTIVATED_ELEM: [[]] * len(self.selected_inputs),
                        self.ENABLE_VARIABLE_BOOL: [False] * len(self.selected_inputs),
                        self.VALUES: [None] * len(self.selected_inputs),
                    })
                    default_design_space[self.NB_POINTS] = default_design_space[self.NB_POINTS].astype(int)
                    design_space_dataframe_descriptor.update({self.NB_POINTS: ('int', None, True)})
                dynamic_inputs.update({
                    proxy.DESIGN_SPACE: {
                        proxy.TYPE: 'dataframe',
                        proxy.DEFAULT: default_design_space,
                        proxy.STRUCTURING: False,
                        proxy.DATAFRAME_DESCRIPTOR: design_space_dataframe_descriptor,
                    }
                })

                # Next lines of code treat the case in which eval inputs change with a previously defined design space,
                # so that the bound are kept instead of set to default None.
                if 'design_space' in disc_in:
                    disc_in['design_space'][proxy.DEFAULT] = default_design_space
                    proxy.dm.set_data(
                        proxy.get_var_full_name(proxy.DESIGN_SPACE, disc_in),
                        proxy.DATAFRAME_DESCRIPTOR,
                        design_space_dataframe_descriptor,
                        check_value=False,
                    )

                    from_design_space = list(disc_in['design_space'][proxy.VALUE][self.VARIABLES])
                    from_eval_inputs = self.selected_inputs

                    df_cols = (
                        [self.VARIABLES, self.LOWER_BOUND, self.UPPER_BOUND]
                        + ([self.NB_POINTS] if proxy.sampling_method == proxy.GRID_SEARCH else [])
                        + ([self.LIST_ACTIVATED_ELEM, self.ENABLE_VARIABLE_BOOL, self.VALUES])
                    )
                    final_dataframe = pd.DataFrame(columns=df_cols)
                    if proxy.sampling_method == proxy.GRID_SEARCH:
                        final_dataframe[self.NB_POINTS] = final_dataframe[self.NB_POINTS].astype(int)
                    for element in from_eval_inputs:
                        default_df = default_design_space[default_design_space[self.VARIABLES] == element]
                        if final_dataframe.empty:
                            final_dataframe = default_df.copy()
                        else:
                            final_dataframe = pd.concat([final_dataframe, default_df], ignore_index=True)
                        if element in from_design_space:
                            to_append = disc_in['design_space'][proxy.VALUE][
                                disc_in['design_space'][proxy.VALUE][self.VARIABLES] == element
                            ]
                            # NB: gridsearch could set up its own space
                            if proxy.sampling_method == proxy.DOE_ALGO:
                                # for DoE need to dismiss self.NB_POINTS
                                to_append = to_append.loc[:, to_append.columns != self.NB_POINTS]
                            elif proxy.sampling_method == proxy.GRID_SEARCH and self.NB_POINTS not in to_append.columns:
                                # for GridSearch need to eventually insert the self.NB_POINTS column
                                to_append.insert(3, self.NB_POINTS, 2)

                            # I want to update the dataframes following the variable name and not the index
                            final_dataframe.set_index('variable', inplace=True)
                            final_dataframe.update(to_append.set_index('variable'), overwrite=True)
                            final_dataframe.reset_index(inplace=True)
                    proxy.dm.set_data(
                        proxy.get_var_full_name(proxy.DESIGN_SPACE, disc_in),
                        proxy.VALUE,
                        final_dataframe,
                        check_value=False,
                    )

    def setup_algo_options(self, dynamic_inputs, proxy):
        """
        Method that setup 'algo_options''
        Arguments:
            dynamic_inputs (dict): the dynamic input dict to be updated
        """
        disc_in = proxy.get_data_in()
        # Dynamic input of algo_options
        if proxy.ALGO in disc_in:
            algo_name = proxy.get_sosdisc_inputs(proxy.ALGO)
            if algo_name is not None:  # and algo_name_has_changed:
                default_dict = self.get_algo_default_options(algo_name)
                algo_options_dict = {
                    proxy.ALGO_OPTIONS: {
                        proxy.TYPE: 'dict',
                        proxy.DEFAULT: default_dict,
                        proxy.DATAFRAME_EDITION_LOCKED: False,
                        proxy.STRUCTURING: True,
                        proxy.DATAFRAME_DESCRIPTOR: {
                            self.VARIABLES: ('string', None, False),
                            self.VALUES: ('string', None, True),
                        },
                    }
                }
                dynamic_inputs.update(algo_options_dict)
                all_options = list(default_dict.keys())
                if (
                    proxy.ALGO_OPTIONS in disc_in
                    and disc_in[proxy.ALGO_OPTIONS][proxy.VALUE] is not None
                    and list(disc_in[proxy.ALGO_OPTIONS][proxy.VALUE].keys()) != all_options
                ):
                    options_map = ChainMap(disc_in[proxy.ALGO_OPTIONS][proxy.VALUE], default_dict)
                    disc_in[proxy.ALGO_OPTIONS][proxy.VALUE] = {key: options_map[key] for key in all_options}

    def get_algo_default_options(self, algo_name):
        """This algo generate the default options to set for a given doe algorithm."""
        # In get_options_and_default_values, it is already checked whether the algo_name belongs to the list of possible Gemseo
        # DoE algorithms
        available_algos = get_available_doe_algorithms()
        if algo_name in available_algos:
            algo_options_desc_in, _ = self.get_options_and_default_values(algo_name)
            return algo_options_desc_in
        msg = f"The DoE algorithm {algo_name} is not available in GEMSEO list :{available_algos}"
        raise ValueError(msg)

    def get_arguments(self, wrapper):
        # Dynamic input of default design space
        algo_name = wrapper.get_sosdisc_inputs(wrapper.ALGO)
        algo_options = wrapper.get_sosdisc_inputs(wrapper.ALGO_OPTIONS)
        dspace_df = wrapper.get_sosdisc_inputs(wrapper.DESIGN_SPACE)
        eval_inputs = wrapper.get_sosdisc_inputs(wrapper.EVAL_INPUTS)
        selected_inputs = eval_inputs.loc[eval_inputs[wrapper.SELECTED_INPUT]][wrapper.FULL_NAME].tolist()
        design_space = self.create_design_space(selected_inputs, dspace_df)
        doe_kwargs = {'sampling_algo_name': algo_name, 'algo_options': algo_options, 'design_space': design_space}
        return [], doe_kwargs

    def create_design_space(self, selected_inputs, dspace_df):
        """
        create_design_space with variables names based on selected_inputs (if dspace_df is not None)

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dspace_df (dataframe): design space in Desc_in format

        Returns:
             design_space (gemseo DesignSpace): gemseo Design Space with names of variables based on selected_inputs
        """
        design_space = None
        if dspace_df is not None:
            dspace_df_updated = self.update_design_space(selected_inputs, dspace_df)
            design_space, _ = dspace_tool.create_gemseo_dspace_from_dspace_df(dspace_df_updated)
        return design_space

    def update_design_space(self, selected_inputs, dspace_df):
        """
        update dspace_df (design space in Desc_in format)

        Arguments:
            selected_inputs (list): list of selected variables (the true variables in eval_inputs Desc_in)
            dspace_df (dataframe): design space in Desc_in format

        Returns:
             dspace_df_updated (dataframe): updated dspace_df

        """
        lower_bounds = dspace_df[self.LOWER_BOUND].tolist()
        upper_bounds = dspace_df[self.UPPER_BOUND].tolist()
        values = lower_bounds

        # FIXME: why are we dismissing some of the user-input values in design_space ?
        enable_variables = [True for _ in selected_inputs]
        return pd.DataFrame({
            self.VARIABLES: selected_inputs,
            self.VALUES: values,
            self.LOWER_BOUND: lower_bounds,
            self.UPPER_BOUND: upper_bounds,
            self.ENABLE_VARIABLE_BOOL: enable_variables,
            self.LIST_ACTIVATED_ELEM: [[True] for _ in selected_inputs],
        })

    def is_ready_to_sample(self, proxy):
        disc_in = proxy.get_data_in()
        return (
            self.selected_inputs
            and proxy.ALGO in disc_in
            and proxy.ALGO_OPTIONS in disc_in
            and proxy.DESIGN_SPACE in disc_in
        )

    def filter_inputs(self, proxy):
        """Filter for the majority of algorithms the"""
        disc_in = proxy.get_data_in()
        if not (proxy.ALGO in disc_in and proxy.get_sosdisc_inputs(proxy.ALGO) in self.TYPE_PERMISSIVE_ALGORITHMS) \
                and  proxy.eval_in_possible_types:
            proxy.eval_in_possible_types = {
                _v: _t for (_v, _t) in proxy.eval_in_possible_types.items() if _t in ('array', 'float')
            }
            proxy.eval_in_possible_values = [
                _v for _v in proxy.eval_in_possible_values if _v in proxy.eval_in_possible_types
            ]
