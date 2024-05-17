'''
Copyright 2022 Airbus SAS
Modifications on 2023/02/23-2024/05/17 Copyright 2023 Capgemini

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

from copy import deepcopy

import chaospy as cp
import numpy as np
import openturns as ot
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)
from sostrades_core.tools.post_processing.post_processing_tools import (
    format_currency_legend,
)


class UncertaintyQuantification(SoSWrapp):
    """
    Generic Uncertainty Quantification class
    """

    # ontology information
    _ontology_data = {
        'label': 'Uncertainty Quantification Model',
        SoSWrapp.TYPE: 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-chart-area',
        'version': '',
    }

    EVAL_INPUTS = 'eval_inputs'
    GATHER_OUTPUTS = 'gather_outputs'
    DEFAULT = SoSWrapp.DEFAULT
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    NB_POINTS = 'nb_points'
    VARIABLE = 'variable'

    eval_df_data_description = {
        SoSWrapp.TYPE: 'dataframe',
        SoSWrapp.DATAFRAME_DESCRIPTOR: {},
        SoSWrapp.DYNAMIC_DATAFRAME_COLUMNS: True,
        SoSWrapp.UNIT: None,
        SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
        SoSWrapp.NAMESPACE: 'ns_evaluator',
    }
    DESC_IN = {
        'samples_inputs_df': eval_df_data_description,
        'samples_outputs_df': eval_df_data_description.copy(),
        'design_space': {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.DATAFRAME_DESCRIPTOR: {
                'shortest_name': ('string', None, False),
                LOWER_BOUND: ('multiple', None, True),
                UPPER_BOUND: ('multiple', None, True),
                VARIABLE: ('multiple', None, True),
                NB_POINTS: ('int', None, True),
                'full_name': ('string', None, False),
            },
            SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
            SoSWrapp.NAMESPACE: 'ns_sample_generator',
            SoSWrapp.STRUCTURING: True
        },
        'confidence_interval': {
            SoSWrapp.TYPE: 'float',
            SoSWrapp.UNIT: '%',
            SoSWrapp.DEFAULT: 90,
            SoSWrapp.RANGE: [0.0, 100.0],
            SoSWrapp.STRUCTURING: False,
            SoSWrapp.NUMERICAL: True,
            SoSWrapp.USER_LEVEL: 2,
        },
        'sample_size': {
            SoSWrapp.TYPE: 'float',
            SoSWrapp.UNIT: None,
            SoSWrapp.DEFAULT: 1000,
            SoSWrapp.STRUCTURING: False,
            SoSWrapp.NUMERICAL: True,
            SoSWrapp.USER_LEVEL: 2,
        },
        'prepare_samples_function': {
            SoSWrapp.TYPE: 'string',
            SoSWrapp.DEFAULT: 'None',
            SoSWrapp.USER_LEVEL: 2,
        },
        EVAL_INPUTS: {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.DATAFRAME_DESCRIPTOR: {
                'selected_input': ('bool', None, True),
                'full_name': ('string', None, False),
                'shortest_name': ('string', None, False),
                # 'ontology_name': ('string', None, False),je
            },
            SoSWrapp.DATAFRAME_EDITION_LOCKED: False,
            SoSWrapp.STRUCTURING: True,
            SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
            SoSWrapp.NAMESPACE: 'ns_sample_generator',
        },
        GATHER_OUTPUTS: {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.DATAFRAME_DESCRIPTOR: {
                'selected_output': ('bool', None, True),
                'full_name': ('string', None, False),
                'shortest_name': ('string', None, False),
                # 'ontology_name': ('string', None, False),
            },
            SoSWrapp.DATAFRAME_EDITION_LOCKED: False,
            SoSWrapp.STRUCTURING: True,
            SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
            SoSWrapp.NAMESPACE: 'ns_evaluator',
        },
    }

    DESC_OUT = {
        'input_parameters_samples_df': {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.UNIT: None,
        },
        'output_interpolated_values_df': {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.UNIT: None,
        },
        'input_parameters_names': {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.UNIT: None,
        },
        'dict_array_float_names': {
            SoSWrapp.TYPE: 'dict',
            SoSWrapp.UNIT: None,
        },
        'pure_float_input_names': {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.UNIT: None,
        },
        'float_output_names': {
            SoSWrapp.TYPE: 'list',
            SoSWrapp.UNIT: None,
        }
    }

    def setup_sos_disciplines(self):
        """setup sos disciplines"""
        data_in = self.get_data_in()
        if data_in != {}:

            dynamic_outputs = {}
            dynamic_inputs = {}
            if (self.EVAL_INPUTS in data_in) & (
                    self.GATHER_OUTPUTS in data_in
            ):
                gather_outputs = self.get_sosdisc_inputs(self.GATHER_OUTPUTS)
                eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)

                if (eval_inputs is not None) & (gather_outputs is not None):

                    selected_inputs = eval_inputs[
                        eval_inputs['selected_input'] == True
                        ]['full_name']

                    in_param = selected_inputs.tolist()
                    # in_param.sort()

                    selected_outputs = gather_outputs[
                        gather_outputs['selected_output'] == True
                        ]['full_name']

                    out_param = selected_outputs.tolist()
                    out_param.sort()

                    parameter_list = in_param + out_param
                    parameter_list = [val.split('.')[-1] for val in parameter_list]

                    conversion_full_ontology = {
                        parameter: [parameter, ''] for parameter in parameter_list
                    }

                    # possible_distrib = ['Normal', 'PERT', 'LogNormal', 'Triangular']

                    # distrib = [possible_distrib[random.randrange(
                    # len(possible_distrib))] for i in range(len(in_param))]
                    # distrib = ['Normal', 'PERT', 'Triangular']
                    # def random_distribution(input):
                    #     return np.random.choice(
                    #         [i for i in range(len(possible_distrib))],
                    #         # p=[1 / len(possible_distrib)
                    #         # for input in
                    #         # possible_distrib])
                    #         p=[0, 1, 0, 0],
                    #     )
                    # 
                    # # distrib = [possible_distrib[random_distribution(input)] for input in selected_inputs.tolist()]
                    distrib = ['PERT' for _ in selected_inputs.tolist()]

                    if ('design_space' in data_in) & (len(in_param) > 0):

                        if data_in['design_space']['value'] is not None:

                            lower_bnd = data_in['design_space']['value'][
                                self.LOWER_BOUND
                            ]
                            upper_bnd = data_in['design_space']['value'][
                                self.UPPER_BOUND
                            ]
                            input_distribution_default = pd.DataFrame(
                                {
                                    'parameter': in_param,
                                    'distribution': distrib,
                                    'lower_parameter': lower_bnd,
                                    'upper_parameter': upper_bnd,
                                    'most_probable_value': [
                                        (a + b) / 2
                                        for a, b in zip(lower_bnd, upper_bnd)
                                    ],
                                }
                            )

                            input_distribution_default.loc[
                                input_distribution_default['distribution'] == 'Normal',
                                'most_probable_value',
                            ] = np.nan
                            input_distribution_default.loc[
                                input_distribution_default['distribution']
                                == 'LogNormal',
                                'most_probable_value',
                            ] = np.nan

                            data_details_default = pd.DataFrame()
                            for input_param in list(set(in_param)):
                                try:
                                    [name, unit] = conversion_full_ontology[
                                        input_param.split('.')[-1]
                                    ]
                                except Exception as ex:
                                    print(
                                        'The following exception occurs when trying to reach Ontology server',
                                        ex,
                                    )
                                    [name, unit] = [input_param, '']
                                input_serie = pd.Series({
                                    SoSWrapp.TYPE: 'input',
                                    'variable': input_param,
                                    'name': name,
                                    SoSWrapp.UNIT: unit,
                                })
                                data_details_default = pd.concat([data_details_default, pd.DataFrame([input_serie])],
                                                                 ignore_index=True)
                            for output_param in list(set(out_param)):
                                try:
                                    [name, unit] = conversion_full_ontology[
                                        output_param.split('.')[-1]
                                    ]
                                except Exception as ex:
                                    print(
                                        'The following exception occurs when trying to reach Ontology server',
                                        ex,
                                    )
                                    [name, unit] = [output_param, None]

                                output_serie = pd.Series({
                                    SoSWrapp.TYPE: 'output',
                                    'variable': output_param,
                                    'name': name,
                                    SoSWrapp.UNIT: unit,
                                })
                                data_details_default = pd.concat([data_details_default, pd.DataFrame([output_serie])],
                                                                 ignore_index=True)

                            dynamic_inputs['input_distribution_parameters_df'] = {
                                SoSWrapp.TYPE: 'dataframe',
                                SoSWrapp.DATAFRAME_DESCRIPTOR: {
                                    'parameter': ('string', None, False),
                                    'distribution': ('string', None, True),
                                    'lower_parameter': ('multiple', None, True),
                                    'upper_parameter': ('multiple', None, True),
                                    'most_probable_value': ('multiple', None, True),
                                },
                                SoSWrapp.UNIT: '-',
                                SoSWrapp.DEFAULT: input_distribution_default,
                                SoSWrapp.STRUCTURING: False,
                            }

                            dynamic_inputs['data_details_df'] = {
                                SoSWrapp.TYPE: 'dataframe',
                                SoSWrapp.DATAFRAME_DESCRIPTOR: {
                                    SoSWrapp.TYPE: ('string', None, False),
                                    'variable': ('string', None, False),
                                    'name': ('string', None, True),
                                    SoSWrapp.UNIT: ('string', None, True),
                                },
                                SoSWrapp.UNIT: None,
                                SoSWrapp.DEFAULT: data_details_default,
                                SoSWrapp.STRUCTURING: False,
                            }

                            if 'input_distribution_parameters_df' in data_in:
                                data_in['input_distribution_parameters_df'][
                                    'value'
                                ] = self.get_sosdisc_inputs(
                                    'input_distribution_parameters_df'
                                )
                                data_in['data_details_df'][
                                    'value'
                                ] = self.get_sosdisc_inputs('data_details_df')
                                design_space_value = self.get_sosdisc_inputs('design_space')
                                input_distribution_parameters_df_value = self.get_sosdisc_inputs(
                                    'input_distribution_parameters_df')
                                if ((design_space_value['variable'].to_list() != in_param)
                                        or (input_distribution_parameters_df_value['lower_parameter'].to_list()
                                            != design_space_value['lower_bnd'].to_list())
                                        or (
                                                self.get_sosdisc_inputs(
                                                    'input_distribution_parameters_df'
                                                )['upper_parameter'].to_list()
                                                != self.get_sosdisc_inputs('design_space')[
                                                    'upper_bnd'
                                                ].to_list()
                                        )
                                ):
                                    data_in['input_distribution_parameters_df'][
                                        'value'
                                    ] = input_distribution_default
                                    data_in['data_details_df'][
                                        'value'
                                    ] = data_details_default
                                if self.get_sosdisc_inputs('data_details_df')[
                                    'variable'
                                ].to_list() != (in_param + out_param):
                                    data_in['data_details_df'][
                                        'value'
                                    ] = data_details_default

            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def check_data_integrity(self):
        """
        Check that eval_inputs and outputs are float
        """

        self.check_eval_in_out_types(self.EVAL_INPUTS, self.IO_TYPE_IN)
        self.check_eval_in_out_types(self.GATHER_OUTPUTS, self.IO_TYPE_OUT)

    def check_eval_in_out_types(self, eval_io_name, io_type):
        """

        Args:
            eval_io_name: evalinputs or gather_outputs
            io_type: 'in' or 'out'

        Returns: CHeck data_integrity for parameter inside eval in or out

        """

        eval_io = self.get_sosdisc_inputs(eval_io_name)
        if eval_io is not None:
            eval_io_full_name = self.get_input_var_full_name(eval_io_name)
            parameter_list = eval_io[eval_io[f'selected_{io_type}put'] == True
                                     ]['full_name'].tolist()
            check_integrity_msg_list = []
            for param in parameter_list:
                param_full_ns_list = self.dm.get_all_namespaces_from_var_name(param)
                for param_full_ns in param_full_ns_list:
                    param_type = self.dm.get_data(param_full_ns, self.TYPE)
                    if param_type not in ['float', 'int']:
                        check_integrity_msg = f'Parameter {param_full_ns} found in eval_{io_type} should be float or' \
                                              f' int for uncertainty quantification'
                        check_integrity_msg_list.append(check_integrity_msg)

            check_integrity_msg = '\n'.join(check_integrity_msg_list)
            self.dm.set_data(
                eval_io_full_name, self.CHECK_INTEGRITY_MSG, check_integrity_msg)

    def prepare_samples(self):
        """
        Prepare the inputs samples for distribution and the output samples for the gridgenerator
        """
        inputs_dict = self.get_sosdisc_inputs()
        samples_inputs_df = inputs_dict['samples_inputs_df']

        samples_inputs_df = self.delete_reference_scenarios(samples_inputs_df)
        self.input_parameters_names = list(samples_inputs_df.columns)[1:]

        samples_outputs_df = inputs_dict['samples_outputs_df']
        samples_outputs_df = self.delete_reference_scenarios(samples_outputs_df)
        self.output_names = list(samples_outputs_df.columns)[1:]

        self.confidence_interval = inputs_dict['confidence_interval'] / 100
        self.sample_size = inputs_dict['sample_size']

        self.input_distribution_parameters_df = deepcopy(
            inputs_dict['input_distribution_parameters_df']
        )

        self.all_samples_df = samples_inputs_df.merge(samples_outputs_df, on='scenario_name', how='left')
        self.breakdown_arrays_to_float()

        self.set_float_input_distribution_parameters_df_values()

    def breakdown_arrays_to_float(self):
        """
        Converts arrays inputs and outputs to lists of floats for ease of manipulation later

        The logic for handling arrays inputs/outputs is the following :
        - break them down into floats at first
        - do the work (interpolation)
        - recombine them into arrays when interpolation is done
        """

        # CONVERTS INPUTS
        self.float_input_names = []
        self.float_input_distribution_parameters_df = pd.DataFrame()

        self.float_all_samples_df = pd.DataFrame()
        self.float_all_samples_df['scenario_name'] = self.all_samples_df['scenario_name']

        self.pure_float_input_names = []
        self.dict_array_float_names = {}

        for input_name in self.input_parameters_names:
            distribution_parameters = self.input_distribution_parameters_df.loc[
                self.input_distribution_parameters_df['parameter'] == input_name]
            lower_bound, upper_bound, distribution = distribution_parameters[['lower_parameter',
                                                                              'upper_parameter',
                                                                              'distribution']].values[0]
            if isinstance(lower_bound, (float, int)) and isinstance(upper_bound, (float, int)):
                self.float_input_distribution_parameters_df = pd.concat([self.float_input_distribution_parameters_df,
                                                                         distribution_parameters])
                self.float_input_names.append(input_name)
                self.pure_float_input_names.append(input_name)
                self.float_all_samples_df[input_name] = self.all_samples_df[input_name]
            elif isinstance(lower_bound, np.ndarray) and isinstance(upper_bound, np.ndarray):
                if lower_bound.ndim != 1 or upper_bound.ndim != 1:
                    raise ValueError("inputs of type array can only be one-dimensional")
                if lower_bound.shape != upper_bound.shape:
                    raise ValueError("'lower_parameter' and 'upper_parameter' must have the same shape in case they are"
                                     " of type numpy.ndarray")
                length = len(lower_bound)
                floats_distributions_parameters = pd.DataFrame({
                    'parameter': [f"{input_name}[{i}]" for i in range(length)],
                    'lower_parameter': list(lower_bound),
                    'upper_parameter': list(distribution_parameters['upper_parameter'].values[0]),
                    'most_probable_value': list(distribution_parameters['most_probable_value'].values[0]),
                    'distribution': [distribution] * length,
                })
                self.float_input_distribution_parameters_df = pd.concat([self.float_input_distribution_parameters_df,
                                                                         floats_distributions_parameters])

                input_values = np.stack(self.all_samples_df[input_name].values)
                self.dict_array_float_names[input_name] = []
                for i in range(length):
                    float_input_name = f"{input_name}[{i}]"
                    self.float_all_samples_df[float_input_name] = input_values[:, i]
                    self.dict_array_float_names[input_name].append(float_input_name)
                    self.float_input_names.append(float_input_name)
            else:
                raise ValueError("'lower_parameter' and 'upper_parameter' must be of same type, available types are: "
                                 "float, int, numpy.ndarray (one-dimensional)")

        # CONVERTS OUTPUTS
        self.float_output_names = []
        for output_name in self.output_names:
            example_value = self.all_samples_df[output_name].values[0]
            if isinstance(example_value, (float, int)):
                self.float_output_names.append(output_name)
            elif isinstance(example_value, np.ndarray):
                if example_value.ndim != 1:
                    raise ValueError("inputs of type array can only be one-dimensional")
                float_output_names = [f"{output_name}[{i}]" for i in range(len(example_value))]
                self.dict_array_float_names[output_name] = float_output_names
                self.float_output_names += self.dict_array_float_names[output_name]
                values = np.stack(self.all_samples_df[output_name].values)
                for i, float_output_name in enumerate(float_output_names):
                    self.float_all_samples_df[float_output_name] = values[:, i]

    def set_float_input_distribution_parameters_df_values(self):
        """Set the values taken by each float input in float_all_samples_df"""
        list_of_unique_values = []
        for float_input_name in self.float_input_names:
            sorted_unique_values = sorted(list(self.float_all_samples_df[float_input_name].unique()))
            list_of_unique_values.append(sorted_unique_values)

        self.float_input_distribution_parameters_df["values"] = list_of_unique_values
        self.float_all_samples_df = self.float_all_samples_df.sort_values(by=self.float_input_names)

    def delete_reference_scenarios(self, samples_df):
        """
        Delete the reference scenario in a df for UQ
        """
        reference_scenario_samples_list = [scen for scen in samples_df['scenario_name'].values if
                                           'reference_scenario' in scen]
        samples_df_wo_ref = samples_df.loc[~samples_df['scenario_name'].isin(reference_scenario_samples_list)]

        return samples_df_wo_ref

    def run(self):
        """run method"""
        self.check_inputs_consistency()

        self.prepare_samples()

        # fixes a particular state of the random generator algorithm thanks to
        # the seed sample_size
        np.random.seed(42)
        ot.RandomGenerator.SetSeed(42)

        distrib_list = self.compute_distribution_list()

        self.compute_montecarlo_distribution(distrib_list)

        self.compute_output_interpolation()

        dict_values = {
            'input_parameters_samples_df': self.float_input_parameters_samples_df,
            'output_interpolated_values_df': self.output_interpolated_values_df,
            'input_parameters_names': self.input_parameters_names,
            'pure_float_input_names': self.pure_float_input_names,
            'dict_array_float_names': self.dict_array_float_names,
            'float_output_names': self.float_output_names,
        }

        self.store_sos_outputs_values(dict_values)

    def compute_distribution_list(self):
        """
        Compute the distribution list for all inputs, and store generated samples into a dataframe
        """
        self.float_input_parameters_samples_df = pd.DataFrame()
        distrib_list = []
        for input_name in self.float_input_names:
            distribution, lower_parameter, upper_parameter, most_probable_value = \
                self.float_input_distribution_parameters_df.loc[
                    self.float_input_distribution_parameters_df['parameter'] == input_name
                    ][['distribution', 'lower_parameter', 'upper_parameter', 'most_probable_value']].values[0]
            distrib = None
            if distribution == 'Normal':
                distrib = self.get_Normal_distrib(
                    lower_parameter,
                    upper_parameter,
                    confidence_interval=self.confidence_interval,
                )
            elif distribution == 'PERT':
                distrib = self.get_PERT_distrib(lower_parameter,
                                                upper_parameter,
                                                most_probable_value,
                                                )
            elif distribution == 'LogNormal':
                distrib = self.get_LogNormal_distrib(
                    lower_parameter,
                    upper_parameter,
                    confidence_interval=self.confidence_interval,
                )
            elif distribution == 'Triangular':
                distrib = self.get_Triangular_distrib(
                    lower_parameter,
                    upper_parameter,
                    most_probable_value,
                )
            else:
                self.logger.exception(
                    'Exception occurred: possible values in distribution are [Normal, PERT, Triangular, LogNormal].'
                )
            if distrib is not None:
                distrib_list.append(distrib)
                self.float_input_parameters_samples_df[input_name] = pd.DataFrame(
                    np.array(distrib.getSample(self.sample_size))
                )
        return distrib_list

    def compute_montecarlo_distribution(self, distrib_list: list):
        """Generate samples based on the distribution list"""
        identity_correlation_matrix = ot.CorrelationMatrix(len(distrib_list))
        copula = ot.NormalCopula(identity_correlation_matrix)
        distribution = ot.ComposedDistribution(distrib_list, copula)
        self.composed_distrib_sample = distribution.getSample(self.sample_size)

    @staticmethod
    def get_Normal_distrib(lower_bnd: float, upper_bnd: float, confidence_interval=0.95):
        """Returns a Normal distribution"""
        norm_val = float(format(1 - confidence_interval, '.2f')) / 2
        ratio = norm.ppf(1 - norm_val) - norm.ppf(norm_val)

        mu = (lower_bnd + upper_bnd) / 2
        sigma = (upper_bnd - lower_bnd) / ratio
        distrib = ot.Normal(mu, sigma)

        return distrib

    @staticmethod
    def get_PERT_distrib(lower_bnd: float, upper_bnd: float, most_probable_val: float):
        """Returns a PERT distribution"""
        chaospy_dist = cp.PERT(lower_bnd, most_probable_val, upper_bnd)
        distrib = ot.Distribution(ot.ChaospyDistribution(chaospy_dist))

        return distrib

    @staticmethod
    def get_Triangular_distrib(lower_bnd: float, upper_bnd: float, most_probable_val: float):
        """Returns a Triangular distribution"""
        distrib = ot.Triangular(int(lower_bnd), int(most_probable_val), int(upper_bnd))

        return distrib

    @staticmethod
    def get_LogNormal_distrib(lower_bnd: float, upper_bnd: float, confidence_interval=0.95):
        """Returns a LogNormal distribution"""
        norm_val = float(format(1 - confidence_interval, '.2f')) / 2
        ratio = norm.ppf(1 - norm_val) - norm.ppf(norm_val)

        mu = (lower_bnd + upper_bnd) / 2
        sigma = (upper_bnd - lower_bnd) / ratio

        distrib = ot.LogNormal()
        distrib.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))

        return distrib

    def compute_output_interpolation(self):
        """
        Perform the interpolation based on the inputs/outputs samples provided

        for each ouput, an interpolation is performed, then all the interpolations are gathered into one dataframe
        """
        input_parameters_single_values_tuple = tuple(
            [
                self.float_input_distribution_parameters_df.loc[
                    self.float_input_distribution_parameters_df['parameter'] == input_name
                    ]['values'].values[0]
                for input_name in self.float_input_names
            ]
        )
        input_dim_tuple = tuple(
            [len(set(sub_t)) for sub_t in input_parameters_single_values_tuple]
        )

        self.output_interpolated_values_df = pd.DataFrame()
        for output_name in self.output_names:
            if output_name in self.float_output_names:  # output is a float
                y = list(self.all_samples_df[output_name])
                # adapt output format to be used by RegularGridInterpolator
                output_values = np.reshape(y, input_dim_tuple)
                f = RegularGridInterpolator(
                    input_parameters_single_values_tuple, output_values, bounds_error=False
                )
                output_interpolated_values = f(self.composed_distrib_sample)
                self.output_interpolated_values_df[f'{output_name}'] = output_interpolated_values
            else:  # output is an array
                output_interpolated_arrays = []
                for float_var_name in self.dict_array_float_names[output_name]:
                    y = list(self.float_all_samples_df[float_var_name])
                    # adapt output format to be used by RegularGridInterpolator
                    output_values = np.reshape(y, input_dim_tuple)
                    f = RegularGridInterpolator(
                        input_parameters_single_values_tuple, output_values, bounds_error=False
                    )
                    output_interpolated_values = f(self.composed_distrib_sample)
                    output_interpolated_arrays.append(output_interpolated_values)
                output_interpolated_arrays = list(np.stack(output_interpolated_arrays).T)
                self.output_interpolated_values_df[f'{output_name}'] = output_interpolated_arrays

    def check_inputs_consistency(self):
        """check consistency between inputs from eval_inputs and samples_inputs_df"""
        inputs_dict = self.get_sosdisc_inputs()
        eval_inputs = inputs_dict[self.EVAL_INPUTS]
        selected_inputs = eval_inputs[eval_inputs['selected_input'] == True][
            'full_name'
        ]
        selected_inputs = selected_inputs.tolist()
        inputs_from_samples = inputs_dict['samples_inputs_df']
        input_from_samples = list(inputs_from_samples.columns)[1:]

        if set(selected_inputs) != set(input_from_samples):
            self.logger.exception(
                'selected inputs from eval inputs must be the same than inputs from the samples inputs dataframe'
            )

    def get_chart_filter_list(self):
        """
        For the outputs, making a graph for tco vs year for each range and for specific
        value of ToT with a shift of five year between then
        """

        chart_filters = []

        in_names = []
        out_names = []
        if 'data_details_df' in self.get_sosdisc_inputs():
            data_df = self.get_sosdisc_inputs(['data_details_df'])
            in_names = data_df.loc[data_df[SoSWrapp.TYPE] == 'input', 'name'].to_list()
        if 'output_interpolated_values_df' in self.get_sosdisc_outputs():
            out_df = (
                self.get_sosdisc_outputs(['output_interpolated_values_df'])
                .keys()
                .to_list()
            )
            out_names = [n.split('.')[-1] for n in out_df]

        names_list = in_names + out_names
        chart_list = [n + ' Distribution' for n in names_list]
        chart_filters.append(ChartFilter('Charts', chart_list, chart_list, 'Charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):
        """For the outputs, making a bar graph with gradients values"""

        instanciated_charts = []
        graphs_list = []

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'Charts':
                    graphs_list = chart_filter.selected_values

        if 'output_interpolated_values_df' in self.get_sosdisc_outputs():
            output_distrib_df = deepcopy(
                self.get_sosdisc_outputs('output_interpolated_values_df')
            )
        if 'input_parameters_samples_df' in self.get_sosdisc_outputs():
            input_parameters_distrib_df = deepcopy(
                self.get_sosdisc_outputs('input_parameters_samples_df')
            )
        if 'data_details_df' in self.get_sosdisc_inputs():
            self.data_details = deepcopy(self.get_sosdisc_inputs(['data_details_df']))
        if 'input_distribution_parameters_df' in self.get_sosdisc_inputs():
            input_distribution_parameters_df = deepcopy(
                self.get_sosdisc_inputs(['input_distribution_parameters_df'])
            )
        if 'confidence_interval' in self.get_sosdisc_inputs():
            confidence_interval = (

                    deepcopy(self.get_sosdisc_inputs(['confidence_interval'])) / 100
            )

        input_parameters_names = self.get_sosdisc_outputs('input_parameters_names')
        pure_float_input_names = self.get_sosdisc_outputs('pure_float_input_names')
        dict_array_float_names = self.get_sosdisc_outputs('dict_array_float_names')
        float_output_names = self.get_sosdisc_outputs('float_output_names')
        for input_name in input_parameters_names:
            input_distrib_name = input_name + ' Distribution'

            if input_distrib_name in graphs_list:
                if input_name in pure_float_input_names:
                    # input is of type float -> historgram
                    input_distrib = list(input_parameters_distrib_df[input_name])
                    new_chart = self.input_histogram_graph(
                        input_distrib,
                        input_name,
                        input_distribution_parameters_df,
                        confidence_interval,
                    )
                    instanciated_charts.append(new_chart)
                else:
                    # input is of type array -> array uncertainty plot
                    input_distrib = list(input_parameters_distrib_df[dict_array_float_names[input_name]].values)
                    new_chart = self.array_uncertainty_plot(list_of_arrays=input_distrib,
                                                            name=input_name)
                    instanciated_charts.append(new_chart)

        for output_name in list(output_distrib_df.columns):
            output_distrib = list(output_distrib_df[output_name])
            output_distrib_name = output_name.split('.')[-1] + ' Distribution'
            if output_name in float_output_names:
                # output type is float -> histograme
                if not all(np.isnan(output_distrib)):
                    if output_distrib_name in graphs_list:
                        new_chart = self.output_histogram_graph(
                            output_distrib, output_name, confidence_interval
                        )
                        instanciated_charts.append(new_chart)
            else:
                # output type is array -> array_uncertainty plot
                if output_distrib_name in graphs_list:
                    new_chart = self.array_uncertainty_plot(list_of_arrays=output_distrib,
                                                            name=output_name,
                                                            is_output=True)
                    instanciated_charts.append(new_chart)

        return instanciated_charts

    def input_histogram_graph(
            self, data, data_name, distrib_param, confidence_interval
    ):
        """generates a histogram plot for input of type float"""
        name, unit = self.data_details.loc[self.data_details["variable"] == data_name][
            ["name", 'unit']
        ].values[0]
        hist_y = go.Figure()
        hist_y.add_trace(go.Histogram(x=list(data), nbinsx=100, histnorm='probability'))

        # statistics on data list
        distribution_type = distrib_param.loc[distrib_param['parameter'] == data_name][
            'distribution'
        ].values[0]
        data_list = [x for x in data if not np.isnan(x)]
        bins = np.histogram_bin_edges(data_list, bins=100)
        hist = np.histogram(data_list, bins=bins)[0]
        norm_hist = hist / np.cumsum(hist)[-1]

        y_max = max(norm_hist)
        median = np.median(data_list)
        y_mean = np.mean(data_list)
        if distribution_type in ['Normal', 'LogNormal']:
            # left boundary confidence interval
            lb = float(format(1 - confidence_interval, '.2f')) / 2
            y_left_boundary = np.nanquantile(list(data), lb)
            y_right_boundary = np.nanquantile(list(data), 1 - lb)
        else:
            y_left_boundary, y_right_boundary = distrib_param.loc[
                distrib_param['parameter'] == data_name
                ][['lower_parameter', 'upper_parameter']].values[0]

        hist_y.update_layout(
            xaxis=dict(title=name, ticksuffix=unit), yaxis=dict(title='Probability')
        )

        hist_y.add_shape(
            type='line',
            xref='x',
            yref='paper',
            x0=y_left_boundary,
            x1=y_left_boundary,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            ),
        )

        hist_y.add_shape(
            type='line',
            xref='x',
            yref='paper',
            x0=y_right_boundary,
            x1=y_right_boundary,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            ),
        )

        hist_y.add_shape(
            type='line',
            xref='x',
            yref='paper',
            x0=y_mean,
            x1=y_mean,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            ),
        )

        hist_y.add_annotation(
            x=y_left_boundary,
            y=y_max,
            font=dict(color="black", size=12),
            text=" Lower parameter ",
            showarrow=False,
            xanchor="right",
        )
        hist_y.add_annotation(
            x=y_right_boundary,
            y=y_max,
            font=dict(color="black", size=12),
            text=" Upper parameter ",
            showarrow=False,
            xanchor="left",
        )
        hist_y.add_annotation(
            x=y_mean,
            y=0.75 * y_max,
            font=dict(color="black", size=12),
            text=" Mean ",
            showarrow=False,
            xanchor="left",
        )
        hist_y.add_annotation(
            x=0.85,
            y=1.15,
            font=dict(family='Arial', color='#7f7f7f', size=10),
            text=f' Mean: {format_currency_legend(y_mean, unit)} <br> Median: {format_currency_legend(median, unit)} ',
            showarrow=False,
            xanchor="left",
            align="right",
            xref='paper',
            yref='paper',
            bordercolor='black',
            borderwidth=1,
        )

        hist_y.update_layout(showlegend=False)

        new_chart = InstantiatedPlotlyNativeChart(
            fig=hist_y,
            chart_name=f'{name} - {distribution_type} Distribution',
            default_legend=False,
        )

        # new_chart.to_plotly().show()

        return new_chart

    def output_histogram_graph(self, data, data_name, confidence_interval):
        """generates an histogram for output of type float"""
        name = data_name
        unit = None

        if len(data_name.split('.')) > 1:
            name = data_name.split('.')[1]

        var_name = data_name
        if var_name is not None:
            try:
                unit = self.data_details.loc[self.data_details["variable"] == var_name][
                    "unit"
                ].values[0]
            except:
                unit = None
        hist_y = go.Figure()
        hist_y.add_trace(go.Histogram(x=list(data), nbinsx=100, histnorm='probability'))

        # statistics on data list
        data_list = [x for x in data if not np.isnan(x)]
        bins = np.histogram_bin_edges(data_list, bins=100)
        hist = np.histogram(data_list, bins=bins)[0]
        norm_hist = hist / np.cumsum(hist)[-1]
        y_max = max(norm_hist)
        median = np.median(data_list)
        y_mean = np.mean(data_list)

        # left boundary confidence interval
        lb = float(format(1 - confidence_interval, '.2f')) / 2
        y_left_boundary = np.nanquantile(list(data), lb)
        y_right_boundary = np.nanquantile(list(data), 1 - lb)
        hist_y.update_layout(
            xaxis=dict(title=name, ticksuffix=unit), yaxis=dict(title='Probability')
        )

        hist_y.add_shape(
            type='line',
            xref='x',
            yref='paper',
            x0=y_left_boundary,
            x1=y_left_boundary,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            ),
        )

        hist_y.add_shape(
            type='line',
            xref='x',
            yref='paper',
            x0=y_right_boundary,
            x1=y_right_boundary,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            ),
        )
        hist_y.add_shape(
            type='line',
            xref='x',
            yref='paper',
            x0=y_mean,
            x1=y_mean,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            ),
        )

        hist_y.add_shape(
            type='rect',
            xref='x',
            yref='paper',
            x0=y_left_boundary,
            x1=y_right_boundary,
            y0=0,
            y1=1,
            line=dict(color="LightSeaGreen"),
            fillcolor="PaleTurquoise",
            opacity=0.2,
        )

        hist_y.add_annotation(
            x=y_left_boundary,
            y=y_max,
            font=dict(color="black", size=12),
            text=f' {format_currency_legend(y_left_boundary, unit)} ',
            showarrow=False,
            xanchor="right",
        )

        hist_y.add_annotation(
            x=y_right_boundary,
            y=y_max,
            font=dict(color="black", size=12),
            text=f' {format_currency_legend(y_right_boundary, unit)}',
            showarrow=False,
            xanchor="left",
        )

        hist_y.add_annotation(
            x=y_mean,
            y=0.75 * y_max,
            font=dict(color="black", size=12),
            text=f' {format_currency_legend(y_mean, unit)} ',
            showarrow=False,
            xanchor="left",
        )

        hist_y.add_annotation(
            x=0.60,
            y=1.15,
            font=dict(family='Arial', color='#7f7f7f', size=10),
            text=f'Confidence Interval: {int(confidence_interval * 100)} % [{format_currency_legend(y_left_boundary, "")}, {format_currency_legend(y_right_boundary, "")}] {unit} <br> Mean: {format_currency_legend(y_mean, unit)} <br> Median: {format_currency_legend(median, unit)} ',
            showarrow=False,
            xanchor="left",
            align="right",
            xref='paper',
            yref='paper',
            bordercolor='black',
            borderwidth=1,
        )

        hist_y.update_layout(showlegend=False)

        new_chart = InstantiatedPlotlyNativeChart(
            fig=hist_y, chart_name=f'{name} - Distribution', default_legend=False
        )

        return new_chart

    def array_uncertainty_plot(self,
                               list_of_arrays: list[np.ndarray],
                               name: str,
                               is_output: bool = False
                               ):
        """
        Returns a chart for 1-dimensional array types inputs/outputs (time series typically), with
        - all the samples (all the time series)
        - the mean time serie
        - if output: the lower and upper quantiles
        - if input: the parameters of the distribution (PERT, Normal, LogNormal)
        """
        arrays_x = list(range(len(list_of_arrays[0])))
        mean_array = np.nanmean(list_of_arrays, axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=arrays_x, y=list_of_arrays[0].tolist(),
                                 line=dict(color='rgba(169,169,169,0.1)'),
                                 name="samples"))
        for time_serie in list_of_arrays[1:]:
            fig.add_trace(go.Scatter(x=arrays_x, y=time_serie.tolist(),
                                     line=dict(color='rgba(169,169,169,0.1)'),
                                     showlegend=False))

        fig.add_trace(go.Scatter(x=arrays_x, y=mean_array.tolist(), name='Mean', line=dict(color='black', dash='dash')))

        input_distribution_parameters_df = self.get_sosdisc_inputs("input_distribution_parameters_df")
        distribution = input_distribution_parameters_df.loc[
            input_distribution_parameters_df["parameter"] == name]['distribution'].values[0] if not is_output \
            else ''
        if distribution == 'PERT':
            lower_parameter, upper_parameter = \
                input_distribution_parameters_df.loc[input_distribution_parameters_df["parameter"] == name][
                    ['lower_parameter', 'upper_parameter']].values[0]
            fig.add_trace(go.Scatter(x=arrays_x, y=list(lower_parameter),
                                     line=dict(color='green', dash='dash'),
                                     name='lower parameter'))
            fig.add_trace(go.Scatter(x=arrays_x, y=list(upper_parameter),
                                     line=dict(color='blue', dash='dash'),
                                     name='upper parameter'))
        elif is_output or distribution in ['Normal', 'LogNormal']:
            confidence_interval = float(self.get_sosdisc_inputs('confidence_interval')) / 100
            ql = float(format(1 - confidence_interval, '.2f')) / 2
            qu = 1 - ql
            quantile_lower = np.nanquantile(list_of_arrays, q=ql, axis=0)
            quantile_upper = np.nanquantile(list_of_arrays, q=qu, axis=0)
            fig.add_trace(go.Scatter(x=arrays_x, y=quantile_lower.tolist(),
                                     line=dict(color='green', dash='dash'),
                                     name=f'quantile {int(100 * ql)}%'))
            fig.add_trace(go.Scatter(x=arrays_x, y=quantile_upper.tolist(),
                                     line=dict(color='blue', dash='dash'),
                                     name=f'quantile {int(100 * qu)}%'))

        fig.update_layout(title='Multiple Time Series')

        new_chart = InstantiatedPlotlyNativeChart(
            fig=fig, chart_name=f'{name} - {distribution} Distribution', default_legend=False
        )

        return new_chart
