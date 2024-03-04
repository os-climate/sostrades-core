'''
Copyright 2024 Capgemini

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
import numpy as np
import pandas as pd

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.gather_discipline import GatherDiscipline
from sostrades_core.execution_engine.sample_generators.tornado_chart_analysis_sample_generator import (
    TornadoChartAnalysisSampleGenerator,
)
from sostrades_core.execution_engine.proxy_sample_generator import ProxySampleGenerator
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.gather.gather_tool import gather_selected_outputs


class TornadoChartAnalysis(SoSWrapp):
    """
    Tornado chart Analysis class
    """

    # ontology information
    _ontology_data = {
        "label": "Tornado chart analysis Model",
        SoSWrapp.TYPE: "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fa-solid fa-chart-area",
        "version": "",
    }
    GATHER_OUTPUTS = GatherDiscipline.GATHER_OUTPUTS
    GATHER_OUTPUTS_DESC = GatherDiscipline.EVAL_OUTPUTS_DESC.copy()
    GATHER_OUTPUTS_DESC[SoSWrapp.NAMESPACE] = ProxySampleGenerator.NS_SAMPLING
    GATHER_OUTPUTS_DESC[SoSWrapp.VISIBILITY] = SoSWrapp.SHARED_VISIBILITY

    OUTPUT_VARIATIONS_SUFFIX = "_variations"
    INPUT_COL = "input"
    VARIATION_INPUT_COL = "input_variation"
    VARIATION_OUTPUT_COL = "output_variation"

    REFERENCE_SCENARIO_NAME = TornadoChartAnalysisSampleGenerator.REFERENCE_SCENARIO_NAME
    SCENARIO_NAME_COL = TornadoChartAnalysisSampleGenerator.SCENARIO_NAMES
    SCENARIO_VARIABLE_VARIATIONS = TornadoChartAnalysisSampleGenerator.SCENARIO_VARIABLE_VARIATIONS
    DESC_IN = {GATHER_OUTPUTS: GATHER_OUTPUTS_DESC, SCENARIO_VARIABLE_VARIATIONS: {SoSWrapp.TYPE: "dataframe"}}

    CHART_FILTER_KEY_SELECTED_OUTPUTS = "outputs"
    CHART_FILTER_KEY_SELECTED_INPUTS = "inputs"

    ACCEPTED_OUTPUT_TYPES = ProxyDiscipline.VAR_TYPE_MAP["float"]

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.selected_outputs_dict = {}

    def setup_sos_disciplines(self):
        """setup sos disciplines"""
        data_in = self.get_data_in()
        if data_in != {}:
            # Add the outputs of the driver eval selected in gather_outputs in input of the disc
            dynamic_outputs = {}
            dynamic_inputs = {}
            if self.GATHER_OUTPUTS in data_in:
                gather_outputs = self.get_sosdisc_inputs(self.GATHER_OUTPUTS)
                # get only variables that are selected
                self.selected_outputs_dict = gather_selected_outputs(gather_outputs, GatherDiscipline.GATHER_SUFFIX)
                # add dynamic input for each output name
                for output_name in self.selected_outputs_dict.values():
                    dynamic_inputs[output_name] = {
                        SoSWrapp.TYPE: "dict",
                        SoSWrapp.NAMESPACE: ProxySampleGenerator.NS_SAMPLING,
                        SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                    }
                    dynamic_outputs[f"{output_name}{self.OUTPUT_VARIATIONS_SUFFIX}"] = {
                        SoSWrapp.TYPE: "dataframe",
                        SoSWrapp.DATAFRAME_DESCRIPTOR: {
                            self.INPUT_COL: ("string", None, False),
                            self.VARIATION_INPUT_COL: ("float", None, False),
                            self.VARIATION_OUTPUT_COL: ("float", None, False),
                        },
                    }

            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def __get_input_variables_list_and_df(self):
        """
        Get the list of input variable for the tornado chart
        """
        variation_data_df = self.get_sosdisc_inputs(self.SCENARIO_VARIABLE_VARIATIONS)
        # get the list of inputs by removing the column of scenario_names
        variables_list = [col for col in variation_data_df.columns if col != self.SCENARIO_NAME_COL]
        return variables_list, variation_data_df

    def run(self):
        dict_values = {}
        if len(self.selected_outputs_dict) > 0:
            variables_list, variation_data_df = self.__get_input_variables_list_and_df()

            for output_name in self.selected_outputs_dict.values():
                output_data = self.get_sosdisc_inputs(output_name)

                # create one output for each output_data
                output_variations_dict = {
                    self.INPUT_COL: [],
                    self.VARIATION_INPUT_COL: [],
                    self.VARIATION_OUTPUT_COL: [],
                }

                # create a dataframe that contains scenario_name, inputs variation, output value per scenario
                variation_with_output_df = variation_data_df.merge(
                    pd.DataFrame({self.SCENARIO_NAME_COL: output_data.keys(), output_name: list(output_data.values())}),
                    on=self.SCENARIO_NAME_COL,
                )
                # get reference value
                reference_value = output_data[self.REFERENCE_SCENARIO_NAME]

                # compute the output variation for each input variation
                for input_name in variables_list:
                    # get the rows where the input variations is not 0%
                    input_variations_df = variation_with_output_df[variation_with_output_df[input_name] != 0.0]

                    # build the variations results: the input, the variation on input, the variation on output
                    output_variations_dict[self.INPUT_COL].extend([input_name] * len(input_variations_df))
                    output_variations_dict[self.VARIATION_INPUT_COL].extend(
                        list(input_variations_df[input_name].values)
                    )

                    # initialize with None values
                    computed_variations = [None] * len(input_variations_df)
                    # compute the variations
                    if reference_value is not None:

                        if isinstance(reference_value, dict):
                            if len(reference_value) > 0 and reference_value.values().first() is not None:

                                # check subtype
                                if isinstance(reference_value.values().first(), dict):
                                    # case dict of dict
                                    computed_variations = [
                                        self._compute_dict_of_dict_outputs(reference_value, output_dict_dict)
                                        for output_dict_dict in input_variations_df[output_name].values
                                    ]

                                elif isinstance(reference_value.values().first(), pd.DataFrame):
                                    # case dict of df
                                    computed_variations = [
                                        self._compute_dict_of_dataframe_outputs(reference_value, output_dict_df)
                                        for output_dict_df in input_variations_df[output_name].values
                                    ]
                                elif isinstance(reference_value.values().first(), float) or isinstance(
                                    reference_value.values().first(), int
                                ):
                                    # case dict
                                    computed_variations = [
                                        self._compute_dict_outputs(reference_value, output)
                                        for output in input_variations_df[output_name].values
                                    ]

                        elif isinstance(reference_value, pd.DataFrame):
                            computed_variations = [
                                self._compute_dataframe_outputs(reference_value, output_df)
                                for output_df in input_variations_df[output_name].values
                            ]
                        elif isinstance(reference_value, list) or isinstance(reference_value, np.ndarray):
                            computed_variations = [
                                self._compute_array_output(reference_value, output)
                                for output in input_variations_df[output_name].values
                            ]
                        elif isinstance(reference_value, float) or isinstance(reference_value, int):
                            computed_variations = [
                                self._compute_output(reference_value, output)
                                for output in input_variations_df[output_name].values
                            ]

                    output_variations_dict[self.VARIATION_OUTPUT_COL].extend(computed_variations)
                dict_values[f"{output_name}{self.OUTPUT_VARIATIONS_SUFFIX}"] = pd.DataFrame(output_variations_dict)

        self.store_sos_outputs_values(dict_values)

    def __get_outputs_compatible_tornado_types(self):
        """
        Get outputs with valid types for tornado chart analysis
        """
        outputs_list = list(self.selected_outputs_dict.values())
        outputs_with_valid_types = []
        for output_name in outputs_list:
            output_df = self.get_sosdisc_outputs(f"{output_name}{self.OUTPUT_VARIATIONS_SUFFIX}")
            values = output_df[TornadoChartAnalysis.VARIATION_OUTPUT_COL].values
            if len(values) > 0:
                if isinstance(values[0], TornadoChartAnalysis.ACCEPTED_OUTPUT_TYPES):
                    outputs_with_valid_types.append(output_name)
        return outputs_with_valid_types

    def _compute_output(self, reference_value: float, output: float) -> float:
        result = 0.0
        if reference_value != 0.0 and reference_value is not None:
            result = 100.0 * (output - reference_value) / reference_value

        return result

    def _compute_array_output(self, reference_value, output):
        result = [
            self._compute_output(reference_value[i], output[i])
            for i in range(0, len(reference_value))
            if reference_value[i] != 0.0 and isinstance(reference_value[i], float)
        ]

        return result

    def _compute_dict_outputs(self, reference_value_dict: dict, output_dict: dict) -> dict:
        """
        Compute the variation of outputs in case of output type is dict of float or int, else return empty array
        :param reference_value_dict_dict: reference value (variation at 0%)
        :type reference_value_dict_dict: dict of float or int
        :param output_dict_dict_data: output values at each variation (+/-%)
        :type output_dict_dict_data: dict of float or int
        :return: computed variations
        :return type: dict of float or int
        """
        output_variations = {
            key: self._compute_output(float(reference_value_dict[key]), float(output_dict[key]))
            for key in output_dict.keys()
            if isinstance(reference_value_dict[key], float) or isinstance(reference_value_dict[key], int)
        }

        return output_variations

    def _compute_dataframe_outputs(self, reference_value: pd.DataFrame, output_df: pd.DataFrame) -> dict:
        """
        Compute the variation of outputs in case of output type is dict of float or int, else return empty array
        :param reference_value_dict_dict: reference value (variation at 0%)
        :type reference_value_dict_dict: dict of float or int
        :param output_dict_dict_data: output values at each variation (+/-%)
        :type output_dict_dict_data: dict of float or int
        :return: computed variations
        :return type: dict of float or int
        """
        return 100.0 * (output_df - reference_value).divide(reference_value, fill_value=0.0)

    def _compute_dict_of_dict_outputs(self, reference_value_dict_dict: dict, output_dict_dict: dict) -> dict:
        """
        Compute the variation of outputs in case of output type is dict of dict of float or int, else return empty array
        :param reference_value_dict_dict: reference value (variation at 0%)
        :type reference_value_dict_dict: dict of dict of float or int
        :param output_dict_dict: output values at each variation (+/-%)
        :type output_dict_dict: dict of dict of float or int
        :return: computed variations
        :return type: dict of dict of float or int
        """
        # check sub type:
        reference_value_dict = reference_value_dict_dict.values().first()
        if isinstance(reference_value_dict.values().first(), float) or isinstance(
            reference_value_dict.values().first(), int
        ):
            output_variations = {
                key: self._compute_dict_of_outputs(reference_value_dict_dict[key], output_dict_dict[key])
                for key in output_dict_dict.keys()
            }

        return output_variations

    def _compute_dict_of_dataframe_outputs(self, reference_value_dict_df: dict, output_dict_df: dict) -> dict:
        """
        Compute the variation of outputs in case of output type is dict of dataframe of float or int, else return empty array
        :param reference_value_dict_df: reference value (variation at 0%)
        :type reference_value_dict_df: dict of dataframe of float or int
        :param output_dict_df_data: output values at each variation (+/-%)
        :type output_dict_df_data: dict of dataframe of float or int
        :return: computed variations
        :return type: dict of dataframe of float or int
        """
        output_variations = {
            key: self._compute_dataframe_outputs(reference_value_dict_df[key], output_dict_df[key])
            for key in output_dict_df.keys()
        }

        return output_variations

    def get_chart_filter_list(self):
        """
        post processing function designed to build filters
        """
        filters = []
        outputs_list = self.__get_outputs_compatible_tornado_types()
        filters.append(
            ChartFilter(
                "Outputs variables", outputs_list, outputs_list, TornadoChartAnalysis.CHART_FILTER_KEY_SELECTED_OUTPUTS
            )
        )

        inputs_list, _ = self.__get_input_variables_list_and_df()
        filters.append(
            ChartFilter(
                "Input variables", inputs_list, inputs_list, TornadoChartAnalysis.CHART_FILTER_KEY_SELECTED_INPUTS
            )
        )
        return filters

    def get_post_processing_list(self, filters=None):
        """
        post processing function designed to build graphs
        """
        # For the outputs, making a bar graph with gradients values

        instanciated_charts = []
        # Default value if no filter
        selected_outputs_list = []
        selected_inputs_list = []
        if filters is None:
            filters = self.get_chart_filter_list()
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == TornadoChartAnalysis.CHART_FILTER_KEY_SELECTED_OUTPUTS:
                    selected_outputs_list = chart_filter.selected_values
                if chart_filter.filter_key == TornadoChartAnalysis.CHART_FILTER_KEY_SELECTED_INPUTS:
                    selected_inputs_list = chart_filter.selected_values

        for selected_output in selected_outputs_list:
            selected_output_df = self.get_sosdisc_outputs(f"{selected_output}{self.OUTPUT_VARIATIONS_SUFFIX}")
            if isinstance(selected_output_df[self.VARIATION_OUTPUT_COL].iloc[0], float):
                instanciated_charts.append(
                    self.__make_tornado_chart(
                        variation_df=selected_output_df,
                        selected_inputs=selected_inputs_list,
                        output_variable_name=selected_output,
                    )
                )
        return instanciated_charts

    @staticmethod
    def __make_tornado_chart(
        variation_df: pd.DataFrame, selected_inputs: list[str], output_variable_name: str
    ) -> TwoAxesInstanciatedChart:
        """
        Make a tornado chart

        :param variation_df: the output variation df
        :type variation_df: pd.DataFrame
        :param selected_inputs: the selected inputs
        :type selected_inputs: list[str]
        :param output_variable_name: the name of the output variable
        :type output_variable_name: str
        :return: The tornado chart as a TwoAxesInstanciatedChart
        :rtype: TwoAxesInstanciatedChart
        """
        # We should only have 2 variations, otherwise there is an issue before
        variations = list(variation_df[TornadoChartAnalysis.VARIATION_INPUT_COL].unique())
        assert len(variations) == 2
        variation_value_neg, variation_value_pos = list(sorted(variations))

        def get_output_variation_value(input_name: str, variation_value: float) -> float:
            """
            Gets the output variation for an input and a given variation
            """
            result = variation_df[
                (variation_df[TornadoChartAnalysis.INPUT_COL] == input_name)
                & (variation_df[TornadoChartAnalysis.VARIATION_INPUT_COL] == variation_value)
            ][TornadoChartAnalysis.VARIATION_OUTPUT_COL]

            # We should have 1 result, otherwise there is an internal error
            # Because we have multiple results for the same variation
            assert len(result) == 1
            return result.iloc[0]

        if abs(variation_value_neg) == abs(variation_value_pos):
            # If both variations are the same, simplify title
            chart_name = f"{output_variable_name} sensitivity to input variations of {abs(variation_value_neg)}%"
        else:
            chart_name = f"{output_variable_name} sensitivity to input variations of [{variation_value_neg}%, {variation_value_pos}%]"
        new_chart = TwoAxesInstanciatedChart(
            "Sensitivity (%)", "", chart_name=chart_name, stacked_bar=False, bar_orientation="h"
        )

        # Compute all bars
        abscissa_list = []
        ordinate_list = []
        ordinate_list_minus = []
        for input_name in selected_inputs:
            pos = get_output_variation_value(input_name, variation_value_pos)
            neg = get_output_variation_value(input_name, variation_value_neg)

            abscissa_list.append(input_name)
            ordinate_list.append(pos)
            ordinate_list_minus.append(neg)

        # Sort by highest positive bar size
        abs_list = [abs(ordinate) for ordinate in ordinate_list]
        abscissa_list = [abscissa for _, abscissa in sorted(zip(abs_list, abscissa_list))]
        ordinate_list = [abscissa for _, abscissa in sorted(zip(abs_list, ordinate_list))]
        ordinate_list_minus = [abscissa for _, abscissa in sorted(zip(abs_list, ordinate_list_minus))]

        # Instanciate and add serie
        new_series = InstanciatedSeries(ordinate_list, abscissa_list, f"{variation_value_pos}%", "bar")
        new_series2 = InstanciatedSeries(ordinate_list_minus, abscissa_list, f"{variation_value_neg}%", "bar")
        new_chart.series.append(new_series)
        new_chart.series.append(new_series2)
        return new_chart
