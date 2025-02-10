'''
Copyright 2025 Capgemini

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

from copy import deepcopy
from enum import auto
from typing import Any, ClassVar, Iterable

import numpy as np
import plotly.graph_objects as go
from gemseo.datasets.io_dataset import IODataset
from gemseo.uncertainty import create_statistics
from numpy import size
from pandas import DataFrame, concat
from strenum import StrEnum

from sostrades_core.execution_engine.disciplines_wrappers.monte_carlo_driver_wrapper import (
    SoSOutputNames as MCOutputNames,
)
from sostrades_core.execution_engine.proxy_monte_carlo_driver import ProxyMonteCarloDriver
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)
from sostrades_core.tools.post_processing.post_processing_tools import format_currency_legend


class SoSInputNames(MCOutputNames):
    """The names of the input parameters."""

    probability_threshold = auto()
    """The threshold value used to compute the probability.

    The analysis will return the probability that the output is above the threshold.
    Can be a single float, or an iterable of the same length as the number of outputs.
    """

    tolerance_confidence = auto()
    """The confidence for the tolerance coverage.

    Must be between 0 and 1.
    Can be a single float, or an iterable of the same length as the number of outputs.
    """

    tolerance_coverage = auto()
    """The proportion of output values that must be in the tolerance interval with the specified confidence.

    Must be between 0 and 1.
    Can be a single float, or an iterable of the same length as the number of outputs.
    """


class SoSOutputNames(StrEnum):
    """The names of the output parameters."""

    statistics = auto()
    """The statistics of the output samples."""


class UncertaintyAnalysis(SoSWrapp):
    """A discipline to perform statistical analysis on a random sampling."""

    # ontology information
    _ontology_data: ClassVar[dict[str, str]] = {
        "label": "Uncertainty Analysis",
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

    SoSInputNames: ClassVar[type[SoSInputNames]] = SoSInputNames

    SoSOutputNames: ClassVar[type[SoSOutputNames]] = SoSOutputNames

    DESC_IN: ClassVar[dict[str, Any]] = ProxyMonteCarloDriver.DESC_OUT

    DESC_IN.update({
        SoSInputNames.probability_threshold: {SoSWrapp.DEFAULT: 0.0},
        SoSInputNames.tolerance_confidence: {SoSWrapp.DEFAULT: 0.95},
        SoSInputNames.tolerance_coverage: {SoSWrapp.DEFAULT: 0.99},
    })

    DESC_OUT: ClassVar[dict[str, Any]] = {
        SoSOutputNames.statistics: {
            SoSWrapp.TYPE: 'dataframe',
            'unit': None,
        },
    }

    _dataset: IODataset
    """The dataset of input/output samples."""

    _numerical_parameters: dict[str, float | Iterable]
    """The numerical parameters for the analysis."""

    def _pre_process_data(self) -> None:
        """Read the input data and pre-process it for the analysis.

        Each inputs and outputs are split into 1D components.
        The data is stored in a IODataset that can be used for statistics and sensitivity analysis.
        """
        self._dataset = IODataset()
        input_samples = self.get_sosdisc_inputs(self.SoSInputNames.input_samples)
        for input_name in input_samples.columns:
            if s := size(input_samples.loc[0, input_name]) == 1:
                self._dataset.add_input_variable(input_name, input_samples[input_name])
                continue
            # Split the variable into 1D components
            for i in range(s):
                self._dataset.add_input_variable(
                    f"{input_name}_{i}", [sample[i] for sample in input_samples[input_name]]
                )
        output_samples = self.get_sosdisc_inputs(self.SoSInputNames.output_samples)
        for output_name in output_samples.columns:
            if s := size(output_samples.loc[0, output_name]) == 1:
                self._dataset.add_output_variable(output_name, output_samples[output_name])
                continue
            # Split the variable into 1D components
            for i in range(s):
                self._dataset.add_output_variable(
                    f"{output_name}_{i}", [sample[i] for sample in output_samples[output_name]]
                )

    def _check_parameters(self) -> None:
        """Check the numerical parameters.

        Raises:
            ValueError: If one of the parameters has the wrong size.
        """
        param_names = (
            SoSInputNames.probability_threshold,
            SoSInputNames.tolerance_confidence,
            SoSInputNames.tolerance_coverage,
        )
        self._numerical_parameters = {param: self.get_sosdisc_inputs(param) for param in param_names}
        required_size = len(self._dataset.output_names)
        msg = ""
        for param, value in self._numerical_parameters.items():
            if size(value) not in (1, required_size):
                msg += f"    Parameter {param} must be of size 1 or {required_size}, got {value}\n"
        if msg:
            msg = "Uncertainty Analysis discipline:\n" + msg
            raise ValueError(msg)

    def _compute_statistics(self) -> None:
        """Compute the statistics of the output samples."""
        analysis = create_statistics(self._dataset, variable_names=self._dataset.output_names)
        mean = DataFrame(analysis.compute_mean())
        std = DataFrame(analysis.compute_standard_deviation())
        cv = DataFrame(analysis.compute_variation_coefficient())
        threshold = self._numerical_parameters[SoSInputNames.probability_threshold]
        proba = DataFrame(analysis.compute_probability(threshold))
        coverage = self._numerical_parameters[SoSInputNames.tolerance_coverage]
        confidence = self._numerical_parameters[SoSInputNames.tolerance_confidence]
        lower_interval = DataFrame(analysis.compute_tolerance_interval(coverage, confidence, side="LOWER"))
        upper_interval = DataFrame(analysis.compute_tolerance_interval(coverage, confidence, side="UPPER"))
        both_sides_interval = DataFrame(analysis.compute_tolerance_interval(coverage, confidence, side="BOTH"))
        df = concat((mean, std, cv, proba, lower_interval, upper_interval, both_sides_interval), axis=0)
        tolerance_interval_index = f"tolerance interval (coverage={coverage * 100} %, confidence={confidence * 100} %)"
        df.index = [
            "mean",
            "standard deviation",
            "coefficient of variation",
            f"P[X] > {threshold}",
            f"lower {tolerance_interval_index}",
            f"upper {tolerance_interval_index}",
            f"both-sides {tolerance_interval_index}",
        ]
        self.store_sos_outputs_values({SoSOutputNames.statistics: df})

    def run(self) -> None:
        """Run the uncertainty analysis."""
        self._pre_process_data()
        self._check_parameters()
        self._compute_statistics()

    # TODO: pas fait
    def get_chart_filter_list(self):
        """Get the available charts.

        For the outputs, making a graph for tco vs year for each range and for specific
        value of ToT with a shift of five year between then.
        """
        chart_filters = []

        in_names = []
        out_names = []
        if "data_details_df" in self.get_sosdisc_inputs():
            data_df = self.get_sosdisc_inputs(["data_details_df"])
            in_names = data_df.loc[data_df[SoSWrapp.TYPE] == "input", "name"].to_list()
        if "output_interpolated_values_df" in self.get_sosdisc_outputs():
            out_df = self.get_sosdisc_outputs(["output_interpolated_values_df"]).keys().to_list()
            out_names = [n.split(".")[-1] for n in out_df]

        names_list = in_names + out_names
        chart_list = [n + " Distribution" for n in names_list]
        chart_filters.append(ChartFilter("Charts", chart_list, chart_list, "Charts"))

        return chart_filters

    def get_post_processing_list(self, filters=None):
        """For the outputs, making a bar graph with gradients values."""
        instanciated_charts = []
        graphs_list = []
        input_distribution_parameters_df = None
        input_parameters_distrib_df = None
        confidence_interval = None
        output_distrib_df = None

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == "Charts":
                    graphs_list = chart_filter.selected_values

        if "output_interpolated_values_df" in self.get_sosdisc_outputs():
            output_distrib_df = deepcopy(self.get_sosdisc_outputs("output_interpolated_values_df"))
        if "input_parameters_samples_df" in self.get_sosdisc_outputs():
            input_parameters_distrib_df = deepcopy(self.get_sosdisc_outputs("input_parameters_samples_df"))
        if "data_details_df" in self.get_sosdisc_inputs():
            self.data_details = deepcopy(self.get_sosdisc_inputs(["data_details_df"]))
        if "input_distribution_parameters_df" in self.get_sosdisc_inputs():
            input_distribution_parameters_df = deepcopy(self.get_sosdisc_inputs(["input_distribution_parameters_df"]))

        if "confidence_interval" in self.get_sosdisc_inputs():
            confidence_interval = deepcopy(self.get_sosdisc_inputs(["confidence_interval"])) / 100

        input_parameters_names = self.get_sosdisc_outputs("input_parameters_names")
        pure_float_input_names = self.get_sosdisc_outputs("pure_float_input_names")
        dict_array_float_names = self.get_sosdisc_outputs("dict_array_float_names")
        float_output_names = self.get_sosdisc_outputs("float_output_names")
        for input_name in input_parameters_names:
            input_distrib_name = input_name + " Distribution"

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
                    new_chart = self.array_uncertainty_plot(list_of_arrays=input_distrib, name=input_name)
                    instanciated_charts.append(new_chart)

        for output_name in list(output_distrib_df.columns):
            output_distrib = list(output_distrib_df[output_name])
            output_distrib_name = output_name.split(".")[-1] + " Distribution"
            if output_name in float_output_names:
                # output type is float -> histograme
                if not all(np.isnan(output_distrib)) and output_distrib_name in graphs_list:
                    new_chart = self.output_histogram_graph(output_distrib, output_name, confidence_interval)
                    instanciated_charts.append(new_chart)
            else:
                # output type is array -> array_uncertainty plot
                if output_distrib_name in graphs_list:
                    new_chart = self.array_uncertainty_plot(
                        list_of_arrays=output_distrib, name=output_name, is_output=True
                    )
                    instanciated_charts.append(new_chart)

        return instanciated_charts

    def input_histogram_graph(self, data, data_name, distrib_param, confidence_interval):
        """Generates a histogram plot for input of type float."""
        name, unit = self.data_details.loc[self.data_details["variable"] == data_name][["name", "unit"]].values[0]
        hist_y = go.Figure()
        hist_y.add_trace(go.Histogram(x=list(data), nbinsx=100, histnorm="probability"))

        # statistics on data list
        distribution_type = distrib_param.loc[distrib_param["parameter"] == data_name]["distribution"].values[0]
        data_list = [x for x in data if not np.isnan(x)]
        bins = np.histogram_bin_edges(data_list, bins=100)
        hist = np.histogram(data_list, bins=bins)[0]
        norm_hist = hist / np.cumsum(hist)[-1]

        y_max = max(norm_hist)
        median = np.median(data_list)
        y_mean = np.mean(data_list)
        if distribution_type in ["Normal", "LogNormal"]:
            # left boundary confidence interval
            lb = float(format(1 - confidence_interval, ".2f")) / 2
            y_left_boundary = np.nanquantile(list(data), lb)
            y_right_boundary = np.nanquantile(list(data), 1 - lb)
        else:
            y_left_boundary, y_right_boundary = distrib_param.loc[distrib_param["parameter"] == data_name][
                ["lower_parameter", "upper_parameter"]
            ].values[0]

        hist_y.update_layout(xaxis={"title": name, "ticksuffix": unit}, yaxis={"title": "Probability"})

        hist_y.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=y_left_boundary,
            x1=y_left_boundary,
            y0=0,
            y1=1,
            line={
                "color": "black",
                "width": 2,
                "dash": "dot",
            },
        )

        hist_y.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=y_right_boundary,
            x1=y_right_boundary,
            y0=0,
            y1=1,
            line={
                "color": "black",
                "width": 2,
                "dash": "dot",
            },
        )

        hist_y.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=y_mean,
            x1=y_mean,
            y0=0,
            y1=1,
            line={
                "color": "black",
                "width": 2,
                "dash": "dot",
            },
        )

        hist_y.add_annotation(
            x=y_left_boundary,
            y=y_max,
            font={"color": "black", "size": 12},
            text=" Lower parameter ",
            showarrow=False,
            xanchor="right",
        )
        hist_y.add_annotation(
            x=y_right_boundary,
            y=y_max,
            font={"color": "black", "size": 12},
            text=" Upper parameter ",
            showarrow=False,
            xanchor="left",
        )
        hist_y.add_annotation(
            x=y_mean,
            y=0.75 * y_max,
            font={"color": "black", "size": 12},
            text=" Mean ",
            showarrow=False,
            xanchor="left",
        )
        hist_y.add_annotation(
            x=0.85,
            y=1.15,
            font={"family": "Arial", "color": "#7f7f7f", "size": 10},
            text=f" Mean: {format_currency_legend(y_mean, unit)} <br> Median: {format_currency_legend(median, unit)} ",
            showarrow=False,
            xanchor="left",
            align="right",
            xref="paper",
            yref="paper",
            bordercolor="black",
            borderwidth=1,
        )

        hist_y.update_layout(showlegend=False)

        return InstantiatedPlotlyNativeChart(
            fig=hist_y,
            chart_name=f"{name} - {distribution_type} Distribution",
            default_legend=False,
        )

        # new_chart.to_plotly().show()

    def output_histogram_graph(self, data, data_name, confidence_interval):
        """Generate an histogram for output of type float."""
        name = data_name
        unit = None

        if len(data_name.split(".")) > 1:
            name = data_name.split(".")[1]

        var_name = data_name
        if var_name is not None:
            try:
                unit = self.data_details.loc[self.data_details["variable"] == var_name]["unit"].values[0]
            except Exception:
                unit = None
        hist_y = go.Figure()
        hist_y.add_trace(go.Histogram(x=list(data), nbinsx=100, histnorm="probability"))

        # statistics on data list
        data_list = [x for x in data if not np.isnan(x)]
        bins = np.histogram_bin_edges(data_list, bins=100)
        hist = np.histogram(data_list, bins=bins)[0]
        norm_hist = hist / np.cumsum(hist)[-1]
        y_max = max(norm_hist)
        median = np.median(data_list)
        y_mean = np.mean(data_list)

        # left boundary confidence interval
        lb = float(format(1 - confidence_interval, ".2f")) / 2
        y_left_boundary = np.nanquantile(list(data), lb)
        y_right_boundary = np.nanquantile(list(data), 1 - lb)
        hist_y.update_layout(xaxis={"title": name, "ticksuffix": unit}, yaxis={"title": "Probability"})

        hist_y.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=y_left_boundary,
            x1=y_left_boundary,
            y0=0,
            y1=1,
            line={
                "color": "black",
                "width": 2,
                "dash": "dot",
            },
        )

        hist_y.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=y_right_boundary,
            x1=y_right_boundary,
            y0=0,
            y1=1,
            line={
                "color": "black",
                "width": 2,
                "dash": "dot",
            },
        )
        hist_y.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=y_mean,
            x1=y_mean,
            y0=0,
            y1=1,
            line={
                "color": "black",
                "width": 2,
                "dash": "dot",
            },
        )

        hist_y.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=y_left_boundary,
            x1=y_right_boundary,
            y0=0,
            y1=1,
            line={"color": "LightSeaGreen"},
            fillcolor="PaleTurquoise",
            opacity=0.2,
        )

        hist_y.add_annotation(
            x=y_left_boundary,
            y=y_max,
            font={"color": "black", "size": 12},
            text=f" {format_currency_legend(y_left_boundary, unit)} ",
            showarrow=False,
            xanchor="right",
        )

        hist_y.add_annotation(
            x=y_right_boundary,
            y=y_max,
            font={"color": "black", "size": 12},
            text=f" {format_currency_legend(y_right_boundary, unit)}",
            showarrow=False,
            xanchor="left",
        )

        hist_y.add_annotation(
            x=y_mean,
            y=0.75 * y_max,
            font={"color": "black", "size": 12},
            text=f" {format_currency_legend(y_mean, unit)} ",
            showarrow=False,
            xanchor="left",
        )
        hist_y.add_annotation(
            x=0.60,
            y=1.15,
            font={"family": "Arial", "color": "#7f7f7f", "size": 10},
            text=f"Confidence Interval: {int(confidence_interval * 100)} "
            f"% [{format_currency_legend(y_left_boundary, '')}, {format_currency_legend(y_right_boundary, '')}] {unit} "
            f"<br> Mean: {format_currency_legend(y_mean, unit)} <br> Median: {format_currency_legend(median, unit)}",
            showarrow=False,
            xanchor="left",
            align="right",
            xref="paper",
            yref="paper",
            bordercolor="black",
            borderwidth=1,
        )

        hist_y.update_layout(showlegend=False)

        return InstantiatedPlotlyNativeChart(fig=hist_y, chart_name=f"{name} - Distribution", default_legend=False)

    def array_uncertainty_plot(self, list_of_arrays: list[np.ndarray], name: str, is_output: bool = False):
        """Plots the output of uncertainty analysis.

        Returns a chart for 1-dimensional array types inputs/outputs (time series typically), with
        - all the samples (all the time series)
        - the mean time serie
        - if output: the lower and upper quantiles
        - if input: the parameters of the distribution (PERT, Normal, LogNormal).
        """
        arrays_x = list(range(len(list_of_arrays[0])))
        mean_array = np.nanmean(list_of_arrays, axis=0)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=arrays_x,
                y=list_of_arrays[0].tolist(),
                line={"color": "rgba(169,169,169,0.1)"},
                name="samples",
            )
        )
        for time_serie in list_of_arrays[1:]:
            fig.add_trace(
                go.Scatter(
                    x=arrays_x,
                    y=time_serie.tolist(),
                    line={"color": "rgba(169,169,169,0.1)"},
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=arrays_x,
                y=mean_array.tolist(),
                name="Mean",
                line={"color": "black", "dash": "dash"},
            )
        )

        input_distribution_parameters_df = self.get_sosdisc_inputs("input_distribution_parameters_df")
        distribution = (
            input_distribution_parameters_df.loc[input_distribution_parameters_df["parameter"] == name][
                "distribution"
            ].values[0]
            if not is_output
            else ""
        )
        if distribution == "PERT":
            lower_parameter, upper_parameter = input_distribution_parameters_df.loc[
                input_distribution_parameters_df["parameter"] == name
            ][["lower_parameter", "upper_parameter"]].values[0]
            fig.add_trace(
                go.Scatter(
                    x=arrays_x,
                    y=list(lower_parameter),
                    line={"color": "green", "dash": "dash"},
                    name="lower parameter",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=arrays_x,
                    y=list(upper_parameter),
                    line={"color": "blue", "dash": "dash"},
                    name="upper parameter",
                )
            )
        elif is_output or distribution in ["Normal", "LogNormal"]:
            confidence_interval = float(self.get_sosdisc_inputs("confidence_interval")) / 100
            ql = float(format(1 - confidence_interval, ".2f")) / 2
            qu = 1 - ql
            quantile_lower = np.nanquantile(list_of_arrays, q=ql, axis=0)
            quantile_upper = np.nanquantile(list_of_arrays, q=qu, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=arrays_x,
                    y=quantile_lower.tolist(),
                    line={"color": "green", "dash": "dash"},
                    name=f"quantile {int(100 * ql)}%",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=arrays_x,
                    y=quantile_upper.tolist(),
                    line={"color": "blue", "dash": "dash"},
                    name=f"quantile {int(100 * qu)}%",
                )
            )

        fig.update_layout(title="Multiple Time Series")

        return InstantiatedPlotlyNativeChart(
            fig=fig,
            chart_name=f"{name} - {distribution} Distribution",
            default_legend=False,
        )
