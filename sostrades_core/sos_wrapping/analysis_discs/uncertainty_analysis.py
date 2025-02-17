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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import plotly.express as px
import plotly.graph_objects as go
from gemseo.datasets.io_dataset import IODataset
from gemseo.uncertainty import create_statistics
from numpy import array, ndarray, size
from pandas import DataFrame, Series, concat

from sostrades_core.execution_engine.disciplines_wrappers.monte_carlo_driver_wrapper import MonteCarloDriverWrapper
from sostrades_core.execution_engine.proxy_monte_carlo_driver import ProxyMonteCarloDriver
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)
from sostrades_core.tools.post_processing.post_processing_tools import format_currency_legend

if TYPE_CHECKING:
    from collections.abc import Iterable

_NAMESPACE_DRIVER = "ns_driver_MC"  # must be here to be used in list comprehension in DESC_IN


@dataclass
class SoSInputNames(MonteCarloDriverWrapper.SoSOutputNames):
    """The names of the input parameters."""

    PROBABILITY_THRESHOLD = "probability_threshold"
    """The threshold value used to compute the probability.

    The analysis will return the probability that the output is above the threshold.
    Can be a single float, or an iterable of the same length as the number of outputs.
    """


@dataclass
class SoSOutputNames:
    """The names of the additional output parameters."""

    STATISTICS = "statistics"
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

    NAMESPACE_DRIVER: str = _NAMESPACE_DRIVER
    """The namespace of the Monte Carlo driver."""

    SoSInputNames: ClassVar[type[SoSInputNames]] = SoSInputNames

    SoSOutputNames: ClassVar[type[SoSOutputNames]] = SoSOutputNames

    DESC_IN: ClassVar[dict[str, Any]] = {
        name: desc | {"namespace": _NAMESPACE_DRIVER, "visibility": SoSWrapp.SHARED_VISIBILITY}
        for name, desc in ProxyMonteCarloDriver.DESC_OUT.items()
    }

    DESC_IN.update({
        SoSInputNames.PROBABILITY_THRESHOLD: {
            SoSWrapp.TYPE: "array",
            SoSWrapp.DEFAULT: array([0.0]),
        },
    })

    DESC_OUT: ClassVar[dict[str, Any]] = {
        SoSOutputNames.STATISTICS: {
            SoSWrapp.TYPE: "dataframe",
            "unit": None,
            # "visibility": SoSWrapp.SHARED_VISIBILITY,
            # "namespace": NAMESPACE_DRIVER,
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
        input_samples = self.get_sosdisc_inputs(self.SoSInputNames.INPUT_SAMPLES)
        for input_name in input_samples.columns:
            if s := size(input_samples.loc[0, input_name]) == 1:
                self._dataset.add_input_variable(input_name, input_samples[input_name])
                continue
            # Split the variable into 1D components
            for i in range(s):
                self._dataset.add_input_variable(
                    f"{input_name}_{i}", [sample[i] for sample in input_samples[input_name]]
                )
        output_samples = self.get_sosdisc_inputs(self.SoSInputNames.OUTPUT_SAMPLES)
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
        param_names = (SoSInputNames.PROBABILITY_THRESHOLD,)
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
        """Compute the statistics of the samples."""
        analysis = create_statistics(self._dataset)

        # Compute the mean, median, std, cv
        mean = DataFrame(analysis.compute_mean())
        median = DataFrame(analysis.compute_median())
        std = DataFrame(analysis.compute_standard_deviation())
        cv = DataFrame(analysis.compute_variation_coefficient())

        # Compute the probability to be above a certain threshold
        threshold = self._numerical_parameters[SoSInputNames.PROBABILITY_THRESHOLD]
        if size(threshold) == 1:
            thresh = dict.fromkeys(analysis.names, threshold[0])
        else:
            thresh = {name: threshold.flatten()[i] for i, name in enumerate(analysis.names)}
        proba = DataFrame(analysis.compute_probability(thresh))

        df = concat((mean, median, std, cv, proba), axis=0)
        df.index = [
            "mean",
            "median",
            "standard deviation",
            "coefficient of variation",
            f"P[X] > {threshold}",
        ]
        self.store_sos_outputs_values({SoSOutputNames.STATISTICS: df})

    def run(self) -> None:
        """Run the uncertainty analysis."""
        self._pre_process_data()
        self._check_parameters()
        self._compute_statistics()

    def get_chart_filter_list(self) -> list[ChartFilter]:
        """Get the available charts.

        Returns:
            A list containing the chart filter.
        """
        var_names = self._dataset.columns
        chart_list = [n + " distribution" for n in var_names]
        chart_list.append("Boxplot")
        return [ChartFilter("Charts", chart_list, chart_list, "Charts")]

    def get_post_processing_list(self) -> list[InstantiatedPlotlyNativeChart]:
        """Create the post-processing plots.

        Returns:
            The list of plots.
        """
        plots = [self.histogram_chart(self._dataset[var_name], var_name) for var_name in self._dataset.columns]
        plots.extend((self.boxplot(), self.scatterplot_matrix()))
        return plots

    def histogram_chart(self, data: ndarray | Series, variable_name: str) -> InstantiatedPlotlyNativeChart:
        """Generates a histogram chart.

        The chart also shows the mean, median and tolerance bounds.

        Args:
            data: The samples.
            variable_name: The variable name.

        Returns:
            The histogram chart.
        """
        name, unit = self.data_details.loc[self.data_details["variable"] == variable_name][["name", "unit"]].values[0]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=list(data), nbinsx=100, histnorm="probability"))
        fig.update_layout(xaxis={"title": name, "ticksuffix": unit}, yaxis={"title": "Probability"}, showlegend=False)

        # Add the statistics
        stats = self.get_sosdisc_outputs(SoSOutputNames.statistics)
        mean = stats.iloc[0, variable_name]
        median = stats.iloc[1, variable_name]

        fig.add_annotation(
            x=0.85,
            y=1.15,
            font={"family": "Arial", "color": "#7f7f7f", "size": 10},
            text=f" Mean: {format_currency_legend(mean, unit)} <br> Median: {format_currency_legend(median, unit)} ",
            showarrow=False,
            xanchor="left",
            align="right",
            xref="paper",
            yref="paper",
            bordercolor="black",
            borderwidth=1,
        )

        for value, text in zip([mean, median], ["mean", "median"]):
            fig.add_vline(
                xref="x",
                yref="paper",
                x=value,
                line_color="black",
                line_width=2,
                line_dash="dot",
            )
            fig.add_annotation(
                xref="x",
                yref="paper",
                x=value,
                y=-0.05,
                font={"color": "black", "size": 12},
                text=text,
                showarrow=False,
                xanchor="right",
            )

        return InstantiatedPlotlyNativeChart(
            fig=fig,
            chart_name=f"{name} distribution",
            default_legend=False,
        )

    def boxplot(self) -> InstantiatedPlotlyNativeChart:
        """Create a boxplot of the input and output samples.

        The color of the box indicates if the variable is an input or an output.

        Returns:
            The boxplot.
        """
        fig = go.Figure()
        for input_name in self._dataset.input_names:
            fig.add_trace(
                go.Box(
                    x=input_name,
                    y=self._dataset[input_name],
                    name=input_name,
                    marker_color="lightseagreen",
                )
            )
        for output_name in self._dataset.output_names:
            fig.add_trace(
                go.Box(
                    x=output_name,
                    y=self._dataset[input_name],
                    name=input_name,
                    marker_color="lightseagreen",
                )
            )

        return InstantiatedPlotlyNativeChart(
            fig=fig,
            chart_name="Boxplots",
            default_legend=False,
        )

    def scatterplot_matrix(self) -> InstantiatedPlotlyNativeChart:
        """Create a scatterplot matrix.

        Returns:
            The plot.
        """
        fig = px.scatter_matrix(self._dataset)
        return InstantiatedPlotlyNativeChart(
            fig=fig,
            chart_name="Scatterplot matrix",
            default_legend=False,
        )
