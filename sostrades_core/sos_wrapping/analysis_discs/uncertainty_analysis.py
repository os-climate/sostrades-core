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

import plotly.graph_objects as go
from gemseo.datasets.io_dataset import IODataset
from gemseo.uncertainty import create_statistics
from numpy import array, size
from pandas import DataFrame, concat

from sostrades_core.execution_engine.disciplines_wrappers.monte_carlo_driver_wrapper import (
    SoSOutputNames as MCOutputNames,
)
from sostrades_core.execution_engine.proxy_monte_carlo_driver import ProxyMonteCarloDriver
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_NAMESPACE_DRIVER = "ns_driver_MC"  # must be here to be used in list comprehension in DESC_IN


@dataclass
class SoSInputNames(MCOutputNames):
    """The names of the input parameters."""

    PROBABILITY_THRESHOLD = "probability_threshold"
    """The threshold value used to compute the probability.

    The analysis will return the probability that the output is above the threshold.
    Can be a single float, or an iterable of the same length as the number of outputs.
    """


@dataclass
class SoSOutputNames:
    """The names of the output parameters."""

    INPUT_SAMPLES_POST = "input_samples_post"
    """The dataframe of input samples, split in 1D components."""

    OUTPUT_SAMPLES_POST = "output_samples_post"
    """The dataframe of output samples, split in 1D components."""

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

    CHART_BOXPLOT_NAME: str = "Boxplot"

    CHART_DISTRIBUTION_NAME: str = "Distribution"

    CHART_FILTER_KEY: str = "Charts"

    CHART_SCATTERPLOTMATRIX_NAME: str = "Scatterplot Matrix"

    NAMESPACE_DRIVER: str = _NAMESPACE_DRIVER
    """The namespace of the Monte Carlo driver."""

    SoSInputNames: ClassVar[type[SoSInputNames]] = SoSInputNames

    SoSOutputNames: ClassVar[type[SoSOutputNames]] = SoSOutputNames

    DESC_IN: ClassVar[dict[str, Any]] = {
        name: desc | {"namespace": _NAMESPACE_DRIVER}
        for name, desc in ProxyMonteCarloDriver.DESC_OUT.items()
    }

    DESC_IN.update({
        SoSInputNames.PROBABILITY_THRESHOLD: {
            SoSWrapp.TYPE: "array",
            SoSWrapp.DEFAULT: array([0.0]),
        },
    })

    DESC_OUT: ClassVar[dict[str, Any]] = {
        SoSOutputNames.INPUT_SAMPLES_POST: {
            SoSWrapp.TYPE: "dataframe",
            SoSWrapp.DYNAMIC_DATAFRAME_COLUMNS: True,
        },
        SoSOutputNames.OUTPUT_SAMPLES_POST: {
            SoSWrapp.TYPE: "dataframe",
            SoSWrapp.DYNAMIC_DATAFRAME_COLUMNS: True,
        },
        SoSOutputNames.STATISTICS: {
            SoSWrapp.TYPE: "dataframe",
            SoSWrapp.DYNAMIC_DATAFRAME_COLUMNS: True,
        },
    }

    _dataset: IODataset
    """The dataset of input/output samples."""

    _numerical_parameters: dict[str, float | Iterable]
    """The numerical parameters for the analysis."""

    def _pre_process_data(self) -> None:
        """
        Read the input data and pre-process it for the analysis.

        Each inputs and outputs are split into 1D components.
        The data is stored in a IODataset that can be used for statistics and sensitivity analysis.
        """
        self._dataset = IODataset()
        input_samples = self.get_sosdisc_inputs(SoSInputNames.INPUT_SAMPLES)
        for input_name in input_samples.columns:
            if s := size(input_samples.loc[0, input_name]) == 1:
                self._dataset.add_input_variable(input_name, input_samples[input_name])
                continue
            # Split the variable into 1D components
            for i in range(s):
                self._dataset.add_input_variable(
                    f"{input_name}_{i}", [sample[i] for sample in input_samples[input_name]]
                )
        output_samples = self.get_sosdisc_inputs(SoSInputNames.OUTPUT_SAMPLES)
        for output_name in output_samples.columns:
            if s := size(output_samples.loc[0, output_name]) == 1:
                self._dataset.add_output_variable(output_name, output_samples[output_name])
                continue
            # Split the variable into 1D components
            for i in range(s):
                self._dataset.add_output_variable(
                    f"{output_name}_{i}", [sample[i] for sample in output_samples[output_name]]
                )
        inputs_df = self._dataset.input_dataset
        inputs_df.columns = inputs_df.columns.droplevel(0).droplevel(1)
        self.store_sos_outputs_values({SoSOutputNames.INPUT_SAMPLES_POST: inputs_df})
        outputs_df = self._dataset.output_dataset
        outputs_df.columns = outputs_df.columns.droplevel(0).droplevel(1)
        self.store_sos_outputs_values({SoSOutputNames.OUTPUT_SAMPLES_POST: outputs_df})

    def _check_parameters(self) -> None:
        """
        Check the numerical parameters.

        Raises:
            ValueError: If one of the parameters has the wrong size.

        """
        param_names = (SoSInputNames.PROBABILITY_THRESHOLD,)
        self._numerical_parameters = {param: self.get_sosdisc_inputs(param) for param in param_names}
        required_size = self._dataset.shape[1]
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
        # Store the original column order
        columns = list(self._dataset.columns.droplevel(0).droplevel(1))

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
            thresh = {name: threshold.flatten()[i] for i, name in enumerate(columns)}
        threshold = DataFrame(thresh, index=[0])
        proba = DataFrame(analysis.compute_probability(thresh))

        df = concat((mean, median, std, cv, threshold, proba), axis=0, ignore_index=True)
        df_index = DataFrame(
            {
                " ": [
                    "mean",
                    "median",
                    "standard deviation",
                    "coefficient of variation",
                    "threshold",
                    "P[X > threshold]",
                ]
            },
            index=list(range(6)),
        )
        df = concat((df_index, df), axis=1)
        # Swap the columns in the original order
        columns = [" ", *columns]
        df = df[columns]
        self.store_sos_outputs_values({SoSOutputNames.STATISTICS: df})

    def run(self) -> None:
        """Run the uncertainty analysis."""
        self._pre_process_data()
        self._check_parameters()
        self._compute_statistics()

    def get_chart_filter_list(self) -> list[ChartFilter]:
        """
        Get the available charts.

        Returns:
            A list containing the chart filter.

        """
        chart_list = [f"{var_type} distributions" for var_type in ["Input", "Output"]]
        chart_list.extend((self.CHART_BOXPLOT_NAME, self.CHART_SCATTERPLOTMATRIX_NAME))
        return [ChartFilter("Charts", chart_list, chart_list, self.CHART_FILTER_KEY)]

    def get_post_processing_list(
        self, chart_filters: list[ChartFilter] | None = None
    ) -> list[InstantiatedPlotlyNativeChart]:
        """
        Create the post-processing plots.

        Returns:
            The list of plots.

        """
        chart_filters = chart_filters or self.get_chart_filter_list()
        chart_list = []
        for _filter in chart_filters:
            if _filter.filter_key == self.CHART_FILTER_KEY:
                chart_list = _filter.selected_values
                break
        instantiated_graphs = []
        input_samples = self.get_sosdisc_outputs(SoSOutputNames.INPUT_SAMPLES_POST)
        output_samples = self.get_sosdisc_outputs(SoSOutputNames.OUTPUT_SAMPLES_POST)
        for chart_name in chart_list:
            if chart_name == self.CHART_BOXPLOT_NAME:
                instantiated_graphs.append(self.boxplot(input_samples, output_samples))
            elif chart_name == self.CHART_SCATTERPLOTMATRIX_NAME:
                instantiated_graphs.append(self.scatterplot_matrix(input_samples, output_samples))
            elif chart_name == "Input distributions":
                var_names = input_samples.columns
                for var in var_names:
                    data = input_samples[var].to_numpy().flatten().tolist()
                    instantiated_graphs.append(self.histogram_chart(data, var))
            elif chart_name == "Output distributions":
                var_names = output_samples.columns
                for var in var_names:
                    data = output_samples[var].to_numpy().flatten().tolist()
                    instantiated_graphs.append(self.histogram_chart(data, var))

        return instantiated_graphs

    def histogram_chart(self, data: list, var_name: str) -> InstantiatedPlotlyNativeChart:
        """
        Generates a histogram chart.

        The chart also shows the mean, median and tolerance bounds.

        Args:
            data: The list containing the data to plot.
            var_name: The variable name.

        Returns:
            The histogram chart.

        """
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=30, histnorm="probability"))
        fig.update_layout(xaxis={"title": var_name}, yaxis={"title": "Probability"}, showlegend=False)

        # Add the statistics
        stats = self.get_sosdisc_outputs(SoSOutputNames.STATISTICS)
        mean = stats[var_name].iloc[0]
        median = stats[var_name].iloc[1]

        fig.add_annotation(
            x=0.85,
            y=1.15,
            font={"family": "Arial", "color": "black", "size": 10},
            text=f" Mean: {mean} <br> Median: {median} ",
            showarrow=False,
            xanchor="left",
            align="right",
            xref="paper",
            yref="paper",
            bordercolor="black",
            borderwidth=1,
        )

        for value, text, y_pos, color in zip(
            [mean, median],
            ["mean", "median"],
            [-0.10, 1.05],
            ["black", "red"],
        ):
            fig.add_vline(
                xref="x",
                x=value,
                line_color=color,
                line_width=2,
                line_dash="dot",
            )
            fig.add_annotation(
                xref="x",
                yref="y domain",
                x=value,
                y=y_pos,
                font={"color": color, "size": 12},
                text=text,
                showarrow=False,
                xanchor="center",
            )

        return InstantiatedPlotlyNativeChart(
            fig=fig,
            chart_name=f"{var_name} distribution",
            default_legend=False,
        )

    @staticmethod
    def boxplot(input_samples: DataFrame, output_samples: DataFrame) -> InstantiatedPlotlyNativeChart:
        """
        Create a boxplot of the input and output samples.

        The color of the box indicates if the variable is an input or an output.

        Args:
            input_samples: The dataframe containing the input samples.
            output_samples: The dataframe containing the output samples.

        Returns:
            The boxplot.

        """
        fig = go.Figure()
        for input_name in input_samples.columns:
            fig.add_trace(
                go.Box(
                    y=input_samples[input_name].to_numpy().flatten().tolist(),
                    name=input_name,
                    marker_color="lightseagreen",
                )
            )
        for output_name in output_samples.columns:
            fig.add_trace(
                go.Box(
                    y=output_samples[output_name].to_numpy().flatten().tolist(),
                    name=output_name,
                    marker_color="indianred",
                )
            )

        return InstantiatedPlotlyNativeChart(
            fig=fig,
            chart_name="Boxplots",
            default_legend=False,
        )

    @staticmethod
    def scatterplot_matrix(input_samples: DataFrame, output_samples: DataFrame) -> InstantiatedPlotlyNativeChart:
        """
        Create a scatterplot matrix.

        Args:
            input_samples: The dataframe containing the input samples.
            output_samples: The dataframe containing the output samples.

        Returns:
            The plot.

        """
        samples = concat((input_samples, output_samples), axis=1)
        dimensions = [
            {"label": var_name, "values": samples[var_name].to_numpy().flatten().tolist()}
            for var_name in samples.columns
        ]
        fig = go.Figure()
        fig.add_trace(go.Splom(dimensions=dimensions))
        return InstantiatedPlotlyNativeChart(
            fig=fig,
            chart_name="Scatterplot matrix",
            default_legend=False,
        )
