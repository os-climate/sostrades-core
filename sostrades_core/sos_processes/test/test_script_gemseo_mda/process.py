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

from plotly.tools import mpl_to_plotly

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder
from sostrades_core.sos_processes.test.test_script_gemseo_mda.plot_newtonraphson_sobieski import script_gemseo
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'MDA Newton Raphson Sobieski from Gemseo script Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        mda = script_gemseo()

        self.ee.ns_manager.add_ns('ns_postproc', self.ee.study_name)
        self.ee.post_processing_manager.add_post_processing_functions_to_namespace(
            'ns_postproc', post_processing_filters, post_processings)
        return mda


def post_processing_filters(execution_engine, namespace):
    """
    Post processing function designed to build a rc vs saleprice 2D chart

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :returns: ChartFilter[]
    """
    filters = []
    return filters


def post_processings(execution_engine, namespace, filters):
    """
    Post processing function designed to build a rc vs saleprice 2D chart

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :params: filters, list of filters to applies to the post processing
    :type: ChartFilter[]

    :returns: list of post processing
    """
    mda = execution_engine.root_process.cls_builder
    chart_list = []
    mda_plot = mda.plot_residual_history(
        n_iterations=10,
        logscale=(1e-8, 10.0),
        save=False,
        show=False,
        fig_size=(10, 2),
        )
    chart_name = "plot_residual_history"
    fig = mpl_to_plotly(mda_plot)
    chart_list.append(InstantiatedPlotlyNativeChart(fig, chart_name))

    return chart_list
