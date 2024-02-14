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

# pylint: disable=line-too-long

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.namespace import Namespace
from sostrades_core.execution_engine.proxy_sample_generator import ProxySampleGenerator
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable
import numpy as np
import pandas as pd

def post_processing_filters(execution_engine:ExecutionEngine, namespace:Namespace):
    """ 
    post processing function designed to build filters
    """
    filters = []
    # TODO CHANGE

    chart_list = ['Scenario table']
    filters.append(ChartFilter(
        'Charts', chart_list, chart_list, filter_key='graphs'))
    return filters

    #chart_filters.append(ChartFilter('Variation List (%)', variation_list, variation_list, 'variation_list'))


def post_processings(execution_engine:ExecutionEngine, namespace:str, filters):
    """ 
    post processing function designed to build graphs
    """
    # For the outputs, making a bar graph with gradients values

    instanciated_charts = []
    
    # TODO samples_df or samples_inputs_df
    samples_df_name = namespace + "." + ProxySampleGenerator.SAMPLES_DF
    samples_df:pd.DataFrame = execution_engine.dm.get_value(var_f_name=samples_df_name)
    
    instanciated_charts.append(InstanciatedTable.from_pd_df(table_name="samples_df", df=samples_df))

    y_dict_name = namespace + ".y_dict"
    y_dict:dict[str:float] = execution_engine.dm.get_value(var_f_name=y_dict_name)
    instanciated_charts.append(InstanciatedTable(table_name="y_dict", header=["key", "value"], cells=list(y_dict.items())))

    return instanciated_charts