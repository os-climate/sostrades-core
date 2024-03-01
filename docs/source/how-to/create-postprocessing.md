# Create a postprocessing in a wrapper discipline


## Document a function
Two methods need to be implemented for post-processings.

### get_chart_filter_list
This method is used to make the list of available filters.
```{eval-rst}
.. automethod:: sostrades_core.execution_engine.sos_wrapp.SoSWrapp.get_chart_filter_list
```

Here is how ChartFilter is defined :

```{eval-rst}
.. autoclass:: sostrades_core.tools.post_processing.charts.chart_filter::ChartFilter
```

* name : string that contains filter name
* filter_values : list of filter items that can be used to filter post processing element
* selected_values : list of filter items currently selected for the given filter
* filter_key : unique key used to identify the current filter

### get_post_processing_list
```{eval-rst}
.. automethod:: sostrades_core.execution_engine.sos_wrapp.SoSWrapp.get_post_processing_list
```

get_post_processing_list conditionally generates the instances of the post processing objects depending on the filters' selected values

Remember that this method should not make any heavy computation so graph data should be processed by the model in the discipline run and stored in output variables if need be

We can create a plotly figure (in the example a table). We then call the method InstantiatedPlotlyChart(plotly_figure). This returned chart is then used as in the previous example


#### TwoAxesInstanciatedChart
Here is an example to make a simple chart
```python
def get_post_processing_list(self, chart_filters=None):
    """
    Gets the charts selected
    """
    instanciated_charts = []
    chart_list = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'charts':
                chart_list = chart_filter.selected_values
    
    if 'sample chart' in chart_list:
        # Get the values
        x = self.get_sosdisc_inputs('x')
        y = self.get_sosdisc_inputs('y')

        # Instanciate chart
        new_chart = TwoAxesInstanciatedChart('x (-)', 'y (-)', chart_name="x vs y")

        # Add data points
        serie = InstanciatedSeries([x], [y], series_name="x vs y", display_type='scatter')

        new_chart.series.append(serie)

        instanciated_charts.append(new_chart)

    return instanciated_charts
```

#### InstantiatedPlotlyChart
To use a plotly figure already created
