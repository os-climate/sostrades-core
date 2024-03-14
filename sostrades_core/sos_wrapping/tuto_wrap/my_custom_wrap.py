from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter

class MyCustomWrap(SoSWrapp):
    # Ontology information
    _ontology_data = {
        'label': 'Label of the discipline',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'version': '',
    }

    # Maturity of the model
    _maturity = 'Fake'

    # Description of inputs
    DESC_IN = {
        'x': {'type': 'float', 'default': 10, 'unit': 'year', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_one'},
        'a': {'type': 'float', 'unit': '-', 'namespace': 'ns_one'},
        'b': {'type': 'float', 'unit': '-',},
    }

    # Description of outputs
    DESC_OUT = {
        'y': {'type': 'float', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_one'}
    }

    # Method that runs the model
    def run(self):
        """
        Method that runs the model
        """
        # get input of discipline
        param_in = self.get_sosdisc_inputs()

        # performs the "computation"
        x = param_in['x']
        a = param_in['a']
        b = param_in['b']

        y = a * x + b

        output_values = {'y': y}

        # store data
        self.store_sos_outputs_values(output_values)

    def get_chart_filter_list(self):
        """
        Gets the charts filter list
        """
        chart_filters = []

        chart_list = ['sample chart']
        
        chart_filters.append(ChartFilter('Charts', chart_list, chart_list, 'charts'))

        return chart_filters

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
