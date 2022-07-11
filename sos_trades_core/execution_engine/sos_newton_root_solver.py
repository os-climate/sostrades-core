# -*-mode: python; py-indent-offset: 4; tab-width:4; coding: iso-8859-1 -*-
# Copyright: Airbus group innovations
# @author: Valentin Joncquieres


from sos_trades_core.tools.grad_solvers.solvers.newton_raphson_problem import NewtonRaphsonProblem
from sos_trades_core.execution_engine.sos_eval import SoSEval
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter

from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import InstantiatedPlotlyNativeChart
from plotly import graph_objects as go


class NewtonRootSolver(SoSEval):
    """
    The Newton Root Solver class is used
    to solve the Newton Raphson problem with a specific residual function and specific residual informations

    residual_function is a list of builder (or a single builder) to execute in order to solve the Newton root solver
    the dict residual_infos give information on the residual variable to check for convergence + the namespace associated to the variable
    If not precised the newton unknowns are located at the same namespace 
    """

    MANDATORY_RESIDUAL_INFOS_KEYS = [
        'residual_variable', 'residual_ns_name', 'unknown_variable']
    OPTIONAL_RESIDUAL_INFOS_KEYS = ['unknown_ns_name']
    RESIDUAL_INFOS_KEYS = MANDATORY_RESIDUAL_INFOS_KEYS + OPTIONAL_RESIDUAL_INFOS_KEYS
    # ontology information
    _ontology_data = {
        'label': 'Newton Root Solver Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'Newton Root Solver with its own residual function',
        'icon': '',
    }
    ERROR_MSG = 'ERROR in NewtonRootSolver'

    DESC_IN = {'x0': {'type': 'array', 'structuring': True},
               'NR_stop_residual': {'type': 'float', 'default': 1e-7},
               'NR_relax_factor': {'type': 'float', 'default': 0.95},
               'NR_max_ite': {'type': 'float', 'default': 20},
               'NR_res0': {'type': 'float', 'default': 1.0}}

    DESC_OUT = {'x_final': {'type': 'array'},
                'residual_history': {'type': 'list'}}

    def __init__(self, sos_name, ee, residual_builders, residual_infos):

        if not isinstance(residual_builders, list):
            residual_builders = [residual_builders]
        SoSEval.__init__(self, sos_name, ee, cls_builder=residual_builders)
        self.nr_solver = None

        self.check_and_assign_residual_infos(residual_infos)

    def check_and_assign_residual_infos(self, residual_infos):
        '''
        Check if residual infos is a dict and has mandatory keys 
        Then if optional keys are not in the dictionary (namespace for unknowns), the same namespace as for residual variable is chosen
        '''
        if not isinstance(residual_infos, dict):
            raise Exception(
                'residual_infos must be a dictionary with informations on residual variables : \n residual_variable,residual_ns_name,unknown_variable keys are mandatory')
        if not set(self.MANDATORY_RESIDUAL_INFOS_KEYS) <= set(residual_infos.keys()):
            raise Exception(
                f'{self.MANDATORY_RESIDUAL_INFOS_KEYS} are mandatory keys for residual_infos parameters')

        self.residual_infos = residual_infos

        if not set(residual_infos.keys()) == set(self.RESIDUAL_INFOS_KEYS):
            for optional_key in self.OPTIONAL_RESIDUAL_INFOS_KEYS:
                if optional_key not in residual_infos.keys():
                    self.residual_infos[optional_key] = residual_infos[optional_key.replace(
                        'unknown', 'residual')]

    def set_eval_possible_values(self):

        pass

    def configure(self):

        SoSEval.configure(self)

        self.check_input_namespaces()
        self.check_variables_exists_and_are_arrays()
        self.set_x0()

    def set_x0(self):

        x0 = self.get_sosdisc_inputs('x0')
        unknown_name = self.get_unknown_namespaced_variable()
        self.dm.set_values_from_dict(
            {unknown_name: x0})

    def check_variables_exists_and_are_arrays(self):

        unknown_name = self.get_unknown_namespaced_variable()
        if not self.dm.check_data_in_dm(unknown_name):
            existing_name_list = self.dm.get_all_namespaces_from_var_name(
                self.residual_infos['unknown_variable'])
            raise Exception(
                f'The unknown variable {unknown_name} does not exist in the inputs of the given residual builders \nMaybe the namespace is not coherent with existing variable names in residual builders : {existing_name_list}')
        else:
            unknown_type = self.dm.get_data(unknown_name)['type']
            if unknown_type != 'array':
                raise Exception(
                    f'Newton Root solver only uses arrays : The unknown variable {unknown_name} must be specified as an array in residual builders for Newton Root Solver')

        residual_name = self.get_residual_namespaced_variable()
        if not self.dm.check_data_in_dm(residual_name):
            existing_name_list = self.dm.get_all_namespaces_from_var_name(
                self.residual_infos['residual_variable'])
            raise Exception(
                f'The residual variable {residual_name} does not exist in the inputs of the given residual builders \nMaybe the namespace is not coherent with existing variable names in residual builders : {existing_name_list}')
        else:
            residual_type = self.dm.get_data(residual_name)['type']
            if residual_type != 'array':
                raise Exception(
                    f'Newton Root solver only uses arrays : The residual variable {residual_name} must be specified as an array in residual builders for Newton Root Solver')

    def check_input_namespaces(self):
        '''
        CHeck if namespaces unknown_ns_name and residual_ns_name have been already declared in the configure of the sub builder
        '''

        for ns in ['unknown_ns_name', 'residual_ns_name']:
            if not self.ee.ns_manager.check_namespace_name_in_ns_manager(
                    self, self.residual_infos[ns]):
                raise Exception(
                    f"The namespace {self.residual_infos[ns]} has not been declared in residual builders, the {ns.split('_')[0]} variable cannot be found with the given {ns}")

    def get_unknown_namespaced_variable(self):
        '''
        Return the namespaced variable  name for unknown of Newton Root Solver
        '''
        unknown_ns_value = self.ee.ns_manager.get_shared_namespace_value(self,
                                                                         self.residual_infos['unknown_ns_name'])

        return f"{unknown_ns_value}.{self.residual_infos['unknown_variable']}"

    def get_residual_namespaced_variable(self):
        '''
        Return the namespaced variable  name for residual of Newton Root Solver
        '''
        residual_ns_value = self.ee.ns_manager.get_shared_namespace_value(self,
                                                                          self.residual_infos['residual_ns_name'])

        return f"{residual_ns_value}.{self.residual_infos['residual_variable']}"

    def get_residual_process(self):
        return self.sos_disciplines[0]

    def run(self):
        '''
            Run the NewtonRaphson solver :
            -configure the Newton Raphson depending on the residual option
            -solve the newton raphson
            -check the results
        '''
        # Configure the Newton Raphson with parameters and initial variable
        self.configure_solver()
        # Solvze the newton raphson
        x_final = self.nr_solver.solve()
        residual_history = self.nr_solver.get_residual_hist()
        dict_values = {'x_final': x_final,
                       'residual_history': residual_history}
        print('x_final', x_final, 'residual_history', residual_history)
        self.store_sos_outputs_values(dict_values)

    def configure_solver(self):
        '''
        Configure the Newton Raphson solver with its parameters and the initial value
        '''
        inputs_dict = self.get_sosdisc_inputs()

        self.nr_solver = NewtonRaphsonProblem(
            inputs_dict['x0'], self.residual_function, None, verbose=0)
        self.nr_solver.set_max_iterations(inputs_dict['NR_max_ite'])
        self.nr_solver.set_stop_residual(inputs_dict['NR_stop_residual'])
        self.nr_solver.set_relax_factor(inputs_dict['NR_relax_factor'])

        # if res_O is None then the residual is normalized with the first found residual
        # if res_0=1 the true residual norm is computed
        self.nr_solver.set_res_0(inputs_dict['NR_res0'])
        #self.nr_solver.bounds = [(-90., 90.), (-10., 1.e20)]

    def residual_function(self, x):
        '''
        Residual for the standard Newton Raphson computation
        '''

        residual_process = self.get_residual_process()
        unknown_name = self.get_unknown_namespaced_variable()

        self.dm.set_values_from_dict({unknown_name: x})

        # Compute the coupling
        local_data = residual_process.execute()

        residual_name = self.get_residual_namespaced_variable()
        idf_residual = local_data[residual_name]

        return idf_residual

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Residuals history']

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        '''
        Simple aero post procs
        '''
        instanciated_charts = []
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Residuals history' in chart_list:

            chart_name = f'Newton solver residuals history wrt iterations'
            fig = go.Figure()
            residual_history = self.get_sosdisc_outputs('residual_history')
            NR_stop_residual = self.get_sosdisc_inputs('NR_stop_residual')
#             new_chart = TwoAxesInstanciatedChart('Iterations [-]', 'Residuals history [-]',
# chart_name='Residual history vs iterations')

#             serie = InstanciatedSeries(
#                 list(range(len(residual_history))), residual_history, 'residuals', 'lines')
#             new_chart.add_series(serie)
#             serie = InstanciatedSeries(
#                 list(range(len(residual_history))), [NR_stop_residual] * len(residual_history), 'stop residual', 'lines')
#             new_chart.add_series(serie)
            fig.add_trace(go.Scatter(x=list(range(len(residual_history))),
                                     y=residual_history, name='Residuals history', visible=True))
            fig.add_trace(go.Scatter(x=list(range(len(residual_history))),
                                     y=[NR_stop_residual] * len(residual_history), name='Stop residual', visible=True))
            fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                              xaxis_title='Iterations', yaxis_title=f'Residuals history')

            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=list([
                            dict(
                                args=[{"yaxis.type": "linear"}],
                                label="Linear",
                                method="relayout"
                            ),
                            dict(
                                args=[{"yaxis.type": "log"}],
                                label="Log",
                                method="relayout"
                            ),
                        ]),
                        type="buttons",
                        direction="right",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        active=0,
                        x=0.0,
                        xanchor="left",
                        y=1.01,
                        yanchor="bottom"
                    ),
                ]
            )

            new_chart = InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name, default_title=True)

            instanciated_charts.append(new_chart)

        return instanciated_charts
