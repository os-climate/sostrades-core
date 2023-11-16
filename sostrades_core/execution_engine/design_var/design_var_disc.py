'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/02 Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.design_var.design_var import DesignVar
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
import numpy as np
import pandas as pd
from plotly import graph_objects as go
import plotly.colors as plt_color

from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter

color_list = plt_color.qualitative.Plotly
color_list.extend(plt_color.qualitative.Alphabet)


class DesignVarDiscipline(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Design Variable Model',
        'type': 'Test',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-drafting-compass fa-fw',
        'version': '',
    }
    WRITE_XVECT = 'write_xvect'
    LOG_DVAR = 'log_designvar'
    EXPORT_XVECT = 'export_xvect'
    OUT_TYPE = 'out_type'
    OUT_NAME = 'out_name'
    OUT_TYPES = ['float', 'array', 'dataframe']
    VARIABLES = "variable"
    VALUES = "value"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    TYPE = "type"
    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"
    INDEX = DesignVar.INDEX
    INDEX_NAME = DesignVar.INDEX_NAME
    DATAFRAME_FILL = DesignVar.DATAFRAME_FILL
    COLUMNS_NAMES = DesignVar.COLUMNS_NAMES
    DATAFRAME_FILL_POSSIBLE_VALUES = DesignVar.DATAFRAME_FILL_POSSIBLE_VALUES
    DESIGN_VAR_DESCRIPTOR = DesignVar.DESIGN_VAR_DESCRIPTOR
    DESIGN_SPACE = DesignVar.DESIGN_SPACE
    FILL_ACTIVATED_ELEMENTS = DesignVar.FILL_ACTIVATED_ELEMENTS
    DESC_IN = {
        DESIGN_VAR_DESCRIPTOR: {'type': 'dict', 'editable': True, 'structuring': True, 'user_level': 3},
        DESIGN_SPACE: {'type': 'dataframe', 'dataframe_descriptor': {VARIABLES: ('string', None, True),
                                                                     VALUES: ('multiple', None, True),
                                                                     LOWER_BOUND: ('multiple', None, True, True),
                                                                     UPPER_BOUND: ('multiple', None, True, True),
                                                                     ENABLE_VARIABLE_BOOL: ('bool', None, True),
                                                                     LIST_ACTIVATED_ELEM: ('list', None, True), },
                       'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_optim'},
        LOG_DVAR: {'type': 'bool', 'default': False, 'user_level': 3},
        WRITE_XVECT: {'type': 'bool', 'default': False, 'user_level': 3},
    }

    DESC_OUT = {
        'design_space_last_ite': {'type': 'dataframe', 'user_level': 3}
    }

    def setup_sos_disciplines(self):

        dynamic_inputs = {}
        dynamic_outputs = {}

        inputs_dict = self.get_sosdisc_inputs()

        # loops over the output descriptor to add dynamic inputs and outputs from the loaded usecase.
        # The structure of the output descriptor dict is checked prior its use
        if self.DESIGN_VAR_DESCRIPTOR in inputs_dict.keys():
            design_var_descriptor = inputs_dict[self.DESIGN_VAR_DESCRIPTOR]
            if design_var_descriptor is not None:
                if self._check_descriptor(design_var_descriptor):
                    for key in design_var_descriptor.keys():
                        # if we are using the dataframe fill method 'one column for key, one for value',
                        # a string column is created and should not be taken into account for gradients or MDA computations
                        # We need to modify the DEFAULT_EXCLUDED_COLUMNS of the Discipline woth the column name for the key column
                        if self.DATAFRAME_FILL in design_var_descriptor[key] and design_var_descriptor[key][
                            self.DATAFRAME_FILL] == self.DATAFRAME_FILL_POSSIBLE_VALUES[1]:
                            self.DEFAULT_EXCLUDED_COLUMNS = SoSWrapp.DEFAULT_EXCLUDED_COLUMNS + [self.COLUMNS_NAMES[0]]

                        dynamic_inputs[key] = {'type': 'array',
                                               'visibility': SoSWrapp.SHARED_VISIBILITY,
                                               'namespace': design_var_descriptor[key]['namespace_in']}
                        dynamic_outputs[design_var_descriptor[key][self.OUT_NAME]] = {
                            'type': design_var_descriptor[key]['out_type'],
                            'visibility': SoSWrapp.SHARED_VISIBILITY,
                            'namespace': design_var_descriptor[key]['namespace_out']}
        if self.WRITE_XVECT in inputs_dict.keys():
            write_xvect = inputs_dict[self.WRITE_XVECT]
            if write_xvect:
                dynamic_outputs['all_iteration_dict'] = {'type': 'dict'}

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)
        self.inst_desc_in = dynamic_inputs
        self.inst_desc_out = dynamic_outputs

        self.iter = 0

    def init_execution(self):
        super().init_execution()
        inputs_dict = self.get_sosdisc_inputs()
        self.design = DesignVar(inputs_dict, self.logger)
        self.all_iterations_dict = {}
        self.dict_last_ite = None

    def run(self):
        inputs_dict = self.get_sosdisc_inputs()

        self.design.configure(inputs_dict)
        outputs_dict = self.design.output_dict

        # retrieve the design space and update values with current iteration
        # values
        dspace_in = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        dspace_out = dspace_in.copy(deep=True)
        dict_diff = {}
        dict_current = {}
        for dvar in dspace_in.variable:
            for disc_var in inputs_dict.keys():
                if dvar in disc_var:
                    val = self.get_sosdisc_inputs(disc_var)
                    if isinstance(val, np.ndarray):
                        dict_current[dvar] = val

                        val = str(val.tolist())
                        # dict_diff[dvar]
                        # dict_diff[dvar]
                    if isinstance(val, list):
                        dict_current[dvar] = np.array(val)
                    dspace_out.loc[dspace_out.variable ==
                                   dvar, "value"] = str(val)

        # option to log difference between two iterations to track optimization
        if inputs_dict[self.LOG_DVAR]:
            if self.dict_last_ite is None:
                self.logger.info('x0' + str(dict_current))

                self.dict_last_ite = dict_current

            else:
                dict_diff = {
                    key: dict_current[key] - self.dict_last_ite[key] for key in dict_current}
                self.logger.info(
                    'difference between two iterations' + str(dict_diff))
        # update output dictionary with dspace
        outputs_dict.update({'design_space_last_ite': dspace_out})

        # dump design space into a csv
        if self.get_sosdisc_inputs(self.WRITE_XVECT):
            dspace_out.to_csv(f"dspace_ite_{self.iter}.csv", index=False)

            # write all iterations into a dictionnary
            self.all_iterations_dict.update({f"iteration {self.iter}": dict_current})
            outputs_dict.update({'all_iteration_dict': self.all_iterations_dict})

        self.store_sos_outputs_values(self.design.output_dict)
        self.iter += 1

    def compute_sos_jacobian(self):

        design_var_descriptor = self.get_sosdisc_inputs(self.DESIGN_VAR_DESCRIPTOR)

        for key in design_var_descriptor.keys():
            out_type = design_var_descriptor[key][self.OUT_TYPE]
            out_name = design_var_descriptor[key][self.OUT_NAME]
            if out_type == 'array':
                self.set_partial_derivative(
                    out_name, key, self.design.bspline_dict[key]['b_array'])
            elif out_type == 'dataframe':
                asset_name = design_var_descriptor[key]['key']
                # specific processing occurs in the partial derivative computation depending on the way the
                # dataframe was filled
                self.set_partial_derivative_for_other_types(
                        (out_name, asset_name), (key,), self.design.bspline_dict[key]['b_array'])
            elif out_type == 'float':
                self.set_partial_derivative(out_name, key, np.array([1.]))
            else:
                raise (ValueError('Output type not yet supported'))

    def _check_descriptor(self, design_var_descriptor):
        """

        :param design_var_descriptor: dict input of Design Var Discipline containing all information necessary to build its dynamic inputs and outputs.
        For each input, data needed are the key, type, and namespace_in and for output out_name, out_type, namespace_out and depending on its type, index, index name and key.
        :return: True if the dict has the requested data, False otherwise.
        """
        test = True

        for key in design_var_descriptor.keys():
            needed_keys = [self.OUT_TYPE, 'namespace_in',
                           self.OUT_NAME, 'namespace_out']
            messages = [f'Supported output types are {self.OUT_TYPES}.',
                        'Please set the input namespace.',
                        'Please set output_name.',
                        'Please set the output namespace.',
                        ]
            dvar_descriptor_key = design_var_descriptor[key]
            for n_key in needed_keys:
                if n_key not in dvar_descriptor_key.keys():
                    test = False
                    raise (ValueError(
                        f'Discipline {self.sos_name} design_var_descriptor[{key}] is missing "{n_key}" element. {messages[needed_keys.index(n_key)]}'))
                else:
                    out_type = dvar_descriptor_key[self.OUT_TYPE]
                    if out_type == 'float':
                        continue
                    elif out_type == 'array':
                        array_needs = [self.INDEX, self.INDEX_NAME]
                        array_mess = [
                            f'Please set an index describing the length of the output array of {key} (index is also used for post proc representations).',
                            f'Please set an index name describing for the output array of {key} (index_name is also used for post proc representations).',
                        ]
                        for k in array_needs:
                            if k not in dvar_descriptor_key.keys():
                                test = False
                                raise (
                                    ValueError(
                                        f'Discipline {self.sos_name} design_var_descriptor[{key}] is missing "{k}" element. {array_mess[array_needs.index(k)]}'))
                    elif out_type == 'dataframe':

                        if self.DATAFRAME_FILL in dvar_descriptor_key:
                            dataframe_fill = dvar_descriptor_key[self.DATAFRAME_FILL]
                            # check if the dataframe fill is among possible values
                            if dataframe_fill not in self.DATAFRAME_FILL_POSSIBLE_VALUES:
                                raise (
                                    ValueError(
                                        f'Discipline {self.sos_name} design_var_descriptor[{self.DATAFRAME_FILL}] is not in {self.DATAFRAME_FILL_POSSIBLE_VALUES}'))

                        else:
                            # if no dataframe fill default is one column per key
                            dataframe_fill = self.DATAFRAME_FILL_POSSIBLE_VALUES[0]
                            # index and key must be in the dataframe_descriptor for both methods
                        dataframe_needs = [self.INDEX, self.INDEX_NAME, 'key']
                        dataframe_mess = [
                            f'Please set an index to the output dataframe of {key} (index is also used for post proc representations).',
                            f'Please set an index name to the output dataframe of {key} (index_name is also used for post proc representations).',
                            f'Please set a "key" name to the output dataframe of {key} (name of the column in which the output will be written).',
                        ]
                        for k in dataframe_needs:
                            if k not in dvar_descriptor_key.keys():
                                test = False
                                raise (
                                    ValueError(
                                        f'Discipline {self.sos_name} design_var_descriptor[{key}] is missing "{k}" element. {dataframe_mess[dataframe_needs.index(k)]}'))

                        if dataframe_fill == self.DATAFRAME_FILL_POSSIBLE_VALUES[1]:
                            # column_names must be in the dataframe_descriptor for 'one column for key,one for value' method
                            if self.COLUMNS_NAMES not in dvar_descriptor_key.keys():
                                test = False
                                raise (
                                    ValueError(
                                        f'Discipline {self.sos_name} design_var_descriptor[{self.COLUMNS_NAMES}] is missing with option {self.DATAFRAME_FILL_POSSIBLE_VALUES[1]}'))
                            # only two column names are required
                            if len(dvar_descriptor_key[self.COLUMNS_NAMES]) != 2:
                                test = False
                                raise (
                                    ValueError(
                                        f'Discipline {self.sos_name} design_var_descriptor[{self.COLUMNS_NAMES}] must be of length 2, the column name for keys and the one for the value'))

                    else:
                        test = False
                        raise (ValueError(
                            f'Discipline {self.sos_name} design_var_descriptor[{key}] out_type is not supported. Supported out_types are {self.OUT_TYPES}'))

        return test

    def get_chart_filter_list(self):

        chart_filters = []
        chart_list = ['BSpline']
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        initial_xvect_list = ['Standard', 'With initial xvect']
        chart_filters.append(ChartFilter(
            'Initial xvect', initial_xvect_list, ['Standard', ], 'initial_xvect'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        # For the outputs, making a graph for block fuel vs range and blocktime vs
        # range

        instanciated_charts = []
        charts = []
        initial_xvect_list = ['Standard', ]
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values
                if chart_filter.filter_key == 'initial_xvect':
                    initial_xvect_list = chart_filter.selected_values
        if 'With initial xvect' in initial_xvect_list:
            init_xvect = True
        else:
            init_xvect = False

        if 'BSpline' in charts:
            list_dv = self.get_sosdisc_inputs(self.DESIGN_VAR_DESCRIPTOR)
            for parameter in list_dv.keys():
                new_chart = self.get_chart_BSpline(parameter, init_xvect)
                if new_chart is not None:
                    instanciated_charts.append(new_chart)

        return instanciated_charts

    def get_chart_BSpline(self, parameter, init_xvect=False):
        """
        Function to create post-proc for the design variables with display of the control points used to 
        calculate the B-Splines.
        The activation/deactivation of control points is accounted for by inserting the values from the design space
        dataframe into the ctrl_pts if need be (activated_elem==False) and at the appropriate index.
        Input: parameter (name), parameter values, design_space
        Output: InstantiatedPlotlyNativeChart
        """

        design_space = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        pts = self.get_sosdisc_inputs(parameter)
        ctrl_pts = list(pts)
        starting_pts = list(
            design_space[design_space['variable'] == parameter]['value'].values[0])
        for i, activation in enumerate(design_space.loc[design_space['variable']
                                                        == parameter, 'activated_elem'].to_list()[0]):
            if not activation and len(
                    design_space.loc[design_space['variable'] == parameter, 'value'].to_list()[0]) > i:
                ctrl_pts.insert(i, design_space.loc[design_space['variable']
                                                    == parameter, 'value'].to_list()[0][i])
        eval_pts = None

        design_var_descriptor = self.get_sosdisc_inputs(self.DESIGN_VAR_DESCRIPTOR)
        out_name = design_var_descriptor[parameter][self.OUT_NAME]
        out_type = design_var_descriptor[parameter][self.OUT_TYPE]
        index = design_var_descriptor[parameter][self.INDEX]
        index_name = design_var_descriptor[parameter][self.INDEX_NAME]

        if out_type == 'array':
            eval_pts = self.get_sosdisc_outputs(out_name)

        elif out_type == 'dataframe':
            df = self.get_sosdisc_outputs(out_name)
            if self.DATAFRAME_FILL in design_var_descriptor[parameter]:
                dataframe_fill = design_var_descriptor[parameter][self.DATAFRAME_FILL]
            else:
                dataframe_fill = self.DATAFRAME_FILL_POSSIBLE_VALUES[0]
            col_name = design_var_descriptor[parameter]['key']
            if dataframe_fill == self.DATAFRAME_FILL_POSSIBLE_VALUES[0]:
                eval_pts = df[col_name].values
            elif dataframe_fill == self.DATAFRAME_FILL_POSSIBLE_VALUES[1]:
                column_names = design_var_descriptor[parameter][self.COLUMNS_NAMES]
                eval_pts = df[df[column_names[0]] == col_name][column_names[1]].values

        if eval_pts is None:
            print('eval pts not found in sos_disc_outputs')
            return None
        else:
            chart_name = f'B-Spline for {parameter}'
            fig = go.Figure()
            if 'complex' in str(type(ctrl_pts[0])):
                ctrl_pts = [np.real(value) for value in ctrl_pts]
            if 'complex' in str(type(eval_pts[0])):
                eval_pts = [np.real(value) for value in eval_pts]
            if 'complex' in str(type(starting_pts[0])):
                starting_pts = [np.real(value) for value in starting_pts]
            x_ctrl_pts = np.linspace(
                index[0], index[-1], len(ctrl_pts))
            marker_dict = dict(size=150 / len(ctrl_pts), line=dict(
                width=150 / (3 * len(ctrl_pts)), color='DarkSlateGrey'))
            fig.add_trace(go.Scatter(x=list(x_ctrl_pts),
                                     y=list(ctrl_pts), name='Poles',
                                     line=dict(color=color_list[0]),
                                     mode='markers',
                                     marker=marker_dict))
            fig.add_trace(go.Scatter(x=list(index), y=list(eval_pts), name='B-Spline',
                                     line=dict(color=color_list[0]), ))
            if init_xvect:
                marker_dict['opacity'] = 0.5
                fig.add_trace(go.Scatter(x=list(x_ctrl_pts),
                                         y=list(starting_pts), name='Initial Poles',
                                         mode='markers',
                                         line=dict(color=color_list[0]),
                                         marker=marker_dict))
            fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                              xaxis_title=index_name, yaxis_title=f'value of {parameter}')
            new_chart = InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name, default_title=True)

            return new_chart
