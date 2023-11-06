'''
Copyright 2022 Airbus SAS
Modifications on 2023/02/23-2023/11/03 Copyright 2023 Capgemini

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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import logging
from sostrades_core.tools.base_functions.compute_len import compute_len
from sostrades_core.execution_engine.design_var.design_var import DesignVar
from numpy import zeros, array, ndarray, complex128
from functools import wraps


# decorator for delegating a method to the ProxyDiscipline object during configuration

class SoSWrappException(Exception):
    pass


class SoSWrapp(object):
    '''**SoSWrapp** is the class from which inherits our model wrapper when using 'SoSTrades' wrapping mode.

    It contains necessary information for the discipline configuration. It is owned by both the MDODisciplineWrapp and
    the SoSMDODiscipline.

    Its methods setup_sos_disciplines, run,... are overloaded by the user-provided Wrapper.

    N.B.: setup_sos_disciplines needs take as argument the proxy and call proxy.add_inputs() and/or proxy.add_outputs().

    Attributes:
        sos_name (string): name of the discipline
        local_data_short_name (Dict[Dict]): short name version of the local data for model input and output
        local_data (Dict[Any]): output of the model last run
    '''
    # -- Disciplinary attributes
    DESC_IN = {}
    DESC_OUT = {}
    TYPE = 'type'
    SUBTYPE = 'subtype_descriptor'
    COUPLING = 'coupling'
    VISIBILITY = 'visibility'
    LOCAL_VISIBILITY = 'Local'
    INTERNAL_VISIBILITY = 'Internal'
    SHARED_VISIBILITY = 'Shared'
    NAMESPACE = 'namespace'
    NS_REFERENCE = 'ns_reference'
    VALUE = 'value'
    DEFAULT = 'default'
    EDITABLE = 'editable'
    USER_LEVEL = 'user_level'
    STRUCTURING = 'structuring'
    POSSIBLE_VALUES = 'possible_values'
    RANGE = 'range'
    UNIT = 'unit'
    NUMERICAL = 'numerical'
    DESCRIPTION = 'description'
    VISIBLE = 'visible'
    CONNECTOR_DATA = 'connector_data'
    VAR_NAME = 'var_name'
    # Dict  ex: {'ColumnName': (column_data_type, column_data_range,
    # column_editable)}
    DATAFRAME_DESCRIPTOR = 'dataframe_descriptor'
    DYNAMIC_DATAFRAME_COLUMNS = 'dynamic_dataframe_columns'
    DATAFRAME_EDITION_LOCKED = 'dataframe_edition_locked'
    DEFAULT_EXCLUDED_COLUMNS = ['year', 'years']
    DATAFRAME_FILL = DesignVar.DATAFRAME_FILL
    ONE_COLUMN_FOR_KEY = DesignVar.ONE_COLUMN_FOR_KEY
    COLUMNS_NAMES = DesignVar.COLUMNS_NAMES
    IO_TYPE = 'io_type'
    IO_TYPE_IN = 'in'
    IO_TYPE_OUT = 'out'
    CHECK_INTEGRITY_MSG = 'check_integrity_msg'
    DYNAMIC_VAR_NAMESPACE_LIST = []

    def __init__(self, sos_name, logger: logging.Logger):
        """
        Constructor.

        Arguments:
            sos_name (string): name of the discipline
        """
        self.sos_name = sos_name
        self.input_full_name_map = {}
        self.output_full_name_map = {}
        self.input_data_names = []
        self.output_data_names = []
        self.attributes = {}
        self.local_data = {}
        self.jac_dict = {}
        self.jac_boundaries = {}
        self.inst_desc_in = {}
        self.inst_desc_out = {}

        # dynamic attributes that easen access to proxy and dm during configuration and get cleaned at runtime
        self.__proxy = None  # stores the proxy during configuration, decorator below to expose methods and properties
        self.dm = AccessOnlyProxy()  # object to proxy the dm during configuration allowing use avoiding wrong referencing
        self.logger = logger

    # decorator to expose methods and properties delegated to ProxyDiscipline object during configuration
    # TODO: change by a decorator outside the class + an AccessOnlyProxy object  ? Or by a __getattr__ overload ?
    def at_proxy(f):  # pylint: disable=E0213
        @wraps(f)
        def proxy_do(self, *args, **kwargs):
            proxy_attr = getattr(self.__proxy, f.__name__)
            if callable(proxy_attr):
                return proxy_attr(*args, **kwargs)
            else:  # otherwise it is a property getter
                return proxy_attr

        return proxy_do

    def clear_proxy(self):
        """
        Clears the ProxyDiscipline instance attribute from the SoSWrapp instance for serialization purposes (so that the
        proxy is not in attribute of the GEMSEO objects during execution).
        """
        del self.__proxy
        self.__proxy = None
        self.dm.clear_ref()

    def assign_proxy(self, proxy):
        """
        Assigns a ProxyDiscipline instance to the self.__proxy attribute (so that the proxy is available to the wrapper
        during the configuration sequence).
        """
        if self.__proxy is None:
            self.__proxy = proxy
            self.dm.set_ref(proxy.dm)

    # methods delegated to the proxy partially (because they might be called during the run)
    def get_sosdisc_inputs(self, *args, **kwargs):
        """
        Interface for the method get_sosdisc_inputs implementing a call to the ProxyDiscipline object, at configuration
        time vs. a call to the homonym protected method of the SoSWrapp object, at runtime.
        """
        if self.__proxy is not None:
            return self.__proxy.get_sosdisc_inputs(*args, **kwargs)
        else:
            return self._get_sosdisc_inputs(*args, **kwargs)

    def get_sosdisc_outputs(self, *args, **kwargs):
        """
        Interface for the method get_sosdisc_outputs implementing a call to the ProxyDiscipline object, at configuration
        time vs. a call to the homonym protected method of the SoSWrapp object, at runtime.
        """
        if self.__proxy is not None:
            return self.__proxy.get_sosdisc_outputs(*args, **kwargs)
        else:
            return self._get_sosdisc_outputs(*args, **kwargs)

    # methods delegated to the proxy totally (that only make sense at configuration time)
    @at_proxy
    def add_inputs(self, input_dict):
        """
        Method add_inputs delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def add_outputs(self, output_dict):
        """
        Method add_inputs delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def clean_variables(self, var_name_list, io_type):
        """
        Remove variables from data_in/data_out, inst_desc_in/inst_desc_out and datamanger

        Arguments:
            var_name_list (List[string]): variable names to clean
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
        """
        pass

    def check_data_integrity(self):
        """
        Method check_data_integrity
        """
        pass

    @at_proxy
    def get_data_in(self):
        """
        Method add_inputs delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_data_out(self):
        """
        Method add_inputs delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def set_dynamic_default_values(self, default_values_dict):
        """
        Method set_dynamic_default_values delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def update_default_value(self, var_name, io_type, new_default_value):
        """
        Method update_default_value delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_disc_full_name(self):
        """
        Method get_disc_full_name delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_disc_display_name(self):
        """
        Method get_disc_display_name delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_input_var_full_name(self, var_name):
        """
        Method get_input_var_full_name delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_output_var_full_name(self, var_name):
        """
        Method get_input_var_full_name delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @property
    @at_proxy
    def config_dependency_disciplines(self):
        """
        Property config_dependency_disciplines delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_inst_desc_in(self):
        """
        Method get_inst_desc_in delegated to associated ProxyDiscipline object during configuration.
        """
        # TODO: expose proxy attributes not only methods to SoSWrapp ? Would also affect properties (see decorator impl)
        pass

    @at_proxy
    def get_father_executor(self):
        """
        Method get_father_executor delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def add_disc_to_config_dependency_disciplines(self):
        """
        Method add_disc_to_config_dependency_disciplines delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_var_full_name(self, short_name, data_io):
        """
        Method get_var_full_name delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def add_new_shared_ns(self, shared_ns):
        """
        Method add_new_shared_ns delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    @at_proxy
    def get_shared_ns_dict(self):
        """
        Method get_shared_ns_dict delegated to associated ProxyDiscipline object during configuration.
        """
        pass

    def setup_sos_disciplines(self):  # type: (...) -> None
        """
        Define the set_up_sos_discipline of its proxy

        To be overloaded by subclasses.

        Arguments:
            proxy (ProxyDiscipline): the proxy discipline for dynamic i/o configuration
        """
        pass

    def init_execution(self):  # type: (...) -> None
        """
        Define the init_execution of its proxy

        To be overloaded by subclasses.

        Arguments:
            proxy (ProxyDiscipline): the proxy discipline for dynamic i/o configuration
        """
        pass

    def run(self):  # type: (...) -> None
        """
        Define the run of the discipline

        To be overloaded by subclasses.
        """
        raise NotImplementedError()

    def _get_sosdisc_inputs(self, keys=None, in_dict=False, full_name_keys=False):
        """
        Accessor for the inputs values as a list or dict to be used by the SoSWrapp object at runtime.

        Arguments:
            keys (List): the input short or full names list (depending on value of full_name_keys)
            in_dict (bool): if output format is dict
            full_name_keys (bool): if keys in args AND returned dictionary are full names or short names. Note that only
                                   True allows to query for variables of the subprocess as well as of the discipline itself.
        Returns:
            The inputs values list or dict
        """
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = self.input_data_names  # discipline and subprocess
            else:
                # discipline only
                keys = list(self.attributes['input_full_name_map'].keys())
            in_dict = True
        inputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_IN, full_name_keys=full_name_keys)
        if in_dict:
            # return inputs in an dictionary
            return inputs
        else:
            # return inputs in an ordered tuple (default)
            if len(inputs) > 1:
                return list(inputs.values())
            else:
                return list(inputs.values())[0]

    def _get_sosdisc_outputs(self, keys=None, in_dict=False, full_name_keys=False):
        """
        Accessor for the outputs values as a list or dict to be used by the SoSWrapp object at runtime.
        Arguments:
            keys (List): the output short or full names list (depending on value of full_name_keys)
            in_dict (bool): if output format is dict
            full_name_keys (bool): if keys in args AND returned dictionary are full names or short names. Note that only
                                   True allows to query for variables of the subprocess as well as of the discipline itself.
        Returns:
            The outputs values list or dict
        """
        if keys is None:
            # if no keys, get all discipline keys and force
            # output format as dict
            if full_name_keys:
                keys = self.output_data_names  # discipline and subprocess
            else:
                # discipline only
                keys = list(self.attributes['output_full_name_map'].keys())
            in_dict = True
        outputs = self._get_sosdisc_io(
            keys, io_type=self.IO_TYPE_OUT, full_name_keys=full_name_keys)
        if in_dict:
            # return outputs in an dictionary
            return outputs
        else:
            # return outputs in an ordered tuple (default)
            if len(outputs) > 1:
                return list(outputs.values())
            else:
                return list(outputs.values())[0]

    def _get_sosdisc_io(self, keys, io_type, full_name_keys):
        """
        Generic method to retrieve sos inputs and outputs

        Arguments:
            keys (List[String]): the data names list in short or full names (depending on value of full_name_keys)
            io_type (string): IO_TYPE_IN or IO_TYPE_OUT
            full_name_keys: if keys in args and returned dict are full names. Note that only True allows to query for
                            variables of the subprocess as well as of the discipline itself.
        Returns:
            dict of keys values
        Raises:
            ValueError if i_o type is not IO_TYPE_IN or IO_TYPE_OUT
            KeyError if asked for an output key when self.local_data is not initialized
        """

        # convert local key names to namespaced ones
        if isinstance(keys, str):
            keys = [keys]

        if full_name_keys:
            query_keys = keys
        else:
            if io_type == self.IO_TYPE_IN:
                query_keys = [self.attributes['input_full_name_map'][key]
                              for key in keys]
            elif io_type == self.IO_TYPE_OUT:
                query_keys = [self.attributes['output_full_name_map'][key]
                              for key in keys]
            else:
                raise ValueError("Unknown io_type :" +
                                 str(io_type))

        values_dict = dict(zip(keys, map(self.local_data.get, query_keys)))
        return values_dict

    def _run(self):
        """
        Run user-defined model.

        Returns:
            local_data (Dict): outputs of the model run
        """
        self.run()
        return self.local_data

    def store_sos_outputs_values(self, dict_values, full_name_keys=False):
        """"
        Store run outputs in the local_data attribute.

        NB: permits coherence with EEV3 wrapper run definition.

        Arguments:
            dict_values (Dict): variables' values to store
        """
        if full_name_keys:
            self.local_data.update(dict_values)
        else:
            outputs = dict(zip(map(
                self.attributes['output_full_name_map'].get, dict_values.keys()), dict_values.values()))
            self.local_data.update(outputs)

    def get_chart_filter_list(self):
        """ Return a list of ChartFilter instance base on the inherited
        class post processing filtering capabilities

        :return: ChartFilter[]
        """
        return []

    def get_post_processing_list(self, filters=None):
        """ Return a list of post processing instance using the ChartFilter list given
        as parameter, to be overload in subclasses

        :params: chart_fiters : filter to apply during post processing making
        :type: ChartFilter[]

        :return post processing instance list
        """
        return []

    def set_partial_derivative(self, y_key, x_key, value):
        """
        Method to fill the jacobian dict attribute of the wrapp with a partial derivative (value) given
        a specific input (x_key) and output (y_key). Input and output keys are returned in full name

        @param y_key: String, short name of the output whose derivative is calculated
        @param x_key: String, short name of the input whose derivative is calculated
        @param value: Array, values of the given derivative its dimensions depends on the input/output sizes
        """
        y_key_full = self.attributes['output_full_name_map'][y_key]
        x_key_full = self.attributes['input_full_name_map'][x_key]
        if y_key_full not in self.jac_dict.keys():
            self.jac_dict[y_key_full] = {}
        self.jac_dict[y_key_full].update({x_key_full: value})

    def set_partial_derivative_for_other_types(self, y_key_column, x_key_column, value):
        '''
        Set the derivative of the column y_key by the column x_key inside the jacobian of GEMS self.jac
        y_key_column = 'y_key, column_name'
        '''
        if len(x_key_column) == 2:
            x_key, x_column = x_key_column
        else:
            x_key = x_key_column[0]
            x_column = None

        lines_nb_x, index_x_column = self.get_boundary_jac_for_columns(
            x_key, x_column, self.IO_TYPE_IN)

        if len(y_key_column) == 2:
            y_key, y_column = y_key_column
        else:
            y_key = y_key_column[0]
            y_column = None

        # in the particular case of design_var_disc, the design_var dataframe can be filled in 2 different ways
        # its properties can be recovered from the self.design object (which does not exist for other disc)
        if hasattr(self, 'design') and self.design.design_var_descriptor is not None and \
                self.DATAFRAME_FILL in self.design.design_var_descriptor[x_key].keys() and \
                self.design.design_var_descriptor[x_key][self.DATAFRAME_FILL] == self.ONE_COLUMN_FOR_KEY:
            dataframefillmethod = self.design.design_var_descriptor[x_key][self.DATAFRAME_FILL]
            lines_nb_y, index_y_column = self.get_boundary_jac_for_design_var_columns(
                y_key, y_column, self.IO_TYPE_OUT, x_key, dataframefillmethod)
        else:
            lines_nb_y, index_y_column = self.get_boundary_jac_for_columns(
                y_key, y_column, self.IO_TYPE_OUT)

        # Convert keys in namespaced keys in the jacobian matrix for GEMS
        y_key_full = self.attributes['output_full_name_map'][y_key]

        x_key_full = self.attributes['input_full_name_map'][x_key]

        if y_key_full not in self.jac_dict.keys():
            self.jac_dict[y_key_full] = {}
        if x_key_full not in self.jac_dict[y_key_full]:
            self.jac_dict[y_key_full][x_key_full] = zeros(
                self.get_jac_matrix_shape(y_key, x_key))
        # Check if value is or has complex
        if type(value[0]) in [complex, complex128]:
            self.jac_dict[y_key_full][x_key_full] = array(
                self.jac_dict[y_key_full][x_key_full], dtype=complex)
        elif type(value[0]) in [array, ndarray]:
            if value.dtype in [complex, complex128]:
                self.jac_dict[y_key_full][x_key_full] = array(
                    self.jac_dict[y_key_full][x_key_full], dtype=complex)

        if index_y_column is not None and index_x_column is not None:
            self.jac_dict[y_key_full][x_key_full][index_y_column * lines_nb_y:(index_y_column + 1) * lines_nb_y,
            index_x_column * lines_nb_x:(index_x_column + 1) * lines_nb_x] = value
            self.jac_boundaries.update({f'{y_key_full},{y_column}': {'start': index_y_column * lines_nb_y,
                                                                     'end': (index_y_column + 1) * lines_nb_y},
                                        f'{x_key_full},{x_column}': {'start': index_x_column * lines_nb_x,
                                                                     'end': (index_x_column + 1) * lines_nb_x}})

        elif index_y_column is None and index_x_column is not None:
            self.jac_dict[y_key_full][x_key_full][:, index_x_column *
                                                     lines_nb_x:(index_x_column + 1) * lines_nb_x] = value

            self.jac_boundaries.update({f'{y_key_full},{y_column}': {'start': 0,
                                                                     'end': -1},
                                        f'{x_key_full},{x_column}': {'start': index_x_column * lines_nb_x,
                                                                     'end': (index_x_column + 1) * lines_nb_x}})
        elif index_y_column is not None and index_x_column is None:
            self.jac_dict[y_key_full][x_key_full][index_y_column * lines_nb_y:(index_y_column + 1) * lines_nb_y,
            :] = value
            self.jac_boundaries.update({f'{y_key_full},{y_column}': {'start': index_y_column * lines_nb_y,
                                                                     'end': (index_y_column + 1) * lines_nb_y},
                                        f'{x_key_full},{x_column}': {'start': 0,
                                                                     'end': -1}})
        else:
            raise Exception(
                'The type of a variable is not yet taken into account in set_partial_derivative_for_other_types')

    def get_jac_matrix_shape(self, y_key, x_key):
        y_value = self.get_sosdisc_outputs(y_key)
        x_value = self.get_sosdisc_inputs(x_key)
        n_out_j = compute_len(y_value)
        n_in_j = compute_len(x_value)
        expected_shape = (n_out_j, n_in_j)

        return expected_shape

    #TODO: see if should generalize the get_boundary_jac method with *args, **kwargs
    def get_boundary_jac_for_design_var_columns(self, ykey, column, io_type, xkey, dataframefillmethod):
        '''
        particular case of the design_var discipline where the design var dataframe has been filled following the
        'one column for key, one for value' method. In this case, all the name for the assets xkey are in the first column
        of the column names and their value in the 2nd column of column names of the dataframe value

        The method finds the number of design var per asset and the index of occurence of the asset in the list of asset for a given
        techno
        '''
        if ykey not in self.DESC_OUT:
            key_type = self.inst_desc_out[ykey]['type']
        else:
            key_type = self.DESC_OUT[ykey]['type']
        value = self.get_sosdisc_outputs(ykey)

        if dataframefillmethod == self.ONE_COLUMN_FOR_KEY:
            column_name_for_asset_name = self.design.design_var_descriptor[xkey][self.COLUMNS_NAMES][0]
            list_assets = list(value[column_name_for_asset_name].unique())
            if len(list_assets) < 1:
                raise ValueError(f'No asset found in get_sosdisc_outputs({ykey})')
            lines_nb = int(len(value) / len(list_assets))
            index_column = list_assets.index(column)
        else:
            raise NotImplementedError

        return lines_nb, index_column

    def get_boundary_jac_for_columns(self, key, column, io_type):
        if io_type == self.IO_TYPE_IN:
            # var_full_name = self.attributes['input_full_name_map'][key]
            if key not in self.DESC_IN:
                key_type = self.inst_desc_in[key]['type']
            else:
                key_type = self.DESC_IN[key]['type']
            value = self.get_sosdisc_inputs(key)
        if io_type == self.IO_TYPE_OUT:
            # var_full_name = self.attributes['output_full_name_map'][key]
            if key not in self.DESC_OUT:
                key_type = self.inst_desc_out[key]['type']
            else:
                key_type = self.DESC_OUT[key]['type']
            value = self.get_sosdisc_outputs(key)

        if key_type == 'dataframe':
            # Get the number of lines and the index of column from the metadata
            # for standard dataframe fill, there is one column of value per asset in the dataframe value
            lines_nb = len(value)
            index_column = [column for column in value.columns if column not in self.DEFAULT_EXCLUDED_COLUMNS].index(
                column)

        elif key_type == 'array' or key_type == 'float':
            lines_nb = None
            index_column = None
        elif key_type == 'dict':
            dict_keys = list(value.keys())
            lines_nb = len(value[column])
            index_column = dict_keys.index(column)
        return lines_nb, index_column


class AccessOnlyProxy:
    """
    Class that proxies an object providing access but avoiding its erroneous referencing. Unrelated to ProxyDiscipline.
    """

    # TODO: move to a tool?
    def __init__(self):
        self.__obj = None

    def __getattr__(self, item):
        return getattr(self.__obj, item)

    def set_ref(self, obj):
        self.__obj = obj

    def clear_ref(self):
        del self.__obj
        self.__obj = None
