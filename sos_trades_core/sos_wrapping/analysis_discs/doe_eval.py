'''
Copyright 2022 Airbus SAS

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
import copy
import re

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_eval import SoSEval
import pandas as pd
from collections import ChainMap


class DoeEval(SoSEval):
    '''
    Generic DOE evaluation class
    '''

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.analysis_discs.doe_eval',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    default_algo_options = {}

    DEFAULT = 'default'

    # Design space dataframe headers
    VARIABLES = "variable"
    VALUES = "value"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    TYPE = "type"
    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"
    POSSIBLE_VALUES = 'possible_values'
    N_SAMPLES = "n_samples"
    DESIGN_SPACE = "design_space"

    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    USER_GRAD = 'user'

    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    NS_SEP = '.'
    INPUT_TYPE = ['float', 'array', 'int', 'string']
    INPUT_MULTIPLIER_TYPE = []
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'

    DESC_IN = {'sampling_algo': {'type': 'string', 'structuring': True},
               'eval_inputs': {'type': 'dataframe',
                               'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                        'full_name': ('string', None, False)},
                               'dataframe_edition_locked': False,
                               'structuring': True,
                               'visibility': SoSDiscipline.SHARED_VISIBILITY,
                               'namespace': 'ns_doe_eval'},
               'eval_outputs': {'type': 'dataframe',
                                'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                         'full_name': ('string', None, False)},
                                'dataframe_edition_locked': False,
                                'structuring': True, 'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                'namespace': 'ns_doe_eval'},
               'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
               'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0},
               }

    DESC_OUT = {
        'samples_inputs_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSDiscipline.SHARED_VISIBILITY,
                              'namespace': 'ns_doe_eval'}
    }
    # We define here the different default algo options in a case of a DOE
    # TODO Implement a generic get_options functions to retrieve the default
    # options using directly the DoeFactory

    # Default values of algorithms
    default_algo_options = {
        'n_samples': 'default',
        'alpha': 'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'seed': 1,
        'center_bb': 'default',
        'center_cc': 'default',
        'criterion': 'default',
        'levels': 'default'
    }

    default_algo_options_lhs = {
        'n_samples': 'default',
        'alpha': 'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'seed': 1,
        'center_bb': 'default',
        'center_cc': 'default',
        'criterion': 'default',
        'levels': 'default'
    }

    default_algo_options_fullfact = {
        'n_samples': 'default',
        'alpha': 'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'seed': 1,
        'center_bb': 'default',
        'center_cc': 'default',
        'criterion': 'default',
        'levels': 'default'
    }
    d = {'col1': [1, 2], 'col2': [3, 4]}
    X_pd = pd.DataFrame(data=d)

    default_algo_options_CustomDOE = {
        'n_processes': 1,
        'wait_time_between_samples': 0.0
    }

    default_algo_options_CustomDOE_file = {
        'eval_jac': False,
        'max_time': 0,
        'samples': None,
        'doe_file': 'X_pd.csv',
        'comments': '#',
        'delimiter': ',',
        'skiprows': 0
    }

    algo_dict = {"lhs": default_algo_options_lhs,
                 "fullfact": default_algo_options_fullfact,
                 "CustomDOE": default_algo_options_CustomDOE,
                 }

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''
        # if 'ns_doe' does not exist in ns_manager, we create this new
        # namespace to store output dictionaries associated to eval_outputs
        if 'ns_doe' not in ee.ns_manager.shared_ns_dict.keys():
            ee.ns_manager.add_ns('ns_doe', ee.study_name)
        super(DoeEval, self).__init__(sos_name, ee, cls_builder)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.DOE')
        self.doe_factory = DOEFactory()
        self.design_space = None
        self.samples = None
        self.customed_samples = None
        self.dict_desactivated_elem = {}
        self.selected_outputs = []
        self.selected_inputs = []
        self.previous_algo_name = ""

    def setup_sos_disciplines(self):
        """
        Overload setup_sos_disciplines to create a dynamic desc_in
        default descin are the algo name and its options
        In case of a CustomDOE', additionnal input is the customed sample ( dataframe)
        In other cases, additionnal inputs are the number of samples and the design space
        """

        dynamic_inputs = {}
        dynamic_outputs = {}
        algo_name_has_changed = False
        selected_inputs_has_changed = False

        # The setup of the discipline can begin once the algorithm we want to use to generate
        # the samples has been set
        if self.ALGO in self._data_in:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            if self.previous_algo_name != algo_name:
                algo_name_has_changed = True
                self.previous_algo_name = algo_name
            eval_outputs = self.get_sosdisc_inputs('eval_outputs')
            eval_inputs = self.get_sosdisc_inputs('eval_inputs')

            # we fetch the inputs and outputs selected by the user
            selected_outputs = eval_outputs[eval_outputs['selected_output']
                                            == True]['full_name']
            selected_inputs = eval_inputs[eval_inputs['selected_input']
                                          == True]['full_name']
            if set(selected_inputs.tolist()) != set(self.selected_inputs):
                selected_inputs_has_changed = True
                self.selected_inputs = selected_inputs.tolist()
            self.selected_outputs = selected_outputs.tolist()

            # doe can be done only for selected inputs and outputs
            if algo_name is not None and len(selected_inputs) > 0 and len(selected_outputs) > 0:
                # we set the lists which will be used by the evaluation
                # function of sosEval
                self.set_eval_in_out_lists(selected_inputs, selected_outputs)

                # setting dynamic outputs. One output of type dict per selected
                # output

                for out_var in self.eval_out_list:
                    dynamic_outputs.update(
                        {f'{out_var.split(self.ee.study_name + ".")[1]}_dict': {'type': 'dict', 'visibility': 'Shared',
                                                                                'namespace': 'ns_doe'}})

                if algo_name == "CustomDOE":
                    default_custom_dataframe = pd.DataFrame(
                        [[NaN for input in range(len(self.selected_inputs))]], columns=self.selected_inputs)
                    dataframe_descriptor = {}
                    for i, key in enumerate(self.selected_inputs):
                        cle = key
                        var = tuple([self.ee.dm.get_data(
                            self.eval_in_list[i], 'type'), None, True])
                        dataframe_descriptor[cle] = var

                    dynamic_inputs.update(
                        {'custom_samples_df': {'type': 'dataframe', self.DEFAULT: default_custom_dataframe,
                                               'dataframe_descriptor': dataframe_descriptor,
                                               'dataframe_edition_locked': False}})
                    if 'custom_samples_df' in self._data_in and selected_inputs_has_changed:
                        self._data_in['custom_samples_df']['value'] = default_custom_dataframe
                        self._data_in['custom_samples_df']['dataframe_descriptor'] = dataframe_descriptor

                else:

                    default_design_space = pd.DataFrame({'variable': selected_inputs,

                                                         'lower_bnd': [[0.0, 0.0] if self.ee.dm.get_data(var,
                                                                                                         'type') == 'array' else 0.0
                                                                       for var in self.eval_in_list],
                                                         'upper_bnd': [[10.0, 10.0] if self.ee.dm.get_data(var,
                                                                                                           'type') == 'array' else 10.0
                                                                       for var in self.eval_in_list]
                                                         })

                    dynamic_inputs.update(
                        {'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space
                                          }})
                    if 'design_space' in self._data_in and selected_inputs_has_changed:
                        self._data_in['design_space']['value'] = default_design_space

                default_dict = self.get_algo_default_options(algo_name)
                dynamic_inputs.update({'algo_options': {'type': 'dict', self.DEFAULT: default_dict,
                                                        'dataframe_edition_locked': False,
                                                        'structuring': True,

                                                        'dataframe_descriptor': {
                                                            self.VARIABLES: ('string', None, False),
                                                            self.VALUES: ('string', None, True)}}})
                all_options = list(default_dict.keys())
                if 'algo_options' in self._data_in and algo_name_has_changed:
                    self._data_in['algo_options']['value'] = default_dict
                if 'algo_options' in self._data_in and self._data_in['algo_options']['value'] is not None and list(
                        self._data_in['algo_options']['value'].keys()) != all_options:
                    options_map = ChainMap(
                        self._data_in['algo_options']['value'], default_dict)
                    self._data_in['algo_options']['value'] = {
                        key: options_map[key] for key in all_options}

                # if multipliers in eval_in
                if (len(self.selected_inputs) > 0) and (any([self.MULTIPLIER_PARTICULE in val for val in self.selected_inputs])):
                    generic_multipliers_dynamic_inputs_list = self.create_generic_multipliers_dynamic_input()
                    for generic_multiplier_dynamic_input in generic_multipliers_dynamic_inputs_list:
                        dynamic_inputs.update(generic_multiplier_dynamic_input)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def create_generic_multipliers_dynamic_input(self):
        dynamic_inputs_list = []
        for selected_in in self.selected_inputs:
            if self.MULTIPLIER_PARTICULE in selected_in:
                multiplier_name = selected_in.split('.')[-1]
                origin_var_name = multiplier_name.split('.')[0].split('@')[0]
                # if
                if len(self.ee.dm.get_all_namespaces_from_var_name(origin_var_name)) > 1:
                    self.logger.exception(
                        'Multiplier name selected already exists!')
                origin_var_fullname = self.ee.dm.get_all_namespaces_from_var_name(origin_var_name)[
                    0]
                origin_var_ns = self.ee.dm.get_data(
                    origin_var_fullname, 'namespace')
                dynamic_inputs_list.append(
                    {
                        f'{multiplier_name}': {
                            'type': 'float',
                            'visibility': 'Shared',
                            'namespace': origin_var_ns,
                            'unit': self.ee.dm.get_data(origin_var_fullname).get('unit', '-'),
                            'default': 100
                        }
                    }
                )
        return dynamic_inputs_list

    def create_design_space(self):
        """
        create_design_space
        """
        dspace = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        design_space = None
        if dspace is not None:
            design_space = self.set_design_space()

        return design_space

    def set_design_space(self):
        """
        reads design space (set_design_space)
        """

        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        # variables = self.eval_in_list

        if 'full_name' in dspace_df:
            variables = dspace_df['full_name'].tolist()
            variables = [f'{self.ee.study_name}.{eval}' for eval in variables]
        else:
            variables = self.eval_in_list

        lower_bounds = dspace_df[self.LOWER_BOUND].tolist()
        upper_bounds = dspace_df[self.UPPER_BOUND].tolist()
        values = lower_bounds
        enable_variables = [True for invar in self.eval_in_list]
        # This won't work for an array with a dimension greater than 2
        activated_elems = [[True, True] if self.ee.dm.get_data(var, 'type') == 'array' else [True] for var in
                           self.eval_in_list]
        dspace_dict_updated = pd.DataFrame({self.VARIABLES: variables,
                                            self.VALUES: values,
                                            self.LOWER_BOUND: lower_bounds,
                                            self.UPPER_BOUND: upper_bounds,
                                            self.ENABLE_VARIABLE_BOOL: enable_variables,
                                            self.LIST_ACTIVATED_ELEM: activated_elems})

        design_space = self.read_from_dataframe(dspace_dict_updated)

        return design_space

    def read_from_dataframe(self, df):
        """Parses a DataFrame to read the DesignSpace

        :param df : design space df
        :returns:  the design space
        """
        names = list(df[self.VARIABLES])
        values = list(df[self.VALUES])
        l_bounds = list(df[self.LOWER_BOUND])
        u_bounds = list(df[self.UPPER_BOUND])
        enabled_variable = list(df[self.ENABLE_VARIABLE_BOOL])
        list_activated_elem = list(df[self.LIST_ACTIVATED_ELEM])
        design_space = DesignSpace()
        for dv, val, lb, ub, l_activated, enable_var in zip(names, values, l_bounds, u_bounds, list_activated_elem,
                                                            enabled_variable):

            # check if variable is enabled to add it or not in the design var
            if enable_var:
                self.dict_desactivated_elem[dv] = {}

                if [type(val), type(lb), type(ub)] == [str] * 3:
                    val = val
                    lb = lb
                    ub = ub
                name = dv
                if type(val) != list and type(val) != ndarray:
                    size = 1
                    var_type = ['float']
                    l_b = array([lb])
                    u_b = array([ub])
                    value = array([val])
                else:
                    # check if there is any False in l_activated
                    if not all(l_activated):
                        index_false = l_activated.index(False)
                        self.dict_desactivated_elem[dv] = {
                            'value': val[index_false], 'position': index_false}

                        val = delete(val, index_false)
                        lb = delete(lb, index_false)
                        ub = delete(ub, index_false)

                    size = len(val)
                    var_type = ['float'] * size
                    l_b = array(lb)
                    u_b = array(ub)
                    value = array(val)
                design_space.add_variable(
                    name, size, var_type, l_b, u_b, value)
        return design_space

    def configure(self):
        """Configuration of the DoeEval and setting of the design space
        """
        SoSEval.configure(self)
        # if self.DESIGN_SPACE in self._data_in:
        #     self.design_space = self.create_design_space()

    def generate_samples_from_doe_factory(self):
        """Generating samples for the Doe using the Doe Factory
        """
        algo_name = self.get_sosdisc_inputs(self.ALGO)
        if algo_name == 'CustomDOE':
            return self.create_samples_from_custom_df()
        else:
            self.design_space = self.create_design_space()
            options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            filled_options = {}
            for algo_option in options:
                if options[algo_option] != 'default':
                    filled_options[algo_option] = options[algo_option]

            if self.N_SAMPLES not in options:
                self.logger.warning("N_samples is not defined; pay attention you use fullfact algo "
                                    "and that levels are well defined")

            self.logger.info(filled_options)
            filled_options[self.DIMENSION] = self.design_space.dimension
            filled_options[self._VARIABLES_NAMES] = self.design_space.variables_names
            filled_options[self._VARIABLES_SIZES] = self.design_space.variables_sizes
            # filled_options['n_processes'] = int(filled_options['n_processes'])
            filled_options['n_processes'] = self.get_sosdisc_inputs(
                'n_processes')
            filled_options['wait_time_between_samples'] = self.get_sosdisc_inputs(
                'wait_time_between_fork')
            algo = self.doe_factory.create(algo_name)

            self.samples = algo._generate_samples(**filled_options)

            unnormalize_vect = self.design_space.unnormalize_vect
            round_vect = self.design_space.round_vect
            samples = []
            for sample in self.samples:
                x_sample = round_vect(unnormalize_vect(sample))
                self.design_space.check_membership(x_sample)
                samples.append(x_sample)
            self.samples = samples

            return self.prepare_samples()

    def prepare_samples(self):

        samples = []
        for sample in self.samples:
            sample_dict = self.design_space.array_to_dict(sample)
            # convert arrays in sample_dict into SoSTrades types
            sample_dict = self._convert_array_into_new_type(sample_dict)
            ordered_sample = []
            for in_variable in self.eval_in_list:
                ordered_sample.append(sample_dict[in_variable])
            samples.append(ordered_sample)
        return samples

    def create_samples_from_custom_df(self):
        """Generation of the samples in case of a customed DOE
        """
        self.customed_samples = self.get_sosdisc_inputs('custom_samples_df')
        self.check_customed_samples()
        samples_custom = []
        for index, rows in self.customed_samples.iterrows():
            ordered_sample = []
            for col in rows:
                ordered_sample.append(col)
            samples_custom.append(ordered_sample)
        return samples_custom

    def check_customed_samples(self):
        """ We check that the columns of the dataframe are the same  that  the selected inputs
        We also check that they are of the same type
        """
        if set(self.selected_inputs) != set(self.customed_samples.columns.to_list()):
            self.logger.error("the costumed dataframe columns must be the same and in the same order than the eval in "
                              "list ")
            
    def run(self):
        '''
            Overloaded SoSEval method
            The execution of the doe
        '''
        # upadte default inputs of children with dm values
        self.update_default_inputs(self.sos_disciplines[0])

        dict_sample = {}
        dict_output = {}

        # We first begin by sample generation
        self.samples = self.generate_samples_from_doe_factory()

        # Then add the reference scenario (initial point ) to the generated
        # samples
        self.samples.append(
            [self.ee.dm.get_value(reference_variable_full_name) for reference_variable_full_name in self.eval_in_list])
        reference_scenario_id = len(self.samples)

        if self.INPUT_MULTIPLIER_TYPE != []:
            origin_vars_to_update_dict = self.create_origin_vars_to_update_dict()
            multipliers_samples = copy.deepcopy(self.samples)
            self.add_multiplied_var_to_samples(
                multipliers_samples, origin_vars_to_update_dict)
            eval_in_with_multiplied_var = self.eval_in_list + \
                list(origin_vars_to_update_dict.keys())

        # evaluation of the samples through a call to samples_evaluation
        evaluation_outputs = self.samples_evaluation(
            self.samples, convert_to_array=False, completed_eval_in_list=eval_in_with_multiplied_var)

        # we loop through the samples evaluated to build dictionnaries needed
        # for output generation
        reference_scenario = f'scenario_{reference_scenario_id}'

        for (scenario_name, evaluated_samples) in evaluation_outputs.items():

            # generation of the dictionnary of samples used
            dict_one_sample = {}
            current_sample = evaluated_samples[0]
            scenario_naming = scenario_name if scenario_name != reference_scenario else 'reference'
            for idx, f_name in enumerate(self.eval_in_list):
                dict_one_sample[f_name] = current_sample[idx]
            dict_sample[scenario_naming] = dict_one_sample

            # generation of the dictionnary of outputs
            dict_one_output = {}
            current_output = evaluated_samples[1]
            for idx, values in enumerate(current_output):
                dict_one_output[self.eval_out_list[idx]] = values
            dict_output[scenario_naming] = dict_one_output

        # construction of a dataframe of generated samples
        # columns are selected inputs
        columns = ['scenario']
        columns.extend(self.selected_inputs)
        samples_all_row = []
        for (scenario, scenario_sample) in dict_sample.items():
            samples_row = [scenario]
            for generated_input in scenario_sample.values():
                samples_row.append(generated_input)
            samples_all_row.append(samples_row)
        samples_dataframe = pd.DataFrame(samples_all_row, columns=columns)

        # construction of a dictionnary of dynamic outputs
        # The key is the output name and the value a dictionnary of results
        # with scenarii as keys
        global_dict_output = {key: {} for key in self.eval_out_list}
        for (scenario, scenario_output) in dict_output.items():
            for full_name_out in scenario_output.keys():
                global_dict_output[full_name_out][scenario] = scenario_output[full_name_out]

        # saving outputs in the dm
        self.store_sos_outputs_values(
            {'samples_inputs_df': samples_dataframe})
        for dynamic_output in self.eval_out_list:
            self.store_sos_outputs_values({
                f'{dynamic_output.split(self.ee.study_name + ".")[1]}_dict':
                    global_dict_output[dynamic_output]})
            
    def update_default_inputs(self, disc):
        '''
        Update default inputs of disc with dm values
        '''
        input_data = {}
        input_data_names = disc.get_input_data_names()
        for data_name in input_data_names:
            val = self.ee.dm.get_value(data_name)
            if val is not None:
                input_data[data_name] = val

        # store mdo_chain default inputs
        if disc.is_sos_coupling:
            disc.mdo_chain.default_inputs.update(input_data)
        disc.default_inputs.update(input_data)

        # recursive update default inputs of children
        for sub_disc in disc.sos_disciplines:
            self.update_default_inputs(sub_disc)

    def create_origin_vars_to_update_dict(self):
        origin_vars_to_update_dict = {}
        for select_in in self.eval_in_list:
            if self.MULTIPLIER_PARTICULE in select_in:
                var_origin_f_name = self.get_names_from_multiplier(select_in)[
                    0]
                if var_origin_f_name not in origin_vars_to_update_dict:
                    origin_vars_to_update_dict[var_origin_f_name] = copy.deepcopy(
                        self.ee.dm.get_data(var_origin_f_name)['value'])
        return origin_vars_to_update_dict

    def add_multiplied_var_to_samples(self, multipliers_samples, origin_vars_to_update_dict):
        for sample_i in range(len(multipliers_samples)):
            x = multipliers_samples[sample_i]
            vars_to_update_dict = {}
            for multiplier_i, x_id in enumerate(self.eval_in_list):
                # for grid search multipliers inputs
                var_name = x_id.split(self.ee.study_name + '.')[-1]
                if self.MULTIPLIER_PARTICULE in var_name:
                    var_origin_f_name = '.'.join(
                        [self.ee.study_name, self.get_names_from_multiplier(var_name)[0]])
                    if var_origin_f_name in vars_to_update_dict:
                        var_to_update = vars_to_update_dict[var_origin_f_name]
                    else:
                        var_to_update = copy.deepcopy(
                            origin_vars_to_update_dict[var_origin_f_name])
                    vars_to_update_dict[var_origin_f_name] = self.apply_muliplier(
                        multiplier_name=var_name, multiplier_value=x[multiplier_i] / 100, var_to_update=var_to_update)
            for multiplied_var in vars_to_update_dict:
                self.samples[sample_i].append(
                    vars_to_update_dict[multiplied_var])

    def apply_muliplier(self, multiplier_name, multiplier_value, var_to_update):
        col_index = multiplier_name.split(self.MULTIPLIER_PARTICULE)[
            0].split('@')[1]
        if any(char.isdigit() for char in col_index):
            col_index = re.findall(r'\d+', col_index)[0]
            cols_list = var_to_update.columns.to_list()
            key = cols_list[int(col_index)]
            var_to_update[key] = multiplier_value * var_to_update[key]
        else:
            if isinstance(var_to_update, dict):
                float_cols_ids_list = [dict_keys for dict_keys in var_to_update if isinstance(
                    var_to_update[dict_keys], float)]
            elif isinstance(var_to_update, pd.DataFrame):
                float_cols_ids_list = [
                    df_keys for df_keys in var_to_update if var_to_update[df_keys].dtype == 'float']
            for key in float_cols_ids_list:
                var_to_update[key] = multiplier_value * var_to_update[key]
        return var_to_update

    def get_algo_default_options(self, algo_name):
        """This algo generate the default options to set for a given doe algorithm
        """

        if algo_name in self.algo_dict.keys():
            return self.algo_dict[algo_name]
        else:
            return self.default_algo_options

    def get_full_names(self, names):
        '''
        get full names of ineq_names and obj_names
        '''
        full_names = []
        for i_name in names:
            full_id_l = self.dm.get_all_namespaces_from_var_name(i_name)
            if full_id_l != []:
                if len(full_id_l) > 1:
                    # full_id = full_id_l[0]
                    full_id = self.get_scenario_lagr(full_id_l)
                else:
                    full_id = full_id_l[0]
                full_names.append(full_id)

        return full_names

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''

        poss_in_values_full = []
        poss_out_values_full = []

        for data_in_key in disc._data_in.keys():
            is_input_type = disc._data_in[data_in_key][self.TYPE] in self.INPUT_TYPE
            is_structuring = disc._data_in[data_in_key].get(
                self.STRUCTURING, False)
            in_coupling_numerical = data_in_key in list(
                SoSCoupling.DESC_IN.keys())
            full_id = disc.get_var_full_name(
                data_in_key, disc._data_in)
            is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
                                           ]['io_type'] == 'in'
            is_input_multiplier_type = disc._data_in[data_in_key][self.TYPE] in self.INPUT_MULTIPLIER_TYPE
            is_editable = disc._data_in[data_in_key]['editable']
            is_None = disc._data_in[data_in_key]['value'] is None
            if is_in_type and not in_coupling_numerical and not is_structuring and is_editable:
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                if is_input_type:
                    poss_in_values_full.append(
                        full_id.split(self.ee.study_name + ".")[1])

                elif is_input_multiplier_type and not is_None:
                    poss_in_values_list = self.set_multipliers_values(
                        disc, full_id, data_in_key)
                    for val in poss_in_values_list:
                        poss_in_values_full.append(val)

        for data_out_key in disc._data_out.keys():
            # Caution ! This won't work for variables with points in name
            # as for ac_model
            in_coupling_numerical = data_out_key in list(
                SoSCoupling.DESC_IN.keys()) or data_out_key == 'residuals_history'
            full_id = disc.get_var_full_name(
                data_out_key, disc._data_out)
            if not in_coupling_numerical:
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                poss_out_values_full.append(
                    full_id.split(self.ee.study_name + ".")[1])

        return poss_in_values_full, poss_out_values_full

    def set_multipliers_values(self, disc, full_id, var_name):
        poss_in_values_list = []
        df_var = disc._data_in[var_name]['value']
        # if df_var is dict : transform dict to df
        if disc._data_in[var_name][self.TYPE] == 'dict':
            dict_var = disc._data_in[var_name]['value']
            df_var = pd.DataFrame(
                dict_var, index=list(dict_var.keys()))
        # check & create float columns list from df
        columns = df_var.columns
        float_cols_list = [col_name for col_name in columns if (
            df_var[col_name].dtype == 'float' and not all(df_var[col_name].isna()))]
        # if df with float columns
        if len(float_cols_list) > 0:
            for col_name in float_cols_list:
                col_name_clean = self.clean_var_name(col_name)
                multiplier_fullname = f'{full_id}@{col_name_clean}{self.MULTIPLIER_PARTICULE}'.split(
                    self.ee.study_name + ".")[1]
                poss_in_values_list.append(multiplier_fullname)
            # if df with more than one float column, create multiplier for all
            # columns also
            if len(float_cols_list) > 1:
                multiplier_fullname = f'{full_id}@allcolumns{self.MULTIPLIER_PARTICULE}'.split(
                    self.ee.study_name + ".")[1]
                poss_in_values_list.append(multiplier_fullname)
        return poss_in_values_list

    def clean_var_name(self, var_name):
        return re.sub(r"[^a-zA-Z0-9]", "_", var_name)

    def get_names_from_multiplier(self, var_name):
        var_origin_name = var_name.split(self.MULTIPLIER_PARTICULE)[
            0].split('@')[0]
        column_name = var_name.split(self.MULTIPLIER_PARTICULE)[
            0].split('@')[1]

        return [var_origin_name, column_name]

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        # the eval process to analyse is stored as the only child of SoSEval
        # (coupling chain of the eval process or single discipline)
        analyzed_disc = self.sos_disciplines[0]

        possible_in_values_full, possible_out_values_full = self.fill_possible_values(
            analyzed_disc)

        possible_in_values_full, possible_out_values_full = self.find_possible_values(
            analyzed_disc, possible_in_values_full, possible_out_values_full)

        # Take only unique values in the list
        possible_in_values_full = list(set(possible_in_values_full))
        possible_out_values_full = list(set(possible_out_values_full))

        # Fill the possible_values of eval_inputs

        possible_in_values_full.sort()
        possible_out_values_full.sort()

        default_in_dataframe = pd.DataFrame({'selected_input': [False for invar in possible_in_values_full],
                                             'full_name': possible_in_values_full})
        default_out_dataframe = pd.DataFrame({'selected_output': [False for invar in possible_out_values_full],
                                              'full_name': possible_out_values_full})

        eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
        eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')
        if eval_input_new_dm is None:
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             'value', default_in_dataframe, check_value=False)
        # check if the eval_inputs need to be updtated after a subprocess
        # configure
        elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
            self.check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
                               is_eval_input=True)
            default_dataframe = copy.deepcopy(default_in_dataframe)
            already_set_names = eval_input_new_dm['full_name'].tolist()
            already_set_values = eval_input_new_dm['selected_input'].tolist()
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
                    index]
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             'value', default_dataframe, check_value=False)

        if eval_output_new_dm is None:
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                             'value', default_out_dataframe, check_value=False)
            # check if the eval_inputs need to be updtated after a subprocess
            # configure
        elif set(eval_output_new_dm['full_name'].tolist()) != (set(default_out_dataframe['full_name'].tolist())):
            self.check_eval_io(eval_output_new_dm['full_name'].tolist(), default_out_dataframe['full_name'].tolist(),
                               is_eval_input=False)
            default_dataframe = copy.deepcopy(default_out_dataframe)
            already_set_names = eval_output_new_dm['full_name'].tolist()
            already_set_values = eval_output_new_dm['selected_output'].tolist()
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_output'] = already_set_values[
                    index]
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                             'value', default_dataframe, check_value=False)

        # filling possible values for sampling algorithm name
        self.dm.set_data(f'{self.get_disc_full_name()}.sampling_algo',
                         self.POSSIBLE_VALUES, self.custom_order_possible_algorithms(self.doe_factory.algorithms))

    def custom_order_possible_algorithms(self, algo_list):
        """ This algo sorts the possible algorithms list so that most used algorithms
        which are fullfact,lhs and CustomDOE appears at the top of the list
        The remaing algorithms are sorted in an alphabetical order
        """
        sorted_algorithms = algo_list[:]
        sorted_algorithms.remove('CustomDOE')
        sorted_algorithms.remove("fullfact")
        sorted_algorithms.remove("lhs")
        sorted_algorithms.sort()
        sorted_algorithms.insert(0, "lhs")
        sorted_algorithms.insert(0, 'CustomDOE')
        sorted_algorithms.insert(0, "fullfact")
        return sorted_algorithms

    def set_eval_in_out_lists(self, in_list, out_list):
        '''
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        '''
        self.eval_in_base_list = [
            element.split(".")[-1] for element in in_list]
        self.eval_out_base_list = [
            element.split(".")[-1] for element in out_list]
        self.eval_in_list = [
            f'{self.ee.study_name}.{element}' for element in in_list]
        self.eval_out_list = [
            f'{self.ee.study_name}.{element}' for element in out_list]

    def check_eval_io(self, given_list, default_list, is_eval_input):
        """
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        """

        for given_io in given_list:
            if given_io not in default_list:
                if is_eval_input:
                    error_msg = f'The input {given_io} in eval_inputs is not among possible values. Check if it is an ' \
                                f'input of the subprocess with the correct full name (without study name at the ' \
                                f'beginning) and within allowed types (int, array, float). Dynamic inputs might  not ' \
                                f'be created. '

                else:
                    error_msg = f'The output {given_io} in eval_outputs is not among possible values. Check if it is an ' \
                                f'output of the subprocess with the correct full name (without study name at the ' \
                                f'beginning). Dynamic inputs might  not be created. '

                self.logger.warning(error_msg)
