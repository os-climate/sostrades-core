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

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.sos_processes.processes_factory import SoSProcessFactory
from importlib import import_module
from os.path import dirname
from os import listdir

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_eval import SoSEval
import pandas as pd


class BuildDoeEval(SoSEval):
    '''
    Generic DOE evaluation class
    Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ repo_of_sub_processes (structuring)
                        |_ sub_process_folder_name (structuring)
                                    |_ usecase_of_sub_process (structuring,dynamic: sub_process_folder_name!='None')
                                    |_ eval_inputs (structuring)
                                    |_ eval_outputs (structuring)
                                    |_ sampling_algo (structuring)
                                            |_ custom_samples_df (dynamic: sampling_algo=="CustomDOE")
                                            |_ design_space (dynamic: sampling_algo!="CustomDOE")
                                            |_ algo_options (dynamic: sampling_algo!='None' and eval_inputs not empty and eval_output not empty)
            |_ n_processes
            |_ wait_time_between_fork
        |_ DESC_OUT
            |_ samples_inputs_df
            |_ all_ns_dict
            |_ <var>_dict (dynamic, for <var> in eval_outputs)
    Description of Desc parameters:
        'repo_of_sub_processes':    folder root of the sub processes to be nested inside the DoE.
                                    If 'None' then it uses the sos_processes python for doe creation.
        'sub_process_folder_name':  selected process folder name to be nested inside the DoE.
                                    If 'None' then it uses the sos_processes python for doe creation.
        'eval_inputs':              selection of input variables to be used for the DoE
        'eval_outputs':             selection of output variables to be used for the DoE (the selected observables)
        'sampling_algo':            method of defining the sampling input dataset for the variable chosen in 'eval_inputs'
        'n_processes':
        'wait_time_between_fork':
        'samples_inputs_df' :       copy of the generated or provided input sample
        'all_ns_dict' :             a map of ns keys: values
        'usecase_of_sub_process' :  either empty or an available usecase of the sub_process
        'custom_samples_df':        provided input sample
        'design_space':             provided design space
        'algo_options':             options depending of the choice of 'sampling_algo'
        <var observable name>_dict':for each selected output observable doe result
                                    associated to sample and the selected observable
    '''

    # ontology information
    _ontology_data = {
        'label': 'DoE_eval driver',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'DoE driver discipline that implements a Design of Experiment on a nested system (Implementation based on SoSEval driver discipline). Remark: the optimization "formulation" capability is not covered',
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
    INPUT_TYPE = ['float', 'array', 'int']

    DESC_IN = {'repo_of_sub_processes': {'type': 'string',
                                         'structuring': True,
                                         'default': 'None',
                                         'possible_values': ['None', 'sos_trades_core.sos_processes.test']
                                         },
               'sub_process_folder_name': {'type': 'string',
                                           'structuring': True,
                                           'default': 'None',
                                           'editable': True
                                           },
               'sampling_algo': {'type': 'string',
                                 'structuring': True},
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
               'n_processes': {'type': 'int',
                               'numerical': True,
                               'default': 1},
               'wait_time_between_fork': {'type': 'float',
                                          'numerical': True,
                                          'default': 0.0},
               }

    DESC_OUT = {
        'samples_inputs_df': {'type': 'dataframe',
                              'unit': None, 'visibility': SoSDiscipline.SHARED_VISIBILITY,
                              'namespace': 'ns_doe_eval'},
        'all_ns_dict': {'type': 'dataframe',
                        'unit': None, 'visibility': SoSDiscipline.SHARED_VISIBILITY,
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
        ee.ns_manager.add_ns('ns_doe', ee.study_name)
        super(BuildDoeEval, self).__init__(sos_name, ee, cls_builder)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.DOE')
        self.doe_factory = DOEFactory()
        self.design_space = None
        self.samples = None
        self.customed_samples = None
        self.dict_desactivated_elem = {}
        self.selected_outputs = []
        self.selected_inputs = []
        self.previous_repo_of_sub_processes = ""
        self.previous_sub_process_folder_name = ""
        self.previous_usecase_of_sub_process = ""
        self.previous_algo_name = ""

    def build(self):
        '''
        Get and build builder from sub_process of doe_eval driver 
        '''
        sub_process_folder_name_has_changed = False
        if 'repo_of_sub_processes' in self.get_data_io_dict_keys('in') and 'sub_process_folder_name' in self.get_data_io_dict_keys('in'):
            repo = self.get_sosdisc_inputs('repo_of_sub_processes')
            sub_process = self.get_sosdisc_inputs(
                'sub_process_folder_name')
            # We come from outside driver process
            if sub_process != self.previous_sub_process_folder_name:
                sub_process_folder_name_has_changed = True
                self.previous_sub_process_folder_name = sub_process
            if repo != 'None' and sub_process != 'None':  # We come from driver process with provided sub process
                if len(self.cls_builder) == 0 or sub_process_folder_name_has_changed:
                    cls_builder = self.get_nested_builders_from_sub_process(
                        repo, sub_process)
                    if not isinstance(cls_builder, list):
                        cls_builder = [cls_builder]
                    self.set_nested_builders(cls_builder)
                    self.update_namespace_list_with_extra_ns_except_driver(
                        'DoE_Eval', after_name=self.ee.study_name)  # --> yes but here we do not want to shift ns_doe_eval and ns_doe already created
        SoSEval.build(self)

    def update_namespace_list_with_extra_ns_except_driver(self, extra_ns, after_name=None, namespace_list=None):
        '''
        Update the value of a list of namespaces with an extra namespace placed behind after_name
        '''
        if namespace_list is None:
            namespace_list = self.ee.ns_manager.ns_list
            namespace_list = [
                elem for elem in namespace_list if f'{self.ee.study_name}.DoE_Eval' not in elem.get_value()]
        for ns in namespace_list:
            self.ee.ns_manager.update_namespace_with_extra_ns(
                ns, extra_ns, after_name)

    def get_nested_builders_from_sub_process(self, repo, sub_process):
        """
        create_nested builders from their nested process
        """
        cls_builder = self.ee.factory.get_builder_from_process(
            repo=repo, mod_id=sub_process)
        return cls_builder

    def set_nested_builders(self, cls_builder):
        """
        Set nested builder to the doe_eval process in case this doe_eval process was instantiated with an empty nested builder
        """
        self.cls_builder = cls_builder
        self.eval_process_builder = self._set_eval_process_builder()
        return

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

        if 'repo_of_sub_processes' in self.get_data_io_dict_keys('in') and 'sub_process_folder_name' in self._data_in:
            repo = self.get_sosdisc_inputs('repo_of_sub_processes')
            if repo == 'None':
                self._data_in['sub_process_folder_name']['editable'] = True
            else:
                self._data_in['sub_process_folder_name']['editable'] = True

        # configure the sub_process_folder_name list
        repo_of_sub_processes_has_changed = False
        if 'repo_of_sub_processes' in self.get_data_io_dict_keys('in'):
            repo = self.get_sosdisc_inputs('repo_of_sub_processes')
            if repo != self.previous_repo_of_sub_processes:
                repo_of_sub_processes_has_changed = True
                self.previous_repo_of_sub_processes = repo
            # if repo != 'None' and repo_of_sub_processes_has_changed:
            if repo != 'None':
                sub_process_folder_name = ['test_disc_hessian']
                sub_process_folder_name += [
                    'test_disc1_disc2_coupling', 'test_sellar_coupling']
                process_factory = SoSProcessFactory(additional_repository_list=[
                    repo], search_python_path=False)
                process_list_dict = process_factory.get_processes_dict()
                filtered_process_list = [
                    proc_name for proc_name in process_list_dict[repo] if 'test_proc_build_' in proc_name]
                sub_process_folder_name += filtered_process_list
                # dynamic_inputs.update(
                #    {'sub_process_folder_name': {'type': 'string', 'default': 'None', 'possible_values': sub_process_folder_name, 'structuring': True
                #                                 }})
                # if 'sub_process_folder_name' in self._data_in and
                # repo_of_sub_processes_has_changed:
                if 'sub_process_folder_name' in self._data_in:
                    self._data_in['sub_process_folder_name']['possible_values'] = sub_process_folder_name
                    #self._data_in['sub_process_folder_name']['value'] = 'None'

        # configure the usecase_of_sub_process list
        if 'repo_of_sub_processes' in self.get_data_io_dict_keys('in') and 'sub_process_folder_name' in self.get_data_io_dict_keys('in'):
            repo = self.get_sosdisc_inputs('repo_of_sub_processes')
            sub_process = self.get_sosdisc_inputs('sub_process_folder_name')
            if repo != 'None' and sub_process != 'None':
                process_usecase_list = ['Empty']
                usecase_list = self.get_usecase_possible_values(
                    repo, sub_process)
                process_usecase_list += usecase_list
                dynamic_inputs.update(
                    {'usecase_of_sub_process': {'type': 'string', 'default': 'Empty', 'possible_values': process_usecase_list, 'structuring': True}})

        if self.ALGO in self._data_in:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            if self.previous_algo_name != algo_name:
                algo_name_has_changed = True
                self.previous_algo_name = algo_name
            eval_outputs = self.get_sosdisc_inputs('eval_outputs')
            eval_inputs = self.get_sosdisc_inputs('eval_inputs')

            # we fetch the inputs and outputs selected by the user
            if not eval_outputs is None:
                selected_outputs = eval_outputs[eval_outputs['selected_output']
                                                == True]['full_name']
                self.selected_outputs = selected_outputs.tolist()
            if not eval_inputs is None:
                selected_inputs = eval_inputs[eval_inputs['selected_input']
                                              == True]['full_name']
                if set(selected_inputs.tolist()) != set(self.selected_inputs):
                    selected_inputs_has_changed = True
                    self.selected_inputs = selected_inputs.tolist()
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

                                                        'dataframe_descriptor': {
                                                            self.VARIABLES: ('string', None, False),
                                                            self.VALUES: ('string', None, True)}}})
                if 'algo_options' in self._data_in and algo_name_has_changed:
                    self._data_in['algo_options']['value'] = default_dict

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)
        self.load_data_from_usecase_of_subprocess()

    def get_usecase_possible_values(self, repo, sub_process):
        '''
            Once subprocess has been selected,
            get the possible values for usecases if any
        '''
        usecase_list = []
        imported_module = import_module(
            '.'.join([repo, sub_process]))
        process_directory = dirname(imported_module.__file__)
        for usecase_py in listdir(process_directory):
            if usecase_py.startswith('usecase'):
                usecase = usecase_py.replace('.py', '')
                usecase_list.append(
                    '.'.join([usecase]))
        return usecase_list

    def load_data_from_usecase_of_subprocess(self):
        """
        load data of the selected sub process usecase and put them as a child of doe eval
        """
        usecase_has_changed = False
        if 'usecase_of_sub_process' in self._data_in and 'repo_of_sub_processes' in self.get_data_io_dict_keys('in') and 'sub_process_folder_name' in self.get_data_io_dict_keys('in'):
            usecase = self.get_sosdisc_inputs('usecase_of_sub_process')
            repository = self.get_sosdisc_inputs('repo_of_sub_processes')
            process = self.get_sosdisc_inputs('sub_process_folder_name')
            if self.previous_usecase_of_sub_process != usecase:
                usecase_has_changed = True
                self.previous_usecase_of_sub_process = usecase
            if usecase != 'Empty' and usecase_has_changed:
                imported_usecase = '.'.join([repository, process, usecase])
                imported_module = import_module(imported_usecase)
                study_tmp = getattr(imported_module, 'Study')(
                    execution_engine=self.ee)
                # we provide the good prefix to the loaded use_case data
                study_tmp.study_name = f'{self.ee.study_name}.DoE_Eval'
                usecase_data = study_tmp.setup_usecase()
                if not isinstance(usecase_data, list):
                    usecase_data = [usecase_data]
                input_dict_to_load = {}
                for uc_d in usecase_data:
                    input_dict_to_load.update(uc_d)
                # print(input_dict_to_load)
                self.ee.dm.set_values_from_dict(
                    input_dict_to_load)

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
        """Configuration of the BuildDoeEval and setting of the design space
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

        dict_sample = {}
        dict_output = {}

        # We first begin by sample generation
        self.samples = self.generate_samples_from_doe_factory()

        # Then add the reference scenario (initial point ) to the generated
        # samples
        self.samples.append(
            [self.ee.dm.get_value(reference_variable_full_name) for reference_variable_full_name in self.eval_in_list])
        reference_scenario_id = len(self.samples)

        # evaluation of the samples through a call to samples_evaluation
        evaluation_outputs = self.samples_evaluation(
            self.samples, convert_to_array=False)

        # we loop through the samples evaluated to build dictionnaries needed
        # for output generation
        reference_scenario = f'scenario_{reference_scenario_id}'
        for (scenario_name, evaluated_samples) in evaluation_outputs.items():

            # generation of the dictionnary of samples used
            dict_one_sample = {}
            current_sample = evaluated_samples[0]
            scenario_naming = scenario_name if scenario_name != reference_scenario else 'reference'
            for idx, values in enumerate(current_sample):
                dict_one_sample[self.eval_in_list[idx]] = values
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
        self.status = 'RUNNING'
        #########################################################
        my_keys = [key for key in self.ee.ns_manager.all_ns_dict]
        my_dict = {}
        for item in my_keys:
            my_dict[item] = self.ee.ns_manager.all_ns_dict[item].to_dict()
        my_all_ns_dict = pd.DataFrame.from_dict(my_dict, orient='index')
        del my_all_ns_dict['dependency_disc_list']
        #########################################################
        self.store_sos_outputs_values(
            {'samples_inputs_df': samples_dataframe,
             'all_ns_dict': my_all_ns_dict
             })
        for dynamic_output in self.eval_out_list:
            self.store_sos_outputs_values({
                f'{dynamic_output.split(self.ee.study_name + ".")[1]}_dict':
                    global_dict_output[dynamic_output]})

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
            in_coupling_numerical = data_in_key in list(
                SoSCoupling.DESC_IN.keys())
            full_id = disc.get_var_full_name(
                data_in_key, disc._data_in)
            is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
                                           ]['io_type'] == 'in'
            if is_input_type and is_in_type and not in_coupling_numerical:
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                poss_in_values_full.append(
                    full_id.split(self.ee.study_name + ".")[1])
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
