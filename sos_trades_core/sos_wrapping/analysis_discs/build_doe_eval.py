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
from collections import ChainMap


class BuildDoeEval(SoSEval):
    '''
    Generic DOE evaluation class

    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SUB_PROCESS_INPUTS (structuring)
            |_ REPO_OF_SUB_PROCESSES (structuring)
                        |_ SUB_PROCESS_NAME (structuring)
                                    |_ USECASE_OF_SUB_PROCESS (structuring,dynamic: SUB_PROCESS_NAME!='None')
                                    |_ EVAL_INPUTS (structuring,dynamic : SUB_PROCESS_NAME!='None') NB: Mandatory not to be empty (If not then warning)
                                    |_ EVAL_OUTPUTS (structuring,dynamic : SUB_PROCESS_NAME!='None') NB: Mandatory not to be empty (If not then warning)
                                    |_ SAMPLING_ALGO (structuring,dynamic : SUB_PROCESS_NAME!='None')
                                            |_ CUSTOM_SAMPLES_DF (dynamic: SAMPLING_ALGO=="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo 
                                            |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO!="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo
                                            |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
            |_ N_PROCESSES
            |_ WAIT_TIME_BETWEEN_FORK
            |_ NS_IN_DF (dynamic: if sub_process_ns_in_build is not None)
        |_ DESC_OUT
            |_ SAMPLES_INPUTS_DF
            |_ <var>_dict (dynamic: sampling_algo!='None' and eval_inputs not empty and eval_outputs not empty, for <var> in eval_outputs)

    2) Description of DESC parameters:
        |_ DESC_IN
           |_ SUB_PROCESS_INPUTS:               All inputs for driver builder in the form of a dictionary of four keys
                                                    'process_repository': folder root of the sub processes to be nested inside the DoE.
                                                    'process_repository': folder root of the sub processes to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    'process_name':       selected process name (in repository) to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    'usecase_name' :      either empty or an available usecase of the sub_process
                                                    'usecase_data' :      anonymized dictionary of usecase inputs to be nested in context
                                                                          it is a temporary input: it will be put to None as soon as                                                                        
                                                                          its content is 'loaded' in the dm. We will have it has editable                                                                             
                                                It is in dict type (specific 'proc_builder_modale' type to have a specific GUI widget) 
            |_ REPO_OF_SUB_PROCESSES:           folder root of the sub processes to be nested inside the DoE.
                                                If 'None' then it uses the sos_processes python for doe creation.
                |_ SUB_PROCESS_NAME:            selected process name to be nested inside the DoE.
                                                If 'None' then it uses the sos_processes python for doe creation.
                    |_ USECASE_OF_SUB_PROCESS : either empty or an available usecase of the sub_process
                    |_ EVAL_INPUTS:             selection of input variables to be used for the DoE
                    |_ EVAL_OUTPUTS:            selection of output variables to be used for the DoE (the selected observables)
                    |_ SAMPLING_ALGO:           method of defining the sampling input dataset for the variable chosen in self.EVAL_INPUTS
                        |_ CUSTOM_SAMPLES_DF:   provided input sample
                        |_ DESIGN_SPACE:        provided design space
                        |_ ALGO_OPTIONS:        options depending of the choice of self.SAMPLING_ALGO
            |_ N_PROCESSES:
            |_ WAIT_TIME_BETWEEN_FORK:
            |_ NS_IN_DF :                       a map of ns name: value
         |_ DESC_OUT
            |_ SAMPLES_INPUTS_DF :              copy of the generated or provided input sample
            |_ <var observable name>_dict':     for each selected output observable doe result
                                                associated to sample and the selected observable
    '''
#################### Begin: Ontology of the discipline ###################
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
        #'icon': 'fas fa-grid-4 fa-fw',  # icon for doe driver
        'icon': 'fas fa-screwdriver-wrench fa-fw',  # icon for proc builder
        'version': ''
    }
#################### End: Ontology of the discipline #####################
#################### Begin: Constants and parameters #####################
    # -- Disciplinary attributes
    REPO_OF_SUB_PROCESSES = 'repo_of_sub_processes'
    SUB_PROCESS_NAME = 'sub_process_name'
    USECASE_OF_SUB_PROCESS = 'usecase_of_sub_process'

    SUB_PROCESSES_LIST_WEB = 'sub_processes_list'
    USECASE_OF_SUB_PROCESS_WEB = 'usecase_of_sub_process_web'

    SUB_PROCESS_INPUTS = 'sub_process_inputs'

    EVAL_INPUTS = 'eval_inputs'  # should be in SOS_EVAL
    EVAL_OUTPUTS = 'eval_outputs'  # should be in SOS_EVAL
    N_PROCESSES = 'n_processes'  # should be in SOS_EVAL
    WAIT_TIME_BETWEEN_FORK = 'wait_time_between_fork'  # should be defined in SOS_EVAL

    SAMPLING_ALGO = 'sampling_algo'
    SAMPLES_INPUTS_DF = 'samples_inputs_df'
    CUSTOM_SAMPLES_DF = 'custom_samples_df'
    NS_IN_DF = 'ns_in_df'

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

    DESC_IN = {REPO_OF_SUB_PROCESSES: {'type': 'string',
                                       'structuring': True,
                                       #'default': 'None',
                                       'possible_values': ['None', 'sos_trades_core.sos_processes.test'],
                                       'user_level': 1,
                                       'optional': False
                                       },
               SUB_PROCESS_NAME: {'type': 'string',
                                  'structuring': True,
                                  #'default': 'None',
                                  #'editable': True,
                                  'user_level': 1,
                                  'optional': False
                                  },
               N_PROCESSES: {'type': 'int',
                             'numerical': True,
                             'default': 1},
               WAIT_TIME_BETWEEN_FORK: {'type': 'float',
                                        'numerical': True,
                                        'default': 0.0},
               }

    DESC_OUT = {
        SAMPLES_INPUTS_DF: {'type': 'dataframe',
                            'unit': None, 'visibility': SoSDiscipline.SHARED_VISIBILITY,
                            'namespace': 'ns_doe_eval'},

    }
    # We define here the different default algo options in a case of a DOE
    # TODO Implement a generic get_options functions to retrieve the default
    # options using directly the DoeFactory

    # Default values of algorithms
    default_algo_options = {
        'n_samples': 1,
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
        'n_samples': 1,
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
        'n_samples': 1,
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
        'n_processes': 1,  # same name as N_PROCESSES ! Redundant ?
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
#################### End: Constants and parameters #######################
#################### Begin: Main methods ################################

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''
        if 'ns_doe' not in ee.ns_manager.shared_ns_dict.keys():
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
        self.previous_sub_process_repo = None
        self.previous_sub_process_name = None
        self.previous_sub_process_usecase_name = 'Empty'
        self.dyn_var_sp_from_import_dict = {}
        self.previous_algo_name = ""
        self.sub_process_ns_in_build = None
        self.sub_proc_build_status = 'Empty_SP'
        self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
        self.sub_process_couple = None
        self.usecase_couple = None  # temporarly added attribute

    def build(self):
        '''
            Overloaded SoSEval method
            Get and build builder from sub_process of doe_eval driver
            Added to provide proc builder capability
            Reached from __configure_io in ee.py: self.factory.build() is going from build to build starting from root
            It comes before congiguring()
        '''
        if self.REPO_OF_SUB_PROCESSES in self._data_in and self.SUB_PROCESS_NAME in self._data_in:
            sub_process_repo = self.get_sosdisc_inputs(
                self.REPO_OF_SUB_PROCESSES)
            sub_process_name = self.get_sosdisc_inputs(
                self.SUB_PROCESS_NAME)
            if sub_process_repo != None and sub_process_name != None:  # a sub_process_full_name is available
                # either Unchanged_SP or Create_SP or Replace_SP
                # 1. set_sub_process_status
                self.set_sub_process_status(
                    sub_process_repo, sub_process_name)
                # 2 build_eval_subproc
                if self.sub_proc_build_status == 'Create_SP' or self.sub_proc_build_status == 'Replace_SP':
                    self.build_eval_subproc(
                        sub_process_repo, sub_process_name)
        SoSEval.build(self)

    def configure(self):
        """
            Overloaded SoSEval method,k
            Configuration of the BuildDoeEval
            Reached from __configure_io in ee.py: self.root_process.configure_io() is going from confiure to configure starting from root
            It comes after build()
        """
        SoSEval.configure(self)

        # Treatment of dynamic subprocess inputs in case of change of usecase
        # of subprocess (Added to provide proc builder capability)
        if len(self.dyn_var_sp_from_import_dict) > 0:
            self.set_configure_status(False)
        else:
            self.set_configure_status(True)

    def setup_sos_disciplines(self):
        """
        Overload setup_sos_disciplines to create a dynamic desc_in
        Reached from configure() of sos_discipline [SoSDiscipline.configure in config() of SoSEval]. 
        It is done upstream of set_eval_possible_values()
        Default desc_in are the algo name and its options
        In case of a CustomDOE', additional input is the custom sample (dataframe)
        In other cases, additional inputs are the number of samples and the design space
        """
        # 1. provide possible values of sub_processes
        if self.REPO_OF_SUB_PROCESSES in self._data_in:
            sub_process_repo = self.get_sosdisc_inputs(
                self.REPO_OF_SUB_PROCESSES)
            # if sub_process_repo != self.previous_sub_process_repo:
            # self.previous_sub_process_repo = sub_process_repo # Not useful
            # because also in build
            if sub_process_repo is not None:
                self.setup_sos_disciplines_possible_sub_processes_in_repo(
                    sub_process_repo)
        dynamic_inputs = {}
        dynamic_outputs = {}
        # Remark: in cas of 'Unchanged_SP', it will do a refresh of available
        # subprocesses
        if self.sub_proc_build_status != 'Empty_SP':
            sub_process_repo = self.get_sosdisc_inputs(
                self.REPO_OF_SUB_PROCESSES)
            sub_process_name = self.get_sosdisc_inputs(
                self.SUB_PROCESS_NAME)
            # 2. provide possible values of usecases
            self.setup_sos_disciplines_building_inputs_sub_process_usecase(
                sub_process_repo, sub_process_name, dynamic_inputs)
            # 3. provide driver inputs based on selected subprocess
            self.setup_sos_disciplines_driver_inputs_depend_on_sub_process(
                dynamic_inputs)
        dynamic_inputs, dynamic_outputs = self.setup_sos_disciplines_driver_inputs_depend_on_sampling_algo(
            dynamic_inputs, dynamic_outputs)
        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)
        # 4. import data from selected sub_process_usecase
        self.manage_import_inputs_from_sub_process()

    def set_eval_possible_values(self):
        '''
            Overloaded SoSEval method : used in SoSEval.configure()
            It is done downstream of setup_sos_disciplines()
            In fact: copy past of the SoSEval method --> not needed !
            Once all disciplines have been "run" through,
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

        eval_input_new_dm = self.get_sosdisc_inputs(self.EVAL_INPUTS)
        eval_output_new_dm = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
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

    def run(self):
        '''
            Overloaded SoSEval method
            SoSEval has no specific run method
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
        self.store_sos_outputs_values(
            {self.SAMPLES_INPUTS_DF: samples_dataframe})
        for dynamic_output in self.eval_out_list:
            self.store_sos_outputs_values({
                f'{dynamic_output.split(self.ee.study_name + ".")[1]}_dict':
                    global_dict_output[dynamic_output]})

#################### End: Main methods ################################
##################### Begin: Sub methods ################################
    def set_sub_process_status(self, sub_process_repo, sub_process_name):
        '''
            State subprocess CRUD status
            The subprocess is defined by its name and repository
            Function needed in build(self)
        '''
        # We come from outside driver process
        if sub_process_name != self.previous_sub_process_name or sub_process_repo != self.previous_sub_process_repo:
            self.previous_sub_process_repo = sub_process_repo
            self.previous_sub_process_name = sub_process_name
            self.sub_process_couple = [
                sub_process_repo, sub_process_name]
        # driver process with provided sub process
            if len(self.cls_builder) == 0:
                self.sub_proc_build_status = 'Create_SP'
            else:
                self.sub_proc_build_status = 'Replace_SP'
        else:
            self.sub_proc_build_status = 'Unchanged_SP'

    def build_eval_subproc(self, sub_process_repo, sub_process_name):
        '''
            Get and build builder from sub_process of eval driver
            The subprocess is defined by its name and repository
            Function needed in build(self)
        '''
        # 1. Clean if needed
        if self.sub_proc_build_status == 'Replace_SP':
            # clean all instances before rebuilt
            self.sos_disciplines[0].clean()
            self.sos_disciplines = []  # Should it be del self.sos_disciplines[0]?
            # We "clean" also all dynamic inputs to be reloaded by
            # the usecase
            self.add_inputs({})
        # 2. Get and set the builder of subprocess
        cls_builder = self.get_nested_builders_from_sub_process(
            sub_process_repo, sub_process_name)
        if not isinstance(cls_builder, list):
            cls_builder = [cls_builder]
        self.set_nested_builders(cls_builder)
        # 3. Optional step : capture the input namespace specified at
        # building step
        my_keys = [
            key for key in self.ee.ns_manager.shared_ns_dict if key != 'ns_doe']
        my_dict = {}
        for item in my_keys:
            my_dict[item] = self.ee.ns_manager.shared_ns_dict[item].get_value()
            self.sub_process_ns_in_build = pd.DataFrame(
                list(my_dict.items()), columns=['Name', 'Value'])
        # 4. Shift namespace due to the 'DoE_Eval' inserted
        self.update_namespace_list_with_extra_ns_except_driver(
            'DoE_Eval', after_name=self.ee.study_name)
        # Here we not shift doe_eval that is already in the namespace dictionary.
        # In a more general frame we would need to compare previous
        # namspace dict and only shift new created namespace keys

    def get_nested_builders_from_sub_process(self, sub_process_repo, sub_process_name):
        """
            Create_nested builders from their nested process.
            Function needed in build_eval_subproc(self)
        """
        cls_builder = self.ee.factory.get_builder_from_process(
            repo=sub_process_repo, mod_id=sub_process_name)
        return cls_builder

    def set_nested_builders(self, cls_builder):
        """
            Set nested builder to the eval process in case this eval process was instantiated with an empty nested builder. 
            Function needed in build_eval_subproc(self)
        """
        self.cls_builder = cls_builder
        self.eval_process_builder = self._set_eval_process_builder()
        return

    def update_namespace_list_with_extra_ns_except_driver(self, extra_ns, after_name=None, namespace_list=None):
        '''
            Update the value of a list of namespaces with an extra namespace placed behind after_name
            In our context, we do not want to shift ns_doe_eval and ns_doe already created before nested sub_process
            Function needed in build_eval_subproc(self)
        '''
        if namespace_list is None:
            namespace_list = self.ee.ns_manager.ns_list
            namespace_list = [
                elem for elem in namespace_list if f'{self.ee.study_name}.DoE_Eval' not in elem.get_value()]
        for ns in namespace_list:
            self.ee.ns_manager.update_namespace_with_extra_ns(
                ns, extra_ns, after_name)

    def setup_sos_disciplines_possible_sub_processes_in_repo(self, sub_process_repo):
        """
            Treatment of REPO_OF_SUB_PROCESSES/SUB_PROCESS_NAME
            Configure list possible_values for the SUB_PROCESS_NAME
            Function needed in setup_sos_disciplines()
        """
        possible_sub_process_name_list = self.get_possible_sub_processes_in_repo(
            sub_process_repo, restricted_list=True)
        if self.SUB_PROCESS_NAME in self._data_in:
            self._data_in[self.SUB_PROCESS_NAME]['possible_values'] = possible_sub_process_name_list

    def setup_sos_disciplines_building_inputs_sub_process_usecase(self, sub_process_repo, sub_process_name, dynamic_inputs):
        """
            Treatment of USECASE_OF_SUB_PROCESS
            Setup the usecase_of_sub_process list
            Function needed in setup_sos_disciplines()
        """
        process_usecase_list = ['Empty']
        usecase_list = self.get_possible_sub_process_usecases(
            sub_process_repo, sub_process_name)
        process_usecase_list += usecase_list
        dynamic_inputs.update({self.USECASE_OF_SUB_PROCESS: {'type': 'string',
                                                             'default': 'Empty',
                                                             'possible_values': process_usecase_list,
                                                             'structuring': True,
                                                             'description': 'usecase set of data inputs',
                                                             }})
        return dynamic_inputs

    def setup_sos_disciplines_driver_inputs_depend_on_sub_process(self, dynamic_inputs):
        """
            Update of USECASE_OF_SUB_PROCESS/SAMPLING_ALGO/EVAL_INPUTS/EVAL_OUTPUTS/NS_IN_DF
            Function needed in setup_sos_disciplines()
        """
        dynamic_inputs.update({self.SAMPLING_ALGO: {'type': 'string',
                                                    'possible_values': self.custom_order_possible_algorithms(self.doe_factory.algorithms),
                                                    'structuring': True}
                               })
        dynamic_inputs.update({self.EVAL_INPUTS: {'type': 'dataframe',
                                                  'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                           'full_name': ('string', None, False)},
                                                  'dataframe_edition_locked': False,
                                                  'structuring': True,
                                                  'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                                  'namespace': 'ns_doe_eval'}
                               })
        dynamic_inputs.update({self.EVAL_OUTPUTS: {'type': 'dataframe',
                                                   'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                                            'full_name': ('string', None, False)},
                                                   'dataframe_edition_locked': False,
                                                   'structuring': True,
                                                   'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                                   'namespace': 'ns_doe_eval'}
                               })
        self.sub_proc_build_status = 'Unchanged_SP'
        # Also provide information about namespace variables provided at
        # building time
        if self.sub_process_ns_in_build is not None:
            dynamic_inputs.update({self.NS_IN_DF: {'type': 'dataframe',
                                                   'unit': None,
                                                   'editable': False,
                                                   'default': self.sub_process_ns_in_build}})
        # Also provide desc_in for core/Web GUI/API link prototyping
        if self.usecase_couple is not None:
            dynamic_inputs.update({self.USECASE_OF_SUB_PROCESS_WEB: {'type': 'dict',
                                                                     'unit': None,
                                                                     'editable': False,
                                                                     'default': self.usecase_couple[1]}})
        if self.sub_process_couple is not None:
            dynamic_inputs.update({self.SUB_PROCESSES_LIST_WEB: {'type': 'list',
                                                                 'unit': None,
                                                                 'editable': False,
                                                                 'default': self.sub_process_couple}})
        if self.sub_process_couple is not None and self.usecase_couple is not None:
            sub_process_inputs_dict = {}
            sub_process_inputs_dict['process_repository'] = self.sub_process_couple[0]
            sub_process_inputs_dict['process_name'] = self.sub_process_couple[1]
            #sub_process_inputs_dict['usecase_name'] = self.usecase_couple[0]
            #sub_process_inputs_dict['usecase_data'] = self.usecase_couple[1]
            dynamic_inputs.update({self.SUB_PROCESS_INPUTS: {'type': 'dict',
                                                             'unit': None,
                                                             'editable': False,
                                                             'default': sub_process_inputs_dict}})
        return dynamic_inputs

    def setup_sos_disciplines_driver_inputs_depend_on_sampling_algo(self, dynamic_inputs, dynamic_outputs):
        """
            Create or update ALGO_OPTIONS
            Create or update either CUSTOM_SAMPLES_DF or DESIGN_SPACE
            Function needed in setup_sos_disciplines()
        """
        # Dealing with eval_inputs and eval_outputs
        # we know that SAMPLING_ALGO/EVAL_INPUTS/EVAL_OUTPUTS keys are set at
        # the same time
        algo_name_has_changed = False
        selected_inputs_has_changed = False
        if self.ALGO in self._data_in:  # and sub_process_name != None
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            if self.previous_algo_name != algo_name:
                algo_name_has_changed = True
                self.previous_algo_name = algo_name
            eval_outputs = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)

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

            if algo_name is not None:  # self.ALGO_OPTIONS has to be set whatever selected_inputs/selected_outputs values
                default_dict = self.get_algo_default_options(algo_name)
                dynamic_inputs.update({self.ALGO_OPTIONS: {'type': 'dict', self.DEFAULT: default_dict,
                                                           'dataframe_edition_locked': False,
                                                           'structuring': True,

                                                           'dataframe_descriptor': {
                                                               self.VARIABLES: ('string', None, False),
                                                               self.VALUES: ('string', None, True)}}})
                all_options = list(default_dict.keys())
                if self.ALGO_OPTIONS in self._data_in and algo_name_has_changed:
                    self._data_in[self.ALGO_OPTIONS]['value'] = default_dict
                if self.ALGO_OPTIONS in self._data_in and self._data_in[self.ALGO_OPTIONS]['value'] is not None and list(
                        self._data_in[self.ALGO_OPTIONS]['value'].keys()) != all_options:
                    options_map = ChainMap(
                        self._data_in[self.ALGO_OPTIONS]['value'], default_dict)
                    self._data_in[self.ALGO_OPTIONS]['value'] = {
                        key: options_map[key] for key in all_options}

            if algo_name is not None and len(selected_inputs) == 0:
                # Warning: selected_inputs cannot be empty
                # Problem: it is not None but it is an "empty dataframe" so has
                # the meaning of None (i.e. Mandatory field)
                self.logger.warning('Selected_inputs cannot be empty!')
            if algo_name is not None and len(selected_outputs) == 0:
                self.logger.warning('Selected_outputs cannot be empty!')

            # we set the lists which will be used by the evaluation
            # function of sosEval
            self.set_eval_in_out_lists(selected_inputs, selected_outputs)

            if len(selected_outputs) > 0:
                # setting dynamic outputs. One output of type dict per selected
                # output
                for out_var in self.eval_out_list:
                    dynamic_outputs.update(
                        {f'{out_var.split(self.ee.study_name + ".")[1]}_dict': {'type': 'dict', 'visibility': 'Shared',
                                                                                'namespace': 'ns_doe'}})

            if algo_name is not None and len(selected_inputs) > 0:
                if algo_name == "CustomDOE":
                    default_custom_dataframe = pd.DataFrame(
                        [[NaN for input in range(len(self.selected_inputs))]], columns=self.selected_inputs)
                    dataframe_descriptor = {}
                    for i, key in enumerate(self.selected_inputs):
                        cle = key
                        var = tuple([self.ee.dm.get_data(
                            self.eval_in_list[i], 'type'), None, True])
                        dataframe_descriptor[cle] = var

                    dynamic_inputs.update({self.CUSTOM_SAMPLES_DF: {'type': 'dataframe', self.DEFAULT: default_custom_dataframe,
                                                                    'dataframe_descriptor': dataframe_descriptor,
                                                                    'dataframe_edition_locked': False
                                                                    }})
                    if self.CUSTOM_SAMPLES_DF in self._data_in and selected_inputs_has_changed:
                        self._data_in[self.CUSTOM_SAMPLES_DF]['value'] = default_custom_dataframe
                        self._data_in[self.CUSTOM_SAMPLES_DF]['dataframe_descriptor'] = dataframe_descriptor

                else:

                    default_design_space = pd.DataFrame({'variable': selected_inputs,

                                                         'lower_bnd': [[0.0, 0.0] if self.ee.dm.get_data(var,
                                                                                                         'type') == 'array' else 0.0
                                                                       for var in self.eval_in_list],
                                                         'upper_bnd': [[10.0, 10.0] if self.ee.dm.get_data(var,
                                                                                                           'type') == 'array' else 10.0
                                                                       for var in self.eval_in_list]
                                                         })

                    dynamic_inputs.update({self.DESIGN_SPACE: {'type': 'dataframe', self.DEFAULT: default_design_space
                                                               }})
                    if self.DESIGN_SPACE in self._data_in and selected_inputs_has_changed:
                        self._data_in[self.DESIGN_SPACE]['value'] = default_design_space
        return dynamic_inputs, dynamic_outputs

    def manage_import_inputs_from_sub_process(self):
        """
            Function needed in setup_sos_disciplines()
        """
        # Set sub_proc_import_usecase_status
        if self.USECASE_OF_SUB_PROCESS in self._data_in:  # and self.sub_proc_build_status != 'Empty_SP'
            sub_process_usecase_name = self.get_sosdisc_inputs(
                self.USECASE_OF_SUB_PROCESS)
            self.set_sub_process_usecase_status_from_user_inputs(
                sub_process_usecase_name)
            possible_values = self.get_data_io_from_key('in', self.USECASE_OF_SUB_PROCESS)[
                self.POSSIBLE_VALUES]
            # would be the case from script but not in the gui
            if sub_process_usecase_name not in possible_values:
                self.logger.error(
                    'Selected use_case is not in the possible_values!')
        else:
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

        # Treat the case of SP_UC_Import
        if self.sub_proc_import_usecase_status == 'SP_UC_Import':
            poc_pre_treatment = True
            if poc_pre_treatment:
                # 1.1 Find usecase sub_process_usecase_full_name
                sub_process_usecase_full_name = self.get_sub_process_usecase_full_name()
                # 1.2 import data in anonymized form
                anonymize_input_dict_from_usecase = self.import_input_data_from_usecase_of_sub_process(
                    sub_process_usecase_full_name)  # variable temporarly added as an attribute to put it as a editable desc_in
                # 1.3 Store usecase_couple
                self.usecase_couple = [self.get_sosdisc_inputs(self.USECASE_OF_SUB_PROCESS),
                                       anonymize_input_dict_from_usecase]
            else:  # To be done for WEB GUI/API
                pass
                sub_process_usecase_name = self.get_sosdisc_inputs(
                    self.SUB_PROCESS_INPUTS)['usecase_name']
                anonymize_input_dict_from_usecase = self.get_sosdisc_inputs(
                    self.SUB_PROCESS_INPUTS)['usecase_data']
            # 2 put anonymized dict in context (unanonimize)
            input_dict_from_usecase = self.put_anonymized_input_dict_in_sub_process_context(
                anonymize_input_dict_from_usecase)
            # 3. treat data because of dynamic keys not in dict
            #    Added treatment for input_dict_from_usecase with dynamic keys
            #   Find dynamic keys and redirect them in
            #   self.dyn_var_sp_from_import_dict and removing from input_dict_from_usecase
            # we so replace
            # self.ee.dm.set_values_from_dict(input_dict_from_usecase) by the
            # following function
            dyn_key_list = self.set_only_static_values_from_dict(
                input_dict_from_usecase)
            for key in dyn_key_list:
                self.dyn_var_sp_from_import_dict[key] = input_dict_from_usecase[key]
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
        # there are still dynamic variables put apart
        elif len(self.dyn_var_sp_from_import_dict) != 0:
            self.ee.dm.set_values_from_dict(self.dyn_var_sp_from_import_dict)
            # Is it also OK in case of a dynamic param of dynamic param ?
            self.dyn_var_sp_from_import_dict = {}

    def get_possible_sub_processes_in_repo(self, sub_process_repo, restricted_list=True):
        """
            Create_nested builders from their nested process.
            Function needed in setup_sos_disciplines_possible_sub_processes_in_repo()
        """
        process_factory = SoSProcessFactory(additional_repository_list=[
            sub_process_repo], search_python_path=False)
        process_list_dict = process_factory.get_processes_dict()
        if restricted_list == True:
            possible_sub_process_name_list = ['test_disc_hessian']
            possible_sub_process_name_list += [
                'test_disc1_disc2_coupling', 'test_sellar_coupling', 'test_disc10_setup_sos_discipline']
            filtered_process_list = [
                proc_name for proc_name in process_list_dict[sub_process_repo] if 'test_proc_build_' in proc_name]
            possible_sub_process_name_list += filtered_process_list
        else:
            possible_sub_process_name_list = process_list_dict
        return possible_sub_process_name_list

    def custom_order_possible_algorithms(self, algo_list):
        """ This algo sorts the possible algorithms list so that most used algorithms
            which are fullfact,lhs and CustomDOE appears at the top of the list
            The remaing algorithms are sorted in an alphabetical order
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sub_process()
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

    def get_possible_sub_process_usecases(self, sub_process_repo, sub_process_name):
        '''
            Once subprocess has been selected,
            get the possible list of usecases if any
            Function needed in setup_sos_disciplines_building_inputs_sub_process_usecase()
        '''
        usecase_list = []
        imported_module = import_module(
            '.'.join([sub_process_repo, sub_process_name]))
        process_directory = dirname(imported_module.__file__)
        for usecase_py in listdir(process_directory):
            if usecase_py.startswith('usecase'):
                usecase = usecase_py.replace('.py', '')
                usecase_list.append(
                    '.'.join([usecase]))
        return usecase_list

    def get_algo_default_options(self, algo_name):
        """
            This algo generate the default options to set for a given doe algorithm
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sampling_algo()
        """
        if algo_name in self.algo_dict.keys():
            return self.algo_dict[algo_name]
        else:
            return self.default_algo_options

    def set_sub_process_usecase_status_from_user_inputs(self, sub_process_usecase_name):
        """
            State subprocess usecase import status
            The subprocess is defined by its name and repository
            Function needed in manage_import_inputs_from_sub_process()
        """
        usecase_has_changed = False
        if self.previous_sub_process_usecase_name != sub_process_usecase_name:
            self.previous_sub_process_usecase_name = sub_process_usecase_name
            if sub_process_usecase_name != 'Empty':
                self.sub_proc_import_usecase_status = 'SP_UC_Import'
        else:
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

    def get_sub_process_usecase_full_name(self):
        """
            Function needed in manage_import_inputs_from_sub_process()
        """
        sub_process_repo = self.get_sosdisc_inputs(
            self.REPO_OF_SUB_PROCESSES)
        sub_process_name = self.get_sosdisc_inputs(
            self.SUB_PROCESS_NAME)
        sub_process_usecase_name = self.get_sosdisc_inputs(
            self.USECASE_OF_SUB_PROCESS)
        #
        sub_process_usecase_repo = '.'.join(
            [sub_process_repo, sub_process_name])
        sub_process_usecase_full_name = '.'.join(
            [sub_process_usecase_repo, sub_process_usecase_name])
        return sub_process_usecase_full_name

    def import_input_data_from_usecase_of_sub_process(self, sub_process_usecase_full_name):
        """
            Load data in anonymized form of the selected sub process usecase
            Function needed in manage_import_inputs_from_sub_process()
        """
        # Get anonymized dict from sub_process_usecase_full_name
        imported_module = import_module(sub_process_usecase_full_name)
        study_tmp = getattr(imported_module, 'Study')(
            execution_engine=self.ee)
        anonymize_input_dict_from_usecase = {}
        # Remark: see def __anonymize_key in execution_engine
        study_tmp.study_name = self.ee.STUDY_PLACEHOLDER_WITHOUT_DOT
        anonymize_usecase_data = study_tmp.setup_usecase()
        if not isinstance(anonymize_usecase_data, list):
            anonymize_usecase_data = [anonymize_usecase_data]
        for uc_d in anonymize_usecase_data:
            anonymize_input_dict_from_usecase.update(uc_d)
        return anonymize_input_dict_from_usecase

    def put_anonymized_input_dict_in_sub_process_context(self, anonymize_input_dict_from_usecase):
        """
            Put_anonymized_input_dict in sub_process context
            Function needed in manage_import_inputs_from_sub_process()
        """
        # Get unanonymized dict (i.e. dict of subprocess in driver context)
        # from anonymized dict and context
        # good prefix in context
        new_study_placeholder = f'{self.ee.study_name}.DoE_Eval'
        input_dict_from_usecase = {}
        for key_to_unanonymize, value in anonymize_input_dict_from_usecase.items():
            converted_key = key_to_unanonymize.replace(
                self.ee.STUDY_PLACEHOLDER_WITHOUT_DOT, new_study_placeholder)  # see def __unanonymize_key  in execution_engine
            uc_d = {converted_key: value}
            input_dict_from_usecase.update(uc_d)
        return input_dict_from_usecase

    def set_only_static_values_from_dict(self, values_dict, full_ns_keys=True):
        ''' Set values in data_dict from dict with namespaced keys 
            if full_ns_keys (not uuid), try to get its uuid correspondency through get_data_id function
            Function needed in manage_import_inputs_from_sub_process()
        '''
        dyn_key_list = []
        keys_to_map = self.ee.dm.data_id_map.keys(
        ) if full_ns_keys else self.ee.dm.data_id_map.values()
        for key, value in values_dict.items():
            if not key in keys_to_map:
                dyn_key_list += [key]
            else:
                k = self.ee.dm.get_data_id(key) if full_ns_keys else key
                VALUE = SoSDiscipline.VALUE
                self.ee.dm.data_dict[k][VALUE] = value
        return dyn_key_list

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
            Function needed in set_eval_possible_values()
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
            if is_input_type and is_in_type and not in_coupling_numerical and not is_structuring:
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

    def set_eval_in_out_lists(self, in_list, out_list):
        '''
            Set the evaluation variable list (in and out) present in the DM
            which fits with the eval_in_base_list filled in the usecase or by the user
            Function needed in set_eval_possible_values()
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
            Function needed in set_eval_possible_values()
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

    def generate_samples_from_doe_factory(self):
        """Generating samples for the Doe using the Doe Factory
            Function needed in run()
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
            # filled_options[self.N_PROCESSES] = int(filled_options[self.N_PROCESSES])
            filled_options[self.N_PROCESSES] = self.get_sosdisc_inputs(
                self.N_PROCESSES)
            filled_options['wait_time_between_samples'] = self.get_sosdisc_inputs(
                self.WAIT_TIME_BETWEEN_FORK)
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

    def create_samples_from_custom_df(self):
        """
            Generation of the samples in case of a customed DOE
            Function needed in generate_samples_from_doe_factory()
        """
        self.customed_samples = self.get_sosdisc_inputs(self.CUSTOM_SAMPLES_DF)
        self.check_customed_samples()
        samples_custom = []
        for index, rows in self.customed_samples.iterrows():
            ordered_sample = []
            for col in rows:
                ordered_sample.append(col)
            samples_custom.append(ordered_sample)
        return samples_custom

    def check_customed_samples(self):
        """
            We check that the columns of the dataframe are the same  that  the selected inputs
            We also check that they are of the same type
            Function needed in create_samples_from_custom_df()
        """
        if set(self.selected_inputs) != set(self.customed_samples.columns.to_list()):
            self.logger.error("the customed dataframe columns must be the same and in the same order than the eval in "
                              "list ")

    def create_design_space(self):
        """
            Create_design_space
            Function needed in generate_samples_from_doe_factory()
        """
        dspace = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        design_space = None
        if dspace is not None:
            design_space = self.set_design_space()
        return design_space

    def set_design_space(self):
        """
            Reads design space (set_design_space)
            Function needed in create_design_space()
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
        """
            Parses a DataFrame to read the DesignSpace
            :param df : design space df
            :returns:  the design space
            Function needed in set_design_space()
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

    def prepare_samples(self):
        """
            Prepare sample
            Function needed in generate_samples_from_doe_factory()
        """
        samples = []
        for sample in self.samples:
            sample_dict = self.design_space.array_to_dict(sample)
            sample_dict = self._convert_array_into_new_type(sample_dict)
            ordered_sample = []
            for in_variable in self.eval_in_list:
                ordered_sample.append(sample_dict[in_variable])
            samples.append(ordered_sample)
        return samples

    def get_full_names(self, names):
        '''
            Get full names of ineq_names and obj_names
            Provided function but not used
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
##################### End: Sub methods ################################
