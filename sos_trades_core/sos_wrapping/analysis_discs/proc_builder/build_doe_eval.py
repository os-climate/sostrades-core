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
from importlib import import_module

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.proc_builder.sos_add_subproc_to_driver import AddSubProcToDriver
from sos_trades_core.execution_engine.proc_builder.build_sos_eval import BuildSoSEval
from sos_trades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
import pandas as pd
from collections import ChainMap


class BuildDoeEval(BuildSoSEval):
    '''
    Generic DOE evaluation class

    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
                |_ EVAL_INPUTS (namespace: 'ns_doe_eval', structuring,dynamic : self.sub_proc_build_status != 'Empty_SP') NB: Mandatory not to be empty (If not then warning)
                |_ EVAL_OUTPUTS (namespace: 'ns_doe_eval', structuring, dynamic : self.sub_proc_build_status != 'Empty_SP') NB: Mandatory not to be empty (If not then warning)
                |_ SAMPLING_ALGO (structuring,dynamic : self.sub_proc_build_status != 'Empty_SP')
                        |_ CUSTOM_SAMPLES_DF (dynamic: SAMPLING_ALGO=="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo 
                        |_ DESIGN_SPACE (dynamic: SAMPLING_ALGO!="CustomDOE") NB: default DESIGN_SPACE depends on EVAL_INPUTS (As to be "Not empty") And Algo
                        |_ ALGO_OPTIONS (structuring, dynamic: SAMPLING_ALGO != None)
                        |_ <var multiplier name> (internal namespace: 'origin_var_ns', dynamic: almost one selected inputs with MULTIPLIER_PARTICULE ('__MULTIPLIER__) in its name, only used in grid_search_eval) 
            |_ N_PROCESSES
            |_ WAIT_TIME_BETWEEN_FORK
        |_ DESC_OUT
            |_ SAMPLES_INPUTS_DF (namespace: 'ns_doe_eval') 
            |_ <var>_dict (internal namspace 'ns_doe', dynamic: sampling_algo!='None' and eval_inputs not empty and eval_outputs not empty, for <var> in eval_outputs)

    2) Description of DESC parameters:
        |_ DESC_IN                                                                           
                    |_ EVAL_INPUTS:             selection of input variables to be used for the DoE
                    |_ EVAL_OUTPUTS:            selection of output variables to be used for the DoE (the selected observables)
                    |_ SAMPLING_ALGO:           method of defining the sampling input dataset for the variable chosen in self.EVAL_INPUTS
                        |_ CUSTOM_SAMPLES_DF:   provided input sample
                        |_ DESIGN_SPACE:        provided design space
                        |_ ALGO_OPTIONS:        options depending on the choice of self.SAMPLING_ALGO
                        |_ <var multiplier name>: for each selected input with MULTIPLIER_PARTICULE in its name (only used in grid_search_eval)
            |_ N_PROCESSES:
            |_ WAIT_TIME_BETWEEN_FORK:
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
        'definition': 'DoE driver discipline that implements a Design of Experiment on a nested system (Implementation based on BuildSoSEval driver discipline). Remark: the optimization "formulation" capability is not covered',
        #'icon': 'fas fa-grid-4 fa-fw',  # icon for doe driver
        'icon': 'fas fa-screwdriver-wrench fa-fw',  # icon for proc builder
        'version': ''
    }
#################### End: Ontology of the discipline #####################
#################### Begin: Constants and parameters #####################
    # -- Disciplinary attributes
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

    default_process_builder_parameter_type = ProcessBuilderParameterType(
        None, None, 'Empty')

    DESC_IN = {
        SUB_PROCESS_INPUTS: {'type': SoSDiscipline.PROC_BUILDER_MODAL,
                             'structuring': True,
                             'default': default_process_builder_parameter_type.to_data_manager_dict(),
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

# Begin: Main methods for proc builder to be specified in specific driver
# ####


### End: Main methods for proc builder to be specified in specific driver ####

#################### Begin: Main methods ################################

    def __init__(self, sos_name, ee, cls_builder, associated_namespaces=[]):
        '''
        Constructor
        '''
        # if 'ns_doe' does not exist in ns_manager, we create this new
        # namespace to store output dictionaries associated to eval_outputs
        if 'ns_doe' not in ee.ns_manager.shared_ns_dict.keys():
            ee.ns_manager.add_ns('ns_doe', ee.study_name)
        super(BuildDoeEval, self).__init__(sos_name, ee, cls_builder,
                                           associated_namespaces=associated_namespaces)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.DOE')
        self.doe_factory = DOEFactory()
        self.design_space = None
        self.samples = None
        self.customed_samples = None
        self.dict_desactivated_elem = {}
        self.selected_outputs = []
        self.selected_inputs = []

        self.previous_algo_name = ""

    def build(self):
        '''
            Overloaded Build method
            Get and build builder from sub_process of doe_eval driver
            Remark: Reached from __configure_io in ee.py: self.factory.build() is going from build to build starting from root
            Remark: It comes before configuring()
            Main method Added to provide proc builder capability

        '''
        AddSubProcToDriver.build(self)

        BuildSoSEval.build(self)

    def configure(self):
        """
            Overloaded configure method
            Configuration of the BuildDoeEval
            Remark: Reached from __configure_io in ee.py: self.root_process.configure_io() is going from confiure to configure starting from root
            Remark: It comes after build()
            Main method Added to provide proc builder capability
        """
        BuildSoSEval.configure(self)

    def set_eval_possible_values(self):
        '''
            Overloaded BuildSoSEval method : used in BuildSoSEval.configure()
            It is done downstream of setup_sos_disciplines()
            In fact: copy past of the BuildSoSEval method --> not needed !
            Once all disciplines have been "run" through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        # the eval process to analyse is stored as the only child of BuildSoSEval
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
        my_ns_doe_eval_path = self.ee.ns_manager.disc_ns_dict[self]['others_ns']['ns_doe_eval'].get_value(
        )
        if eval_input_new_dm is None:
            self.dm.set_data(f'{my_ns_doe_eval_path}.eval_inputs',
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
            self.dm.set_data(f'{my_ns_doe_eval_path}.eval_inputs',
                             'value', default_dataframe, check_value=False)

        if eval_output_new_dm is None:
            self.dm.set_data(f'{my_ns_doe_eval_path}.eval_outputs',
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
            self.dm.set_data(f'{my_ns_doe_eval_path}.eval_outputs',
                             'value', default_dataframe, check_value=False)

    def run(self):
        '''
            Overloaded BuildSoSEval method
            BuildSoSEval has no specific run method
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

        # Added treatment for multiplier
        eval_in_with_multiplied_var = None
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
            {self.SAMPLES_INPUTS_DF: samples_dataframe})
        for dynamic_output in self.eval_out_list:
            self.store_sos_outputs_values({
                f'{dynamic_output.split(self.ee.study_name + ".",1)[1]}_dict':
                    global_dict_output[dynamic_output]})

#################### End: Main methods ################################

##################### Begin: Sub methods ################################
# Remark: those sub methods should be private functions

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

    def get_algo_default_options(self, algo_name):
        """
            This algo generate the default options to set for a given doe algorithm
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sampling_algo()
        """
        if algo_name in self.algo_dict.keys():
            return self.algo_dict[algo_name]
        else:
            return self.default_algo_options

    def create_generic_multipliers_dynamic_input(self):
        """
            Multiplier: this algo create the generic multiplier dynamic input
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sampling_algo()
        """
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
            is_editable = disc._data_in[data_in_key]['editable']
            is_None = disc._data_in[data_in_key]['value'] is None
            if is_in_type and not in_coupling_numerical and not is_structuring and is_editable:
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                if is_input_type:
                    poss_in_values_full.append(
                        full_id.split(self.ee.study_name + ".", 1)[1])

                # Added treatment for multiplier
                is_input_multiplier_type = disc._data_in[data_in_key][self.TYPE] in self.INPUT_MULTIPLIER_TYPE
                if is_input_multiplier_type and not is_None:
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
                    full_id.split(self.ee.study_name + ".", 1)[1])
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

    def update_default_inputs(self, disc):
        '''
        Update default inputs of disc with dm values
            Function needed in run()
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

    def generate_samples_from_doe_factory(self):
        """Generating samples for the Doe using the Doe Factory
            Function needed in run()
        """
        algo_name = self.get_sosdisc_inputs(self.SAMPLING_ALGO)
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
        self.customed_samples = self.get_sosdisc_inputs(
            self.CUSTOM_SAMPLES_DF).copy()
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
        if not set(self.selected_inputs).issubset(set(self.customed_samples.columns.to_list())):
            missing_eval_in_variables = set.union(set(self.selected_inputs), set(
                self.customed_samples.columns.to_list())) - set(self.customed_samples.columns.to_list())
            msg = f'the columns of the custom samples dataframe must include all the the eval_in selected list of variables. Here the following selected eval_in variables {missing_eval_in_variables} are not in the provided sample.'
            # To do: provide also the list of missing eval_in variables:
            self.logger.error(msg)
            raise ValueError(msg)
        else:
            not_relevant_columns = set(
                self.customed_samples.columns.to_list()) - set(self.selected_inputs)
            msg = f'the following columns {not_relevant_columns} of the custom samples dataframe are filtered because they are not in eval_in.'
            self.logger.warning(msg)
            if len(not_relevant_columns) != 0:
                self.customed_samples.drop(
                    not_relevant_columns, axis=1, inplace=True)
            self.selected_inputs.sort()
            self.customed_samples = self.customed_samples[self.selected_inputs]

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
            # convert arrays in sample_dict into SoSTrades types
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

    def create_origin_vars_to_update_dict(self):
        '''
        Multiplier: 
        Function needed in run
        '''
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
        '''
        Multiplier: multiplied var to sample
        Function needed in run
        '''
        for sample_i in range(len(multipliers_samples)):
            x = multipliers_samples[sample_i]
            vars_to_update_dict = {}
            for multiplier_i, x_id in enumerate(self.eval_in_list):
                # for grid search multipliers inputs
                var_name = x_id.split(self.ee.study_name + '.', 1)[-1]
                if self.MULTIPLIER_PARTICULE in var_name:
                    var_origin_f_name = '.'.join(
                        [self.ee.study_name, self.get_names_from_multiplier(var_name)[0]])
                    if var_origin_f_name in vars_to_update_dict:
                        var_to_update = vars_to_update_dict[var_origin_f_name]
                    else:
                        var_to_update = copy.deepcopy(
                            origin_vars_to_update_dict[var_origin_f_name])
                    vars_to_update_dict[var_origin_f_name] = self.apply_muliplier(
                        multiplier_name=var_name, multiplier_value=x[multiplier_i] / 100.0, var_to_update=var_to_update)
            for multiplied_var in vars_to_update_dict:
                self.samples[sample_i].append(
                    vars_to_update_dict[multiplied_var])

    def apply_muliplier(self, multiplier_name, multiplier_value, var_to_update):
        '''
        Multiplier: apply multiplier
        Function needed in add_multiplied_var_to_samples
        '''
        # if dict or dataframe to be multiplied
        if '@' in multiplier_name:
            col_name_clean = multiplier_name.split(self.MULTIPLIER_PARTICULE)[
                0].split('@')[1]
            if col_name_clean == 'allcolumns':
                if isinstance(var_to_update, dict):
                    float_cols_ids_list = [dict_keys for dict_keys in var_to_update if isinstance(
                        var_to_update[dict_keys], float)]
                elif isinstance(var_to_update, pd.DataFrame):
                    float_cols_ids_list = [
                        df_keys for df_keys in var_to_update if var_to_update[df_keys].dtype == 'float']
                for key in float_cols_ids_list:
                    var_to_update[key] = multiplier_value * var_to_update[key]
            else:
                keys_clean = [self.clean_var_name(var)
                              for var in var_to_update.keys()]
                col_index = keys_clean.index(col_name_clean)
                col_name = var_to_update.keys()[col_index]
                var_to_update[col_name] = multiplier_value * \
                    var_to_update[col_name]
        # if float to be multiplied
        else:
            var_to_update = multiplier_value * var_to_update
        return var_to_update

    def set_multipliers_values(self, disc, full_id, var_name):
        '''
        Multiplier: set multipliers values
        Function needed in fill_possible_values
        '''
        poss_in_values_list = []
        # if local var
        if 'namespace' not in disc._data_in[var_name]:
            origin_var_ns = disc._data_in[var_name]['ns_reference'].value
        else:
            origin_var_ns = disc._data_in[var_name]['namespace']

        disc_id = ('.').join(full_id.split('.')[:-1])
        ns_disc_id = ('__').join([origin_var_ns, disc_id])
        if ns_disc_id in disc.ee.ns_manager.all_ns_dict:
            full_id_ns = ('.').join(
                [disc.ee.ns_manager.all_ns_dict[ns_disc_id].value, var_name])
        else:
            full_id_ns = full_id

        if disc._data_in[var_name][self.TYPE] == 'float':
            multiplier_fullname = f'{full_id_ns}{self.MULTIPLIER_PARTICULE}'.split(
                self.ee.study_name + ".")[1]
            poss_in_values_list.append(multiplier_fullname)

        else:
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
                    multiplier_fullname = f'{full_id_ns}@{col_name_clean}{self.MULTIPLIER_PARTICULE}'.split(
                        self.ee.study_name + ".")[1]
                    poss_in_values_list.append(multiplier_fullname)
                # if df with more than one float column, create multiplier for all
                # columns also
                if len(float_cols_list) > 1:
                    multiplier_fullname = f'{full_id_ns}@allcolumns{self.MULTIPLIER_PARTICULE}'.split(
                        self.ee.study_name + ".")[1]
                    poss_in_values_list.append(multiplier_fullname)
        return poss_in_values_list

    def clean_var_name(self, var_name):
        '''
        Multiplier: clean var name
        Function needed in set_multipliers_values
        '''
        return re.sub(r"[^a-zA-Z0-9]", "_", var_name)

    def get_names_from_multiplier(self, var_name):
        '''get names from multiplier
        Function needed in create_origin_vars_to_update_dict
        '''
        column_name = None
        var_origin_name = var_name.split(self.MULTIPLIER_PARTICULE)[
            0].split('@')[0]
        if '@' in var_name:
            column_name = var_name.split(self.MULTIPLIER_PARTICULE)[
                0].split('@')[1]

        return [var_origin_name, column_name]

##################### End: Sub methods ################################

### Begin: Sub methods for proc builder to wrap specific driver dynamic inputs ####
    # come from setup_sos_disciplines_driver_inputs_depend_on_sub_process
    def setup_desc_in_dict_of_driver(self):
        """
            Create desc_in_dict for dynamic inputs of the driver depending on sub process
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sub_process()
            Function to be specified per driver
            Update of SAMPLING_ALGO/EVAL_INPUTS/EVAL_OUTPUTS
        """
        desc_in_dict = {}
        desc_in_dict[self.SAMPLING_ALGO] = {'type': 'string',
                                                    'possible_values': self.custom_order_possible_algorithms(self.doe_factory.algorithms),
                                                    'structuring': True}
        desc_in_dict[self.EVAL_INPUTS] = {'type': 'dataframe',
                                          'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                                   'full_name': ('string', None, False)},
                                                  'dataframe_edition_locked': False,
                                                  'structuring': True,
                                                  'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                                  'namespace': 'ns_doe_eval'}
        desc_in_dict[self.EVAL_OUTPUTS] = {'type': 'dataframe',
                                           'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                                    'full_name': ('string', None, False)},
                                                   'dataframe_edition_locked': False,
                                                   'structuring': True,
                                                   'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                                   'namespace': 'ns_doe_eval'}
        return desc_in_dict

    def setup_sos_disciplines_driver_inputs_independent_on_sub_process(self, dynamic_inputs, dynamic_outputs):
        """
            setup_dynamic inputs when driver parameters depending on the SP selection are already set:
            here SAMPLING_ALGO/EVAL_INPUTS/EVAL_OUTPUTS
            Manage update of EVAL_INPUTS/EVAL_OUTPUTS
            Create or update ALGO_OPTIONS
            Create or update either CUSTOM_SAMPLES_DF or DESIGN_SPACE
            Function needed in setup_sos_disciplines()
            Function to be specified per driver
        """
        # Dealing with eval_inputs and eval_outputs
        # we know that SAMPLING_ALGO/EVAL_INPUTS/EVAL_OUTPUTS keys are set at
        # the same time
        algo_name_has_changed = False
        selected_inputs_has_changed = False
        # we check that SAMPLING_ALGO/EVAL_INPUTS/EVAL_OUTPUTS are available
        # evan if we test only self.Algo
        if self.SAMPLING_ALGO in self._data_in:  # and sub_process_name != None
            algo_name = self.get_sosdisc_inputs(self.SAMPLING_ALGO)
            # 1. Manage update of SAMPLING_ALGO
            if self.previous_algo_name != algo_name:
                algo_name_has_changed = True
                self.previous_algo_name = algo_name
            # 2. Set ALGO_OPTIONS depending on the selected algo_name
            if algo_name is not None:
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
            # 3. Prepare update of selected_inputs and selected_outputs
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
            # 4. Manage empty selection in EVAL_INPUTS/EVAL_OUTPUTS
            if algo_name is not None and len(selected_inputs) == 0:
                # Warning: selected_inputs cannot be empty
                # Problem: it is not None but it is an "empty dataframe" so has
                # the meaning of None (i.e. Mandatory field)
                self.logger.warning('Selected_inputs cannot be empty!')
            if algo_name is not None and len(selected_outputs) == 0:
                self.logger.warning('Selected_outputs cannot be empty!')
            # 5. Manage update of EVAL_INPUTS/EVAL_OUTPUTS
            # we set the lists which will be used by the evaluation
            # function of BuildSoSEval
            self.set_eval_in_out_lists(selected_inputs, selected_outputs)
            # 6. Manage update of multiplier
            # if multipliers in eval_in
            if (len(self.selected_inputs) > 0) and (any([self.MULTIPLIER_PARTICULE in val for val in self.selected_inputs])):
                generic_multipliers_dynamic_inputs_list = self.create_generic_multipliers_dynamic_input()
                for generic_multiplier_dynamic_input in generic_multipliers_dynamic_inputs_list:
                    dynamic_inputs.update(generic_multiplier_dynamic_input)
            # 7. Manage different types of output dict tables depending on selected_outputs
            # doe can be done only for selected outputs
            if len(selected_outputs) > 0:
                # setting dynamic outputs. One output of type dict per selected
                # output
                for out_var in self.eval_out_list:
                    dynamic_outputs.update(
                        {f'{out_var.split(self.ee.study_name + ".")[1]}_dict': {'type': 'dict', 'visibility': 'Shared',
                                                                                'namespace': 'ns_doe'}})
            # 6. Manage different types of algo_names : CUSTOM_SAMPLES_DF or DESIGN_SPACE
            # doe can be done only for selected inputs
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
# End: Sub methods for proc builder  to wrap specific driver dynamic
# inputs ####

### Begin: Sub methods for proc builder to be specified in specific driver ####

    def get_cls_builder(self):
        '''
            Specific function of the driver to get the cls_builder
            Function needed in set_sub_process_status()
            Function to be specified per driver
        '''
        cls_builder = self.cls_builder
        return cls_builder

    def set_cls_builder(self, value):
        '''
            Specific function of the driver to set the cls_builder with value
            Function needed in set_nested_builders
            Function to be specified per driver
        '''
        self.cls_builder = value

    def set_ref_discipline_full_name(self):
        '''
            Specific function of the driver to define the full name of the reference disvcipline
            Function needed in _init_ of the driver
            Function to be specified per driver
        '''
        driver_name = self.name
        self.ref_discipline_full_name = f'{self.ee.study_name}.{driver_name}'

        return

    def clean_driver_before_rebuilt(self):  # come from build_eval_subproc
        '''
            Specific function of the driver to clean all instances before rebuild and reset any needed parameter
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
            Function to be specified per driver
        '''
        self.sos_disciplines[0].clean()
        self.sos_disciplines = []  # Should it be del self.sos_disciplines[0]?
        # We "clean" also all dynamic inputs to be reloaded by
        # the usecase
        self.add_inputs({})  # is it needed ?
        return

    def get_ns_of_driver(self):  # come from build_eval_subproc
        '''
            Specific function of the driver to get ns of driver
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
            Function to be specified per driver
        '''
        ns_of_driver = ['ns_doe', 'ns_doe_eval']
        return ns_of_driver
#### End: Sub methods for proc builder to be specified in specific driver ####
