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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from copy import deepcopy
import pandas as pd

from gemseo.core.doe_scenario import DOEScenario
from gemseo.formulations.formulations_factory import MDOFormulationsFactory

from sos_trades_core.execution_engine.sos_scenario import SoSScenario
from sos_trades_core.api import get_sos_logger


class SoSDOEScenario(SoSScenario, DOEScenario):
    """
    Generic implementation of DOE Scenario
    """
    # Default values of algorithms
    default_algo_options = {
        'n_samples': None,
        'alpha':  'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'n_processes': 1,
        'seed': 1,
        'wait_time_between_samples': 0.0,
        'center_bb': None,
        'center_cc': None,
        'criterion': None,
        'levels': None
    }

    default_algo_options_lhs = {
        'n_samples': None,
        'alpha':  'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'n_processes': 1,
        'seed': 1,
        'wait_time_between_samples': 0.0,
        'center_bb': None,
        'center_cc': None,
        'criterion': None,
        'levels': None
    }

    default_algo_options_fullfact = {
        'n_samples': None,
        'alpha':  'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'n_processes': 1,
        'seed': 1,
        'wait_time_between_samples': 0.0,
        'center_bb': None,
        'center_cc': None,
        'criterion': None,
        'levels': None
    }

    d = {'col1': [1, 2], 'col2': [3, 4]}
    X_pd = pd.DataFrame(data=d)

    default_algo_options_CustomDOE = {
        'eval_jac': False,
        'max_time': 0,
        'n_processes': 1,
        'wait_time_between_samples': 0.0,
        'samples': X_pd,
        'doe_file': None,
        'comments': '#',
        'delimiter': ',',
        'skiprows': 0
    }

    default_algo_options_CustomDOE_file = {
        'eval_jac': False,
        'max_time': 0,
        'n_processes': 1,
        'wait_time_between_samples': 0.0,
        'samples': None,
        'doe_file': 'X_pd.csv',
        'comments': '#',
        'delimiter': ',',
        'skiprows': 0
    }

    algo_dict = {"lhs": default_algo_options_lhs,
                 "fullfact": default_algo_options_fullfact,
                 "CustomDOE": default_algo_options_CustomDOE_file,
                 }

    DOE_DS_IO = 'doe_ds_io'
    OPTIM_RESULT = 'optim_result'

    # DESC_I/O
    DESC_IN = {'n_samples': {'type': 'float'},
               }
    DESC_IN.update(SoSScenario.DESC_IN)

    DESC_OUT = {'doe_ds_io': {'type': 'dataframe'},
                'optim_result': {'type': 'dict'},
                }
    DESC_OUT.update(SoSScenario.DESC_OUT)

    def __init__(self, sos_name, ee, cls_builder):
        """
        Constructor
        """
        SoSScenario.__init__(self, sos_name, ee, cls_builder)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.SoSDOEScenario')
        self.ALGO_MANDATORY_FIELDS = [self.ALGO, self.N_SAMPLES]

    def setup_sos_disciplines(self):

        SoSScenario.setup_sos_disciplines(self)

        if 'eval_mode' in self._data_in:
            eval_mode = self.get_sosdisc_inputs('eval_mode')
            if eval_mode:
                self._data_in[self.N_SAMPLES][self.EDITABLE] = False
                self._data_out[self.DOE_DS_IO][self.EDITABLE] = False
                self._data_out[self.OPTIM_RESULT][self.EDITABLE] = False

                self._data_in[self.N_SAMPLES][self.OPTIONAL] = True
                self._data_out[self.DOE_DS_IO][self.OPTIONAL] = True
                self._data_out[self.OPTIM_RESULT][self.OPTIONAL] = True

    def set_scenario(self):

        # pre-set scenario
        design_space, formulation, obj_full_name = self.pre_set_scenario()

        if None not in [design_space, formulation, obj_full_name]:
            # DOEScenario creation (GEMS object)
            DOEScenario.__init__(self, self.sos_disciplines, formulation,
                                 obj_full_name, design_space, name=self.sos_name,
                                 grammar_type=SoSScenario.SOS_GRAMMAR_TYPE)

            self.activated_variables = self.formulation.design_space.variables_names
            self.set_diff_method()

    def run(self):
        # update default inputs of the couplings
        # TODO: to delete when MDA initialization is improved
        for disc in self.sos_disciplines:
            if disc.is_sos_coupling:
                self._set_default_inputs_from_dm(disc)

        # set design space values to complex
        dspace = deepcopy(self.opt_problem.design_space)
        diff_method = self.get_sosdisc_inputs('differentiation_method')
        if diff_method == self.COMPLEX_STEP:
            curr_x = dspace._current_x
            for var in curr_x:
                curr_x[var] = curr_x[var].astype('complex128')
        self.opt_problem.design_space = dspace
        eval_mode = self.get_sosdisc_inputs('eval_mode')
        eval_jac = self.get_sosdisc_inputs('eval_jac')
        execute_at_opt = self.get_sosdisc_inputs('execute_at_xopt')
        design_space = self.get_sosdisc_inputs('design_space')
        if eval_mode:
            self.opt_problem.evaluate_functions(
                eval_jac=eval_jac, normalize=False)
            # if eval mode design space was not modified
            self.store_sos_outputs_values(
                {'design_space_out': design_space})
            d = {'design_parameters': ['Eval mode as not io table of outputs'], 'functions': [
                'Eval mode as not io table of outputs']}
            XY_pd = pd.DataFrame(data=d)
            optim_res_dict = {}
            optim_res_dict['x_opt'] = 'eval mode'
            optim_res_dict['f_opt'] = 'eval mode'
        else:
            DOEScenario._run(self)
            self.update_design_space_out()

            opt_P = self.opt_problem  # added lines TBC_DM to ease output
            dataset = opt_P.export_to_dataset("dataset_name")
            XY_pd = dataset.export_to_dataframe()
            res_dict = self.get_optimum().__dict__
            optim_res_dict = {}
            optim_res_dict['x_opt'] = res_dict['x_opt']
            optim_res_dict['f_opt'] = res_dict['f_opt']

        dict_values = {'doe_ds_io': XY_pd,
                       'optim_result': optim_res_dict}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def _run_algorithm(self):

        problem = self.formulation.opt_problem
        # Clears the database when multiple runs are performed (bi level)
        # if self.clear_history_before_run: #TBC_DM
        #    problem.database.clear()
        algo_name = self.get_sosdisc_inputs(self.ALGO)
        n_samples = self.get_sosdisc_inputs(self.N_SAMPLES)
        options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)

        if options is None:
            options = {}
        if self.N_SAMPLES in options:
            self.logger.warning("Double definition of algorithm option " +
                                "max_iter, keeping value: " + str(n_samples))
            options.pop(self.N_SAMPLES)
        lib = self._algo_factory.create(algo_name)
        self.logger.info(options)

        print("my_options")
        print(options)

        self.optimization_result = lib.execute(problem, algo_name=algo_name,
                                               n_samples=n_samples,
                                               **options)

        return self.optimization_result

    def set_eval_possible_values(self):

        analyzed_disc = self.sos_disciplines
        possible_out_values = self.fill_possible_values(
            analyzed_disc)  # possible_in_values

        possible_out_values = self.find_possible_values(
            analyzed_disc, possible_out_values)  # possible_in_values

        # Take only unique values in the list
        possible_out_values = list(set(possible_out_values))

        # Fill the possible_values of obj and constraints
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.OBJECTIVE_NAME}',
                         self.POSSIBLE_VALUES, possible_out_values)
#         self.dm.set_data(f'{self.get_disc_full_name()}.{self.INEQ_CONSTRAINTS}',
#                          self.POSSIBLE_VALUES, possible_out_values)
#        self.dm.set_data(f'{self.get_disc_full_name()}.{self.EQ_CONSTRAINTS}',
#                         self.POSSIBLE_VALUES, possible_out_values)
        # fill the possible values of algos
        self._init_algo_factory()
        avail_algos = self._algo_factory.algorithms
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.ALGO}',
                         self.POSSIBLE_VALUES, avail_algos)
        # fill the possible values of formulations
        self._form_factory = MDOFormulationsFactory()
        avail_formulations = self._form_factory.formulations
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.FORMULATION}',
                         self.POSSIBLE_VALUES, avail_formulations)
