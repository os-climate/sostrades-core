'''
Copyright 2022 Airbus SAS
Modifications on {} Copyright 2024 Capgemini
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
Condition function to choose between GaussSeidel or Newton Raphson sub mda class if GSorNewtonMDA is chosen
if condition is True then we use GaussSeidel
else we use NewtonRaphson
'''


def true_func(self):
    """
    Always returns True.

    Args:
        self: An instance of the class.

    Returns:
        bool: Always True.

    """
    return True


def lagrangian_objective_func(self):
    """
    Checks if the Lagrangian objective exceeds a predefined threshold.

    Args:
        self: An instance containing discipline data and optimization parameters.

    Returns:
        bool: True if the Lagrangian objective is greater than 1.0e10, otherwise False.

    """
    ns_objective = self._disciplines[0].dm.get_all_namespaces_from_var_name(
        'objective_lagrangian')[0]

    condition = self._disciplines[0].dm.get_value(ns_objective)[0] > 1.0e10
    print('condition on objective:', condition, 'objective > 1.0e10 :',
          self.disciplines[0].dm.get_value(ns_objective)[0])
    return condition


def max_ite_func(self):
    """
    Determines whether the maximum number of iterations has been reached.

    Args:
        self: An instance containing discipline data and optimization parameters.

    Returns:
        bool: True if the maximum iteration condition is satisfied, otherwise False.

    """
    try:
        ns_objective_list = self._disciplines[0].dm.get_all_namespaces_from_var_name(
            'optim_output_df')
        scenario_name = self._disciplines[0].get_disc_full_name(
        ).replace('.' + self._disciplines[0].sos_name, '')

        ns_objective = [
            ns_obj for ns_obj in ns_objective_list if ns_obj.startswith(scenario_name)][0]
        condition = self._disciplines[0].dm.get_value(
            ns_objective)['iteration'].values.max() < 10

        print('condition on iterations :', condition, 'objective <10  :',
              self._disciplines[0].dm.get_value(
                  ns_objective)['iteration'].values.max())
    except:
        print('Using MDAGaussSeidel method, max ite function')
        # return condition so that it uses GaussSeidel
        condition = True

    return condition


def lagrangian_objective_and_max_ite_func(self):
    """
    Evaluates the stopping condition based on the maximum iteration function and Lagrangian objective function.

    Args:
        self: An instance containing the necessary attributes for evaluation.

    Returns:
        bool: True if either the maximum iteration condition or the Lagrangian objective condition is met,
        otherwise False.

    """
    condition = max_ite_func(self)

    condition2 = lagrangian_objective_func(self)

    return condition or condition2
