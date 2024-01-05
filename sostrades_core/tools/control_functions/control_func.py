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
Condition function to choose between GaussSeidel or Newton Raphson sub mda class if GSorNewtonMDA is chosen
if condition is True then we use GaussSeidel
else we use NewtonRaphson
'''


def true_func(self):
    return True


def lagrangian_objective_func(self):
    ns_objective = self._disciplines[0].dm.get_all_namespaces_from_var_name(
        'objective_lagrangian')[0]

    condition = self._disciplines[0].dm.get_value(ns_objective)[0] > 1.0e10
    print('condition on objective:', condition, 'objective > 1.0e10 :',
          self.disciplines[0].dm.get_value(ns_objective)[0])
    return condition


def max_ite_func(self):
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
    condition = max_ite_func(self)

    condition2 = lagrangian_objective_func(self)

    return condition or condition2
