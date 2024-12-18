'''
Copyright 2024 Capgemini
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
import importlib
import inspect
import os
import pickle
import time

from tqdm import tqdm

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest

GENERATED_TEST_FOLDERNAME = 'generated_jacobian_tests'


def check_each_discpline_jacobians_in_process(usecase_path: str):
    """
    This function executes the usecase and test each discipline gradients at the point obtained after execution.

    When testing a discpline gradients, the coupling outputs are tested wrt the coupling inputs.
    The coupling caracter is based on if the variables are coupling or not in the complete process.
    """
    name = 'jacobianIsolatedDiscTest'
    ee = ExecutionEngine(name)

    process_file_path = '.'.join(usecase_path.split('.')[:-1]) + '.process'
    imported_module = importlib.import_module(usecase_path)
    builder_module = importlib.import_module(process_file_path)
    builder_class_instance = getattr(builder_module, 'ProcessBuilder')(ee=ee)
    builders = builder_class_instance.get_builders()

    ee.factory.set_builders_to_coupling_builder(builders)
    ee.configure()

    usecase = getattr(imported_module, 'Study')(execution_engine=ee)
    usecase.study_name = name
    usecase.init_from_subusecase = True
    # First step : Dump data to a temp folder

    values = usecase.setup_usecase()
    values_dict = {}
    if isinstance(values, list):
        for val in values:
            values_dict.update(val)
    else:
        values_dict = values

    ee.load_study_from_input_dict(values_dict)
    ee.execute()

    ids_discipline_to_test = []
    for id_disc, val in usecase.ee.dm.disciplines_dict.items():
        module_path_discipline = val['model_name_full_path']
        if 'sostrades_core' not in module_path_discipline:
            ids_discipline_to_test.append(id_disc)

    gloal_data_dict = usecase.ee.dm.get_data_dict_values()
    namespaces_dict = {key: ns_obj.value for key, ns_obj in usecase.ee.ns_manager.shared_ns_dict.items()}
    all_coupling_variables = list(
        filter(lambda key: usecase.ee.dm.reduced_dm[key]['coupling'], usecase.ee.dm.reduced_dm.keys()))

    discipline_with_wrong_gradients = []
    for disc_id in tqdm(ids_discipline_to_test):
        discipline_module = usecase.ee.dm.disciplines_dict[disc_id]['model_name_full_path']
        proxy_discipline = usecase.ee.dm.disciplines_dict[disc_id]['reference']
        discipline_class_name = get_discipline_classname_from_module(discipline_module)
        discipline_class_path = f"{discipline_module}.{discipline_class_name}"

        disc_study_path = usecase.ee.dm.disciplines_dict[disc_id]['disc_label']
        print("TESTING :", disc_study_path)
        model_name = disc_study_path.split('.')[-1]
        test_name = disc_study_path.split(f'.{model_name}')[0]

        inputs_disc = list(proxy_discipline.get_sosdisc_inputs(full_name_keys=True).keys())
        outputs_disc = list(proxy_discipline.get_sosdisc_outputs(full_name_keys=True).keys())

        coupling_inputs = list(filter(lambda input_var_disc: input_var_disc in all_coupling_variables, inputs_disc))
        coupling_outputs = list(filter(lambda output_var_disc: output_var_disc in all_coupling_variables, outputs_disc))

        discipline_inputs_dict = {key: gloal_data_dict[key] for key in inputs_disc}

        success_discipline_gradients = one_test_gradients_discipline(test_name=test_name,
                                                                     model_name=model_name,
                                                                     discipline_class_path=discipline_class_path,
                                                                     inputs=discipline_inputs_dict,
                                                                     namespaces=namespaces_dict,
                                                                     coupling_inputs=coupling_inputs,
                                                                     coupling_outputs=coupling_outputs)
        if not success_discipline_gradients:
            discipline_with_wrong_gradients.append(discipline_class_path)
        time.sleep(2.)

    if len(discipline_with_wrong_gradients) > 0:
        print_msg = '\n'.join(discipline_with_wrong_gradients)

        raise ValueError(f'Following disciplines have incorrect gradients : \n{print_msg}')
    else:
        if os.path.exists(GENERATED_TEST_FOLDERNAME):
            os.remove(GENERATED_TEST_FOLDERNAME)
        if os.path.exists("jacobian_pkls"):
            os.remove("jacobian_pkls")
        print("No gradients errors in disciplines of process")


def get_discipline_classname_from_module(module_path):
    """
    Takes a module path as a string and returns a list of all available classes in that module.

    Args:
    module_path (str): The path to the module, e.g., 'package.subpackage.module'

    Returns:
    list: A list of class objects found in the module
    """
    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Get all members of the module
        members = inspect.getmembers(module)

        # Filter for classes defined in this module
        classes = [
            member[1].__name__ for member in members
            if inspect.isclass(member[1]) and member[1].__module__ == module_path
        ]

        return classes[0]

    except ImportError:
        print(f"Error: Could not import module '{module_path}'")
        return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []


def one_test_gradients_discipline(test_name: str,
                                  model_name: str,
                                  discipline_class_path: str,
                                  inputs: dict,
                                  namespaces: dict,
                                  coupling_inputs: list[str],
                                  coupling_outputs: list[str],
                                  ):
    if len(coupling_inputs) == 0 or len(coupling_outputs) == 0:
        return True

    """Tests the gradients of the discipline at exact point of usecase ending point"""
    class MyClass(AbstractJacobianUnittest):
        if not os.path.exists('jacobian_pkls'):
            os.mkdir('jacobian_pkls')

        def analytic_grad_entry(self):
            return []

        def setUp(self) -> None:
            self.ee = ExecutionEngine(test_name)
            self.ee.ns_manager.add_ns_def(namespaces)

            builder = self.ee.factory.get_builder_from_module(model_name, discipline_class_path)
            self.ee.factory.set_builders_to_coupling_builder(builder)
            self.ee.configure()

            self.ee.load_study_from_input_dict(inputs)

        def test(self):
            self.ee.execute()
            result = True

            self.override_dump_jacobian = True
            disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline
            pickle_filename = f'{test_name}.{model_name}'.replace('.', '_') + '.pkl'
            if not os.path.exists(GENERATED_TEST_FOLDERNAME):
                os.mkdir(GENERATED_TEST_FOLDERNAME)
                os.mkdir(os.path.join(GENERATED_TEST_FOLDERNAME, 'jacobian_pkls'))
            try:
                self.check_jacobian(location=os.path.join(os.path.abspath(os.curdir), GENERATED_TEST_FOLDERNAME),
                                    filename=pickle_filename,
                                    discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                                    local_data=disc_techno.local_data,
                                    inputs=coupling_inputs,
                                    outputs=coupling_outputs)
                pkl_to_remove = os.path.join(os.path.abspath(os.curdir), GENERATED_TEST_FOLDERNAME, "jacobian_pkls", pickle_filename)
                os.remove(pkl_to_remove)
                print(f'Gradients OK for {discipline_class_path}')
            except AssertionError as e:
                handle_discipline_with_wrong_gradients(coupling_inputs=coupling_inputs,
                                                       coupling_outputs=coupling_outputs,
                                                       discipline_module_path=discipline_class_path,
                                                       discipline_inputs=inputs,
                                                       model_name=model_name,
                                                       ns_dict=namespaces,
                                                       name=test_name,
                                                       jacobian_pkl_name=pickle_filename)
                print(f'WRONG Gradient for {discipline_class_path}')
                result = False

            return result

    mytest = MyClass()
    mytest.setUp()
    return mytest.test()


def handle_discipline_with_wrong_gradients(coupling_inputs: list[str],
                                           coupling_outputs: list[str],
                                           discipline_module_path: str,
                                           discipline_inputs: dict,
                                           ns_dict: dict,
                                           model_name: str,
                                           name: str,
                                           jacobian_pkl_name: str):
    """Prepares a dedicated test file to help debug the gradients"""

    pickle_to_dump = {
        'ns_dict': ns_dict,
        'values_dict': discipline_inputs,
        'coupling_inputs': coupling_inputs,
        'coupling_outputs': coupling_outputs,
        'mod_path': discipline_module_path,
        'model_name': model_name
    }
    generated_test_filename = discipline_module_path.replace('.', '_') + '.pkl'
    generated_test_data_folder = os.path.join(GENERATED_TEST_FOLDERNAME, 'data')

    if not os.path.exists(generated_test_data_folder):
        os.mkdir(generated_test_data_folder)
    path_generated_test_file_data = os.path.join(generated_test_data_folder, generated_test_filename)

    with open(path_generated_test_file_data, 'wb') as f:
        pickle.dump(pickle_to_dump, f)

    path_pickle_in_test_file = os.path.join("data", generated_test_filename)

    generated_code = f"""
import pickle
from os.path import dirname

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)


class MyGeneratedTest(AbstractJacobianUnittest):

    def analytic_grad_entry(self):
        return []

    def test_execute(self):
        with open('{path_pickle_in_test_file}', 'rb') as f:
            file_dict = pickle.load(f)
            ns_dict = file_dict['ns_dict']
            mod_path = file_dict['mod_path']
            model_name = file_dict['model_name']
            values_dict = file_dict['values_dict']
            coupling_inputs = file_dict['coupling_inputs']
            coupling_ouputs = file_dict['coupling_outputs']

        self.name = '{name}'
        self.ee = ExecutionEngine(self.name)

        self.ee.ns_manager.add_ns_def(ns_dict)

        builder = self.ee.factory.get_builder_from_module(
            model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='{jacobian_pkl_name}',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=coupling_inputs,
                            outputs=coupling_ouputs)
    """
    genretated_test_file_name = discipline_module_path.replace('.', '_') + '.py'
    path_generated_test_file = os.path.join(GENERATED_TEST_FOLDERNAME, genretated_test_file_name)

    with open(path_generated_test_file, 'w') as f:
        f.write(generated_code)

    print(f'Dumped ready to use Jacobian test class at {path_generated_test_file}')
