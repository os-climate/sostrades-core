'''
Copyright 2023 Capgemini

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

from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
import os
import sys
import ast
import pathlib

"""
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
"""


class SelectorDiscipline(ProxyCoupling):
    """
    Generic Uncertainty Quantification class
    """

    # ontology information
    _ontology_data = {
        'label': 'Selector Discipline',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-chart-area',
        'version': '',
    }

    DESC_IN = {
        'repository': {
            ProxyCoupling.TYPE: 'string',
            ProxyCoupling.STRUCTURING: True,
            ProxyCoupling.POSSIBLE_VALUES: ['sostrades_core', 'energy_models', 'climateeconomics']
        },
        'discipline_name': {
            ProxyCoupling.TYPE: 'string',
            ProxyCoupling.DEFAULT: 'Disc'
        },
    }
    DESC_IN.update(ProxyCoupling.DESC_IN)
    new_variables = False

    def setup_sos_disciplines(self):
        """setup sos disciplines"""

        dynamic_inputs = {}
        data_in = self.get_data_in()
        if data_in != {}:
            repository = self.get_sosdisc_inputs('repository')

            if repository is not None:
                disciplines_in_repo = find_disciplines_in_folder(repository)
                self.label_todisc_dict = self.create_label_to_disc_dict(repository, disciplines_in_repo)

                dynamic_inputs['discipline'] = {
                    ProxyCoupling.TYPE: 'string',
                    ProxyCoupling.STRUCTURING: True,
                    ProxyCoupling.POSSIBLE_VALUES: list(self.label_todisc_dict.keys())
                }
                if self.new_variables:
                    self.new_variables = False
                else:
                    self.new_variables = True
        self.add_inputs(dynamic_inputs)
        ProxyCoupling.setup_sos_disciplines(self)

    def create_label_to_disc_dict(self, repository, disciplines_in_repo):
        label_todisc_dict = {}

        for disc in disciplines_in_repo:
            try:
                inst_class = self.ee.factory.get_disc_class_from_module(f'{repository}.{disc}')
                label_todisc_dict[inst_class._ontology_data['label']] = disc
            except:
                label_todisc_dict[disc] = disc
        return label_todisc_dict

    def is_configured(self):
        '''
        Return False if at least one sub discipline needs to be configured, True if not
        '''
        return self.get_configure_status() and not self.check_structuring_variables_changes() and (
                self.get_disciplines_to_configure() == []) and not self.new_variables

    def prepare_build(self):
        '''
        Prepare the builder to be build according to the discipline chosen by the user
        '''

        if 'discipline' in self.get_data_in():
            inputs_dict = self.get_sosdisc_inputs()
            disc_name = inputs_dict['discipline_name']
            discipline_label = inputs_dict['discipline']
            discipline = self.label_todisc_dict[discipline_label]
            repo = inputs_dict['repository']
            disc_path = f'{repo}.{discipline}'
            if discipline is not None:
                integrity_msg = self.disc_in_possible_values()
                if integrity_msg == '':
                    if self.cls_builder == []:
                        self.cls_builder = [self.ee.factory.get_builder_from_module(
                            disc_name, disc_path)]
                        self.create_namespaces_at_root(disc_path)
                else:
                    raise Exception(integrity_msg)
            else:
                self.cls_builder = []
        else:
            self.cls_builder = []
        return self.cls_builder

    def disc_in_possible_values(self):

        var_data_dict = self.get_data_in()['discipline']
        check_integrity_msg = self.check_data_integrity_cls.check_variable_value(
            var_data_dict, True)

        return check_integrity_msg

    def create_namespaces_at_root(self, disc_path):
        namespace_list = []
        cls = self.ee.factory.get_disc_class_from_module(disc_path)
        for inp in cls.DESC_IN.values():
            if 'namespace' in inp:
                namespace_list.append(inp['namespace'])
        for out in cls.DESC_OUT.values():
            if 'namespace' in out:
                namespace_list.append(out['namespace'])

        self.ee.ns_manager.add_ns_def({namespace: self.ee.study_name for namespace in set(namespace_list)})


def find_disciplines_in_folder(folder_name):
    module_paths = []
    base_class = SoSWrapp

    for path in sys.path:
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            path = pathlib.Path(folder_path)
            for file_path in path.glob('**/*.py'):
                classes = find_classes_in_file(file_path, base_class)
                if len(classes) == 1:
                    string_path = os.path.splitext(os.path.relpath(file_path, path))[0].replace(os.path.sep, '.')
                    module_paths.append(f'{string_path}.{classes[0]}')
            break
    return module_paths


def find_classes_in_file(file_path, base_class):
    with open(file_path, 'r') as file:
        source_code = file.read()

    module_ast = ast.parse(source_code)
    class_nodes = [node for node in module_ast.body if isinstance(node, ast.ClassDef)]
    subclasses = []

    for class_node in class_nodes:
        class_name = class_node.name
        bases = [base.id for base in class_node.bases if isinstance(base, ast.Name)]
        if base_class.__name__ in bases:
            subclasses.append(class_name)

    return subclasses
