'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/05/16 Copyright 2023 Capgemini

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
from __future__ import annotations

from typing import Any

from sostrades_core.execution_engine.ns_manager import NamespaceManager
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder


class SoSBuilderException(Exception):
    pass

class SoSBuilder:
    '''
    Class that stores a class and associated attributes to be built afterwards
    '''

    NS_NAME_SEPARATOR = NamespaceManager.NS_NAME_SEPARATOR
    SPECIFIC_PROXYS = [
        'ProxyCoupling',
        'ProxyDisciplineGather',
        'ProxyOptim',
        'ArchiBuilder',
        'ProxyMonoInstanceDriver',
        'ProxyMultiInstanceDriver',
        'SelectorDiscipline',
        'ProxySampleGenerator',
    ]

    def __init__(self, disc_name, ee, cls, is_executable=True):
        '''
        Constructor
        :param cls: class that will be instantiated by the builder
        :type cls: class
        '''
        self.__disc_name = disc_name
        self.disc = None
        self.__ee = ee
        self.logger = ee.logger.getChild(self.__class__.__name__)
        self.__args = {
            'sos_name': self.__disc_name,
            'ee': self.__ee,
            'cls_builder': cls,
        }
        self.cls = cls
        # A builder can build several disciplines (ex: scatter)
        self.discipline_dict = {}
        # flag to determine if the associated discipline has a run method
        # (True by default)
        self._is_executable = is_executable
        self.__associated_namespaces_dict = {}

    @property
    def sos_name(self):
        return self.__disc_name

    @property
    def ee(self):
        return self.__ee

    @property
    def args(self):
        return self.__args

    @property
    def associated_namespaces(self):
        return list(self.__associated_namespaces_dict.values())

    def set_builder_info(self, key_info: str, value_info: Any):
        '''Sets the arguments that will be needed to instantiate self.cls
        :param args: list of arguments to instantiate the self.cls class
        :type args: list
        '''
        self.__args[key_info] = value_info

    def associate_namespaces(self, ns_list):
        '''
        Associate namespaces to a builder, rule to instantiate the disciplines
        '''
        if isinstance(ns_list, str):
            self.add_namespace_list_in_associated_namespaces([ns_list])
        elif isinstance(ns_list, list):
            self.add_namespace_list_in_associated_namespaces(ns_list)
        else:
            raise SoSBuilderException('Should specify a list of strings or a string to associate namespaces')
        # self.__args['associated_namespaces'] = self.__associated_namespaces

    def set_disc_name(self, new_disc_name):
        self.__disc_name = new_disc_name
        self.__args['sos_name'] = self.__disc_name

    def build(self):
        '''
        Instantiates the class self.cls
        '''
        current_ns = self.__ee.ns_manager.current_disc_ns

        # If we are in the builder of the high level coupling the current ns is None and
        # we have to check if the coupling has already been created
        # The future disc_name will be created without ns then
        if current_ns is None:
            future_new_ns_disc_name = self.sos_name
        else:
            future_new_ns_disc_name = f'{current_ns}.{self.sos_name}'

        if self.disc is None or future_new_ns_disc_name not in self.discipline_dict:
            self.create_disc(future_new_ns_disc_name)
        else:
            self.disc = self.discipline_dict[future_new_ns_disc_name]

        if issubclass(self.cls, ProxyDisciplineBuilder):
            self.build_sub_discs(current_ns, future_new_ns_disc_name)

        return self.disc

    def create_disc(self, future_new_ns_disc_name):
        if self.cls.__name__ in self.SPECIFIC_PROXYS:
            self.disc = self.cls(**self.__args)
        else:
            self.disc = ProxyDiscipline(**self.__args)

        self.disc.father_builder = self

        self.discipline_dict[future_new_ns_disc_name] = self.disc
        self.configure_associated_namespaces()

    def configure_associated_namespaces(self):
        for ns_id in self.associated_namespaces:
            self.__ee.ns_manager.add_disc_in_dependency_list_of_namespace(ns_id, self.disc.disc_id)
        self.__ee.ns_manager.associate_display_values_to_new_local_namespaces(self)

    def build_sub_discs(self, current_ns, future_new_ns_disc_name):
        # for disc in self.discipline_list:
        disc = self.discipline_dict[future_new_ns_disc_name]
        self.__ee.ns_manager.set_current_disc_ns(disc.get_disc_full_name())
        disc.build()
        self.__ee.ns_manager.set_current_disc_ns(current_ns)

    def remove_discipline(self, disc):
        full_name = disc.get_disc_full_name()
        del self.discipline_dict[full_name]

    def add_namespace_list_in_associated_namespaces(self, ns_list):
        '''
        Add a namespace in associated namespaces list but check if one already exists with the same name
        If yes then the new one has the priority :
        we do this with a dict for performances the update gives the priority to the new one
        '''
        new_ns_dict = {ns.split(self.NS_NAME_SEPARATOR)[0]: ns for ns in ns_list}
        self.__associated_namespaces_dict.update(new_ns_dict)

        self.__args['associated_namespaces'] = self.associated_namespaces

    def delete_all_associated_namespaces(self):
        '''
        Add a namespace in associated namespaces list but check if one already exists with the same name
        If yes then the new one has the priority :
        we do this with a dict for performances the update gives the priority to the new one
        '''
        self.__associated_namespaces_dict = {}

        self.__args['associated_namespaces'] = self.associated_namespaces

    def update_associated_namespaces_with_extra_name(self, extra_name, after_name=None):
        '''

        Args:
            extra_name: extra_name to add after the after name for each associated namespaces to the builder
            after_name: name after which you add the extra_name for each associated namespaces to the builder

        Create the new updated namespaces, then delete the old one and associate the new ones

        '''
        namespace_object_list = [self.__ee.ns_manager.all_ns_dict[ns_id] for ns_id in self.associated_namespaces]
        new_associated_namespaces = self.__ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_name, after_name=after_name, namespace_list=namespace_object_list, clean_existing=False
        )
        self.delete_all_associated_namespaces()
        self.associate_namespaces(new_associated_namespaces)
        # remove the now unused initial namespace
        for ns in namespace_object_list:
            self.__ee.ns_manager.clean_namespace_from_process(ns)

    def update_associated_namespaces_with_extra_name_rec(self, extra_name, after_name=None):
        """
        Recursively update_associated_namespaces_with_extra_name for builder and sub-builders.
        Args:
            extra_name: extra_name to add after the after name for each associated namespaces to the builder
            after_name: name after which you add the extra_name for each associated namespaces to the builder
        """
        # NB: no check so that a ns associated to two builders won't be updated twice.
        self.update_associated_namespaces_with_extra_name(extra_name, after_name)
        sub_builders = self.args["cls_builder"]
        try:
            sub_builders_iterator = iter(sub_builders)
        except TypeError:
            pass
        else:
            for builder in sub_builders_iterator:
                if isinstance(builder, SoSBuilder):
                    builder.update_associated_namespaces_with_extra_name_rec(extra_name, after_name)
