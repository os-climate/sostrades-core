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
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder


class SoSBuilder(object):
    '''
    Class that stores a class and associated attributes to be built afterwards
    '''

    def __init__(self, disc_name, ee, cls, is_executable=True):
        '''
        Constructor
        :param cls: class that will be instantiated by the builder
        :type cls: class
        '''
        self.__disc_name = disc_name
        self.disc = None
        self.__ee = ee
        self.__args = {'sos_name': self.__disc_name, 'ee': self.__ee, 'cls_builder': cls}
        self.cls = cls
        # A builder can build several disciplines (ex: scatter)
        self.discipline_dict = {}
        # flag to determine if the associated discipline has a run method
        # (True by default)
        self._is_executable = is_executable

    @property
    def sos_name(self):
        return self.__disc_name

    @property
    def args(self):
        return self.__args

    def set_builder_info(self, key_info, value_info):
        ''' Sets the arguments that will be needed to instantiate self.cls
        :param args: list of arguments to instantiate the self.cls class
        :type args: list
        '''
        self.__args[key_info] = value_info

    def set_disc_name(self, new_disc_name):

        self.__disc_name = new_disc_name
        self.__args['sos_name'] = self.__disc_name

    def build(self):
        ''' Instantiates the class self.cls
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
        if self.cls.__name__ in ['ProxyCoupling', 'ProxyDisciplineScatter', 'ProxyDisciplineGather','ProxyDoeEval']:
            self.disc = self.cls(**self.__args)
        else:
            self.disc = ProxyDiscipline(**self.__args)

        # if self.cls.__name__ == 'ProxyDiscipline':
        #     self.disc = ProxyDiscipline(**self.__args)
        # else:
        #     self.disc = self.cls(**self.__args)

        self.disc.father_builder = self
        self.discipline_dict[future_new_ns_disc_name] = self.disc

    def build_sub_discs(self, current_ns, future_new_ns_disc_name):

        # for disc in self.discipline_list:
        disc = self.discipline_dict[future_new_ns_disc_name]
        self.__ee.ns_manager.set_current_disc_ns(
            disc.get_disc_full_name())
        disc.build()
        self.__ee.ns_manager.set_current_disc_ns(current_ns)

    def remove_discipline(self, disc):

        full_name = disc.get_disc_full_name()
        del self.discipline_dict[full_name]
