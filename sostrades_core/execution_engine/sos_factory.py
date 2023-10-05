'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/07-2023/11/02 Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.proxy_optim import ProxyOptim

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import inspect
import os
from importlib import import_module

from pandas.core.common import flatten

from sostrades_core.execution_engine.sos_builder import SoSBuilder
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.sos_processes.processes_factory import BUILDERS_MODULE_NAME
from sostrades_core.sos_wrapping.selector_discipline import SelectorDiscipline


class SosFactoryException(Exception):
    pass


class SosFactory:
    """
    Specification: SosFactory allows to manage builders and disciplines to instantiate a process
    """

    EE_PATH = 'sostrades_core.execution_engine'
    GENERIC_MODS_PATH = 'sostrades_core.sos_wrapping'
    BUILDERS_FUNCTION_NAME = 'get_builders'
    SETUP_FUNCTION_NAME = 'setup_process'
    PROCESS_BUILDER = 'ProcessBuilder'

    @staticmethod
    def build_module_name(repository, process_identifier):
        """Return the built process module name using given arguments

        :params: repository, repository name
        :type: str
        :params: process_identifier, process identifier
        :type: str

        :return: str
        """
        return f'{repository}.{process_identifier}.{BUILDERS_MODULE_NAME}'

    def __init__(self, execution_engine, sos_name):
        """Constructor

        :params: execution_engine (current execution engine instance)
        :type: ExecutionEngine

        :params: sos_name (discipline name)
        :type: string
        """

        self.__sos_name = sos_name
        self.__execution_engine = execution_engine
        self.__ns_manager = execution_engine.ns_manager
        self.__tool_factory = execution_engine.tool_factory
        self.__proxy_disciplines = []

        self.__created_namespace = None
        self.__root = None
        self.__sos_builders = None
        self.__optimization_scenarii = None
        self.__current_discipline = None
        self.__repository = None
        self.__process_identifier = None

        self.coupling_disc = None
        self.is_sos_coupling = True
        self.__logger = self.__execution_engine.logger.getChild(self.__class__.__name__)

        self.__reset()

    def __reset(self):
        """Reinitialize members variables"""
        self.__proxy_disciplines = []
        self.__optimization_scenarii = []
        self.__created_namespace = []
        self.__sos_builders = []
        self.__repository = None
        self.__process_identifier = None

        self.__current_discipline = None

    def init_execution(self):  # type: (...) -> None
        """
        init_execution delegated to the wrapper using the proxy for i/o configuration.
        """
        for proxy in self.__proxy_disciplines:
            factory = proxy.mdo_discipline_wrapp
            if factory.wrapper is not None:
                factory.wrapper.init_execution()

    @property
    def sos_name(self):
        return self.__sos_name

    @property
    def root(self):
        return self.__root

    @property
    def tool_factory(self):
        return self.__tool_factory

    def set_builders_to_coupling_builder(self, builders):
        """add builders to builder list

        :params: builders, list of builders to add
        :type: list
        """

        self.coupling_builder = self.create_builder_coupling(self.__sos_name)
        if isinstance(builders, list):
            self.coupling_builder.set_builder_info(
                'cls_builder', list(flatten(builders))
            )
        else:
            if builders.cls == SelectorDiscipline:
                self.coupling_builder = builders

            self.coupling_builder.set_builder_info('cls_builder', [builders])

    def add_discipline(self, discipline):
        """
        Add a discipline to the list of factory disciplines AND to the sos_discipline of the current sos_coupling
        """
        # Useful to debug but not right , in theory you can add a discipline everywhere you want
        #         if self.__current_discipline.get_disc_full_name() not in discipline.get_disc_full_name():
        #             raise Exception(
        # f'The discipline {discipline.get_disc_full_name()} is not added at the
        # right place : {self.__current_discipline.get_disc_full_name()}')

        self.__current_discipline.add_discipline(discipline)
        self.__proxy_disciplines.append(discipline)

    def add_discipline_list(self, disciplines):
        self.__current_discipline.add_discipline_list(disciplines)
        self.__proxy_disciplines.extend(disciplines)

    def remove_discipline(self, disc):
        """remove one discipline from coupling
        :param disc: sos discipline to remove
        :type: SoSDiscipline Object
        """
        disc_id = disc.get_disc_id_from_namespace()
        disc.clean_dm_from_disc(disc)
        self.proxy_disciplines.remove(disc)
        self.__ns_manager.remove_dependencies_after_disc_deletion(
            disc, disc_id)

    @property
    def current_discipline(self):
        return self.__current_discipline

    @current_discipline.setter
    def current_discipline(self, disc):
        """set current discipline on which subdiscipline will be attached to
        :param disc: sos discipline to remove
        :type: SoSDiscipline Object
        """
        self.__current_discipline = disc
        self.__execution_engine.ns_manager.set_current_disc_ns(
            disc.get_disc_full_name()
        )

    @property
    def proxy_disciplines(self):
        """Return all sostrades disciplines manage by the factory

        :returns: list of sostrades disciplines
        :type: SoSDisciplines[]
        """

        return self.__proxy_disciplines

    @property
    def contains_mdo(self) -> bool:
        mdo_disciplines = list(filter(lambda disc: isinstance(disc, ProxyOptim), self.proxy_disciplines))
        return len(mdo_disciplines) > 0

    @property
    def contains_mda_with_strong_couplings(self) -> bool:
        mda_disciplines_with_strong_couplings = len(list(
            filter(lambda disc: isinstance(disc, ProxyCoupling) and len(disc.strong_couplings) > 0,
                   self.proxy_disciplines))) >0

        ee_with_strong_couplings = len(self.__execution_engine.root_process.strong_couplings)

        # TODO: second bool shouldnt be necessary  if the root_process is in self.proxy_disciplines
        return mda_disciplines_with_strong_couplings or ee_with_strong_couplings

    @property
    def repository(self):
        """Return the repository used to create the process"""
        return self.__repository

    @repository.setter
    def repository(self, value):
        """Set the repository used to create the process"""
        self.__repository = value

    @property
    def process_identifier(self):
        """Return the process identifier used to create the
        process inside the defined repository
        Return None if no process has been loaded
        """
        return self.__process_identifier

    @process_identifier.setter
    def process_identifier(self, value):
        """Set the process identifier used to create the
        process inside the defined repository
        Return None if no process has been loaded
        """
        self.__process_identifier = value

    @property
    def process_module(self):
        """Return the full module name of the loaded process
        Return None if no process has been loaded
        """

        if self.repository is None or self.process_identifier is None:
            return None
        else:
            return f'{self.repository}.{self.process_identifier}.{BUILDERS_MODULE_NAME}'

    def set_root_process(self):
        self.__root = self.coupling_disc
        self.__execution_engine.set_root_process(self.__root)


    def build(self):
        """Method that build the root process"""
        self.__execution_engine.ns_manager.reset_current_disc_ns()
        self.coupling_disc = self.coupling_builder.build()
        self.set_root_process()

    # get buildersf

    def get_builder_from_process(self, repo, mod_id, **args):
        """
        Return the list of builders of the process in the repo with the specific base_id
        if additional args are given we use them to setup the process before get builders function
        """

        pb_cls = getattr(
            import_module(SosFactory.build_module_name(repo, mod_id)),
            self.PROCESS_BUILDER,
        )
        pb_ist = pb_cls(self.__execution_engine)

        if len(args) != 0:
            process_setup = getattr(pb_ist, self.SETUP_FUNCTION_NAME)
            process_setup(**args)

        process_func = getattr(pb_ist, self.BUILDERS_FUNCTION_NAME)
        proc_list = process_func()

        return proc_list

    def get_pb_ist_from_process(self, repo, mod_id):

        pb_cls = getattr(
            import_module(SosFactory.build_module_name(repo, mod_id)),
            self.PROCESS_BUILDER,
        )
        pb_ist = pb_cls(self.__execution_engine)
        return pb_ist

    def get_builder_from_module(self, sos_name, mod_path):
        """
        Get a builder which is defined by the class in the mod_path
        """
        cls = self.get_disc_class_from_module(mod_path)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)
        return builder

    def add_gather_builder(self, sos_name):
        '''
        Add Gather Discipline builder
        '''
        module_struct_list = f'{self.EE_PATH}.gather_discipline.GatherDiscipline'
        builder = self.get_builder_from_module(sos_name, module_struct_list)

        return builder

    def add_uq_builder(self, sos_name):
        '''
        Add Uncertainty Quantification builder
        '''
        mod_path = 'sostrades_core.sos_wrapping.analysis_discs.uncertainty_quantification.UncertaintyQuantification'
        builder = self.get_builder_from_module(
            sos_name, mod_path)

        return builder

    # TODO: these implementations are WIP to test class split (adding of mod)
    def create_mono_instance_driver(self, sos_name, cls_builder, with_sample_generator=False, display_options=None):
        module_struct_list = f'{self.EE_PATH}.proxy_mono_instance_driver.ProxyMonoInstanceDriver'
        driver_wrapper_mod = f'{self.EE_PATH}.disciplines_wrappers.mono_instance_driver_wrapper.MonoInstanceDriverWrapper'
        return self.create_driver(sos_name=sos_name,
                                   cls_builder=cls_builder,
                                   map_name=None,
                                   with_sample_generator=with_sample_generator,
                                   flatten_subprocess=False,
                                   display_options=display_options,
                                   module_struct_list=module_struct_list,
                                  driver_wrapper_mod=driver_wrapper_mod)

    def create_multi_instance_driver(self, sos_name, cls_builder, with_sample_generator=False, display_options=None,
                                     map_name=None):
        module_struct_list = f'{self.EE_PATH}.proxy_multi_instance_driver.ProxyMultiInstanceDriver'
        driver_wrapper_mod = f'{self.EE_PATH}.disciplines_wrappers.multi_instance_driver_wrapper.MultiInstanceDriverWrapper'
        return self.create_driver(sos_name=sos_name,
                                   cls_builder=cls_builder,
                                   map_name=map_name,
                                   with_sample_generator=with_sample_generator,
                                   flatten_subprocess=True,
                                   display_options=display_options,
                                   module_struct_list=module_struct_list,
                                  driver_wrapper_mod=driver_wrapper_mod)

    def create_driver(self, sos_name, cls_builder, map_name=None, with_sample_generator=False, flatten_subprocess=False,
                      display_options=None, module_struct_list=None, driver_wrapper_mod=None):
        '''

        Args:
            sos_name: Name of the driver
            cls_builder: sub process builder list to evaluate
            map_name (optional): Map associated to scatter_tool (in multiinstance mode)
            with_sample_generator (optional): Add a sample generator and associate it to the driver evaluator
            flatten_subprocess (optional): Create all subprocess disciplines at high level coupling node
                to allow coupling between all disciplines (useful in multiinstance mode when scenarios
                are not independant btw each other)
            display_options (optional): Dictionary of display_options for multiinstance mode (value True or False) with options :
                'autogather' : will create an automatic gather discipline which will gather
                            all cls_builder outputs at driver node
                'hide_under_coupling' : Hide all disciplines created under the coupling at scenario name node for display purpose
                'hide_coupling_in_driver': Hide the coupling (scenario_name node) under the driver for display purpose
                'group_scenarios_under_disciplines' : Invert the order of scenario and disciplines for display purpose
                                                      Scenarios will be under discipline for the display treeview

        Returns: A driver evaluator with all the parameters

        '''
        if module_struct_list is None:
            module_struct_list = f'{self.EE_PATH}.proxy_driver_evaluator.ProxyDriverEvaluator'
        cls = self.get_disc_class_from_module(module_struct_list)
        if driver_wrapper_mod is None:
            driver_wrapper_mod = f'{self.EE_PATH}.disciplines_wrappers.driver_evaluator_wrapper.DriverEvaluatorWrapper'
        driver_wrapper_cls = self.get_disc_class_from_module(
            driver_wrapper_mod)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)

        if cls_builder is not None:
            if isinstance(cls_builder, list):
                builder.set_builder_info(
                    'cls_builder', list(flatten(cls_builder)))
            else:
                builder.set_builder_info('cls_builder', [cls_builder])

        builder.set_builder_info('map_name', map_name)
        builder.set_builder_info('flatten_subprocess', flatten_subprocess)
        builder.set_builder_info('display_options', display_options)
        builder.set_builder_info('driver_wrapper_cls', driver_wrapper_cls)

        builder_list = [builder]
        if with_sample_generator:
            sampling_builder = self.get_builder_from_module('SampleGenerator',
                                                            'sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper.SampleGeneratorWrapper')
            self.__execution_engine.ns_manager.add_ns('ns_sampling',
                                                      self.__execution_engine.ns_manager.get_shared_ns_dict()[
                                                          'ns_eval'].value)
            builder_list.insert(0, sampling_builder)

        return builder_list

    def create_custom_driver_builder(self, sos_name, cls_builder, driver_wrapper_mod):
        # TODO: recode when driver classes are properly merged, at the moment
        # custom driver wrapper is off (won't build)
        module_struct_list = f'{self.EE_PATH}.proxy_driver_evaluator.ProxyDriverEvaluator'
        cls = self.get_disc_class_from_module(module_struct_list)
        driver_wrapper_cls = self.get_disc_class_from_module(
            driver_wrapper_mod)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)
        if isinstance(cls_builder, list):
            builder.set_builder_info('cls_builder', list(flatten(cls_builder)))
        else:
            builder.set_builder_info('cls_builder', [cls_builder])
        builder.set_builder_info('driver_wrapper_cls', driver_wrapper_cls)
        return builder

    def create_architecture_builder(
            self, builder_name, architecture_df, custom_vb_folder_list=None
    ):
        """
        create a builder  defined by a type ArchiBuilder
        """
        mod_path = f'{self.EE_PATH}.archi_builder.ArchiBuilder'
        cls = self.get_disc_class_from_module(mod_path)
        # is_executable flag is False because the archi discipline has no
        # run method
        builder = SoSBuilder(
            builder_name, self.__execution_engine, cls, is_executable=False
        )
        builder.set_builder_info('architecture_df', architecture_df)
        # add custom value block folder if specified
        if custom_vb_folder_list is not None:
            builder.set_builder_info(
                'custom_vb_folder_list', custom_vb_folder_list)

        return builder

    def create_scatter_tool_builder(self, tool_name, map_name, display_options=None):
        """
        create a scatter tool builder with the tool factory
        """
        scatter_tool = self.tool_factory.create_tool_builder(tool_name, 'ScatterTool', map_name=map_name,
                                                             display_options=display_options)

        return scatter_tool

    def create_scatter_data_builder(self, sos_name, map_name):
        """
        create a builder defined by a scatter data type SoSScatterData
        """
        module_struct_list = f'{self.EE_PATH}.scatter_data.SoSScatterData'
        cls = self.get_disc_class_from_module(module_struct_list)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)
        builder.set_builder_info('map_name', map_name)

        return builder

    def create_gather_data_builder(self, sos_name, map_name):
        """
        create a builder defined by a gather data type SoSGatherData
        """
        module_struct_list = f'{self.EE_PATH}.gather_data.SoSGatherData'
        cls = self.get_disc_class_from_module(module_struct_list)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)
        builder.set_builder_info('map_name', map_name)

        return builder

    def create_builder_coupling(self, sos_name: str) -> SoSBuilder:
        """
        create a builder  defined by a coupling type SoSCoupling
        """
        mod_path = f'{self.EE_PATH}.proxy_coupling.ProxyCoupling'
        cls = self.get_disc_class_from_module(mod_path)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)
        return builder

    def create_builder_selector_disc(self, sos_name: str) -> SoSBuilder:
        """
        create a builder  defined by a selectordiscipline
        """
        mod_path = f'{self.GENERIC_MODS_PATH}.selector_discipline.SelectorDiscipline'
        cls = self.get_disc_class_from_module(mod_path)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)
        return builder

    def create_optim_builder(self, sos_name, cls_builder) -> SoSBuilder:
        """creates the builder of the optim scenario"""
        mod_path = f'{self.EE_PATH}.proxy_optim.ProxyOptim'
        cls = self.get_disc_class_from_module(mod_path)
        builder = SoSBuilder(sos_name, self.__execution_engine, cls)
        builder.set_builder_info('cls_builder', cls_builder)
        return builder

    def get_disc_class_from_module(self, module_path):
        """
        Get the disc class from the module_path
        """
        module_struct_list = module_path.split('.')
        import_name = '.'.join(module_struct_list[:-1])
        # print('import_name = ',import_name)
        try:
            m = import_module(import_name)

        except Exception as e:
            raise (e)
        return getattr(m, module_struct_list[-1])

    def get_module_class_path(self, class_name, folder_list):
        """
        Return the module path of a class in a list of directories
        Return the first found for now .. TODO
        """

        module_class_path = None
        for folder in folder_list:
            # Get the module of the folder
            try:
                module = import_module(folder)
                folder_path = os.path.dirname(module.__file__)
            except:
                raise Warning(f'The folder {folder} is not a module')

            # Get all files in the folder_path
            file_list = os.listdir(folder_path)
            # Find all submodules in the path
            sub_module_list = [
                import_module('.'.join([folder, file.split('.')[0]]))
                for file in file_list
            ]

            for sub_module in sub_module_list:
                # Find all members of each submodule which are classes
                # belonging to the sub_module
                class_list = [
                    value
                    for value, cls in inspect.getmembers(sub_module)
                    if inspect.isclass(getattr(sub_module, value))
                       and cls.__module__ == sub_module.__name__
                ]
                # CHeck if the following class is in the list
                if class_name in class_list:
                    module_class_path = '.'.join(
                        [sub_module.__name__, class_name])
                    break
            else:
                continue
            break

        return module_class_path

    def get_builder_from_class_name(self, sos_name, mod_name, folder_list):
        """
        Get builder only using class name and retrievind the module path from the function get_module_class_path
        """
        mod_path = self.get_module_class_path(mod_name, folder_list)

        if mod_path is None:
            raise Exception(
                f'The builder {mod_name} has not been found in the folder list {folder_list}'
            )
        return self.get_builder_from_module(sos_name, mod_path)

    def clean_discipline_list(self, disciplines, current_discipline=None):
        """
        Clean all disciplines in the proxy_disciplines list of the factory and of the current_discipline
        """
        if current_discipline is None:
            current_discipline = self.__current_discipline

        for disc in disciplines:
            # We check if we have a discipline that can build other disciplines,
            # If it's the case, then we have to clean all its children as well
            # Furthermore, we have to check the specific class from whom the
            # discipline to clean is an instance.
            # TODO : We have to streamline the clean method so that the check for the specific ProxyDisciplineBuilder
            # won't be needed anymore

            if isinstance(disc, ProxyDisciplineBuilder):
                # case of the sosCoupling
                if isinstance(disc, ProxyCoupling):
                    self.clean_discipline_list(
                        disc.proxy_disciplines, current_discipline=disc
                    )

            disc.father_builder.remove_discipline(disc)
            self.__proxy_disciplines.remove(disc)

        current_discipline.remove_discipline_list(disciplines)

    def update_builder_with_extra_name(self, builder, extra_name):
        """
        Update the name of builder with an extra name which will be placed just after the variable after_name
        """
        new_builder_name = f'{extra_name}.{builder.sos_name}'
        builder.set_disc_name(new_builder_name)

    def update_builder_list_with_extra_name(self, extra_name, builder_list=None):
        """
        Update the name of a list of builders with an extra name placed behind after_name
        """
        if builder_list is None:
            builder_list = [self.coupling_builder]
        for builder in builder_list:
            self.update_builder_with_extra_name(builder, extra_name)

    def convert_builder_to_list(self, cls_builder):
        """
        Return list of builders
        """
        if isinstance(cls_builder, list):
            return cls_builder
        else:
            return [cls_builder]

    def remove_sos_discipline(self, discipline):
        """
        Delete discipline from the factory proxy_disciplines and from the discipline father builder
        """

        self.__proxy_disciplines.remove(discipline)

    def remove_discipline_from_father_executor(self, discipline):
        """
        Delete a discipline from its coupling children
        """
        try:
            discipline.father_executor.proxy_disciplines.remove(discipline)

        except:
            print("discipline already deleted from coupling children ")
