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
from win32gui import BringWindowToTop
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_discipline_builder import SoSDisciplineBuilder
from sos_trades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
import pandas as pd


class AddSubProcToDriver(SoSDisciplineBuilder):
    '''
    Generic DOE evaluation class

    1) Strucrure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ SUB_PROCESS_INPUTS (structuring)


    2) Description of DESC parameters:
        |_ DESC_IN
           |_ SUB_PROCESS_INPUTS:               All inputs for driver builder in the form of ProcessBuilderParameterType type
                                                    PROCESS_REPOSITORY:   folder root of the sub processes to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    PROCESS_NAME:         selected process name (in repository) to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    USECASE_INFO:         either empty or an available data source of the sub_process
                                                    USECASE_NAME:         children of USECASE_INFO that contains data source name (can be empty)
                                                    USECASE_TYPE:         children of USECASE_INFO that contains data source type (can be empty)
                                                    USECASE_IDENTIFIER:   children of USECASE_INFO that contains data source identifier (can be empty)
                                                    USECASE_DATA:         anonymized dictionary of usecase inputs to be nested in context
                                                                          it is a temporary input: it will be put to None as soon as
                                                                          its content is 'loaded' in the dm. We will have it has editable
                                                It is in dict type (specific 'proc_builder_modale' type to have a specific GUI widget) 
    '''
#################### Begin: Ontology of the discipline ###################
    # ontology information
    _ontology_data = {
        'label': 'Driver to add a subprocess to a selected driver',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'Driver that allows to nest a subprocess to a selected driver and to import data from associated usecases or studies of the subprocess',
        'icon': 'fas fa-screwdriver-wrench fa-fw',  # icon for proc builder
        'version': ''
    }
#################### End: Ontology of the discipline #####################
#################### Begin: Constants and parameters #####################
    # -- Disciplinary attributes
    SUB_PROCESS_INPUTS = 'sub_process_inputs'
    default_process_builder_parameter_type = ProcessBuilderParameterType(
        None, None, 'Empty')

    DESC_IN = {
        SUB_PROCESS_INPUTS: {'type': SoSDiscipline.PROC_BUILDER_MODAL,
                             'structuring': True,
                             'default': default_process_builder_parameter_type.to_data_manager_dict(),
                             'user_level': 1,
                             'optional': False
                             }}
#################### End: Constants and parameters #######################
#################### Begin: Main methods ################################

    def __init__(self, sos_name, ee, associated_namespaces=[]):
        '''
        Constructor
        '''
        SoSDisciplineBuilder.__init__(
            self, sos_name, ee, associated_namespaces=associated_namespaces)
        self.previous_sub_process_repo = None
        self.previous_sub_process_name = None

        self.previous_sub_process_usecase_name = 'Empty'
        self.previous_sub_process_usecase_data = {}
        self.dyn_var_sp_from_import_dict = {}

        self.sub_process_ns_in_build = None
        self.ns_of_sub_proc = []  # (vsMS) --> used in  build_driver_subproc

        # Possible values: 'Empty_SP', 'Create_SP', 'Replace_SP',
        # 'Unchanged_SP'
        self.sub_proc_build_status = 'Empty_SP'

        # Possible values: 'No_SP_UC_Import', 'SP_UC_Import'
        self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

        self.set_ref_discipline_full_name()

    def build_for_proc_builder(self):
        '''
            Overloaded SoSEval method
            Get and build builder from sub_process of doe_eval driver
            Added to provide proc builder capability
            Reached from __configure_io in ee.py: self.factory.build() is going from build to build starting from root
            It comes before configuring()
        '''
        if self.SUB_PROCESS_INPUTS in self._data_in:
            sub_process_inputs_dict = self.get_sosdisc_inputs(
                self.SUB_PROCESS_INPUTS)
            sub_process_repo = sub_process_inputs_dict[ProcessBuilderParameterType.PROCESS_REPOSITORY]
            sub_process_name = sub_process_inputs_dict[ProcessBuilderParameterType.PROCESS_NAME]
            if sub_process_repo != None and sub_process_name != None:  # a sub_process_full_name is available
                # either Unchanged_SP or Create_SP or Replace_SP
                # 1. set_sub_process_status
                self.set_sub_process_status(
                    sub_process_repo, sub_process_name)
                # 2 build_eval_subproc
                if self.sub_proc_build_status == 'Create_SP' or self.sub_proc_build_status == 'Replace_SP':
                    self.build_driver_subproc(
                        sub_process_repo, sub_process_name)

    def configure_for_proc_builder(self):
        """
            Overloaded SoSDiscipline method
            Configuration of the Build driver
            Reached from __configure_io in ee.py: self.root_process.configure_io() is going from confiure to configure starting from root
            It comes after build()
        """
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

        dynamic_inputs = {}
        dynamic_outputs = {}
        # Remark: in case of 'Unchanged_SP', it will do a refresh of available
        # subprocesses
        if self.sub_proc_build_status != 'Empty_SP':
            self.get_sosdisc_inputs(self.SUB_PROCESS_INPUTS)
            # 1. provide driver inputs based on selected subprocess
            dynamic_inputs = self.setup_sos_disciplines_driver_inputs_depend_on_sub_process(
                dynamic_inputs)
            dynamic_inputs, dynamic_outputs = self.setup_sos_disciplines_driver_inputs_independent_on_sub_process(
                dynamic_inputs, dynamic_outputs)
            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)
            # 2. import data from selected sub_process_usecase
            self.manage_import_inputs_from_sub_process()

#################### End: Main methods ################################
##################### Begin: Sub methods for build #######################
# Remark: those sub methods should be private functions

    def set_sub_process_status(self, sub_process_repo, sub_process_name):
        '''
            State subprocess CRUD status
            The subprocess is defined by its name and repository
            Function needed in build_for_proc_builder(self)
        '''
        # We come from outside driver process
        if (sub_process_name != self.previous_sub_process_name or sub_process_repo != self.previous_sub_process_repo):
            self.previous_sub_process_repo = sub_process_repo
            self.previous_sub_process_name = sub_process_name
        # driver process with provided sub process
            if len(self.get_cls_builder()) == 0:
                self.sub_proc_build_status = 'Create_SP'
            else:
                self.sub_proc_build_status = 'Replace_SP'
        else:
            self.sub_proc_build_status = 'Unchanged_SP'

        return self.sub_proc_build_status

    def build_driver_subproc(self, sub_process_repo, sub_process_name):
        '''
            Get and build builder from sub_process of the driver
            The subprocess is defined by its name and repository
            Function needed in build_for_proc_builder(self)
        '''
        # 1. Clean if needed
        if self.sub_proc_build_status == 'Replace_SP':
            # clean all instances before rebuilt and and reset any needed
            # parameter
            self.clean_driver_before_rebuilt()
        # 2. Get and set the builder of subprocess
        cls_builder = self.get_nested_builders_from_sub_process(
            sub_process_repo, sub_process_name)
        self.set_nested_builders(cls_builder)
        # 3. Capture the input namespace specified at
        # building step
        ns_of_driver = self.get_ns_of_driver()
        self.capture_ns_of_sub_proc_and_associated_values(ns_of_driver)
        # 4. Put ns of subproc in driver context
        self.put_ns_of_subproc_in_driver_context(ns_of_driver, assign_ns=False)

    def get_nested_builders_from_sub_process(self, sub_process_repo, sub_process_name):
        """
            Create_nested builders from their nested process.
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
        """
        cls_builder = self.ee.factory.get_builder_from_process(
            repo=sub_process_repo, mod_id=sub_process_name)
        if not isinstance(cls_builder, list):
            cls_builder = [cls_builder]
        return cls_builder

    def set_nested_builders(self, cls_builder):
        """
            Set nested builder to the driver process in case this driver process was instantiated with an empty nested builder. 
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
        """
        self.set_cls_builder(cls_builder)
        self.driver_process_builder = self.set_driver_process_builder()
        return

    def capture_ns_of_sub_proc_and_associated_values(self, ns_of_driver):
        '''
            Specific function of the driver to capture the ns of the subprocess and their values
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
        '''
        ns_of_sub_proc = [
            key for key in self.ee.ns_manager.shared_ns_dict if key not in ns_of_driver]
        self.ns_of_sub_proc = ns_of_sub_proc

        ns_of_sub_proc_dict = {}
        for item in ns_of_sub_proc:
            ns_of_sub_proc_dict[item] = self.ee.ns_manager.shared_ns_dict[item].get_value(
            )
        self.sub_process_ns_in_build = pd.DataFrame(
            list(ns_of_sub_proc_dict.items()), columns=['Name', 'Value'])

        return ns_of_sub_proc_dict

    def put_ns_of_subproc_in_driver_context(self, ns_of_driver, assign_ns=False):
        '''
            Specific function of the driver to shift namespace of the subprocess with driver name and possibly assign it to the driver 
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
        '''
        #   1 Shift ns with driver name
        driver_name = self.name
        self.update_namespace_list_with_extra_ns_except_driver(
            driver_name, ns_of_driver, after_name=self.ee.study_name)
        #   2 Then add ns keys to driver discipline
        if assign_ns == True:
            for item in self.ns_of_sub_proc:
                self.ee.ns_manager.update_others_ns_with_shared_ns(item)
        return

    def update_namespace_list_with_extra_ns_except_driver(self, extra_ns, driver_ns_list, after_name=None, namespace_list=None):
        '''
            Update the value of a list of namespaces with an extra namespace placed behind after_name
            In our context, we do not want to shift ns of driver_ns_list  already created before nested sub_process
            Function needed in put_ns_of_subproc_in_driver_context(self, ns_of_driver, assign_ns = False)
        '''
        if namespace_list is None:
            #namespace_list = self.ee.ns_manager.ns_list
            namespace_list = list(self.ee.ns_manager.shared_ns_dict.values())
            namespace_list = [
                elem for elem in namespace_list if elem.__dict__['name'] not in driver_ns_list]
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_ns, after_name, namespace_list=namespace_list)

    def setup_sos_disciplines_driver_inputs_depend_on_sub_process(self, dynamic_inputs):
        """
            Update of dynamic inputs of the driver depending on sub process
            Update of NS_IN_DF
            Function needed in setup_sos_disciplines()
        """
        # Provide dynamic driver parameters triggered by the SP creation
        desc_in_dict = self.setup_desc_in_dict_of_driver()
        dynamic_inputs.update(desc_in_dict)

        # Set status
        self.sub_proc_build_status = 'Unchanged_SP'

        # Also provide information about namespace variables provided at
        # building time
        desc_in_dict = {}
        desc_in_dict[self.NS_IN_DF] = {'type': 'dataframe',
                                       'unit': None,
                                       'editable': False,
                                       'default': self.sub_process_ns_in_build}

        if self.sub_process_ns_in_build is not None:
            dynamic_inputs.update(desc_in_dict)

        return dynamic_inputs

    def manage_import_inputs_from_sub_process(self):
        """
            Function needed in setup_sos_disciplines()
        """
        # Set sub_proc_import_usecase_status
        if self.SUB_PROCESS_INPUTS in self._data_in:  # and self.sub_proc_build_status != 'Empty_SP'
            sub_process_inputs_dict = self.get_sosdisc_inputs(
                self.SUB_PROCESS_INPUTS)
            sub_process_usecase_name = sub_process_inputs_dict[
                ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_NAME]
            sub_process_usecase_data = sub_process_inputs_dict[ProcessBuilderParameterType.USECASE_DATA]
            self.set_sub_process_usecase_status_from_user_inputs(
                sub_process_usecase_name, sub_process_usecase_data)
        else:
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

        # Treat the case of SP_UC_Import
        if self.sub_proc_import_usecase_status == 'SP_UC_Import':
            # 1 get anonymized dict
            sub_process_inputs_dict = self.get_sosdisc_inputs(
                self.SUB_PROCESS_INPUTS)
            sub_process_usecase_name = sub_process_inputs_dict[
                ProcessBuilderParameterType.USECASE_INFO][ProcessBuilderParameterType.USECASE_NAME]
            anonymize_input_dict_from_usecase = sub_process_inputs_dict[
                ProcessBuilderParameterType.USECASE_DATA]
            # 2 put anonymized dict in context (unanonymize)
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
            # Set the status to No_SP_UC_Import' and empty the anonymized dict
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'
            sub_process_inputs_dict[ProcessBuilderParameterType.USECASE_DATA] = {
            }
            self.dm.set_data(f'{self.get_disc_full_name()}.{self.SUB_PROCESS_INPUTS}',
                             self.VALUES, sub_process_inputs_dict, check_value=False)
            self.previous_sub_process_usecase_data = {}
        # there are still dynamic variables put apart
        elif len(self.dyn_var_sp_from_import_dict) != 0:
            self.ee.dm.set_values_from_dict(self.dyn_var_sp_from_import_dict)
            # Is it also OK in case of a dynamic param of dynamic param ?
            self.dyn_var_sp_from_import_dict = {}

    def set_sub_process_usecase_status_from_user_inputs(self, sub_process_usecase_name, sub_process_usecase_data):
        """
            State subprocess usecase import status
            The uscase is defined by its name and its anonimized dict
            Function needed in manage_import_inputs_from_sub_process()
        """
        if self.previous_sub_process_usecase_name != sub_process_usecase_name or self.previous_sub_process_usecase_data != sub_process_usecase_data:
            self.previous_sub_process_usecase_name = sub_process_usecase_name
            self.previous_sub_process_usecase_data = sub_process_usecase_data
            # means it is not an empty dictionary
            if sub_process_usecase_name != 'Empty' and not not sub_process_usecase_data:
                self.sub_proc_import_usecase_status = 'SP_UC_Import'
        else:
            self.sub_proc_import_usecase_status = 'No_SP_UC_Import'

    def put_anonymized_input_dict_in_sub_process_context(self, anonymize_input_dict_from_usecase):
        """
            Put_anonymized_input_dict in sub_process context
            Function needed in manage_import_inputs_from_sub_process()
        """
        # Get unanonymized dict (i.e. dict of subprocess in driver context)
        # from anonymized dict and context
        # good prefix in context

        #driver_name = self.name
        # Following treatment of substitution of the new_study_placeholder of value self.ref_discipline_full_name is
        # OK if variable is a variable with ns  in ns_to_update
        # If not then we should shift only of f'{self.ee.study_name}.{driver_name}
        # For this we need to find ns of each variable
        # self.ee.ns_manager.get_shared_namespace(disc, var_ns)

        input_dict_from_usecase = {}
        new_study_placeholder = self.ref_discipline_full_name
        for key_to_unanonymize, value in anonymize_input_dict_from_usecase.items():
            converted_key = key_to_unanonymize.replace(
                self.ee.STUDY_PLACEHOLDER_WITHOUT_DOT, new_study_placeholder)
            # see def __unanonymize_key  in execution_engine
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

##################### End: Sub methods for build #########################


######### Begin: Sub methods for build to be specified in build #####

    def get_cls_builder(self):
        '''
            Specific function of the driver to get the cls_builder
            Function needed in set_sub_process_status()
            Function to be specified per driver
        '''
        cls_builder = None
        # cls_builder = xxx
        return cls_builder

    def set_cls_builder(self, value):
        '''
            Specific function of the driver to set the cls_builder with value
            Function needed in set_nested_builders()
            Function to be specified per driver 
        '''
        pass

    def set_ref_discipline_full_name(self):
        '''
            Specific function of the driver to define the full name of the reference discipline
            i.e. f'{self.ee.study_name}.{driver_name}' for doe_eval
                 f'{self.ee.study_name}.{driver_name}.{self.REFERENCE}' for vsMS
            Function needed in _init_ of the driver
            Function to be specified per driver
        '''
        driver_name = self.name
        # default value
        self.ref_discipline_full_name = f'{self.ee.study_name}.{driver_name}'
        # update the above line
        # self.ref_discipline_full_name = xxxxxxx

        return

    def clean_driver_before_rebuilt(self):
        '''
            Specific function of the driver to clean all instances before rebuild and reset any needed parameter
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
            Function to be specified per driver
        '''
        # xxxx

        # We "clean" also all dynamic inputs to be reloaded by
        # the usecase
        self.add_inputs({})  # is it needed ?
        return

    def get_ns_of_driver(self):
        '''
            Specific function of the driver to get ns of driver
            Function needed in build_driver_subproc(self, sub_process_repo, sub_process_name)
            Function to be specified per driver
        '''
        ns_of_driver = []
        # xxxx
        return ns_of_driver

    def set_driver_process_builder(self):
        '''
            Create the driver process builder
            Function needed in set_nested_builders
            Function that may need to be specified per driver
        '''
        disc_builder = None
        disc_builder = self._set_driver_process_builder()
        return disc_builder

    def setup_desc_in_dict_of_driver(self):
        """
            Create desc_in_dict for dynamic inputs of the driver depending on sub process
            Function needed in setup_sos_disciplines_driver_inputs_depend_on_sub_process()
            Function to be specified per driver
        """
        desc_in_dict = {}
        # xxxx
        return desc_in_dict

    def setup_sos_disciplines_driver_inputs_independent_on_sub_process(self, dynamic_inputs, dynamic_outputs):
        """
            setup_dynamic inputs when driver parameters depending on the SP selection are already set
            Manage update of XXX,YYY parameters
            Function needed in setup_sos_disciplines()
            Function to be specified per driver
        """
        # xxxx
        return dynamic_inputs, dynamic_outputs


#####  End: Sub methods for build to be specified in build #####
