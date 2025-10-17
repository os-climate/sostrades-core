'''
Copyright 2025 Capgemini

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

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    """
    Complete DoE process for FakeCarModel with ValueAnalysis
    Includes proper DoE initialization with continuous parameters
    """

    # ontology information
    _ontology_data = {
        'label': 'TChTestB FakeCarModel DoE ValueAnalysis Process',
        'description': 'Process for testing FakeCarModel DoE with ValueAnalysis',
        'category': '',
        #'category': 'Test',
        'version': '1.0',
    }

    def get_builders(self):
        """
        Create DoE process with proper namespace structure:
        - DoE namespace containing Eval[FakeCarModel] (SampleGenerator created automatically)
        - ValueAnalysis at root level to consume DoE results via samples_df connection
        """
        # Create DoE namespace for proper structure
        doe_ns_dict = {
            'ns_public': f'{self.ee.study_name}.DoE',
        }

        # Create FakeCarModel discipline for inside DoE.Eval
        mods_dict = {
            'FakeCarModel': 'sostrades_core.sos_wrapping.test_discs.FakeCarModel.FakeCarModel_discipline.FakeCarModelDiscipline'
        }
        fakecar_builder_list = self.create_builder_list(mods_dict, ns_dict=doe_ns_dict)

        # Create mono-instance driver for DoE evaluation (SampleGenerator created automatically when with_sample_generator=True)
        eval_builder = self.ee.factory.create_mono_instance_driver('Eval', fakecar_builder_list)

        # Create ValueAnalysis at root level to receive DoE evaluation results
        root_ns_dict = {
            'ns_public': self.ee.study_name, 'ns_eval': f'{self.ee.study_name}.Eval',
        }
        va_mods_dict = {
            'ValueAnalysis': 'sostrades_core.sos_wrapping.analysis_discs.value_analysis.ValueAnalysis_discipline.ValueAnalysisDiscipline'
        }
        value_analysis_builder_list = self.create_builder_list(va_mods_dict, ns_dict=root_ns_dict)

        # Return both eval_builder and ValueAnalysis builders
        return [eval_builder] + value_analysis_builder_list
