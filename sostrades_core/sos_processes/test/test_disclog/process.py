from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test DiscLog Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        mods_dict = {
            'DiscLog': 'sostrades_core.sos_wrapping.test_discs.disc_log_generation.DiscLogGeneration'}
        builder_list = self.create_builder_list(mods_dict, ns_dict={'ns_ac': self.ee.study_name})
        return builder_list
