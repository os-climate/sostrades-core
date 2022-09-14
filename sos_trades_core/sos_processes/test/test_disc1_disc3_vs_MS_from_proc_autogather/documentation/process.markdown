# test disc1_disc3_vs_MS (from proc) with autogather
This "test disc1_disc3_vs_MS" specifies an example of a very symple multi scenario process.
It uses a scatter based on :


1) a scenario build map

		scenario_map = 
				{'input_name': 'scenario_list',
				'input_ns': 'ns_scatter_scenario',
				'output_name': 'scenario_name',
				'scatter_ns': 'ns_scenario',
				'gather_ns': 'ns_scatter_scenario',
				'ns_to_update': ['ns_disc3', 'ns_out_disc3','ns_ac']}
				
- the namespace table 
				'ns_scatter_scenario' = f'{self.ee.study_name}.vs_MS'
				'ns_scenario' : not needed
				
2) the process 'test_disc1_scenario' based on 
- the discipline test_discs.disc1_scenario.Disc1

		DESC_IN = 
				{'x': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_data_ac'},
				'a': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_data_ac'},
				'b': {'type': 'float', 'unit': '-'}}

		DESC_OUT =
				{'indicator': {'type': 'float', 'unit': '-'},
				'y': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'}}
	
		'indicator' = a * b
		'y': a * x + b
- the namespace table 

				'ns_ac' = self.ee.study_name
				'ns_data_ac' =  self.ee.study_name


3) the discipline disc3_scenario.Disc3

		DESC_IN = 
				{'z': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_disc3'},
				'constant': {'type': 'float', 'unit': '-'},
				'power': {'type': 'int', 'unit': '-'}}

		DESC_OUT =
				{'o': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_out_disc3'}}
	
		'o'=  constant + z**power
		
- the namespace table 
				'ns_disc3' = f'{self.ee.study_name}.vs_MS.Disc3)
				'ns_out_disc3'=  f'{self.ee.study_name}.vs_MS'