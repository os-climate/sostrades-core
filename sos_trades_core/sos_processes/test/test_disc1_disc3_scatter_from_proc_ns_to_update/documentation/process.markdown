# test test_scatter-disc1_disc3_vs_MS_from_proc with a different ns_to_update
This "test_disc1_disc3_scatter_from_proc specifies an example of a scatter process applied on (disc1,disc3).
It uses a scatter based on :

1) a scatter build map:

		ac_map = 
				{'input_name': 'scenario_list',
				'input_ns': 'ns_scatter_scenario',
				'output_name': 'ac_name',
				'scatter_ns': 'ns_ac',
				'gather_ns': 'ns_scenario',
				'ns_to_update': ['ns_data_ac']}
				
2) The name space variables:
- the namespace table 

				'ns_scatter_scenario' = f'{self.ee.study_name}'
				'ns_scenario' = f'{self.ee.study_name}'

3) the discipline test_discs.disc1_scenario.Disc1

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


4) the discipline disc3_scenario.Disc3

		DESC_IN = 
				{'z': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_disc3'},
				'constant': {'type': 'float', 'unit': '-'},
				'power': {'type': 'int', 'unit': '-'}}

		DESC_OUT =
				{'o': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_out_disc3'}}
	
		'o'=  constant + z**power
		
- the namespace table 

				'ns_disc3' = f'{self.ee.study_name}.Disc3
				'ns_out_disc3'=  f'{self.ee.study_name}'