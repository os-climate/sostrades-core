# Discipline disc3_scenario.Disc3
This disc3_scenario.Disc3 disciplineis specified as follows:


	DESC_IN = 
			{'z': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_disc3'},
			'constant': {'type': 'float', 'unit': '-'},
			'power': {'type': 'int', 'unit': '-'}}

	DESC_OUT =
			{'o': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_out_disc3'}}

	'o'=  constant + z**power