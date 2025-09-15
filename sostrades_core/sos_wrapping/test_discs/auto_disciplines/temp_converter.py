'''
Copyright 2024 Capgemini
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

from sostrades_core.tools.discipline_decorator.function_discipline_decorator import auto_sos_discipline


@auto_sos_discipline(outputs={'fahrenheit': float, 'kelvin': float})
def TempConverter(celsius: float = 25.0) -> dict:
    """Convert temperature from Celsius to Fahrenheit and Kelvin."""
    fahrenheit = celsius * 9/5 + 32
    kelvin = celsius + 273.15
    return {'fahrenheit': fahrenheit, 'kelvin': kelvin}
