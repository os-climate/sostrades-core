'''
Copyright 2023 Capgemini

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

import csv


def glossary_to_csv(obj, filename):
    # Create a list of dictionaries containing the variable information
    variable_list = []
    for variable_name in dir(obj):
        # Ignore built-in attributes and methods
        if not variable_name.startswith("__") and not callable(getattr(obj, variable_name)):
            variable_value = getattr(obj, variable_name)
            # Check if the variable is a dictionary
            if isinstance(variable_value, dict):
                variable_dict = {}
                variable_dict["id"] = getattr(obj, variable_name).get("var_name", "")
                variable_dict["label"] = variable_name
                variable_dict["unit"] = getattr(obj, variable_name).get("unit", "")
                variable_dict["definition"] = getattr(obj, variable_name).get("description", "")
                variable_dict["definitionSource"] = ""
                variable_dict["ACLTag"] = ""
                variable_list.append(variable_dict)

    # Write the list of dictionaries to the CSV file
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "unit", "definition", "definitionSource", "ACLTag"])
        writer.writeheader()
        writer.writerows(variable_list)
