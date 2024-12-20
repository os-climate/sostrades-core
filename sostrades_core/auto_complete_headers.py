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
import json

from sostrades_core.tools.check_headers import HeaderTools


def main() -> None:
    """
    Main function to read the headers_ignore_config.json file and write headers if needed in the repository.
    """
    try:
        with open("./headers_ignore_config.json", "r", encoding="utf-8") as f:
            headers_ignore_config = json.load(f)
            ht = HeaderTools()
            ht.set_verbose_mode(False)
            ht.write_headers_if_needed_in_repo(
                headers_ignore_config["extension_to_ignore"],
                headers_ignore_config["files_to_ignore"],
                headers_ignore_config["airbus_rev_commit"],
                ".",
            )
    except FileNotFoundError as ex:
        print("headers_ignore_config.json must be available where this command is launched")
        raise ex


if __name__ == "__main__":
    main()
