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

import base64
import os
import re
from os import listdir
from os.path import join, dirname, isdir, isfile


def get_markdown_documentation(filepath):
    """
    Get markdown documentation from a file.md file within a "documentation" folder and with image files in the same folder
    """
    # Manage markdown documentation

    doc_folder_path = join(dirname(filepath), 'documentation')
    filename = os.path.basename(filepath).split('.')[0]
    markdown_data = ""
    if isdir(doc_folder_path):
        # look for markdown file with extension .markdown or .md
        markdown_list = [join(doc_folder_path, md_file) for md_file in listdir(doc_folder_path) if ((
            md_file.endswith(r".markdown") or md_file.endswith(r".md")) and md_file.startswith(filename))]

        if len(markdown_list) > 0:
            # build file path
            markdown_filepath = markdown_list[0]

            if isfile(markdown_filepath):
                markdown_data = ''

                with open(markdown_filepath, 'r+t', encoding='utf-8') as f:
                    markdown_data = f.read()

                # Find file reference in markdown file
                place_holder = f'!\\[(.*)\\]\\((.*)\\)'
                matches = re.finditer(place_holder, markdown_data)

                images_base_64 = {}
                base64_image_tags = []

                for matche in matches:
                    # Format:
                    # (0) => full matche line
                    # (1) => first group (place holder name)
                    # (2) => second group (image path/name)

                    image_name = matche.group(2)

                    # Convert markdown image link to link to base64 image
                    image_filepath = join(doc_folder_path, image_name)

                    if isfile(image_filepath):
                        image_data = open(image_filepath, 'r+b').read()
                        encoded = base64.b64encode(
                            image_data).decode('utf-8')

                        images_base_64.update({image_name: encoded})

                        # first replace the matches
                        matche_value = matche.group(1)
                        matches_replace = f'![{matche_value}]({image_name})'
                        matches_replace_by = f'![{matche_value}][{image_name}]'

                        base64_image_tag = f'[{image_name}]:data:image/png;base64,{images_base_64[image_name]}'
                        base64_image_tags.append(base64_image_tag)

                        markdown_data = markdown_data.replace(
                            matches_replace, matches_replace_by)

                for image_tag in base64_image_tags:
                    markdown_data = f'{markdown_data}\n\n{image_tag}'

    return markdown_data