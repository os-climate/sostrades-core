'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

from pprint import PrettyPrinter


class Database(dict):
    """ Database implementation
    """
    COMMENT = 'Comment'
    UNIT = 'Unit'
    DATABASE_ID = 'database_id'
    VALUE = 'Value'
    SKIP = 'SKIP'
    LEVEL = 'LEVEL'
    CONVERT = 'Convert'
    MIN = 'Min'
    MAX = 'Max'
    HEADERS = [COMMENT, UNIT, VALUE, SKIP, LEVEL, CONVERT, MIN, MAX]

    def __init__(self, labels=None):

        super(Database, self).__init__()
        if labels is not None:
            self.update_from_keys(*labels)
#             for key in labels:
#                 self._is_str(key)
#                 self.__setitem__(key, self._default_dict())

#     def __initialize(self, labels):
#         def build_default():
#             return defaultdict(
#                 dict.fromkeys(headers))
#         self.update(build_default())

    def _default_dict(self):
        return dict.fromkeys(self.HEADERS)

    def __getitem__(self, label):
        return super(Database, self).__getitem__(label)

    def __setitem__(self, label, v):
        self._check_inputs(label, v)
        return super(Database, self).__setitem__(label, v)

    def __delitem__(self, label):
        return super(Database, self).__delitem__(label)

    def get(self, label, default=None):
        return super(Database, self).get(label, default)

    def get_value(self, label, default=None):
        return self.__getitem__(label).get(self.VALUE, default)

    def get_comment(self, label):
        return self.__getitem__(label)[self.COMMENT]

    def get_unit(self, label):
        return self.__getitem__(label)[self.UNIT]

    def set_value(self, label, val):
        self.__getitem__(label)[self.VALUE] = val
        return self

    def set_comment(self, label, val):
        self._is_str(val)
        self.__getitem__(label)[self.COMMENT] = val
        return self

    def set_unit(self, label, val):
        self._is_str(val)
        self.__getitem__(label)[self.UNIT] = val
        return self

#     def setdefault(self, k, default=None):
#         return super(Database, self).setdefault(k, default)

    def update(self, **kwargs):
        for label, val in kwargs.items():
            self._check_inputs(label, val)
            super(Database, self).update({label: val})

    def update_from_keys(self, *args):
        for key in args:
            self._is_str(key)
            self.__setitem__(key, self._default_dict())

    def _check_inputs(self, label, input_dict):

        key_errors = set(input_dict.keys()) - set(self.HEADERS)
        if len(key_errors) > 0:
            raise KeyError("Header(s) %s of label %s not in %s" %
                           (str(key_errors), label, self.HEADERS))

    def _is_str(self, val):
        return isinstance(val, str)

    def __contains__(self, label):
        return super(Database, self).__contains__(label)

    def __str__(self):
        pp = PrettyPrinter()
        return pp.pformat(self)


# _labels = ['a', 'b']
# dico = Database(_labels)
# print(dico.keys())
# print(dico.values())
# ff = {
#     'a': {
#         'Comment': 10,
#         'Unit': 10,
#         'Value': 10},
#     'b': {
#         'Comment': 5,
#         'Unit': 5,
#         'Value': 5},
#     'c': {
#         'Comment': 5,
#         'Unit': 5,
#         'Value': 5},
#     'x': {
#         'Comment': 4,
#         'Unit': 4}}
# print(dico.update(**ff))
# print(dico)
# print(dico.get_value('b'))
# print(dico.get_value('x', 12))
# print(dico.set_value('c', 99))
