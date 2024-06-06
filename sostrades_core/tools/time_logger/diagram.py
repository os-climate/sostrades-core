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
import os
import re

import matplotlib.pylab as plt


class Diagram(object):
    '''
    Diagram class to construct diagram for plot visualization
    '''

    def __init__(self, logger, name, x_axis, y_axis,
                 save_dir=None, sub_save_dir=None, y_log_scale=False):
        '''
        Constructor for the Diagram class
        '''
        self.__logger = logger
        self.__name = name
        self.__x_axis = x_axis
        self.__y_label = ""
        self.__x_label = x_axis + " " + self.__logger.get_units(x_axis)

        self.__y_log_scale = y_log_scale

        if isinstance(y_axis, list):
            self.__y_axis = y_axis
        else:
            self.__y_axis = [y_axis]

        base_save_dir = self.__logger.get_full_save_dir(save_dir=save_dir)

        if sub_save_dir is None:
            self.__save_dir = base_save_dir
        else:
            self.__save_dir = os.path.join(base_save_dir, sub_save_dir)

        if not os.path.isdir(self.__save_dir):
            os.makedirs(self.__save_dir)

    def add_curve(self, y_axis):
        '''
        Add a curve to the y_axis vector
        '''
        self.__y_axis.append(y_axis)

    def plot(self, y_lim=None):
        '''
        Plot figures with matplotlib
        '''
        x_list = self.__logger.get_data(self.__x_axis)

        single_curve = len(self.__y_axis) == 1

        for y_ax in self.__y_axis:
            y_list = self.__logger.get_data(y_ax)

            found = re.findall(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+)$", y_ax)
            if len(found) > 0:
                label = found[0]
            else:
                label = y_ax
            if self.__y_log_scale:
                plt.semilogy(x_list, y_list, label=label)

            if len(y_list) <= 1.:
                plt.plot(x_list, y_list, label=label, marker='s')
            else:
                plt.plot(x_list, y_list, label=label)

        plt.xlabel(self.__x_label)

        if self.__y_label == "":
            label = self.__y_axis[0].split('.')[-1]
            self.__y_label = label + " " + \
                self.__logger.get_units(self.__y_axis[0])
        plt.ylabel(self.__y_label)

        if y_lim is not None:
            plt.ylim(y_lim)

        if not single_curve:
            lgd = plt.legend(bbox_to_anchor=(1, 0), loc="lower left")
        plt.minorticks_on()
        plt.grid(which='major', alpha=0.7)
        plt.grid(which='minor', alpha=0.4)
        plt.rc("font", size=12)
        plt.title(self.__name)
        file_name = os.path.join(self.__save_dir, self.__name + ".png")

        if not single_curve:
            plt.savefig(file_name, format='png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig(file_name, format='png')
        plt.close()
