# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Jun 10, 2021"


class JobName(str):
    @staticmethod
    def _filter_string(val):
        return str(val).replace('.', '_').replace('-', 'm')

    def __new__(cls, val):
        return super().__new__(cls, cls._filter_string(val))

    def append(self, other):
        return JobName(super(JobName, self).__add__('_' + self._filter_string(other)))

    @property
    def string(self):
        return str(self)

    @staticmethod
    def _round(number):
        return round(number, ndigits=2)

    def T(self, temperature):
        return self.append(f'{self._round(temperature)}K')

    @property
    def min(self):
        return self.append('min')


class HasProject:
    def __init__(self, project):
        self._project = project
