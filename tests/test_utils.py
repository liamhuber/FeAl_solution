# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from unittest import TestCase
from pyiron_feal.utils import JobName


class TestJobName(TestCase):

    def test_filter(self):
        self.assertEqual('foo_mbar', JobName._filter_string('foo.-bar'), msg="Unexpected replacements.")

    def test_init(self):
        self.assertEqual('foo_mbar', JobName('foo.-bar'))

    def test_add(self):
        self.assertEqual('foobar', JobName('foo') + 'bar')
        self.assertIsInstance(JobName('foo') + 'bar', str, msg="Did you override __add__?")

    def test_append(self):
        self.assertEqual('foo_bar', JobName('foo').append('bar'))
        self.assertIsInstance(JobName('foo').append('bar'), JobName,
                              msg="Did you accidentally return a regular string?")

    def test_temperature(self):
        self.assertEqual('foo_273K', JobName('foo').T(273))
        self.assertEqual('foo_273_0K', JobName('foo').T(273.0))

    def test_tags(self):
        self.assertEqual(
            'foo_min',
            JobName('foo').min
        )
