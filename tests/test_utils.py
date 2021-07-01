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

    def test_other_tags(self):
        self.assertEqual('foo_potl42', JobName('foo').potl(42))
        self.assertEqual('foo_cAl33_33', JobName('foo').c_Al(0.33333333))
        self.assertEqual('foo_cAl33_333', JobName('foo').c_Al(0.33333333, ndigits=3))
        self.assertEqual('foo_rep4', JobName('foo').repeat(4))
        self.assertEqual('foo', JobName('foo').repeat(None))
        self.assertEqual('foo_trl3', JobName('foo').trial(3))
        self.assertEqual('foo_bcc', JobName('foo').bcc)
        self.assertEqual('foo_fcc', JobName('foo').fcc)
        self.assertEqual('foo_b2', JobName('foo').b2)
        self.assertEqual('foo_d03', JobName('foo').d03)
        self.assertEqual('foo_P0_0', JobName('foo').P(0.))
        self.assertEqual('foo_a4_36', JobName('foo').a(4.355))

    def test_call(self):
        name = JobName('foo')
        self.assertEqual(
            'foo_potl42_bcc_rep4_trl3_273K_P0_0_cDAl2Fe11_1',
            name(
                potl_index=42,
                bcc=True,
                repeat=4,
                trial=3,
                temperature=273,
                pressure=0.,
                c_D03_anti_Al_to_Fe=0.1111111,
                ndigits=1
            )
        )
        self.assertEqual('foo', name.string, msg="Calling shouldn't overwrite the base object.")
