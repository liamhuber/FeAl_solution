# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal.utils import HasProject, JobName
import numpy as np
from pandas import DataFrame

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


class ZeroK(HasProject):
    def __init__(self, project):
        super().__init__(project)
        self._job_dict = None
        self._results = _Results(self)

    @property
    def results(self):
        return self._results

    def jobname(self, potential, structure_routine):
        potl_index = np.argwhere(self.project.input.potentials == potential)[0][0]
        return JobName(f'zerok').potl(potl_index).append(structure_routine.__name__)

    def get_jobs(self, delete_existing_job=False):
        """
        A nested dictionary of jobs with the outer loop for all the potentials specified in the project input, and the
        inner loop over a collection of structures.

        Tries to load the jobs and if this fails, creates them.
        Creation can be forced with the `delete_existing_job` boolean kwarg.
        """
        return {
            potl: {
                routine.__name__: self._get_job(potl, routine, delete_existing_job=delete_existing_job)
                for routine in self.structure_routines
            }
            for potl in self.project.input.potentials
        }

    @property
    def structure_routines(self):
        sf = self.project.create.structure.FeAl
        return [sf.BCC, sf.FCC, sf.B2, sf.D03, sf.random_BCC, sf.random_FCC]

    def _get_job(self, potential, structure_routine, delete_existing_job=False):
        name = self.jobname(potential, structure_routine)
        job = self.project.load(name)
        if job is None or delete_existing_job:
            job = self._create_job(name, structure_routine(), potential)
        return job

    def _create_job(self, name, structure, potential):
        job = self.project.create.job.Lammps(name, delete_existing_job=True)
        job.structure = structure
        job.potential = potential
        job.calc_minimize(pressure=0)
        return job

    @property
    def job_dict(self):
        """To force re-load the job dictionary, e.g. if the jobs were run remotely, use `get_jobs`."""
        self._job_dict = self.get_jobs() if self._job_dict is None else self._job_dict
        return self._job_dict

    def run(self, run_again=False):
        for struct_based_dict in self.get_jobs(delete_existing_job=run_again).values():
            for job in struct_based_dict.values():
                job.run()


class _Results:
    def __init__(self, zerok: ZeroK):
        self._zerok = zerok
        self._table = None

    @property
    def table(self):
        self._table = self._read_table() if self._table is None else self._table
        return self._table

    def _read_table(self):
        df = DataFrame(columns=['potential', 'structure', 'n_atoms', 'n_Al', 'E_pot'])
        for potl in self._zerok.project.input.potentials:
            for routine in self._zerok.structure_routines:
                job = self._zerok.job_dict[potl][routine.__name__]
                df = df.append({
                    'potential': potl,
                    'structure': routine.__name__,
                    'n_atoms': len(job.structure),
                    'n_Al': np.sum(job.structure.get_chemical_symbols() == 'Al'),
                    'E_pot': job.output.energy_pot[-1],
                }, ignore_index=True)
        return df

    def get_energy(self, potential, structure, dmu_Al=0):
        row = self.table.loc[(self.table['potential'] == potential) & (self.table['structure'] == structure)]
        e_pot = row['E_pot'].values[0]
        n_atoms = row['n_atoms'].values[0]
        n_Al = row['n_Al'].values[0]
        return (e_pot + n_Al * dmu_Al) / n_atoms

    @property
    def FCC_BCC_stability(self):
        df = DataFrame(columns=['potential', 'FCC-BCC [eV/atom]'])
        for potential in np.unique(self.table['potential']):
            df = df.append({
                'potential': potential,
                'FCC-BCC [eV/atom]': self.get_energy(potential, 'FCC') - self.get_energy(potential, 'BCC')
            }, ignore_index=True)
        return df
