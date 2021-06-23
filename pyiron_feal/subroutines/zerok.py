# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal.utils import HasProject, JobName
import numpy as np
from pandas import DataFrame
from functools import lru_cache
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]
import matplotlib.pyplot as plt
import seaborn as sns

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

    @staticmethod
    def _get_peratom_energy(job_creator, potl_index=0, run_again=False, **other_job_kwargs):
        job = job_creator(potl_index=potl_index, delete_existing_job=run_again, **other_job_kwargs)
        job.run()
        return job.output.energy_pot[-1] / len(job.structure)

    @lru_cache()
    def get_BCC_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.BCC,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_FCC_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.FCC,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_B2_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.B2,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.D03,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @staticmethod
    def _get_size_converged_point_defect_energy(
            ref_energy_per_atom, job_creator, stderr=1e-3, potl_index=0, run_again=False, **other_job_kwargs
    ):
        reps = 0
        actual_err = np.inf
        energies = [np.inf]
        while actual_err > stderr:
            reps += 1
            job = job_creator(
                potl_index=potl_index,
                repeat=reps,
                delete_existing_job=run_again,
                **other_job_kwargs
            )
            job.run()
            energies.append(job.output.energy_pot[-1] - len(job.structure) * ref_energy_per_atom)
            actual_err = np.abs(energies[-1] - energies[-2])
        return energies[-1], reps, actual_err

    @lru_cache()
    def get_formation_energy(self, stderr=1e-3, run_again=False, potl_index=0, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_BCC_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.dilute,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_B2_antisite_energy_Fe_to_Al(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_B2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.B2_anti_Fe_to_Al,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_B2_antisite_energy_Al_to_Fe(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_B2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.B2_anti_Al_to_Fe,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_antisite_energy_Al_to_Fe(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.D03_anti_Al_to_Fe,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_antisite_energy_aFe_to_Al(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.D03_anti_aFe_to_Al,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_antisite_energy_bFe_to_Al(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.D03_anti_bFe_to_Al,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_solid_solution_repeats(self, potl_index=0, stderr=1e-3, run_again=False, n_trials=10, c_Al=None):
        c_Al = self.project.input.experimental_data['c_Al'] if c_Al is None else c_Al
        reps = 0
        actual_err = np.inf
        while actual_err > stderr:
            reps += 1
            if int(2 * reps ** 3 * c_Al) < 1:
                continue
            print(f"running {reps} repetitions...")
            energies = []
            for n in range(n_trials):
                print(f"\ttrial {n}")
                job = self.project.create.job.minimize.random_BCC(
                    potl_index=potl_index,
                    repeat=reps,
                    c_Al=c_Al,
                    trial=n,
                    delete_existing_job=run_again
                )
                job.run()
                energies.append(job.output.energy_pot[-1] / len(job.structure))
            actual_err = np.std(energies) / np.sqrt(n_trials)
            print(f"stderr = {actual_err}")
        return reps, actual_err

    def plot_phases_0K(self, potl_index=0, ax=None, beautify=True):
        (fig, ax) = plt.subplots() if ax is None else (None, ax)
        E_FCC = self.get_FCC_peratom_energy(potl_index=potl_index)
        E_BCC = self.get_BCC_peratom_energy(potl_index=potl_index)
        E_D03 = self.get_D03_peratom_energy(potl_index=potl_index)
        E_B2 = self.get_B2_peratom_energy(potl_index=potl_index)

        c_Al = [0, 0, 0.25, 0.5]
        energies = [E_FCC, E_BCC, E_D03, E_B2]
        ax.plot(c_Al, energies, marker='o')
        if beautify:
            ax.set_xlabel('$c_\mathrm{Al}$')
            ax.set_ylabel('$E$ [eV/atom]')
            for c, E, label in zip(c_Al, energies, ['FCC', 'BCC', 'D03', 'B2']):
                ax.annotate(label, (c, E))
        return ax

    @property
    def results(self):
        return self._results

    def jobname(self, potential, structure_routine, reps=None, concentration=None):
        potl_index = np.argwhere(self.project.input.potentials == potential)[0][0]
        sname = structure_routine.__name__
        return JobName(f'zerok').potl(potl_index).append(sname).reps(reps).c_Al(concentration)

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
                for routine in self._structure_routines
            }
            for potl in self.project.input.potentials
        }

    @property
    def _structure_routines(self):
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
        for potl, struct_based_dict in self._zerok.job_dict.items():
            for struct, job in struct_based_dict.items():
                df = df.append({
                    'potential': potl,
                    'structure': struct,
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

    def plot_stability(self, dmu=None, figsize=None):
        dmu = dmu if dmu is not None else np.linspace(-1, 2, 100)
        potentials = np.unique(self.table['potential'].values)
        fig, axes = plt.subplots(len(potentials), sharex=True, figsize=figsize if figsize is not None else (4, 8))
        for i, potl in enumerate(potentials):
            e_ref = self.get_energy(potl, 'BCC')
            for struct in ['B2', 'D03', 'random_BCC']:
                axes[i].plot(dmu, self.get_energy(potl, struct, dmu_Al=dmu) - e_ref, label=struct)
            axes[i].set_title(potl)
            axes[i].set_ylabel('$U_\mathrm{struct} - U_\mathrm{BCC}$ [eV/atom]')
            axes[i].axvline(0, color='k', linestyle='--')
            axes[i].axhline(0, color='k', linestyle='--')
        axes[0].legend()
        axes[-1].set_xlabel('$\mu_{Al} - \mu_{Fe}$ [eV]')
        fig.tight_layout()
        return fig, axes
