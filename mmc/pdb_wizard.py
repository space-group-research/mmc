#!/usr/bin/python3
#
# PDB Wizard v0.3.3
# copyright Adam Hogan 2021-2024
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import copy
import io
import os
import tempfile
import sys
from typing import Any, Iterator, Optional, TextIO

import numpy as np
from numpy import array, cos, pi, sin, sqrt

list_of_elements: list[str] = [
    "Ac",
    "Ag",
    "Al",
    "Am",
    "Ar",
    "As",
    "At",
    "Au",
    "B",
    "Ba",
    "Be",
    "Bh",
    "Bi",
    "Bk",
    "Br",
    "C",
    "Ca",
    "Cd",
    "Ce",
    "Cf",
    "Cl",
    "Cm",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Db",
    "Dy",
    "Er",
    "Es",
    "Eu",
    "F",
    "Fe",
    "Fm",
    "Fr",
    "Ga",
    "Gd",
    "Ge",
    "H",
    "He",
    "Hf",
    "Hg",
    "Ho",
    "Hs",
    "I",
    "In",
    "Ir",
    "K",
    "Kr",
    "La",
    "Li",
    "Lr",
    "Lu",
    "Md",
    "Mg",
    "Mn",
    "Mo",
    "Mt",
    "N",
    "Na",
    "Nb",
    "Nd",
    "Ne",
    "Ni",
    "No",
    "Np",
    "O",
    "Os",
    "P",
    "Pa",
    "Pb",
    "Pd",
    "Pm",
    "Po",
    "Pr",
    "Pt",
    "Pu",
    "Ra",
    "Rb",
    "Re",
    "Rf",
    "Rh",
    "Rn",
    "Ru",
    "S",
    "Sb",
    "Sc",
    "Se",
    "Sg",
    "Si",
    "Sm",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Te",
    "Th",
    "Ti",
    "Tl",
    "Tm",
    "U",
    "V",
    "W",
    "Xe",
    "Y",
    "Yb",
    "Zn",
    "Zr",
    "Da",
    "X",
]

element_masses: dict[str, float] = {
    "H": 1.00797,
    "He": 4.0026,
    "Li": 6.941,
    "Be": 9.01218,
    "B": 10.81,
    "C": 12.011,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.998403,
    "Ne": 20.179,
    "Na": 22.98977,
    "Mg": 24.305,
    "Al": 26.98154,
    "Si": 28.0855,
    "P": 30.97376,
    "S": 32.06,
    "Cl": 35.453,
    "K": 39.0983,
    "Ar": 39.948,
    "Ca": 40.08,
    "Sc": 44.9559,
    "Ti": 47.9,
    "V": 50.9415,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.847,
    "Ni": 58.7,
    "Co": 58.9332,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.72,
    "Ge": 72.59,
    "As": 74.9216,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.8,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.9059,
    "Zr": 91.22,
    "Nb": 92.9064,
    "Mo": 95.94,
    "Tc": 98,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.4,
    "Ag": 107.868,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.69,
    "Sb": 121.75,
    "I": 126.9045,
    "Te": 127.6,
    "Xe": 131.3,
    "Cs": 132.9054,
    "Ba": 137.33,
    "La": 138.9055,
    "Ce": 140.12,
    "Pr": 140.9077,
    "Nd": 144.24,
    "Pm": 145,
    "Sm": 150.4,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.9254,
    "Dy": 162.5,
    "Ho": 164.9304,
    "Er": 167.26,
    "Tm": 168.9342,
    "Yb": 173.04,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.9479,
    "W": 183.85,
    "Re": 186.207,
    "Os": 190.2,
    "Ir": 192.22,
    "Pt": 195.09,
    "Au": 196.9665,
    "Hg": 200.59,
    "Tl": 204.37,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Po": 209,
    "At": 210,
    "Rn": 222,
    "Fr": 223,
    "Ra": 226.0254,
    "Ac": 227.0278,
    "Pa": 231.0359,
    "Th": 232.0381,
    "Np": 237.0482,
    "U": 238.029,
    "Pu": 242,
    "Am": 243,
    "Bk": 247,
    "Cm": 247,
    "No": 250,
    "Cf": 251,
    "Es": 252,
    "Hs": 255,
    "Mt": 256,
    "Fm": 257,
    "Md": 258,
    "Lr": 260,
    "Rf": 261,
    "Bh": 262,
    "Db": 262,
    "Sg": 263,
    "Da": 0,
    "X": 0,
}

atomic_numbers: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
    "Da": 0,
    "X": 0,
}


class Atom:
    name: str
    element: str
    x: np.ndarray
    bond_r: float
    vdw: float
    charge: float
    alpha: float
    epsilon: float
    sigma: float
    c6: float
    c8: float
    c10: float
    mass: float
    atomic_number: int
    id: int

    def __init__(self, x, y, z, name) -> None:
        self.name = str(name).strip()
        self.x = array([float(x), float(y), float(z)])
        element = "".join([i for i in self.name[:2] if i.isalpha()])
        element = element.lower().capitalize()

        if element not in list_of_elements:
            element = element[0]
            if element not in list_of_elements:
                print(f"!!! Invalid element {name} !!!")

        self.element = element
        self.charge = 0.0
        self.alpha = 0.0
        self.epsilon = 0.0
        self.sigma = 0.0
        self.c6 = 0.0
        self.c8 = 0.0
        self.c10 = 0.0
        self.mass = element_masses[self.element]
        self.atomic_number = atomic_numbers[self.element]
        self.id = 0

        if element == "H":
            self.bond_r = 1.0
            self.vdw = 1.2
        elif element == "O":
            self.bond_r = 1.3
            self.vdw = 1.8
        elif element == "N" or element == "C":
            self.bond_r = 1.6
            self.vdw = 2.0
        elif self.atomic_number >= 11:
            self.bond_r = 2.2
            self.vdw = 3.4
        else:
            self.bond_r = 2.0
            self.vdw = 3.0

    def __str__(self) -> str:
        return f"Atom instance {self.element} {self.id}"


class PBC:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    inverse_volume: float
    basis_matrix: np.ndarray
    reciprocal_basis_matrix: np.ndarray

    def __init__(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        basis00 = a
        basis01 = 0.0
        basis02 = 0.0
        basis10 = b * cos(pi / 180.0 * gamma)
        basis11 = b * sin(pi / 180.0 * gamma)
        basis12 = 0.0
        basis20 = c * cos(pi / 180.0 * beta)
        basis21 = ((b * c * cos(pi / 180.0 * alpha)) - (basis10 * basis20)) / basis11
        basis22 = sqrt(c * c - basis20 * basis20 - basis21 * basis21)

        self.basis_matrix = array(
            [
                [basis00, basis01, basis02],
                [basis10, basis11, basis12],
                [basis20, basis21, basis22],
            ]
        )

        self.volume = basis00 * (basis11 * basis22 - basis12 * basis21)
        self.volume += basis01 * (basis12 * basis20 - basis10 * basis22)
        self.volume += basis02 * (basis10 * basis21 - basis11 * basis20)

        self.inverse_volume = 1.0 / self.volume

        reciprocal_basis00 = self.inverse_volume * (
            basis11 * basis22 - basis12 * basis21
        )
        reciprocal_basis01 = self.inverse_volume * (
            basis02 * basis21 - basis01 * basis22
        )
        reciprocal_basis02 = self.inverse_volume * (
            basis01 * basis12 - basis02 * basis11
        )
        reciprocal_basis10 = self.inverse_volume * (
            basis12 * basis20 - basis10 * basis22
        )
        reciprocal_basis11 = self.inverse_volume * (
            basis00 * basis22 - basis02 * basis20
        )
        reciprocal_basis12 = self.inverse_volume * (
            basis02 * basis10 - basis00 * basis12
        )
        reciprocal_basis20 = self.inverse_volume * (
            basis10 * basis21 - basis11 * basis20
        )
        reciprocal_basis21 = self.inverse_volume * (
            basis01 * basis20 - basis00 * basis21
        )
        reciprocal_basis22 = self.inverse_volume * (
            basis00 * basis11 - basis01 * basis10
        )

        self.reciprocal_basis_matrix = array(
            [
                [reciprocal_basis00, reciprocal_basis01, reciprocal_basis02],
                [reciprocal_basis10, reciprocal_basis11, reciprocal_basis12],
                [reciprocal_basis20, reciprocal_basis21, reciprocal_basis22],
            ]
        )

    def update(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        basis00 = a
        basis01 = 0.0
        basis02 = 0.0
        basis10 = b * cos(pi / 180.0 * gamma)
        basis11 = b * sin(pi / 180.0 * gamma)
        basis12 = 0.0
        basis20 = c * cos(pi / 180.0 * beta)
        basis21 = ((b * c * cos(pi / 180.0 * alpha)) - (basis10 * basis20)) / basis11
        basis22 = sqrt(c * c - basis20 * basis20 - basis21 * basis21)

        self.basis_matrix = array(
            [
                [basis00, basis01, basis02],
                [basis10, basis11, basis12],
                [basis20, basis21, basis22],
            ]
        )

        self.volume = basis00 * (basis11 * basis22 - basis12 * basis21)
        self.volume += basis01 * (basis12 * basis20 - basis10 * basis22)
        self.volume += basis02 * (basis10 * basis21 - basis11 * basis20)

        self.inverse_volume = 1.0 / self.volume

        reciprocal_basis00 = self.inverse_volume * (
            basis11 * basis22 - basis12 * basis21
        )
        reciprocal_basis01 = self.inverse_volume * (
            basis02 * basis21 - basis01 * basis22
        )
        reciprocal_basis02 = self.inverse_volume * (
            basis01 * basis12 - basis02 * basis11
        )
        reciprocal_basis10 = self.inverse_volume * (
            basis12 * basis20 - basis10 * basis22
        )
        reciprocal_basis11 = self.inverse_volume * (
            basis00 * basis22 - basis02 * basis20
        )
        reciprocal_basis12 = self.inverse_volume * (
            basis02 * basis10 - basis00 * basis12
        )
        reciprocal_basis20 = self.inverse_volume * (
            basis10 * basis21 - basis11 * basis20
        )
        reciprocal_basis21 = self.inverse_volume * (
            basis01 * basis20 - basis00 * basis21
        )
        reciprocal_basis22 = self.inverse_volume * (
            basis00 * basis11 - basis01 * basis10
        )

        self.reciprocal_basis_matrix = array(
            [
                [reciprocal_basis00, reciprocal_basis01, reciprocal_basis02],
                [reciprocal_basis10, reciprocal_basis11, reciprocal_basis12],
                [reciprocal_basis20, reciprocal_basis21, reciprocal_basis22],
            ]
        )

    def min_image(self, dx: np.ndarray) -> np.ndarray:
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.round(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        r = np.sqrt(np.dot(dx_return, dx_return))
        return r

    def wrap(self, dx: np.ndarray) -> np.ndarray:
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.round(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        return dx_return

    def wrap_forward(self, dx: np.ndarray) -> np.ndarray:
        img = np.matmul(dx, self.reciprocal_basis_matrix)
        img = np.floor(img)
        di = np.matmul(img, self.basis_matrix)
        dx_return = dx - di
        return dx_return


def progressbar(
    it: list, prefix: str = "", size: int = 60, out: TextIO = sys.stdout
) -> Iterator:
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(
            f"{prefix}[{'█' * x}{'.' * (size - x)}] {j}/{count}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0)

    for i, item in enumerate(it):
        yield item
        show(i + 1)

    print("", flush=True, file=out)


def get_forcefield(name: int) -> Any:
    ffs = []
    opls_aa_uff = {
        "H": {
            "alpha": 0.41380,
            "sigma": 2.42,
            "epsilon": 15.11,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "C": {
            "alpha": 1.2866,
            "sigma": 3.55,
            "epsilon": 35.25,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "N": {
            "alpha": 0.97157,
            "sigma": 3.25000,
            "epsilon": 85.60000,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "O": {
            "alpha": 0.852,
            "sigma": 3.118,
            "epsilon": 30.19,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "P": {
            "alpha": 3.35,
            "sigma": 3.69456,
            "epsilon": 153.48197,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Cl": {
            "alpha": 2.40028,
            "sigma": 3.516377,
            "epsilon": 114.23084,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Zn": {
            "alpha": 1.98870,
            "sigma": 2.46155,
            "epsilon": 62.39923,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Pt": {
            "alpha": 8.56281,
            "sigma": 2.45353,
            "epsilon": 40.25756,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Pd": {
            "alpha": 5.25926,
            "sigma": 2.5827,
            "epsilon": 24.1545,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "B": {
            "alpha": 0.6634,
            "sigma": 3.63754,
            "epsilon": 90.57952,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Ni": {
            "alpha": 0.38980,
            "sigma": 2.52480,
            "epsilon": 7.55330,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Cr": {
            "alpha": 3.50740,
            "sigma": 2.69320,
            "epsilon": 7.55330,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Cu": {
            "alpha": 2.19630,
            "sigma": 3.11400,
            "epsilon": 2.51600,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Co": {
            "alpha": 3.26440,
            "sigma": 2.55870,
            "epsilon": 7.04980,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "F": {
            "alpha": 0.444747,
            "sigma": 2.996983,
            "epsilon": 25.160979,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Si": {
            "alpha": 2.133000,
            "sigma": 3.826410,
            "epsilon": 202.294269,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Br": {
            "alpha": 3.493000,
            "sigma": 3.732000,
            "epsilon": 126.392600,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "Ti": {
            "alpha": 3.24280,
            "sigma": 2.82860,
            "epsilon": 8.56050,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "In": {
            "alpha": 2.00000,
            "sigma": 3.97600,
            "epsilon": 301.40000,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "W": {
            "alpha": 3.65453,
            "sigma": 2.73420,
            "epsilon": 33.73830,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
        "S": {
            "alpha": 2.474448,
            "sigma": 3.594776,
            "epsilon": 137.882163,
            "c6": 0.0,
            "c8": 0.0,
            "c10": 0.0,
        },
    }
    ffs.append(opls_aa_uff)

    phahst = {
        "Cu": {
            "alpha": 0.29252,
            "sigma": 2.73851,
            "epsilon": 8.82345,
            "c6": 6.96956,
            "c8": 262.82938,
            "c10": 13951.49740,
        },
        "C": {
            "alpha": 0.71317,
            "sigma": 3.35929,
            "epsilon": 4.00147,
            "c6": 11.88969,
            "c8": 547.51694,
            "c10": 27317.97855,
        },
        "O": {
            "alpha": 1.68064,
            "sigma": 3.23867,
            "epsilon": 3.89544,
            "c6": 27.70093,
            "c8": 709.36452,
            "c10": 19820.89339,
        },
        "H": {
            "alpha": 0.02117,
            "sigma": 1.87446,
            "epsilon": 3.63874,
            "c6": 0.16278,
            "c8": 5.03239,
            "c10": 202.99322,
        },
    }
    ffs.append(phahst)
    return ffs[name]


def set_atom_ids(system: list[Atom]) -> None:
    for ind, atom in enumerate(system):
        atom.id = ind + 1


def gcd_list(my_list: list) -> Any:
    result = my_list[0]
    for x in my_list[1:]:
        if result < x:
            temp = result
            result = x
            x = temp
        while x != 0:
            temp = x
            x = result % x
            result = temp
    return result


def read_pdb(file: TextIO) -> tuple[list[Atom], Optional[PBC]]:
    line = file.readline()
    pbc = None
    system = []
    try:
        for line in progressbar(file.readlines(), f"Reading file {sys.argv[1]} "):
            tokens = line.split()

            # ignore these lines
            if (
                tokens[0] == "REMARK"
                or tokens[0] == "HEADER"
                or tokens[0] == "MODEL"
                or tokens[0] == "TER"
                or tokens[0] == "TITLE"
                or tokens[0] == "AUTHOR"
                or tokens[0] == "EXPDTA"
                or tokens[0] == "SEQRES"
                or tokens[0] == "ANISOU"
                or tokens[0] == "SCALE1"
                or tokens[0] == "SCALE2"
                or tokens[0] == "SCALE3"
            ):
                continue

            # stop at END or ENDMDL
            if tokens[0] == "END" or tokens[0] == "ENDMDL":
                break

            # create atoms
            if tokens[0] == "ATOM" or tokens[0] == "HETATM":
                atom = Atom(line[30:38], line[38:46], line[46:54], line[12:16])
                if atom.atomic_number > 0:
                    system.append(atom)

            # load box
            if tokens[0] == "CRYST1":
                a = float(tokens[1])
                b = float(tokens[2])
                c = float(tokens[3])
                alpha = float(tokens[4])
                beta = float(tokens[5])
                gamma = float(tokens[6])
                pbc = PBC(a, b, c, alpha, beta, gamma)

    except ValueError:
        sys.exit(f"Error reading line:\n{line}\n")

    print("")
    set_atom_ids(system)
    file.close()
    return system, pbc


def read_xyz(file: TextIO) -> tuple[list[Atom], Optional[PBC]]:
    file.readline()
    line = file.readline()
    pbc = None
    system = []

    try:
        tokens = line.split()
        if len(tokens) != 6:
            raise ValueError
        a = float(tokens[0])
        b = float(tokens[1])
        c = float(tokens[2])
        alpha = float(tokens[3])
        beta = float(tokens[4])
        gamma = float(tokens[5])
        pbc = PBC(a, b, c, alpha, beta, gamma)
    except ValueError:
        print("Couldn't locate a b c alpha beta gamma on second line of .xyz file")

    try:
        for line in progressbar(file.readlines(), f"Reading file {sys.argv[1]} "):
            # ignore blank lines
            if line == "" or line == "\n":
                continue

            tokens = line.split()
            atom = Atom(tokens[1], tokens[2], tokens[3], tokens[0])

            # attempt to read the fifth column as the charge, pass otherwise
            try:
                charge = tokens[4]
                atom.charge = float(charge)
            except (ValueError, IndexError):
                pass

            system.append(atom)

    except ValueError:
        sys.exit(f"Error reading line:\n{line}")

    set_atom_ids(system)
    file.close()
    return system, pbc


def overlap_detector(system: list[Atom], pbc: PBC) -> list[Atom]:
    messages = []
    overlapping_atoms = True
    while overlapping_atoms:
        overlapping_atoms = False
        for atom in progressbar(system, "Detecting overlapping atoms "):
            for atom2 in system:
                if atom.id != atom2.id and not overlapping_atoms:
                    dx = atom.x - atom2.x
                    r = pbc.min_image(dx)
                    if r < 0.05:
                        overlapping_atoms = True
                        messages.append(
                            f"Deleting overlapping atoms\n"
                            f"{atom.element:>3} {atom.id:>5} --- {atom2.element:>3} {atom2.id:>5}\n"
                            f"{atom.x}\n {atom2.x}"
                        )
                        system.remove(atom2)
    set_atom_ids(system)
    for message in messages:
        print(message)
    return system


def list_close_contacts(system: list[Atom], pbc: PBC) -> None:
    messages = []
    for atom in progressbar(system):
        for atom2 in system:
            if atom2.id > atom.id:
                dx = atom.x - atom2.x
                r = pbc.min_image(dx)
                vdw = 0.5 * (atom.vdw + atom2.vdw)
                bond_r = 0.5 * (atom.bond_r + atom2.bond_r)
                if vdw > r > bond_r:
                    element_string = f"{atom.element}-{atom2.element}"
                    messages.append(
                        f"{element_string:<5} {atom.id:>5} {atom2.id:>5}   r = {np.round(r, 6)}"
                    )
    print("\nClose contacts:\n")
    for message in messages:
        print(message)


def list_bonds(system: list[Atom], pbc: PBC) -> None:
    messages = []
    for atom in progressbar(system):
        for atom2 in system:
            if atom2.id > atom.id:
                dx = atom.x - atom2.x
                r = pbc.min_image(dx)
                bond_r = 0.5 * (atom.bond_r + atom2.bond_r)
                if r < bond_r:
                    element_string = f"{atom.element}-{atom2.element}"
                    messages.append(
                        f"{element_string:<5} {atom.id:>5} {atom2.id:>5}   r = {np.round(r, 6)}"
                    )
    print("\nBonded atoms:\n")
    for message in messages:
        print(message)


def list_angles(system: list[Atom], pbc: PBC) -> None:
    messages = []
    mols = find_molecules(system, pbc)
    for mol in progressbar(mols):
        for atom in mol:
            for atom2 in mol:
                for atom3 in mol:
                    if (
                        atom2.id == atom.id
                        or atom3.id == atom2.id
                        or atom3.id == atom.id
                    ):
                        continue
                    dx1 = atom2.x - atom.x
                    r1 = pbc.min_image(dx1)
                    bond_r1 = 0.5 * (atom.bond_r + atom2.bond_r)
                    dx2 = atom2.x - atom3.x
                    r2 = pbc.min_image(dx2)
                    bond_r2 = 0.5 * (atom2.bond_r + atom3.bond_r)
                    if r1 < bond_r1 and r2 < bond_r2:
                        u_dx1 = pbc.wrap(dx1)
                        u_dx2 = pbc.wrap(dx2)
                        u_dx1 = u_dx1 / np.sqrt(np.dot(u_dx1, u_dx1))
                        u_dx2 = u_dx2 / np.sqrt(np.dot(u_dx2, u_dx2))
                        bond_angle = np.degrees(
                            np.arccos(np.clip(np.dot(u_dx1, u_dx2), -1.0, 1.0))
                        )
                        element_string = (
                            f"{atom.element}-{atom2.element}-{atom3.element}"
                        )
                        messages.append(
                            f"{element_string:<7} {atom.id:>5} {atom2.id:>5} {atom3.id:>5}   "
                            f"angle = {np.round(bond_angle, 2):>6} r1, r2 = {np.round(r1, 3):>6}, {np.round(r2, 3):>6}"
                        )
    print("\nAngles:\n")
    for message in messages:
        print(message)


def list_lone_atoms(system: list[Atom], pbc: PBC) -> None:
    lone_atoms = []
    for atom in progressbar(system):
        lone_atom = True
        for atom2 in system:
            if atom2.id != atom.id:
                dx = atom.x - atom2.x
                r = pbc.min_image(dx)
                vdw = 0.5 * (atom.vdw + atom2.vdw)
                if r < vdw:
                    lone_atom = False
        if lone_atom:
            lone_atoms.append(atom)
    if len(lone_atoms) == 0:
        print("\nNo lone atoms found\n")
    else:
        print("\nLone atoms:\n")
        for atom in lone_atoms:
            print(f"{atom.element:>3} {atom.id:>5} {atom.x}")


def delete_lone_atoms(system: list[Atom], pbc: PBC) -> list[Atom]:
    lone_atoms = []
    for atom in progressbar(system):
        lone_atom = True
        for atom2 in system:
            if atom2.id != atom.id:
                dx = atom.x - atom2.x
                r = pbc.min_image(dx)
                vdw = 0.5 * (atom.vdw + atom2.vdw)
                if r < vdw:
                    lone_atom = False
        if lone_atom:
            lone_atoms.append(atom)
    if len(lone_atoms) == 0:
        print("\nNo lone atoms found\n")
    else:
        print("\nDeleting lone atoms:\n")
        for atom in lone_atoms:
            print("{atom.element:>3} {atom.id:>5} {atom.x}")
            system.remove(atom)
    return system


def write_xyz(system: list[Atom], pbc: PBC, out: TextIO) -> None:
    out.write(str(len(system)))
    out.write(f"\n{pbc.a} {pbc.b} {pbc.c} {pbc.alpha} {pbc.beta} {pbc.gamma}\n")
    for atom in system:
        out.write(f"{atom.element} {atom.x[0]} {atom.x[1]} {atom.x[2]}\n")


def write_standard_pdb(system: list[Atom], pbc: PBC, out: TextIO, skip_mols_step: bool = False) -> None:
    if skip_mols_step:
        mols = [system]
    else:
        system = sort(system, pbc)
        mols = find_molecules(system, pbc)

    out.write("MODEL        1\n")
    out.write(f"COMPND    {'':<69}\n")
    out.write("AUTHOR    GENERATED BY PDB WIZARD\n")
    out.write(
        f"CRYST1  {round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
        f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6} P 1           1\n"
    )

    atom_id = 1
    for idx, mol in enumerate(mols):
        mol_name = "UNK"
        
        mol_elements = [atom.element for atom in mol]
        
        if len(mol_elements) == 1:
            mol_name = mol_elements[0].upper()
        
        if not skip_mols_step:
            mol.sort(key=lambda atom: atom.element)
            mol_elements = [atom.element for atom in mol]
            if mol_elements == ["H", "H", "O"]:
                mol_name = "HOH"
            elif mol_elements == ["H", "H"]:
                mol_name = "H2"
            elif mol_elements == ["H", "H", "H", "H", "C"]:
                mol_name = "MET"
            elif mol_elements == ["N", "N"]:
                mol_name = "N2"
            elif mol_elements == ["H", "H", "C", "C"]:
                mol_name = "ACE"
            elif mol_elements == ["H", "H", "H", "H", "C", "C"]:
                mol_name = "ENE"
            elif mol_elements == ["H", "H", "H", "H", "H", "H", "C", "C"]:
                mol_name = "ETH"
            elif mol_elements == ["Zn"]:
                mol_name = "ZNA"
            elif mol_elements == ["Cl", "Cl", "Cl", "Cl", "Zn"]:
                mol_name = "ZNC"

        base_atom = mol[-1]
        additional_tag = 1

        for atom in mol:
            atom.id = atom_id

            other_elements = [
                other_atom.element for other_atom in mol if atom is not other_atom
            ]
            if atom.element in other_elements and not skip_mols_step:
                atom.name = atom.element + str(additional_tag)
            else:
                atom.name = atom.element

            dx = atom.x - base_atom.x

            dx = pbc.wrap(dx)

            atom.x = base_atom.x + dx

            out.write(
                f"HETATM {atom.id:>4}  {atom.name:<3} {mol_name:>3} A {idx + 1:>4}    "
                f"{round(atom.x[0], 3):>7} {round(atom.x[1], 3):>7} {round(atom.x[2], 3):>7}  1.00  0.00          "
                f"{atom.element:>2}\n"
            )

            atom_id += 1
            additional_tag += 1

    out.write("END\n")
    set_atom_ids(system)


def find_molecules(system: list[Atom], pbc: PBC) -> list[list[Atom]]:
    set_atom_ids(system)
    bonds = {}
    for atom1 in progressbar(system, "Finding molecules "):
        for atom2 in system:
            if atom2.id != atom1.id:
                dx = atom1.x - atom2.x
                r = pbc.min_image(dx)
                bond_r = 0.5 * (atom1.bond_r + atom2.bond_r)
                if r < bond_r:
                    if atom1.id not in bonds.keys():
                        bonds[atom1.id] = [atom2.id]
                    else:
                        if atom2.id not in bonds[atom1.id]:
                            bonds[atom1.id].append(atom2.id)
                    if atom2.id not in bonds.keys():
                        bonds[atom2.id] = [atom1.id]
                    else:
                        if atom1.id not in bonds[atom2.id]:
                            bonds[atom2.id].append(atom1.id)
    
    mols_by_atom_id = []
    
    for atom1 in system:
    
        atom_already_in_mol = False
        
        for mol in mols_by_atom_id:
            if atom1.id in mol:
               atom_already_in_mol = True
               
        if atom_already_in_mol:
            continue
        
        if atom1.id not in bonds.keys():
            mols_by_atom_id.append([atom1.id])
            continue
        
        new_mol = [atom1.id]
        
        while True:
            old_num_of_atoms_in_mol = len(new_mol)
            
            for current_id in new_mol:
                for new_id in bonds[current_id]:
                    if new_id not in new_mol:
                        new_mol.append(new_id)
            
            if len(new_mol) == old_num_of_atoms_in_mol:
                break
        
        mols_by_atom_id.append(new_mol)
    
    mols = [[system[atom_id-1] for atom_id in mol] for mol in mols_by_atom_id]

    return mols


def apply_ff_to_system(system: list[Atom], ff) -> list[Atom]:
    for atom in system:
        try:
            atom.alpha = ff[atom.element]["alpha"]
            atom.epsilon = ff[atom.element]["epsilon"]
            atom.sigma = ff[atom.element]["sigma"]
            atom.c6 = ff[atom.element]["c6"]
            atom.c8 = ff[atom.element]["c8"]
            atom.c10 = ff[atom.element]["c10"]
        except KeyError:
            print(
                f"!!! atom {atom.element} not found in forcefield, parameters set to all zeros !!!"
            )
            atom.alpha = 0.0
            atom.epsilon = 0.0
            atom.sigma = 0.0
            atom.c6 = 0.0
            atom.c8 = 0.0
            atom.c10 = 0.0
    return system


def write_mpmc_pdb(
    system: list[Atom],
    pbc: PBC,
    filename: str,
    write_charges: bool = False,
    write_params: bool = False,
) -> None:
    system = sort(system, pbc)
    out = open(filename, "w")
    out.write("MODEL        1\n")
    out.write(f"COMPND    {'':<69}\n")
    out.write("AUTHOR    GENERATED BY PDB WIZARD\n")
    out.write(
        f"CRYST1  {round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
        f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6} P 1           1\n"
    )
    for idx, atom in enumerate(progressbar(system)):
        out.write(
            f"ATOM {idx+1:>6} {atom.element:<4} MOF F    1    "
            f"{round(atom.x[0], 3):>7} {round(atom.x[1], 3):>7} {round(atom.x[2], 3):>7}"
        )
        if write_params is True or write_params is True:
            out.write(f" {atom.mass:>9.6}")
            out.write(f" {atom.charge:>8.4}")
        if write_params is True:
            out.write(
                f" {atom.alpha:>8.4} {atom.epsilon:>8.4} {atom.sigma:>8.4} "
                f"0.0 0.0 {atom.c6:>8.4} {atom.c8:>10.4} {atom.c10:>10.2}\n"
            )
        else:
            out.write(f" xxx{atom.name.strip()}xxx\n")
    borders = [
        array([0, 0, 0]),
        array([1, 0, 0]),
        array([0, 1, 0]),
        array([0, 0, 1]),
        array([1, 1, 0]),
        array([1, 0, 1]),
        array([0, 1, 1]),
        array([1, 1, 1]),
    ]
    for ind, pos in enumerate(borders):
        border_pos = np.matmul(pos, pbc.basis_matrix)
        out.write(
            f"ATOM {len(system) + ind + 1:>6} {'X':<4} BOX F    2    {round(border_pos[0], 3):>7} "
            f"{round(border_pos[1], 3):>7} {round(border_pos[2], 3):>7} 0.0 0.0 0.0 0.0 0.0\n"
        )
    connections = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [1, 5],
        [2, 4],
        [2, 6],
        [4, 7],
        [5, 7],
        [6, 7],
        [3, 6],
        [3, 5],
    ]
    for connection in connections:
        out.write(
            f"CONECT {len(system) + connection[0]:>4} {len(system) + connection[1]:>4}\n"
        )
    out.write(
        f"REMARK BOX BASIS[0] = {pbc.basis_matrix[0][0]:20.14f} {pbc.basis_matrix[0][1]:20.14f} "
        f"{pbc.basis_matrix[0][2]:20.14f}\n"
    )
    out.write(
        f"REMARK BOX BASIS[1] = {pbc.basis_matrix[1][0]:20.14f} {pbc.basis_matrix[1][1]:20.14f} "
        f"{pbc.basis_matrix[1][2]:20.14f}\n"
    )
    out.write(
        f"REMARK BOX BASIS[2] = {pbc.basis_matrix[2][0]:20.14f} {pbc.basis_matrix[2][1]:20.14f} "
        f"{pbc.basis_matrix[2][2]:20.14f}\n"
    )
    out.write("END\n")
    out.close()
    print(f"Wrote {filename}")


def print_info(system: list[Atom], pbc: PBC, filename: str) -> None:
    print("")
    print(r"   ___  ___  ___   __    __ _                  _ ")
    print(r"  / _ \/   \/ __\ / / /\ \ (_)______ _ _ __ __| |")
    print(r" / /_)/ /\ /__\// \ \/  \/ / |_  / _` | '__/ _` |")
    print(r"/ ___/ /_// \/  \  \  /\  /| |/ / (_| | | | (_| |")
    print(r"\/  /___,'\_____/   \/  \/ |_/___\__,_|_|  \__,_|")
    print(f"\nfilename: {filename}")
    print(
        f"\nCell:\n{round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
        f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6}\n"
    )
    print(
        f"{pbc.basis_matrix[0][0]:20.14f} {pbc.basis_matrix[0][1]:20.14f} {pbc.basis_matrix[0][2]:20.14f}"
    )
    print(
        f"{pbc.basis_matrix[1][0]:20.14f} {pbc.basis_matrix[1][1]:20.14f} {pbc.basis_matrix[1][2]:20.14f}"
    )
    print(
        f"{pbc.basis_matrix[2][0]:20.14f} {pbc.basis_matrix[2][1]:20.14f} {pbc.basis_matrix[2][2]:20.14f}"
    )
    print(
        f"Volume: {pbc.volume:10.2f} A^3 Density: "
        f"{sum([atom.mass for atom in system]) * 1.66054 / pbc.volume:10.4} g/cm^3"
    )
    print_formula_unit(system)


def print_info_movie(systems: list[list[Atom]], pbcs: list[PBC], filename: str) -> None:
    print("")
    print(r"   ___  ___  ___   __    __ _                  _ ")
    print(r"  / _ \/   \/ __\ / / /\ \ (_)______ _ _ __ __| |")
    print(r" / /_)/ /\ /__\// \ \/  \/ / |_  / _` | '__/ _` |")
    print(r"/ ___/ /_// \/  \  \  /\  /| |/ / (_| | | | (_| |")
    print(r"\/  /___,'\_____/   \/  \/ |_/___\__,_|_|  \__,_|")
    print("")
    print(
        f"Trajectory detected\n{len(systems)} frames\nExtract a single frame for more detailed options"
    )


def print_formula_unit(system: list[Atom]) -> None:
    atom_dict = {}
    for atom in system:
        if atom.element not in atom_dict:
            atom_dict[atom.element] = 1
        else:
            atom_dict[atom.element] += 1
    print("\nTotal number of atoms:\n")
    for ele in atom_dict:
        print(f"{ele} {atom_dict[ele]}")
    atom_n = [atom_dict[i] for i in atom_dict]
    atoms_gcd = gcd_list(atom_n)
    print("\nFormula unit\n")
    for ele in atom_dict:
        print(f"{ele} {int(atom_dict[ele] / atoms_gcd)}")


def sort(system: list[Atom], pbc: PBC) -> list[Atom]:
    atom_sorts = [
        {"name": "element", "key": lambda atom: atom.element, "reverse": False}
    ]

    mol_sorts = [
        {
            "name": "first atom's element",
            "key": lambda mol: mol[0].element,
            "reverse": False,
        },
        {"name": "length of molecule", "key": lambda mol: len(mol), "reverse": True},
        {
            "name": "if molecule contains heavy atoms",
            "key": lambda mol: 0 if np.max([atom.mass for atom in mol]) > 16 else 1,
            "reverse": False,
        },
    ]

    print("sorting inside molecules:")

    for sorting_options in reversed(atom_sorts):
        print(
            f"    by {sorting_options['name']} - reverse: {sorting_options['reverse']}"
        )

    print("sorting molecules:")

    for sorting_options in reversed(mol_sorts):
        print(
            f"    by {sorting_options['name']} - reverse: {sorting_options['reverse']}"
        )

    mols = find_molecules(system, pbc)

    for mol in mols:
        for sorting_options in atom_sorts:
            mol.sort(key=sorting_options["key"], reverse=sorting_options["reverse"])

    for sorting_options in mol_sorts:
        mols.sort(key=sorting_options["key"], reverse=sorting_options["reverse"])

    system = []
    for mol in mols:
        for atom in mol:
            system.append(atom)

    return system


def wrapall_forward(system: list[Atom], pbc: PBC) -> list[Atom]:
    for atom in progressbar(system):
        atom.x = pbc.wrap_forward(atom.x)
    print("\nWrapped atoms forward of origin")
    return system


def wrapall(system: list[Atom], pbc: PBC) -> list[Atom]:
    for atom in progressbar(system):
        atom.x = pbc.wrap(atom.x)
    print("\nWrapped atoms around origin")
    return system


def menu_single_extend_axis(system: list[Atom], pbc: PBC) -> tuple[list[Atom], PBC]:
    print(
        f"\nCurrent cell:\n{round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
        f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6}\n"
    )
    print(
        f"{pbc.basis_matrix[0][0]:20.14f} {pbc.basis_matrix[0][1]:20.14f} {pbc.basis_matrix[0][2]:20.14f}"
    )
    print(
        f"{pbc.basis_matrix[1][0]:20.14f} {pbc.basis_matrix[1][1]:20.14f} {pbc.basis_matrix[1][2]:20.14f}"
    )
    print(
        f"{pbc.basis_matrix[2][0]:20.14f} {pbc.basis_matrix[2][1]:20.14f} {pbc.basis_matrix[2][2]:20.14f}"
    )
    while True:
        try:
            axis = input(
                "\nWhat axis would you like to extend? 0, 1, 2 or x, y, z or q(uit)\n\n> "
            )
            if axis.lower() == "q" or axis.lower() == "quit":
                return system, pbc
            if axis.lower() == "x":
                axis = 0
            elif axis.lower() == "y":
                axis = 1
            elif axis.lower() == "z":
                axis = 2
            axis = int(axis)
            if axis < 0 or axis > 2:
                raise ValueError
            times = input("\nHow many times would you like to extend it?\n\n> ")
            times = int(times)
            if times < 1:
                raise ValueError
            break
        except ValueError:
            print("!!! Error converting input to int or x, y, z !!!")

    new_atoms = []
    for i in np.arange(times):
        for atom in system:
            new_atom = copy.deepcopy(atom)
            if axis == 0:
                new_atom.x += (i + 1) * pbc.basis_matrix[0]
            elif axis == 1:
                new_atom.x += (i + 1) * pbc.basis_matrix[1]
            elif axis == 2:
                new_atom.x += (i + 1) * pbc.basis_matrix[2]
            new_atoms.append(new_atom)

    for atom in new_atoms:
        system.append(atom)

    if axis == 0:
        pbc.update((times + 1) * pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma)
    elif axis == 1:
        pbc.update(pbc.a, (times + 1) * pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma)
    elif axis == 2:
        pbc.update(pbc.a, pbc.b, (times + 1) * pbc.c, pbc.alpha, pbc.beta, pbc.gamma)
    set_atom_ids(system)

    return system, pbc


def menu_move_extend_axis(
    systems: list[list[Atom]], pbcs: list[PBC]
) -> tuple[list[list[Atom]], list[PBC]]:
    pbc = pbcs[0]
    print(
        f"\nFirst cell in trajectory:\n{round(pbc.a, 3):>7}  {round(pbc.b, 3):7}  {round(pbc.c, 3):7} "
        f"{round(pbc.alpha, 2):>6} {round(pbc.beta, 2):>6} {round(pbc.gamma, 2):>6}\n"
    )
    print(
        f"{pbc.basis_matrix[0][0]:20.14f} {pbc.basis_matrix[0][1]:20.14f} {pbc.basis_matrix[0][2]:20.14f}"
    )
    print(
        f"{pbc.basis_matrix[1][0]:20.14f} {pbc.basis_matrix[1][1]:20.14f} {pbc.basis_matrix[1][2]:20.14f}"
    )
    print(
        f"{pbc.basis_matrix[2][0]:20.14f} {pbc.basis_matrix[2][1]:20.14f} {pbc.basis_matrix[2][2]:20.14f}"
    )
    while True:
        try:
            axis = input(
                "\nWhat axis would you like to extend? 0, 1, 2 or x, y, z or q(uit)\n\n> "
            )
            if axis.lower() == "q" or axis.lower() == "quit":
                return systems, pbcs
            if axis.lower() == "x":
                axis = 0
            elif axis.lower() == "y":
                axis = 1
            elif axis.lower() == "z":
                axis = 2
            axis = int(axis)
            if axis < 0 or axis > 2:
                raise ValueError
            times = input("\nHow many times would you like to extend it?\n\n> ")
            times = int(times)
            if times < 1:
                raise ValueError
            break
        except ValueError:
            print("!!! Error converting input to int or x, y, z !!!")

    for system, pbc in zip(systems, pbcs):
        new_atoms = []
        for i in np.arange(times):
            for atom in system:
                new_atom = copy.deepcopy(atom)
                if axis == 0:
                    new_atom.x += (i + 1) * pbc.basis_matrix[0]
                elif axis == 1:
                    new_atom.x += (i + 1) * pbc.basis_matrix[1]
                elif axis == 2:
                    new_atom.x += (i + 1) * pbc.basis_matrix[2]
                new_atoms.append(new_atom)

        for atom in new_atoms:
            system.append(atom)
        if axis == 0:
            pbc.update(
                (times + 1) * pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma
            )
        elif axis == 1:
            pbc.update(
                pbc.a, (times + 1) * pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma
            )
        elif axis == 2:
            pbc.update(
                pbc.a, pbc.b, (times + 1) * pbc.c, pbc.alpha, pbc.beta, pbc.gamma
            )
        set_atom_ids(system)
    return systems, pbcs


def list_coords(system: list[Atom]) -> None:
    for atom in system:
        print(f"{atom.element} {atom.x}")


def vmd_preview(system: list[Atom], pbc: PBC) -> None:
    write_mpmc_pdb(system, pbc, "pdb_wizard.tmp.pdb")
    os.system("vmd pdb_wizard.tmp.pdb")
    os.system("rm pdb_wizard.tmp.pdb")


def edit_h_dist(system: list[Atom], pbc: PBC) -> list[Atom]:
    while True:
        second_element = "XX"
        distance = 0
        try:
            second_element = input(
                "\nLook for hydrogens bonded with which element? (e.g. C, O, N, etc)\n\n> "
            )
            if second_element not in list_of_elements:
                raise ValueError
            distance = input(
                f"\nWhat distance (in angstroms) shall {second_element}-H bonds be set to?\n\n> "
            )
            distance = float(distance)
            break
        except ValueError:
            print("!!! Error finding element or reading distance !!!")
    messages = []
    for atom in progressbar(system):
        for atom2 in system:
            if atom2.id > atom.id:
                if (atom.element == "H" and atom2.element == second_element) or (
                    atom2.element == "H" and atom.element == second_element
                ):
                    if atom.element == "H":
                        h_atom = atom
                        other_atom = atom2
                    else:
                        h_atom = atom2
                        other_atom = atom
                    dx = h_atom.x - other_atom.x
                    r = pbc.min_image(dx)
                    bond_r = 0.5 * (atom.bond_r + atom2.bond_r)
                    if r < bond_r:
                        element_string = f"{atom.element}-{atom2.element}"
                        messages.append(
                            f"{element_string:<5} {atom.id:>5} {atom2.id:>5}"
                        )
                        dx = pbc.wrap(dx)
                        dx *= distance / r
                        h_atom.x = other_atom.x + dx

    for message in messages:
        print(message)

    return system


def write_mpmc_options(system: list[Atom], pbc: PBC) -> None:
    write_charges: int
    write_force_field: int
    force_field: int

    while True:
        try:
            write_charges_in: str = input(
                "\nWould you like to read in charges?\n"
                "('yes', 'y', 1 or 'no', 'n', 0)\n\n> "
            )
            if write_charges_in == "yes" or write_charges_in == "y":
                write_charges = 1
            elif write_charges_in == "no" or write_charges_in == "n":
                write_charges = 0
            else:
                write_charges = int(write_charges_in)
            if write_charges > 1 or write_charges < 0:
                raise ValueError
            break
        except ValueError:
            print("!!! Error reading input !!!")

    if write_charges == 1:
        while True:
            print("\nEnter a resp file or a valid column of raw charges")
            charges_filename = input("charges file name > ")
            try:
                charges = []
                if charges_filename.endswith(".resp"):
                    print("\nYou have entered a file ending with .resp. This program expects the resp format.")
                    print("If you have a column of raw charges, use any other file extension name.\n")
                    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tmp:
                        with open(charges_filename, "r") as file:
                           # Skip the lines
                            # RESP charges:
                            # Type |   Atom   |    Charge
                            next(file)
                            next(file)
                            for line in file:
                                if line.strip():
                                    words = line.split()
                                    if words[0] == 'Total':
                                        continue
                                    tmp.write(words[-1] + '\n')

                        charges_filename = tmp.name

                for line in open(charges_filename, "r").readlines():
                    charges.append(float(line))
                if len(charges) == len(system):
                    print("Applying charges ...")
                    for ind, atom in enumerate(system):
                        atom.charge = charges[ind]
                elif len(system) % len(charges) == 0:
                    print(
                            "Number of charges a multiple of the number of atoms, applying charges recursively ..."
                            )
                    for ind, atom in enumerate(system):
                        i = ind % len(charges)
                        atom.charge = charges[i]
                else:
                    raise ValueError
                break
            except TypeError:
                print("!!! Something went wrong reading charges file !!!")
            except ValueError:
                print(
                    "!!! Number of charges doesn't match (a multiple) of the number of atoms !!!\n"
                    "(or something else went wrong reading charges file)"
                )
            except FileNotFoundError:
                print("!!! File not found !!!")

    while True:
        try:
            write_force_field_in: str = input(
                "\nWould you like to automatically apply a forcefield to this MPMC .pbd file?\n"
                "('yes', 'y', 1 or 'no', 'n', 0)\n\n> "
            )
            if write_force_field_in == "yes" or write_force_field_in == "y":
                write_force_field = 1
            elif write_force_field_in == "no" or write_force_field_in == "n":
                write_force_field = 0
            else:
                write_force_field = int(write_force_field_in)
            if write_force_field > 1 or write_force_field < 0:
                raise ValueError
            break
        except ValueError:
            print("!!! Error reading input !!!")

    if write_force_field == 1:
        while True:
            try:
                force_field_in: str = input(
                    "\nWhich force field?\n"
                    "valid answers are 'OPLSAA' (0) or 'PHAHST' (1)\n\n> "
                )
                if force_field_in == "OPLSAA":
                    force_field = 0
                elif force_field_in == "PHAHST":
                    force_field = 1
                else:
                    force_field = int(force_field_in)
                if force_field > 1 or force_field < 0:
                    raise ValueError
                apply_ff_to_system(system, get_forcefield(force_field))
                break
            except ValueError:
                print("!!! Error reading input !!!")

    filename = input("\noutput filename > ")

    if write_charges == 1:
        bool_charges = True
    else:
        bool_charges = False
    if write_force_field == 1:
        bool_ff = True
    else:
        bool_ff = False

    write_mpmc_pdb(
        system, pbc, filename, write_charges=bool_charges, write_params=bool_ff
    )


def list_molecules(system: list[Atom], pbc: PBC) -> None:
    mols = find_molecules(system, pbc)
    for mol in mols:
        mol.sort(key=lambda atom: atom.atomic_number, reverse=True)
        elements = [atom.element for atom in mol]
        print("".join(elements))


def menu_single_geom_analysis(system: list[Atom], pbc: PBC) -> tuple[list[Atom], PBC]:
    while True:
        option = 0
        try:
            option = input(
                "\nWhat would you like to do?\n\n"
                "1 = list bonds\n"
                "2 = list close vdw contacts\n"
                "3 = list angles\n"
                "4 = list lone atoms\n"
                "5 = delete lone atoms\n"
                "6 = list molecules\n"
                "7 = list coordinates\n"
                "8 = edit hydrogen bond distances\n"
                "9 = preview with VMD\n"
                "0 = back to main menu\n\n> "
            )
            option = int(option)
        except ValueError:
            print("!!! Error converting input to int !!!")
        if option == 1:
            list_bonds(system, pbc)
        elif option == 2:
            list_close_contacts(system, pbc)
        elif option == 3:
            list_angles(system, pbc)
        elif option == 4:
            list_lone_atoms(system, pbc)
        elif option == 5:
            system = delete_lone_atoms(system, pbc)
        elif option == 6:
            list_molecules(system, pbc)
        elif option == 7:
            list_coords(system)
        elif option == 8:
            system = edit_h_dist(system, pbc)
        elif option == 9:
            vmd_preview(system, pbc)
        elif option == 0:
            return system, pbc


def menu_single_write_files(system: list[Atom], pbc: PBC) -> None:
    while True:
        option = -1
        try:
            option = input(
                "\nWhat would you like to do?\n\n"
                "1 = write xyz file\n"
                "2 = write MPMC PDB file\n"
                "3 = write standard PDB file\n"
                "0 = back to main menu\n\n> "
            )
            option = int(option)
            if option == 1:
                filename = input("\noutput filename > ")
                out = open(filename, "w")
                write_xyz(system, pbc, out)
                out.close()
                print(f"wrote {filename}")
            elif option == 2:
                write_mpmc_options(system, pbc)
            elif option == 3:
                filename = input("\noutput filename > ")
                out = open(filename, "w")
                write_standard_pdb(system, pbc, out)
                out.close()
                print(f"wrote {filename}")
            elif option == 0:
                return
        except ValueError:
            print("!!! Error converting input to int !!!")


def menu_single_update_pbc(pbc: PBC) -> PBC:
    while True:
        try:
            a = input("Enter cell information\na>     ")
            a = float(a)
            b = input("b>     ")
            b = float(b)
            c = input("c>     ")
            c = float(c)
            alpha = input("alpha> ")
            alpha = float(alpha)
            beta = input("beta>  ")
            beta = float(beta)
            gamma = input("gamma> ")
            gamma = float(gamma)
            break
        except ValueError:
            print("!!! Error converting input to float !!!\n")
    pbc.update(a, b, c, alpha, beta, gamma)
    return pbc


def menu_single_extend_wrap(system: list[Atom], pbc: PBC) -> tuple[list[Atom], PBC]:
    while True:
        option = -1
        try:
            option = input(
                "\nWhat would you like to do?\n\n"
                "1 = extend along axis\n"
                "2 = wrap atoms from (0, 0, 0) to (1, 1, 1)\n"
                "3 = wrap atoms from (-1/2, -1/2, -1/2) to (1/2, 1/2, 1/2)\n"
                "4 = sort atoms\n"
                "0 = back to main menu\n\n> "
            )
            option = int(option)
            if option == 1:
                menu_single_extend_axis(system, pbc)
            elif option == 2:
                wrapall_forward(system, pbc)
            elif option == 3:
                wrapall(system, pbc)
            elif option == 4:
                system = sort(system, pbc)
            elif option == 0:
                return system, pbc
        except ValueError:
            print("!!! Error converting input to int !!!")


def menu_movie_extend_wrap(
    systems: list[list[Atom]], pbcs: list[PBC]
) -> tuple[list[list[Atom]], list[PBC]]:
    while True:
        option = -1
        try:
            option = input(
                "\nWhat would you like to do?\n\n"
                "1 = extend along axis\n"
                "2 = wrap atoms from (0, 0, 0) to (1, 1, 1)\n"
                "3 = wrap atoms from (-1/2, -1/2, -1/2) to (1/2, 1/2, 1/2)\n"
                "4 = sort atoms\n"
                "0 = back to main menu\n\n> "
            )
            option = int(option)
            if option == 1:
                systems, pbcs = menu_move_extend_axis(systems, pbcs)
            elif option == 2:
                for system, pbc in zip(systems, pbcs):
                    wrapall_forward(system, pbc)
            elif option == 3:
                for system, pbc in zip(systems, pbcs):
                    wrapall(system, pbc)
            elif option == 4:
                for system, pbc in zip(systems, pbcs):
                    sort(system, pbc)
            elif option == 0:
                return systems, pbcs
        except ValueError:
            print("!!! Error converting input to int !!!")


def menu_movie_write_files(systems: list[list[Atom]], pbcs: list[PBC]) -> None:
    while True:
        option = -1
        try:
            option = input(
                "\nWhat would you like to do?\n\n"
                "1 = write xyz file\n"
                "2 = write standard PDB file\n"
                "3 = write Xpress PDB file (no mol finding)\n"
                "0 = back to main menu\n\n> "
            )
            option = int(option)
            if option == 1:
                filename = input("\noutput filename > ")
                out = open(filename, "w")
                for system, pbc in progressbar(list(zip(systems, pbcs)), "Writing frame "):
                    write_xyz(system, pbc, out)
                out.close()
                print(f"wrote {filename}")
            elif option == 2:
                filename = input("\noutput filename > ")
                out = open(filename, "w")
                for system, pbc in zip(systems, pbcs):
                    write_standard_pdb(system, pbc, out)
                out.close()
                print(f"wrote {filename}")
            elif option == 3:
                filename = input("\noutput filename > ")
                out = open(filename, "w")
                for system, pbc in progressbar(list(zip(systems, pbcs)), "Writing frame "):
                    write_standard_pdb(system, pbc, out, skip_mols_step=True)
                out.close()
                print(f"wrote {filename}")
            elif option == 0:
                return
        except ValueError:
            print("!!! Error converting input to int !!!")


def menu_movie_create_movie(systems: list[list[Atom]], pbcs: list[PBC]) -> None:
    while True:
        try:
            x_rotation = input("X rotation (degrees): ")
            y_rotation = input("Y rotation (degrees): ")
            z_rotation = input("Z rotation (degrees): ")
            x_translation = input("X translation (angstroms): ")
            y_translation = input("Y translation (angstroms): ")
            z_translation = input("Z translation (angstroms): ")
            break
        except ValueError:
            print("!!! Error converting input !!!")
    print(
        f"\nMovie options:\n\nRotations: \nX {x_rotation} degrees\nY {y_rotation} degrees\nZ {z_rotation} degrees\n"
    )
    print(
        f"Translations:\nX {x_translation} A\nY {y_translation} A\nZ {z_translation} A\n"
    )
    os.system("mkdir vmd_movie_tmp")
    vmd_commands = open("vmd_movie_tmp/vmd_commands", "w")
    vmd_commands.write(f"rot x to {x_rotation}\n")
    vmd_commands.write(f"rot y to {y_rotation}\n")
    vmd_commands.write(f"rot z to {z_rotation}\n")
    vmd_commands.write(f"trans to {x_translation} {y_translation} {z_translation}\n")
    for i in range(len(systems)):
        vmd_commands.write(f"render snapshot {i:06d}.png\n")
        if i != len(systems) - 1:
            vmd_commands.write(f"next\n")
    out = open("vmd_movie_tmp/out.xyz", "w")
    for system, pbc in zip(systems, pbcs):
        write_xyz(system, pbc, out)
    out.close()
    os.system("vmd vmd_movie_tmp/out.xyz -e vmd_movie_tmp/vmd_commands")
    os.system(
        "ffmpeg -framerate 30 -pattern_type glob -i 'vmd_movie_tmp/*.png'"
        " -c:v libx264 -pix_fmt yuv420p vmd_movie_tmp/out.mp4"
    )


def main_loop_movie(systems: list[list[Atom]], pbcs: list[PBC]) -> None:
    while True:
        print_info_movie(systems, pbcs, sys.argv[1])
        option = -1
        try:
            option = input(
                "\nWhat would you like to do?\n\n"
                "1 = extend, wrap, sort\n"
                "2 = write files\n"
                "3 = create movie\n"
                "0 = quit\n\n> "
            )
            option = int(option)
            if option == 1:
                systems, pbcs = menu_movie_extend_wrap(systems, pbcs)
            elif option == 2:
                menu_movie_write_files(systems, pbcs)
            elif option == 3:
                menu_movie_create_movie(systems, pbcs)
            elif option == 0:
                return
        except ValueError:
            print("!!! Error converting input to int !!!")


def main_loop_single(system: list[Atom], pbc: PBC) -> None:
    system = overlap_detector(system, pbc)

    while True:
        print_info(system, pbc, sys.argv[1])
        option: int = -1
        try:
            option_in: str = input(
                "\nWhat would you like to do?\n\n"
                "1 = geometry analysis\n"
                "2 = extend axis, wrap, or sort\n"
                "3 = write files\n"
                "4 = update unit cell\n"
                "0 = quit\n\n> "
            )
            option = int(option_in)
        except ValueError:
            print("!!! Error converting input to int !!!")
        if option == 1:
            system, pbc = menu_single_geom_analysis(system, pbc)
        elif option == 2:
            system, pbc = menu_single_extend_wrap(system, pbc)
        elif option == 3:
            menu_single_write_files(system, pbc)
        elif option == 4:
            pbc = menu_single_update_pbc(pbc)
        elif option == 0:
            break
        else:
            print("\nInvalid option!")


def check_xyz_trajectory(filename: str) -> bool:
    f = open(filename, "r")
    try:
        n_atoms = int(f.readline())
        f.readline()
        for i in range(n_atoms):
            if f.readline() == "":
                return False
        line = f.readline()
        if line == "":
            return False
        n_atoms = int(line)
        f.readline()
        for i in range(n_atoms):
            if f.readline() == "":
                return False
    except ValueError:
        return False

    return True


def check_pdb_trajectory(filename: str) -> bool:
    f = open(filename, "r")
    try:
        line = f.readline()
        n_frames = 0
        while line != "":
            if line[:6] == "MODEL ":
                n_frames += 1
            line = f.readline()
    except ValueError:
        return False

    if n_frames > 1:
        return True
    else:
        return False


def read_xyz_trajectory(file: TextIO) -> tuple[list[list[Atom]], list[Optional[PBC]]]:
    systems: list[list[Atom]] = []
    pbcs: list[Optional[PBC]] = []

    default_pbc = None

    line = file.readline()

    try:
        while line != "" and line != "\n":
            n_atoms: int = int(line)
            pbc: Optional[PBC] = None
            system: list[Atom] = []
            line = file.readline()
            try:
                tokens = line.split()
                if len(tokens) != 6:
                    raise ValueError
                a = float(tokens[0])
                b = float(tokens[1])
                c = float(tokens[2])
                alpha = float(tokens[3])
                beta = float(tokens[4])
                gamma = float(tokens[5])
                pbc = PBC(a, b, c, alpha, beta, gamma)
            except ValueError:
                if default_pbc is None:
                    print("Couldn't locate a b c alpha beta gamma on second line of .xyz file")
                    default_pbc = PBC(1000000, 1000000, 1000000, 90, 90, 90)
                    default_pbc = menu_single_update_pbc(default_pbc)
                pbc = copy.deepcopy(default_pbc)

            try:
                for _ in range(n_atoms):
                    line = file.readline()
                    tokens = line.split()
                    atom = Atom(tokens[1], tokens[2], tokens[3], tokens[0])

                    # attempt to read the fifth column as the charge, pass otherwise
                    try:
                        charge = tokens[4]
                        atom.charge = float(charge)
                    except (ValueError, IndexError):
                        pass

                    system.append(atom)

            except ValueError:
                sys.exit(f"Error reading line {line}")

            systems.append(system)
            pbcs.append(pbc)
            line = file.readline()

    except ValueError:
        sys.exit("Error reading file")

    return systems, pbcs


def read_pdb_trajectory(file: TextIO) -> tuple[list[list[Atom]], list[Optional[PBC]]]:
    systems = []
    pbcs = []

    one_frame = io.StringIO()
    line = file.readline()
    one_frame.write(line)
    while line != "":
        line = file.readline()
        one_frame.seek(0)
        if line[:6] == "MODEL " and len(one_frame.readlines()) > 3:
            one_frame.seek(0)
            system, pbc = read_pdb(one_frame)
            systems.append(system)
            pbcs.append(pbc)
            one_frame.close()
            one_frame = io.StringIO()
        one_frame.seek(0, 2)
        one_frame.write(line)
    one_frame.seek(0)
    system, pbc = read_pdb(one_frame)
    systems.append(system)
    pbcs.append(pbc)
    one_frame.close()
    
    for idx, pbc in enumerate(pbcs):
        if pbc is None:
            print(f"Couldn't locate a b c alpha beta gamma in frame {idx} of pdb file")
            if idx == 0:
                pbcs[idx] = PBC(1000000, 1000000, 1000000, 90, 90, 90)
            else:
                pbcs[idx] = copy.copy(pbcs[0])

    return systems, pbcs


def load_file_and_run_menu() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 pdb_wizard.py <filename.[xyz|pdb]>")

    input_filename = sys.argv[1]

    is_trajectory = False

    if input_filename.split(".")[-1] == "xyz":
        is_trajectory = check_xyz_trajectory(input_filename)
    elif (
        input_filename.split(".")[-1] == "pdb" or input_filename.split(".")[-1] == "ent" or input_filename.split(".")[-1] == "pqr"
    ):
        is_trajectory = check_pdb_trajectory(input_filename)
    elif input_filename.split(".")[-1] == "cif":
        sys.exit("Use Mercury to save a .cif as a .xyz or .pdb first")
    else:
        sys.exit("Unable to determine if input file is xyz or pdb (please rename)")

    if is_trajectory:
        if input_filename.split(".")[-1] == "xyz":
            file = open(input_filename, "r")
            systems, pbcs = read_xyz_trajectory(file)
            file.close()
        elif (
            input_filename.split(".")[-1] == "pdb"
            or input_filename.split(".")[-1] == "ent"
            or input_filename.split(".")[-1] == "pqr"
        ):
            file = open(input_filename, "r")
            systems, pbcs = read_pdb_trajectory(file)
            file.close()
        else:
            sys.exit(1)

        main_loop_movie(systems, pbcs)

    else:
        if input_filename.split(".")[-1] == "xyz":
            file = open(input_filename, "r")
            system, pbc = read_xyz(file)
            file.close()
        elif (
            input_filename.split(".")[-1] == "pdb"
            or input_filename.split(".")[-1] == "ent"
            or input_filename.split(".")[-1] == "pqr"
        ):
            file = open(input_filename, "r")
            system, pbc = read_pdb(file)
            file.close()
        else:
            sys.exit(1)

        if pbc is None:
            while True:
                try:
                    a = input(
                        f"Cell information not found in {input_filename}"
                        "Enter cell information\n"
                        "a>     "
                    )
                    a = float(a)
                    b = input("b>     ")
                    b = float(b)
                    c = input("c>     ")
                    c = float(c)
                    alpha = input("alpha> ")
                    alpha = float(alpha)
                    beta = input("beta>  ")
                    beta = float(beta)
                    gamma = input("gamma> ")
                    gamma = float(gamma)
                    break
                except ValueError:
                    print("!!! Error converting input to float !!!\n")
            pbc = PBC(a, b, c, alpha, beta, gamma)

        main_loop_single(system, pbc)


if __name__ == "__main__":
    load_file_and_run_menu()
    sys.exit()
