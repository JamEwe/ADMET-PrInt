#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# padelpy/functions.py
# v.0.1.10
# Developed in 2021 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains various functions commonly used with PaDEL-Descriptor
#
# Edited i modified by E Jamrozik for data featurization

import warnings
import random
# stdlib. imports
from collections import OrderedDict
from csv import DictReader
from datetime import datetime
from os import remove
from re import IGNORECASE, compile
from time import sleep

# PaDELPy imports
from padelpy import padeldescriptor

__all__ = [
    "from_mdl",
    "from_smiles",
    "from_sdf",
]


def from_smiles(smiles,
                output_csv: str = None,
                descriptors: bool = True,
                fingerprints: bool = False,
                timeout: int = 60,
                maxruntime: int = -1,
                threads: int = -1,
                fingerprint_type: str = None
                ) -> OrderedDict:
    """ from_smiles: converts SMILES string to QSPR descriptors/fingerprints.

    Args:
        smiles (str, list): SMILES string for a given molecule, or a list of
            SMILES strings
        output_csv (str): if supplied, saves descriptors to this CSV file
        descriptors (bool): if `True`, calculates descriptors
        fingerprints (bool): if `True`, calculates fingerprints
        timeout (int): maximum time, in seconds, for conversion
        maxruntime (int): maximum running time per molecule in seconds. default=-1.
        threads (int): number of threads to use; defaults to -1 for max available
        fingerprint_type: (str) KlekFp or Pubchem

    Returns:
        list or OrderedDict: if multiple SMILES strings provided, returns a
            list of OrderedDicts, else single OrderedDict; each OrderedDict
            contains labels and values for each descriptor generated for each
            supplied molecule
    """
    # unit conversion for maximum running time per molecule
    # seconds -> milliseconds
    if maxruntime != -1:
        maxruntime = maxruntime * 1000

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")#[:-3]
    filename = timestamp + str(random.randint(1e8,1e9))

    with open("{}.smi".format(filename), "w") as smi_file:
        if type(smiles) == str:
            smi_file.write(smiles)
        elif type(smiles) == list:
            smi_file.write("\n".join(smiles))
        else:
            raise RuntimeError("Unknown input format for `smiles`: {}".format(
                type(smiles)
            ))
    smi_file.close()

    save_csv = True
    if output_csv is None:
        save_csv = False
        output_csv = "{}.csv".format(timestamp)

    fingerprint_descriptortypes = 'KlekotaRothFingerprinter.xml'

    for attempt in range(3):
        try:
            if fingerprint_type == 'pubchem':
                padeldescriptor(
                    mol_dir="{}.smi".format(filename),
                    d_file=output_csv,
                    convert3d=True,
                    retain3d=True,
                    d_2d=descriptors,
                    d_3d=descriptors,
                    fingerprints=fingerprints,
                    sp_timeout=timeout,
                    retainorder=True,
                    maxruntime=maxruntime,
                    threads=threads
                )
            elif fingerprint_type == 'klek':
                padeldescriptor(
                    mol_dir="{}.smi".format(filename),
                    d_file=output_csv,
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    fingerprints=True
                )
            break
        except RuntimeError as exception:
            if attempt == 2:
                remove("{}.smi".format(filename))
                if not save_csv:
                    sleep(0.5)
                    try:
                        remove(output_csv)
                    except FileNotFoundError as e:
                        warnings.warn(e, RuntimeWarning)
                raise RuntimeError(exception)
            else:
                continue
        except KeyboardInterrupt as kb_exception:
            remove("{}.smi".format(filename))
            if not save_csv:
                try:
                    remove(output_csv)
                except FileNotFoundError as e:
                    warnings.warn(e, RuntimeWarning)
            raise kb_exception

    with open(output_csv, "r", encoding="utf-8") as desc_file:
        reader = DictReader(desc_file)
        rows = [row for row in reader]
    desc_file.close()

    remove("{}.smi".format(filename))
    if not save_csv:
        remove(output_csv)

    if type(smiles) == list and len(rows) != len(smiles):
        raise RuntimeError("PaDEL-Descriptor failed on one or more mols." +
                           " Ensure the input structures are correct.")
    elif type(smiles) == str and len(rows) == 0:
        raise RuntimeError(
            "PaDEL-Descriptor failed on {}.".format(smiles) +
            " Ensure input structure is correct."
        )

    for idx, r in enumerate(rows):
        if len(r) == 0:
            raise RuntimeError(
                "PaDEL-Descriptor failed on {}.".format(smiles[idx]) +
                " Ensure input structure is correct."
            )

    for idx in range(len(rows)):
        del rows[idx]["Name"]

    if type(smiles) == str:
        return rows[0]
    return rows