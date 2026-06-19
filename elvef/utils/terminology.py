#!/usr/bin/env python
# pylint: disable=no-else-return,too-few-public-methods,line-too-long,consider-using-f-string

import re
from logging import getLogger
from pathlib import Path
from typing import List, Union

logger = getLogger(__name__)


class AuscultationPosition:
    """class to manage auscultation position name

    Attributes:
        name_2RSB (str): name of 2rsb. Defaults to "2rsb".
        name_2LSB (str): name of 2lsb. Defaults to "2lsb".
        name_4LSB (str): name of 4lsb. Defaults to "4lsb".
        name_5LMCL (str): name of 5lmcl. Defaults to "5lmcl".
    """

    name_2RSB: str = "2rsb"
    name_2LSB: str = "2lsb"
    name_4LSB: str = "4lsb"
    name_5LMCL: str = "5lmcl"

    @staticmethod
    def normalize_name(name: str) -> str:
        """normalize terminology

        Args:
            name (str): unnormalized name

        Raises:
            ValueError: couldn't estimate valid position name

        Returns:
            str: normalized name
        """
        if name.lower() in ["2rsb"]:
            return AuscultationPosition.name_2RSB

        elif name.lower() in ["2lsb"]:
            return AuscultationPosition.name_2LSB

        elif name.lower() in ["4lsb"]:
            return AuscultationPosition.name_4LSB

        elif name.lower() in ["5lmcl"]:
            return AuscultationPosition.name_5LMCL

        else:
            raise ValueError("Got unknown ausclutation position name: " + name.lower())

    @staticmethod
    def get_hospital_name(bin_file: Union[Path, str]) -> str:
        """get hospital name

        Args:
            bin_file (Union[Path, str]): input bin file

        Raises:
            ValueError: Unexpected file format

        Returns:
            str: hospital name

        """
        name = Path(bin_file).stem.split("_")[0]
        name = re.sub(r"[0-9]+", "", name)

        if name == "":
            raise ValueError("Unexpected file format")
        return name

    @staticmethod
    def get_position_name(bin_file: Union[str, Path], to_upper: bool = False) -> str:
        """extract position name from bin/wav file path

        Args:
            bin_file (Union[str, Path]): binary file path

        Raises:
            ValueError: failed to estimate position name

        Returns:
            str: position name
        """
        position = Path(bin_file).stem

        candidates = re.findall(r"2rsb|2lsb|4lsb|5lmcl", position.lower())
        if len(candidates) != 1:
            raise ValueError("Failed to get position name: {}".format(position))

        position = AuscultationPosition.normalize_name(candidates[0])
        return position.upper() if to_upper else position

    @staticmethod
    def available_names() -> List[str]:
        """return all auscultation position

        Returns:
            List[str]: list of auscultation locations
        """
        return [
            AuscultationPosition.name_2RSB,
            AuscultationPosition.name_2LSB,
            AuscultationPosition.name_4LSB,
            AuscultationPosition.name_5LMCL,
        ]


get_hospital_name = AuscultationPosition.get_hospital_name
get_position_name = AuscultationPosition.get_position_name
