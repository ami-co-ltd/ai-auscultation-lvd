#!/usr/bin/env python

# import dataclasses

# @dataclasses.dataclass
# class DecoderParam:
#     """dataclass for Decoder

#     Attributes:
#         decoder_type (str): decoder module name
#     """

#     decoder_type: str

#     def __post_init__(self) -> None:
#         """init"""
#         self.check_parameters()

#     def check_parameters(self) -> None:
#         """check parameters

#         Raises:
#             ValueError: Decoder type is not supported
#         """

#         if self.decoder_type not in ["maxpooling"]:
#             raise ValueError("Decoder type is not supported")
