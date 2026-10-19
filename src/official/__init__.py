"""Verbatim copies of official SKIPP'D code (MIT, see LICENSE_SKIPPD). Do not edit these files.

relative_op_func.py == reference-skippd/models/Relative_op_func.py (byte-identical).
Known quirks, kept on purpose for faithful reproduction:
  * DST start/end days are computed from the calendar of the *current* month, not March/November.
  * Theoretical power floor 0.05 kW, PV floor 0.05 kW, relative output capped at 1.5.
"""
