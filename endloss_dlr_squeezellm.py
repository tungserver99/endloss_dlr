#!/usr/bin/env python3
"""Deprecated wrapper for the corrected from-scratch EndLoss_DLR entrypoint."""

from __future__ import annotations

import warnings

from endloss_dlr_quantize import main


if __name__ == "__main__":
    warnings.warn(
        "endloss_dlr_squeezellm.py is deprecated. EndLoss_DLR is a from-scratch quantizer; "
        "use endloss_dlr_quantize.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
