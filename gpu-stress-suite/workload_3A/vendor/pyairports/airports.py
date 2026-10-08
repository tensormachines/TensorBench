"""Vendored minimal `pyairports` shim.

WHY THIS EXISTS
---------------
vLLM 0.5.4 imports `outlines` (guided decoding). Every `outlines` version in
vLLM 0.5.4's allowed range (>=0.0.43,<0.1) executes, at module import time:

    from pyairports.airports import AIRPORT_LIST            # outlines/types/airports.py
    AIRPORT_IATA_LIST = list({(a[3], a[3]) for a in AIRPORT_LIST if a[3] != ""})
    IATA = Enum("Airport", AIRPORT_IATA_LIST)

The only `pyairports` distribution reachable from the pinned package index is
`pyairports==0.0.1`, whose wheel is structurally broken: it ships no
`pyairports` module at all (only dist-info and a stray `sample/` package), so
`import pyairports` always fails and therefore `import vllm` always fails.

No working `pyairports` is available from a trusted index in the target
environment, so instead of pulling an unverified third-party package (against
the engagement's trusted-source policy) we vendor this transparent, reviewable
first-party shim. It is COPY'd over the broken install in the image.

SAFETY / SCOPE
--------------
Workloads 1C and 3A use greedy sampling with explicit token-id prompts and
never invoke guided decoding's airport-code type. This shim only has to make
the import succeed. One sentinel row keeps `AIRPORT_IATA_LIST` non-empty so
`Enum("Airport", ...)` is well-formed; the value is otherwise unused.
"""

# Each row mirrors the real pyairports tuple shape; index [3] is the IATA code.
AIRPORT_LIST = [
    ("", "", "", "ZZZ", "", "", "", ""),
]
