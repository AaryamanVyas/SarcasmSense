#!/usr/bin/env python
"""Pre-flight check. Run this BEFORE `pip install`.

    python scripts/check_environment.py

Verifies the two things that account for almost every failed installation:

1. **Python version.** The pinned scientific stack (numpy 1.26, scipy 1.14,
   torch 2.5, faiss-cpu 1.9) publishes wheels for CPython 3.10-3.12 only. On a
   newer interpreter pip cannot find a wheel, falls back to building from
   source, and fails with a confusing message about a missing Fortran compiler
   or C++ toolchain. The real cause is the interpreter, not the compiler.

2. **Virtual environment.** Installing into a system Python usually triggers
   "Defaulting to user installation because normal site-packages is not
   writeable", which scatters packages into a user directory and makes version
   conflicts hard to diagnose.

This file is deliberately written in very conservative Python (no f-strings, no
modern syntax, standard library only) so that it runs and prints a useful
message on *any* interpreter, including one too old or too new for the project.
"""

import os
import platform
import sys

MIN_VERSION = (3, 10)
MAX_VERSION = (3, 12)  # inclusive
RECOMMENDED = "3.11"

RULE = "=" * 74


def _supported_python():
    """Return (ok, message) for the running interpreter's version."""
    version = sys.version_info[:2]
    printable = "%d.%d.%d" % sys.version_info[:3]
    if version < MIN_VERSION:
        return False, (
            "Python " + printable + " is too old.\n"
            "  This project needs CPython 3.10-3.12 (3." + str(MIN_VERSION[1])
            + " introduced syntax the code relies on)."
        )
    if version > MAX_VERSION:
        return False, (
            "Python " + printable + " is too new for the pinned dependencies.\n"
            "\n"
            "  numpy 1.26, scipy 1.14, torch 2.5 and faiss-cpu 1.9 do not publish\n"
            "  wheels for Python %d.%d. pip will try to COMPILE them from source and\n"
            "  fail with an error about a missing Fortran or C++ compiler. That error\n"
            "  is a symptom: installing a compiler will not fix it, because torch has\n"
            "  no source build for this interpreter either."
        ) % version
    return True, "Python " + printable + " is supported."


def _in_virtualenv():
    """Whether we are running inside a venv/virtualenv/conda environment."""
    if hasattr(sys, "real_prefix"):
        return True
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return True
    return bool(os.environ.get("CONDA_DEFAULT_ENV"))


def _windows_instructions():
    return (
        "  1. Install Python " + RECOMMENDED + " (keep your existing Python; they coexist):\n"
        "\n"
        "       winget install -e --id Python.Python.3.11\n"
        "\n"
        "     ...or download it from https://www.python.org/downloads/release/python-3119/\n"
        "     (choose 'Windows installer (64-bit)').\n"
        "\n"
        "  2. Close and reopen PowerShell, then from the project root:\n"
        "\n"
        "       py -3.11 -m venv .venv\n"
        "       .\\.venv\\Scripts\\Activate.ps1\n"
        "       python -m pip install --upgrade pip\n"
        "       pip install -r requirements-training.txt\n"
        "\n"
        "     If Activate.ps1 is blocked by execution policy, run this first:\n"
        "       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass\n"
        "\n"
        "  3. Confirm the prompt now starts with (.venv), then re-run:\n"
        "       python scripts/check_environment.py\n"
    )


def _posix_instructions():
    return (
        "  1. Install Python " + RECOMMENDED + ":\n"
        "       macOS:          brew install python@3.11\n"
        "       Debian/Ubuntu:  sudo apt install python3.11 python3.11-venv\n"
        "\n"
        "  2. From the project root:\n"
        "\n"
        "       python3.11 -m venv .venv\n"
        "       source .venv/bin/activate\n"
        "       python -m pip install --upgrade pip\n"
        "       pip install -r requirements-training.txt\n"
        "\n"
        "  3. Confirm the prompt now starts with (.venv), then re-run:\n"
        "       python scripts/check_environment.py\n"
    )


def main():
    """Print the environment report. Returns a POSIX exit code."""
    print("")
    print(RULE)
    print("ENVIRONMENT PRE-FLIGHT CHECK")
    print(RULE)
    print("  interpreter : " + sys.executable)
    print("  version     : " + platform.python_version() + "  (" + platform.python_implementation() + ")")
    print("  platform    : " + platform.platform())
    print("  virtual env : " + ("yes" if _in_virtualenv() else "NO"))
    print("")

    problems = []

    version_ok, version_message = _supported_python()
    print(("  [OK]   " if version_ok else "  [FAIL] ") + version_message)
    if not version_ok:
        problems.append("python-version")

    if _in_virtualenv():
        print("  [OK]   Running inside a virtual environment.")
    else:
        print(
            "  [WARN] Not in a virtual environment.\n"
            "         pip will install into your user site-packages, which makes\n"
            "         version conflicts hard to diagnose and can break other projects."
        )
        problems.append("no-venv")

    if platform.python_implementation() != "CPython":
        print("  [WARN] Non-CPython interpreter; the pinned wheels target CPython.")
        problems.append("implementation")

    print("")
    if not problems:
        print(RULE)
        print("  Ready. Next:")
        print("      pip install -r requirements-training.txt")
        print("      python scripts/quickstart.py")
        print(RULE)
        print("")
        return 0

    print(RULE)
    print("HOW TO FIX")
    print(RULE)
    if os.name == "nt":
        print(_windows_instructions())
    else:
        print(_posix_instructions())

    if "python-version" in problems and sys.version_info[:2] > MAX_VERSION:
        print(
            "  Note: you do NOT need to uninstall Python "
            + platform.python_version()
            + ". The venv above\n"
            "  pins this project to 3." + str(MAX_VERSION[1] - 1) + " without touching your"
            " system default.\n"
        )
    print(RULE)
    print("")
    return 1


if __name__ == "__main__":
    sys.exit(main())
