#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import sys
from openevolve.cli import main

# examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config_azure.yaml

if __name__ == "__main__":
    sys.exit(main())
