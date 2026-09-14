"""Run the registered PRISM v2 capacity comparison, separate from existing suites."""

from run_residual_gate_controls import CAPACITY_SPEC
from run_residual_gate_controls import main as suite_main


def main():
    suite_main(CAPACITY_SPEC)


if __name__ == "__main__":
    main()
