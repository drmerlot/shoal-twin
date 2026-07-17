"""
Step 3c / 4a: Validate SWMM .inp with pyswmm
=============================================
Loads a skeleton or storm-specific .inp, runs a short simulation,
and checks that pyswmm accepts it without fatal errors.

Usage:
    .venv/bin/python model/validate_inp.py                         # skeleton
    .venv/bin/python model/validate_inp.py --storm halloween_2013  # storm INP

Note:
    DO NOT use 'uv run python' for pyswmm — macOS Gatekeeper kills the
    swmm-toolkit dylib when launched via uv (codesign sandbox issue).
    Use .venv/bin/python directly instead.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STORM_CHOICES = ["halloween_2013", "memorial_day_2015", "halloween_2015"]


def resolve_inp(storm_id=None) -> Path:
    if storm_id:
        return Path(f"model/shoal_creek_{storm_id}.inp")
    return Path("model/shoal_creek_skeleton.inp")


def main():
    parser = argparse.ArgumentParser(description="Validate SWMM .inp with pyswmm")
    parser.add_argument("--storm", choices=STORM_CHOICES, default=None,
                        help="Validate storm-specific INP file")
    args = parser.parse_args()

    inp_path = resolve_inp(args.storm)
    label = args.storm or "skeleton"
    log.info("Step 3c/4a: Validate SWMM .inp  [%s]", label)

    if not inp_path.exists():
        log.error("INP file not found: %s", inp_path)
        if args.storm:
            log.error("Run: uv run python model/build_swmm_inp.py --storm %s", args.storm)
        else:
            log.error("Run: uv run python model/build_swmm_inp.py")
        sys.exit(1)

    size_kb = inp_path.stat().st_size / 1024
    log.info("INP file: %s  (%.1f KB)", inp_path, size_kb)

    try:
        import pyswmm
        from pyswmm import Simulation
    except ImportError:
        log.error("pyswmm not installed. Run: uv add pyswmm")
        sys.exit(1)

    log.info("pyswmm version: %s", pyswmm.__version__)

    errors = []
    routing_err = runoff_err = None
    step_count = 0

    try:
        with Simulation(str(inp_path)) as sim:
            log.info("Simulation loaded successfully")
            log.info("  Start: %s", sim.start_time)
            log.info("  End:   %s", sim.end_time)

            if args.storm:
                # Full execution for storm runs to get real error stats
                log.info("  Running full simulation (storm mode)...")
                sim.execute()
                log.info("  Simulation complete")
                step_count = -1  # full run
            else:
                # Quick 5-step check for skeleton
                sim.step_advance(300)
                for step in sim:
                    step_count += 1
                    if step_count >= 5:
                        break
                log.info("  Stepped %d iterations without error", step_count)

            # Check for routing errors (pyswmm 2.x API)
            routing_err = sim.flow_routing_error
            runoff_err  = sim.runoff_error
            log.info("  Routing error: %.4f%%", (routing_err or 0) * 100)
            log.info("  Runoff error:  %.4f%%", (runoff_err  or 0) * 100)

    except Exception as exc:
        errors.append(str(exc))
        log.error("Simulation error: %s", exc)

    print("\n" + "=" * 60)
    print("SWMM Validation Result")
    print("=" * 60)
    if not errors:
        print("  STATUS: PASS")
        print(f"  File:   {inp_path}  ({size_kb:.1f} KB)")
        if routing_err is not None:
            print(f"  Routing error: {(routing_err or 0)*100:.4f}%")
            print(f"  Runoff error:  {(runoff_err  or 0)*100:.4f}%")
        print()
        if args.storm:
            print(f"  Storm '{args.storm}' INP loads and runs without fatal errors.")
            print("  Ready for Step 4 calibration (tune Manning n + Green-Ampt).")
        else:
            print("  Skeleton INP valid. Run with --storm to validate storm files.")
    else:
        print("  STATUS: FAIL")
        for e in errors:
            print(f"  ERROR: {e}")
        print()
        print("  Fix errors above, then re-run:")
        if args.storm:
            print(f"    uv run python model/build_swmm_inp.py --storm {args.storm}")
            print(f"    .venv/bin/python model/validate_inp.py --storm {args.storm}")
        else:
            print("    uv run python model/build_swmm_inp.py")
            print("    .venv/bin/python model/validate_inp.py")
    print("=" * 60)

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
