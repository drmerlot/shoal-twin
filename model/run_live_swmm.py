import sys
import json
import argparse
from pathlib import Path
import os

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.generate_mc_dataset import process_single_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rain", type=float, required=True)
    parser.add_argument("--dur", type=float, required=True)
    parser.add_argument("--api", type=float, required=True)
    args = parser.parse_args()

    with open(ROOT / "model" / "best_params.json") as f:
        best_params = json.load(f)
    
    with open(ROOT / "model" / "shoal_creek_memorial_day_2015.inp") as f:
        inp_template = f.read()

    row_dict = {
        "run_id": "live_demo",
        "rain_depth_mm": args.rain * 25.4,
        "storm_dur_hr": args.dur,
        "api_5_mm": args.api
    }
    
    # process_single_run might print things to stdout (e.g. via pyswmm or loggers)
    # We want to make sure we only print our JSON at the very end
    # Temporarily suppress stdout
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        res = process_single_run(row_dict, inp_template, best_params)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    if res:
        _, _, meta_row, elapsed = res
        result = {
            "success": True,
            "peak_cfs": meta_row["peak_cfs"],
            "elapsed": elapsed
        }
    else:
        result = {"success": False}

    print(json.dumps(result))

if __name__ == "__main__":
    main()
