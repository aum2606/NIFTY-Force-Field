

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DASH_HOST, DASH_PORT


def main():
    print("=" * 60)
    print("  NIFTY FORCE FIELD")
    print("  Indian Market Quant Research Engine")
    print("=" * 60)
    print()

    # Import and precompute
    from dashboard.app import app, precompute

    print("  Running full pipeline (data + models + backtest)...")
    print("  This may take 1-2 minutes on first run...\n")

    try:
        precompute()
    except Exception as e:
        print(f"\n  ERROR during pipeline: {e}")
        print("  Check your internet connection for yfinance downloads.")
        import traceback
        traceback.print_exc()
        return

    print(f"\n  Dashboard ready at: http://{DASH_HOST}:{DASH_PORT}")
    print("  Press Ctrl+C to stop.\n")

    app.run(debug=False, host=DASH_HOST, port=DASH_PORT)


if __name__ == "__main__":
    main()
