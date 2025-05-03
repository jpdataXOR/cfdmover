import streamlit as st
from datetime import datetime, timezone
from dukascopy_util import fetch_stock_indices_data # Assuming this is available from your files
import pandas as pd
import numpy as np
import time

# --- Configuration ---
# Updated instrument list with codes and a mapping to human-readable names
INSTRUMENT_CODES = [
    "BTC/USD",
    "E_XJO-ASX",
    "E_NQ-100",
    "E_DAAX",
    "E_CAAC-40",
    "E_DJE50XX",
    "E_Futsee-100",
    "E_H-Kong",
    "E_IBC-MAC",
    "E_N225Jap",
    "E_SandP-500",
    "VOL.IDX%2FUSD"
]

# Mapping from instrument code to human-readable name
INSTRUMENT_NAMES = {
    "BTC/USD": "Bitcoin / US Dollar",
    "E_XJO-ASX": "Australia 200 (ASX)",
    "E_NQ-100": "Nasdaq 100",
    "E_DAAX": "Germany 40 (DAX)",
    "E_CAAC-40": "France 40 (CAC)",
    "E_DJE50XX": "Euro Stoxx 50",
    "E_Futsee-100": "UK 100 (FTSE)",
    "E_H-Kong": "Hong Kong 50 (Hang Seng)",
    "E_IBC-MAC": "Spain 35 (IBEX)",
    "E_N225Jap": "Japan 225 (Nikkei)",
    "E_SandP-500": "US 500 (S&P)",
    "VOL.IDX%2FUSD": "Volatility Index (VIX)" # Assuming this is the VIX
}

INTERVAL_OPTIONS = {
    "15 Minute": "15MIN",
    "1 Hour": "1HOUR",
    "1 Day": "1DAY"
}
# Fetch a reasonable amount of historical data for analysis
FETCH_LIMIT_ANALYSIS = 1000
# Refresh interval for the main analysis tab (15 minutes = 900 seconds)
REFRESH_INTERVAL_SECONDS = 15 * 60

# Multiplier for defining a "large" move - This will be controlled by a slider

# --- Analysis Function (Used by Tab 1) ---
def analyze_price_moves(df: pd.DataFrame, instrument_code: str, interval_name: str, multiplier: float):
    """
    Analyzes price movements for a given DataFrame and interval.

    Args:
        df (pd.DataFrame): DataFrame with historical price data, indexed by Date.
                          Must contain a 'Close' column.
        instrument_code (str): The code of the instrument (e.g., "BTC/USD").
        interval_name (str): The name of the interval (e.g., "15 Minute").
        multiplier (float): The multiplier for the median move to define a "large" move.

    Returns:
        dict: A dictionary containing the analysis results for one interval and instrument.
    """
    # Use the human-readable name in the results dictionary
    instrument_name = INSTRUMENT_NAMES.get(instrument_code, instrument_code) # Get name, default to code if not found

    results = {
        "Instrument": instrument_name, # Use human-readable name here
        "Interval": interval_name,
        "Median Move (%)": None,
        "Total Bars": 0,
        "Large Pos Moves Count": 0,
        "Large Pos Moves Freq (%)": "N/A",
        "Avg Gap Pos Moves": "N/A",
        "Large Neg Moves Count": 0,
        "Large Neg Moves Freq (%)": "N/A",
        "Avg Gap Neg Moves": "N/A",
        "Last Pos Move Time": "Never", # Keep for internal calculation if needed
        "Bars Since Last Pos Move": "N/A",
        "Last Neg Move Time": "Never", # Keep for internal calculation if needed
        "Bars Since Last Neg Move": "N/A",
        "Spike Pending Ratio (Pos)": "N/A",
        "Spike Pending Ratio (Neg)": "N/A"
    }

    if df.empty or 'Close' not in df.columns:
        results["Status"] = "No Data"
        return results

    results["Total Bars"] = len(df)

    # Calculate percentage price change between consecutive bars
    percentage_price_change = df['Close'].pct_change().dropna() * 100

    if percentage_price_change.empty:
         results["Status"] = "No Price Change Data"
         return results

    # Calculate the absolute percentage price change
    abs_percentage_price_change = percentage_price_change.abs()

    # Calculate the median absolute percentage price move
    median_move_percent = abs_percentage_price_change.median()
    results["Median Move (%)"] = f"{median_move_percent:.4f}" if median_move_percent is not None else None

    if median_move_percent is None or median_move_percent == 0:
        results["Status"] = "Median Move is Zero"
        return results

    # Calculate the threshold for large moves based on the median percentage move
    large_move_threshold_percent = median_move_percent * multiplier

    # Identify large positive and negative moves based on percentage change
    large_positive_moves = percentage_price_change[percentage_price_change > large_move_threshold_percent]
    large_negative_moves = percentage_price_change[percentage_price_change < -large_move_threshold_percent]


    # Count frequency
    results["Large Pos Moves Count"] = len(large_positive_moves)
    results["Large Neg Moves Count"] = len(large_negative_moves)

    # Calculate frequency percentage
    if results["Total Bars"] > 0:
        results["Large Pos Moves Freq (%)"] = f"{(results['Large Pos Moves Count'] / results['Total Bars']) * 100:.2f}"
        results["Large Neg Moves Freq (%)"] = f"{(results['Large Neg Moves Count'] / results['Total Bars']) * 100:.2f}"

    # --- Calculate Average Gap ---
    if len(large_positive_moves) > 1:
        pos_move_original_indices = df.index.intersection(large_positive_moves.index)
        if len(pos_move_original_indices) > 1:
            pos_move_positions_in_df = [df.index.get_loc(idx) for idx in pos_move_original_indices]
            gaps = np.diff(pos_move_positions_in_df)
            if len(gaps) > 0:
                 results["Avg Gap Pos Moves"] = f"{gaps.mean():.2f}"


    if len(large_negative_moves) > 1:
        neg_move_original_indices = df.index.intersection(large_negative_moves.index)
        if len(neg_move_original_indices) > 1:
            neg_move_positions_in_df = [df.index.get_loc(idx) for idx in neg_move_original_indices]
            gaps = np.diff(neg_move_positions_in_df)
            if len(gaps) > 0:
                results["Avg Gap Neg Moves"] = f"{gaps.mean():.2f}"


    # Find the last occurrence and bars since
    if not large_positive_moves.empty:
        last_pos_idx = large_positive_moves.index[-1]
        results["Last Pos Move Time"] = last_pos_idx.strftime('%Y-%m-%d %H:%M:%S') # Keep for internal use if needed
        pos_in_df = df.index.get_loc(last_pos_idx)
        results["Bars Since Last Pos Move"] = len(df) - 1 - pos_in_df
    else:
         results["Bars Since Last Pos Move"] = "N/A" # Ensure N/A if no moves

    if not large_negative_moves.empty:
        last_neg_idx = large_negative_moves.index[-1]
        results["Last Neg Move Time"] = last_neg_idx.strftime('%Y-%m-%d %H:%M:%S') # Keep for internal use if needed
        neg_in_df = df.index.get_loc(last_neg_idx)
        results["Bars Since Last Neg Move"] = len(df) - 1 - neg_in_df
    else:
         results["Bars Since Last Neg Move"] = "N/A" # Ensure N/A if no moves


    # --- Calculate Spike Pending Ratios ---
    # Positive Ratio
    pos_count = results["Large Pos Moves Count"]
    bars_since_pos = results["Bars Since Last Pos Move"]

    if isinstance(bars_since_pos, int):
        denominator_pos = bars_since_pos + 1
        results["Spike Pending Ratio (Pos)"] = pos_count / denominator_pos
    elif pos_count > 0:
         results["Spike Pending Ratio (Pos)"] = 0.0

    if isinstance(results["Spike Pending Ratio (Pos)"], (int, float)):
        results["Spike Pending Ratio (Pos)"] = f"{results['Spike Pending Ratio (Pos)']:.4f}"

    # Negative Ratio
    neg_count = results["Large Neg Moves Count"]
    bars_since_neg = results["Bars Since Last Neg Move"]

    if isinstance(bars_since_neg, int):
        denominator_neg = bars_since_neg + 1
        results["Spike Pending Ratio (Neg)"] = neg_count / denominator_neg
    elif neg_count > 0:
         results["Spike Pending Ratio (Neg)"] = 0.0

    if isinstance(results["Spike Pending Ratio (Neg)"], (int, float)):
        results["Spike Pending Ratio (Neg)"] = f"{results['Spike Pending Ratio (Neg)']:.4f}"


    results["Status"] = "Success" # Keep status in results dictionary for potential debugging
    return results

# --- Streamlit UI Setup ---
st.title("📊 Market Analysis Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["🔬 Spike Ratio Comparison (Live)", "📈 Other Analysis / Backtesting"]) # Renamed and reordered tabs

# --- Spike Ratio Comparison Tab (Tab 1 - Live) ---
with tab1:
    st.header("Spike Ratio Comparison Across Instruments")
    st.write(f"Compares the Spike Pending Ratios across all instruments for each time interval, sorted by ratio. This data refreshes every {REFRESH_INTERVAL_SECONDS // 60} minutes.")

    # Add a slider for the large move multiplier specifically for this tab's analysis
    large_move_multiplier_comparison = st.slider(
        "Large Move Multiplier (x Median) for Comparison",
        min_value=0.1, # Minimum value for the slider
        max_value=5.0, # Maximum value (adjust as needed)
        value=1.33,    # Default value
        step=0.01,     # Step size
        key='large_move_multiplier_comparison_tab1' # Use a unique key for this slider
    )

    # Create a placeholder for the dynamic tables display
    comparison_tables_placeholder = st.empty()

    # --- Periodic Update Loop for Tab 1 ---
    while True:
        with comparison_tables_placeholder.container():
            st.write(f"⏰ Last analysis completed at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"🔹 Running comparison analysis for all instruments with multiplier {large_move_multiplier_comparison:.2f}...")

            all_comparison_results = [] # List to hold results for all instruments and intervals

            # Use a spinner for the entire comparison process
            with st.spinner("Analyzing..."):
                # Iterate through all instrument codes
                for instrument_code in INSTRUMENT_CODES:
                    # Iterate through all intervals
                    for interval_name, interval_key in INTERVAL_OPTIONS.items():
                        try:
                            df_interval = fetch_stock_indices_data(
                                instrument=instrument_code, # Use the code for fetching
                                offer_side="B", # Using Bid side for consistency
                                interval=interval_key,
                                limit=FETCH_LIMIT_ANALYSIS, # Use the larger limit for analysis
                                time_direction="P"
                            )

                            # Perform the analysis, passing instrument code
                            results = analyze_price_moves(df_interval, instrument_code, interval_name, large_move_multiplier_comparison)
                            # Store the relevant comparison data
                            all_comparison_results.append({
                                "Instrument": results.get("Instrument", instrument_code), # Use the name from results, default to code
                                "Interval": interval_name,
                                "Spike Pending Ratio (Pos)": results.get("Spike Pending Ratio (Pos)", "N/A"),
                                "Spike Pending Ratio (Neg)": results.get("Spike Pending Ratio (Neg)", "N/A"),
                                "Median Move (%)": results.get("Median Move (%)", "N/A"),
                                "Bars Since Last Pos Move": results.get("Bars Since Last Pos Move", "N/A"),
                                "Bars Since Last Neg Move": results.get("Bars Since Last Neg Move", "N/A"),
                                "Avg Gap Pos Moves": results.get("Avg Gap Pos Moves", "N/A"),
                                "Avg Gap Neg Moves": results.get("Avg Gap Neg Moves", "N/A"),
                                "Status": results.get("Status", "Unknown") # Keep status in the list for potential filtering/debugging
                            })


                        except Exception as e:
                            # Use the human-readable name even if error
                            instrument_name = INSTRUMENT_NAMES.get(instrument_code, instrument_code)
                            # Append an error result for this instrument and interval
                            all_comparison_results.append({
                                "Instrument": instrument_name,
                                "Interval": interval_name,
                                "Spike Pending Ratio (Pos)": "N/A",
                                "Spike Pending Ratio (Neg)": "N/A",
                                "Median Move (%)": "N/A",
                                "Bars Since Last Pos Move": "N/A",
                                "Bars Since Last Neg Move": "N/A",
                                "Avg Gap Pos Moves": "N/A",
                                "Avg Gap Neg Moves": "N/A",
                                "Status": f"Error: {e}"
                            })

            st.subheader("📈 Spike Ratio Comparison Results")

            if all_comparison_results:
                # Convert the list of dictionaries to a pandas DataFrame
                comparison_df = pd.DataFrame(all_comparison_results)

                # Ensure ratio columns are numeric for sorting, coercing errors to NaN
                comparison_df['Spike Pending Ratio (Pos)'] = pd.to_numeric(comparison_df['Spike Pending Ratio (Pos)'], errors='coerce')
                comparison_df['Spike Pending Ratio (Neg)'] = pd.to_numeric(comparison_df['Spike Pending Ratio (Neg)'], errors='coerce')
                # Ensure other numeric columns are numeric, coercing errors to NaN
                comparison_df['Median Move (%)'] = pd.to_numeric(comparison_df['Median Move (%)'], errors='coerce')
                # Convert Bars Since and Avg Gap to numeric, handle NaN, then convert to string for display
                comparison_df['Bars Since Last Pos Move'] = pd.to_numeric(comparison_df['Bars Since Last Pos Move'], errors='coerce').fillna("N/A").astype(str).replace('nan', 'N/A')
                comparison_df['Bars Since Last Neg Move'] = pd.to_numeric(comparison_df['Bars Since Last Neg Move'], errors='coerce').fillna("N/A").astype(str).replace('nan', 'N/A')
                comparison_df['Avg Gap Pos Moves'] = pd.to_numeric(comparison_df['Avg Gap Pos Moves'], errors='coerce').fillna("N/A").astype(str).replace('nan', 'N/A')
                comparison_df['Avg Gap Neg Moves'] = pd.to_numeric(comparison_df['Avg Gap Neg Moves'], errors='coerce').fillna("N/A").astype(str).replace('nan', 'N/A')


                # Display separate tables for each interval and ratio type, sorted by ratio
                for interval_name in INTERVAL_OPTIONS.keys():
                    # Filter data for the current interval
                    interval_df = comparison_df[comparison_df['Interval'] == interval_name].copy()

                    if not interval_df.empty:
                        st.write(f"#### {interval_name}")

                        # Table for Positive Ratios, sorted by least - Display requested columns
                        st.write(f"##### Positive Ratios (Sorted by Least)")
                        pos_ratio_table = interval_df[['Instrument', 'Spike Pending Ratio (Pos)', 'Median Move (%)', 'Bars Since Last Pos Move', 'Avg Gap Pos Moves']].sort_values(by='Spike Pending Ratio (Pos)', ascending=True).reset_index(drop=True)
                        st.dataframe(pos_ratio_table, use_container_width=True)

                        # Table for Negative Ratios, sorted by least - Display requested columns
                        st.write(f"##### Negative Ratios (Sorted by Least)")
                        neg_ratio_table = interval_df[['Instrument', 'Spike Pending Ratio (Neg)', 'Median Move (%)', 'Bars Since Last Neg Move', 'Avg Gap Neg Moves']].sort_values(by='Spike Pending Ratio (Neg)', ascending=True).reset_index(drop=True)
                        st.dataframe(neg_ratio_table, use_container_width=True)
                    else:
                        st.info(f"No comparison data available for {interval_name}.")


            else:
                st.info("No comparison results to display. Analysis will run shortly...") # Message while waiting for first run

        # --- Pause before the next refresh ---
        time.sleep(REFRESH_INTERVAL_SECONDS)


# --- Other Analysis / Backtesting Tab (Tab 2 - Placeholder) ---
with tab2:
    st.header("Other Analysis / Backtesting")
    st.write("This tab is reserved for other analysis or strategy backtesting features.")
    st.write("The Spike Ratio Comparison is available in the first tab and refreshes periodically.")
    # You can add your previous strategy backtesting code here if desired.
    # For now, it's just a placeholder.
