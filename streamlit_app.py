import streamlit as st
from datetime import datetime, timezone
from dukascopy_util import fetch_stock_indices_data # Assuming this is available from your files
import pandas as pd
import numpy as np # Import numpy for median and other calculations
import time # For potential future use, though not strictly needed for this static analysis

# --- Configuration ---
# Add BTC/USD to the instrument list and change E_NQ-10 to E_NQ-100
INSTRUMENT_LIST = ["BTC/USD", "E_XJO-ASX", "E_NQ-100"]
INTERVAL_OPTIONS = {
    "15 Minute": "15MIN",
    "1 Hour": "1HOUR",
    "1 Day": "1DAY"
}
# Fetch a reasonable amount of historical data for analysis
FETCH_LIMIT = 1000
# Multiplier for defining a "large" move - This will now be controlled by a slider

# --- Analysis Function (Keep the existing one) ---
def analyze_price_moves(df: pd.DataFrame, instrument_name: str, interval_name: str, multiplier: float):
    """
    Analyzes price movements for a given DataFrame and interval.

    Args:
        df (pd.DataFrame): DataFrame with historical price data, indexed by Date.
                          Must contain a 'Close' column.
        instrument_name (str): The name of the instrument (e.g., "BTC/USD").
        interval_name (str): The name of the interval (e.g., "15 Minute").
        multiplier (float): The multiplier for the median move to define a "large" move.

    Returns:
        dict: A dictionary containing the analysis results for one interval and instrument.
    """
    results = {
        "Instrument": instrument_name, # Include instrument name in results
        "Interval": interval_name,
        "Median Move (%)": None,
        "Total Bars": 0,
        "Large Pos Moves Count": 0,
        "Large Pos Moves Freq (%)": "N/A",
        "Avg Gap Pos Moves": "N/A",
        "Large Neg Moves Count": 0,
        "Large Neg Moves Freq (%)": "N/A",
        "Avg Gap Neg Moves": "N/A",
        "Last Pos Move Time": "Never", # Keep for internal calculation if needed, but won't display in comparison tables
        "Bars Since Last Pos Move": "N/A",
        "Last Neg Move Time": "Never", # Keep for internal calculation if needed, but won't display in comparison tables
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


    results["Status"] = "Success"
    return results

# --- Streamlit UI Setup ---
st.title("📊 Market Spike Ratio Comparison")

st.header("Spike Ratio Comparison Across Instruments and Intervals")
st.write("Compares the Spike Pending Ratios for positive and negative large moves across all instruments for each time interval, sorted by ratio.")

# Add a slider for the large move multiplier
large_move_multiplier_comparison = st.slider(
    "Large Move Multiplier (x Median) for Comparison",
    min_value=0.1, # Minimum value for the slider
    max_value=5.0, # Maximum value (adjust as needed)
    value=1.33,    # Default value
    step=0.01,     # Step size
    key='large_move_multiplier_comparison' # Use a unique key for this slider
)

if st.button("Run Spike Ratio Comparison", key='run_spike_comparison'):
    st.write("🔹 Running comparison analysis for all instruments and intervals...")

    all_comparison_results = [] # List to hold results for all instruments and intervals

    # Use a spinner for the entire comparison process
    with st.spinner(f"Analyzing spike ratios for all instruments with multiplier {large_move_multiplier_comparison:.2f}..."):
        # Iterate through all instruments
        for instrument in INSTRUMENT_LIST:
            # Iterate through all intervals
            for interval_name, interval_key in INTERVAL_OPTIONS.items():
                try:
                    df_interval = fetch_stock_indices_data(
                        instrument=instrument,
                        offer_side="B", # Using Bid side for consistency
                        interval=interval_key,
                        limit=FETCH_LIMIT, # Fetch a good amount of data
                        time_direction="P"
                    )

                    if not df_interval.empty:
                        # Perform the analysis, passing instrument name
                        results = analyze_price_moves(df_interval, instrument, interval_name, large_move_multiplier_comparison)
                        # Store only the relevant comparison data
                        all_comparison_results.append({
                            "Instrument": instrument,
                            "Interval": interval_name,
                            "Spike Pending Ratio (Pos)": results.get("Spike Pending Ratio (Pos)", "N/A"), # Use .get to handle potential missing keys
                            "Spike Pending Ratio (Neg)": results.get("Spike Pending Ratio (Neg)", "N/A"),
                            "Median Move (%)": results.get("Median Move (%)", "N/A"), # Include Median Move (%)
                            "Bars Since Last Pos Move": results.get("Bars Since Last Pos Move", "N/A"), # Include Bars Since Last Pos Move
                            "Bars Since Last Neg Move": results.get("Bars Since Last Neg Move", "N/A"), # Include Bars Since Last Neg Move
                            "Status": results.get("Status", "Unknown")
                        })
                    else:
                         all_comparison_results.append({
                            "Instrument": instrument,
                            "Interval": interval_name,
                            "Spike Pending Ratio (Pos)": "N/A",
                            "Spike Pending Ratio (Neg)": "N/A",
                            "Median Move (%)": "N/A", # Include Median Move (%)
                            "Bars Since Last Pos Move": "N/A", # Include Bars Since Last Pos Move
                            "Bars Since Last Neg Move": "N/A", # Include Bars Since Last Neg Move
                            "Status": "No Data"
                        })


                except Exception as e:
                    # Append an error result for this instrument and interval
                    all_comparison_results.append({
                        "Instrument": instrument,
                        "Interval": interval_name,
                        "Spike Pending Ratio (Pos)": "N/A",
                        "Spike Pending Ratio (Neg)": "N/A",
                        "Median Move (%)": "N/A", # Include Median Move (%)
                        "Bars Since Last Pos Move": "N/A", # Include Bars Since Last Pos Move
                        "Bars Since Last Neg Move": "N/A", # Include Bars Since Last Neg Move
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
        comparison_df['Bars Since Last Pos Move'] = pd.to_numeric(comparison_df['Bars Since Last Pos Move'], errors='coerce').fillna("N/A").astype(str).replace('nan', 'N/A') # Handle potential NaN after coerce
        comparison_df['Bars Since Last Neg Move'] = pd.to_numeric(comparison_df['Bars Since Last Neg Move'], errors='coerce').fillna("N/A").astype(str).replace('nan', 'N/A') # Handle potential NaN after coerce


        # Display separate tables for each interval and ratio type, sorted by ratio
        for interval_name in INTERVAL_OPTIONS.keys():
            # Filter data for the current interval
            interval_df = comparison_df[comparison_df['Interval'] == interval_name].copy()

            if not interval_df.empty:
                st.write(f"#### {interval_name}")

                # Table for Positive Ratios, sorted by least - Include Median Move (%) and Bars Since Last Moves
                st.write(f"##### Positive Ratios (Sorted by Least)")
                pos_ratio_table = interval_df[['Instrument', 'Spike Pending Ratio (Pos)', 'Median Move (%)', 'Bars Since Last Pos Move', 'Bars Since Last Neg Move', 'Status']].sort_values(by='Spike Pending Ratio (Pos)', ascending=True).reset_index(drop=True)
                st.dataframe(pos_ratio_table, use_container_width=True)

                # Table for Negative Ratios, sorted by least - Include Median Move (%) and Bars Since Last Moves
                st.write(f"##### Negative Ratios (Sorted by Least)")
                neg_ratio_table = interval_df[['Instrument', 'Spike Pending Ratio (Neg)', 'Median Move (%)', 'Bars Since Last Pos Move', 'Bars Since Last Neg Move', 'Status']].sort_values(by='Spike Pending Ratio (Neg)', ascending=True).reset_index(drop=True)
                st.dataframe(neg_ratio_table, use_container_width=True)
            else:
                st.info(f"No comparison data available for {interval_name}.")


    else:
        st.info("No comparison results to display. Click 'Run Spike Ratio Comparison' to start.")

