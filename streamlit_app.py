import streamlit as st
from datetime import datetime, timezone
from dukascopy_util import fetch_stock_indices_data # Assuming this is available from your files
import pandas as pd
import numpy as np # Import numpy for median and other calculations
import time # For potential future use, though not strictly needed for this static analysis

# --- Configuration ---
# Add BTC/USD to the instrument list
INSTRUMENT_LIST = ["BTC/USD", "E_XJO-ASX", "E_NQ-10"]
INTERVAL_OPTIONS = {
    "15 Minute": "15MIN",
    "1 Hour": "1HOUR",
    "1 Day": "1DAY"
}
# Fetch a reasonable amount of historical data for analysis
FETCH_LIMIT = 1000
# Multiplier for defining a "large" move - This will now be controlled by a slider

# --- Analysis Function ---
def analyze_price_moves(df: pd.DataFrame, interval_name: str, multiplier: float):
    """
    Analyzes price movements for a given DataFrame and interval.

    Args:
        df (pd.DataFrame): DataFrame with historical price data, indexed by Date.
                          Must contain a 'Close' column.
        interval_name (str): The name of the interval (e.g., "15 Minute").
        multiplier (float): The multiplier for the median move to define a "large" move.

    Returns:
        dict: A dictionary containing the analysis results for one interval.
    """
    results = {
        "Interval": interval_name, # Changed key for table header
        "Median Move (%)": None, # Changed key and name
        "Total Bars": 0,
        "Large Pos Moves Count": 0, # Changed key
        "Large Pos Moves Freq (%)": "N/A", # Changed key and name
        "Avg Gap Pos Moves": "N/A", # Changed key
        "Large Neg Moves Count": 0, # Changed key
        "Large Neg Moves Freq (%)": "N/A", # Changed key and name
        "Avg Gap Neg Moves": "N/A", # Changed key
        "Last Pos Move Time": "Never", # Changed key
        "Bars Since Last Pos Move": "N/A", # Changed key
        "Last Neg Move Time": "Never", # Changed key
        "Bars Since Last Neg Move": "N/A", # Changed key
        "Spike Pending Ratio (Pos)": "N/A", # Changed key and name
        "Spike Pending Ratio (Neg)": "N/A" # Added key for negative ratio
    }

    if df.empty or 'Close' not in df.columns:
        results["Status"] = "No Data" # Add status for table
        return results

    results["Total Bars"] = len(df)

    # Calculate percentage price change between consecutive bars
    percentage_price_change = df['Close'].pct_change().dropna() * 100

    if percentage_price_change.empty:
         results["Status"] = "No Price Change Data" # Add status for table
         return results

    # Calculate the absolute percentage price change
    abs_percentage_price_change = percentage_price_change.abs()

    # Calculate the median absolute percentage price move
    median_move_percent = abs_percentage_price_change.median()
    results["Median Move (%)"] = f"{median_move_percent:.4f}" if median_move_percent is not None else None # Format here

    if median_move_percent is None or median_move_percent == 0:
        results["Status"] = "Median Move is Zero" # Add status for table
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


    # Find the last occurrence
    if not large_positive_moves.empty:
        last_pos_idx = large_positive_moves.index[-1]
        results["Last Pos Move Time"] = last_pos_idx.strftime('%Y-%m-%d %H:%M:%S')
        pos_in_df = df.index.get_loc(last_pos_idx)
        results["Bars Since Last Pos Move"] = len(df) - 1 - pos_in_df


    if not large_negative_moves.empty:
        last_neg_idx = large_negative_moves.index[-1]
        results["Last Neg Move Time"] = last_neg_idx.strftime('%Y-%m-%d %H:%M:%S')
        neg_in_df = df.index.get_loc(last_neg_idx)
        results["Bars Since Last Neg Move"] = len(df) - 1 - neg_in_df

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

    # Negative Ratio (New)
    neg_count = results["Large Neg Moves Count"]
    bars_since_neg = results["Bars Since Last Neg Move"]

    if isinstance(bars_since_neg, int):
        denominator_neg = bars_since_neg + 1
        results["Spike Pending Ratio (Neg)"] = neg_count / denominator_neg
    elif neg_count > 0:
         results["Spike Pending Ratio (Neg)"] = 0.0

    if isinstance(results["Spike Pending Ratio (Neg)"], (int, float)):
        results["Spike Pending Ratio (Neg)"] = f"{results['Spike Pending Ratio (Neg)']:.4f}"


    results["Status"] = "Success" # Add success status
    return results

# --- Streamlit UI Setup ---
st.title("📊 Market Analysis Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["🔬 Price Move Analysis", "📈 Strategy Backtesting"]) # Renamed tabs

# --- Price Move Analysis Tab ---
with tab1:
    st.header("Price Movement Analysis by Period")
    st.write("Analyzing historical price movements for selected instruments and periods.")

    # Inputs for Analysis
    selected_instrument_analysis = st.selectbox(
        "Select Instrument",
        INSTRUMENT_LIST,
        index=0,
        key='analysis_instrument'
    )

    # Add a slider for the large move multiplier
    large_move_multiplier_slider = st.slider(
        "Large Move Multiplier (x Median)",
        min_value=0.1, # Minimum value for the slider
        max_value=5.0, # Maximum value (adjust as needed)
        value=1.33,    # Default value
        step=0.01,     # Step size
        key='large_move_multiplier_slider'
    )

    st.write(f"A 'large move' is defined as a price change > **{large_move_multiplier_slider:.2f}x** the median **percentage** price move for that period.")


    if st.button("Run Price Move Analysis", key='run_price_analysis'):
        if not selected_instrument_analysis:
            st.warning("Please select an instrument.")
            st.stop()

        # Use a spinner to indicate processing without verbose messages
        with st.spinner(f"Running analysis for {selected_instrument_analysis} with multiplier {large_move_multiplier_slider:.2f}..."):
            analysis_results_list = [] # List to hold results for the table

            # Iterate through the desired intervals and fetch data for each
            for interval_name, interval_key in INTERVAL_OPTIONS.items():
                try:
                    df_interval = fetch_stock_indices_data(
                        instrument=selected_instrument_analysis,
                        offer_side="B", # Using Bid side for consistency in analysis
                        interval=interval_key,
                        limit=FETCH_LIMIT, # Fetch a good amount of data
                        time_direction="P"
                    )

                    # Perform the analysis for this interval, passing the slider value
                    results = analyze_price_moves(df_interval, interval_name, large_move_multiplier_slider)
                    analysis_results_list.append(results)

                except Exception as e:
                    # Append an error result to the list for this interval
                    analysis_results_list.append({
                        "Interval": interval_name,
                        "Status": f"Error: {e}"
                    })

        st.subheader("📈 Analysis Results")

        if analysis_results_list:
            # Convert the list of dictionaries to a pandas DataFrame
            analysis_df = pd.DataFrame(analysis_results_list)

            # Set the 'Interval' column as the index before transposing
            analysis_df = analysis_df.set_index('Interval')

            # Transpose the DataFrame
            analysis_df_transposed = analysis_df.T

            # Display the transposed DataFrame as a table
            st.dataframe(analysis_df_transposed, use_container_width=True)

        else:
            st.info("No analysis results to display. Select an instrument and click 'Run Price Move Analysis'.")


# --- Strategy Backtesting Tab (Placeholder) ---
# This tab remains as a placeholder for your previous strategy analysis code
with tab2:
    st.header("Strategy Backtesting and Analysis")
    st.write("This tab is reserved for strategy backtesting features.")
    st.write("You can integrate the code from your previous analysisapp.py and charting.py here.")
    # Add your previous strategy backtesting code here if desired.
    # For now, it's just a placeholder.
    # Example:
    # instrument = st.selectbox("Instrument", instrument_list, index=0, key='analyzer_instrument')
    # ... rest of your strategy analysis code ...
