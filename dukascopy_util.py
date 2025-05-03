"""
dukascopy_util.py

Utility module to fetch historical chart data from Dukascopy via JSONP.
"""
import requests
import random
import string
import time
import json
import pandas as pd

# Default browser-like User-Agent
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/134.0.0.0 Safari/537.36"
)


def generate_jsonp_callback() -> str:
    """Generate a random JSONP callback function name."""
    prefix = "_callbacks____"
    suffix = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return prefix + suffix


def get_current_utc_timestamp_ms() -> str:
    """Return current UTC timestamp in milliseconds as string."""
    return str(int(time.time() * 1000))


def extract_json_from_jsonp(jsonp_text: str) -> object:
    """Extract the JSON payload from a JSONP response text."""
    start = jsonp_text.find("(") + 1
    end = jsonp_text.rfind(")")
    if start <= 0 or end <= start:
        raise ValueError("Invalid JSONP response format.")
    payload = jsonp_text[start:end]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")


def fetch_stock_indices_data(
    instrument: str,
    offer_side: str = "B",
    interval: str = "15MIN",
    limit: int = 25,
    time_direction: str = "P",
    user_agent: str = DEFAULT_USER_AGENT
) -> pd.DataFrame:
    """
    Fetch historical price data for a given instrument from Dukascopy.

    Parameters
    ----------
    instrument : str
        Instrument symbol, e.g., "EUR/USD".
    offer_side : str, default "B"
        "B" for bid, "A" for ask.
    interval : str, default "15MIN"
        Data interval: "TICK", "1SEC", "10SEC", "30SEC", "1MIN", "5MIN", "15MIN", "30MIN", "1H", "4H", "1D", "1W", "1M".
    limit : int, default 25
        Number of data points to retrieve.
    time_direction : str, default "P"
        "P" for past data; "N" for future (rarely used).
    user_agent : str
        HTTP User-Agent header to mimic a browser.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with columns [Open, High, Low, Close, Volume].
    """
    # Build URL
    timestamp = get_current_utc_timestamp_ms()
    callback = generate_jsonp_callback()
    encoded_inst = requests.utils.quote(instrument, safe="")
    url = (
        "https://freeserv.dukascopy.com/2.0/index.php"
        f"?path=chart%2Fjson3"
        f"&instrument={encoded_inst}"
        f"&offer_side={offer_side}"
        f"&interval={interval}"
        "&splits=true"
        "&stocks=true"
        f"&limit={limit}"
        f"&time_direction={time_direction}"
        f"&timestamp={timestamp}"
        f"&jsonp={callback}"
    )

    # Headers to mimic a browser request
    headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Referer": "https://freeserv.dukascopy.com",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "no-cors",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    # Perform HTTP GET
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    jsonp_text = response.text

    # Parse JSONP and extract data
    data = extract_json_from_jsonp(jsonp_text)

    # Normalize into DataFrame
    if isinstance(data, list) and data and isinstance(data[0], list):
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
        # Identify the time column
        time_col = "timestamp" if "timestamp" in df.columns else next(
            (col for col in df.columns if "date" in col.lower()), "timestamp"
        )
        df[time_col] = pd.to_datetime(df[time_col], unit="ms")
        df.rename(columns={time_col: "timestamp"}, inplace=True)
    else:
        raise ValueError(f"Unexpected JSON format: {type(data)}")

    # Finalize DataFrame
    df.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)
    df.set_index("Date", inplace=True)

    # Sort the index to ensure chronological order (ADDED)
    df.sort_index(ascending=True, inplace=True)


    return df