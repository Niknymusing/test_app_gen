"""Time related helper functions collection that can be useful for agents"""

import httpx
from email.utils import parsedate_to_datetime


async def fetch_utc_date_from_google() -> str:
    """
    Fetches current UTC date from Google's HTTP Date header.
    Returns a string in MM/DD/YYYY format, or 'Unknown' on failure.
    """
    url = "https://www.google.com"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.head(url)
            date_header = resp.headers.get("date")
            if date_header:
                dt = parsedate_to_datetime(date_header)
                return dt.strftime("%m/%d/%Y")
            return "Unknown"
    except Exception as e:
        print(f"Error fetching date: {e}")
        return "Unknown"
