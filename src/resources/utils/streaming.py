"""Streaming related utility functions"""

from langgraph.types import StreamWriter


async def stream_text_as_sse(
    text: str,
    writer: StreamWriter,
    event_name: str = "token",
    chunk_size: int = 5,
    send_stream_end_event=False,
):
    """
    Stream a string in chunks as server-sent events (SSE).

    Args:
        text (str): The full text to stream.
        writer (StreamWriter): An instance of LangGraph's stream writer.
        event_name (str, optional): The event key for each chunk. Defaults to "token".
        chunk_size (int, optional): Number of characters to send per chunk. Defaults to 5.
        send_stream_end_event (bool, optional): If True, sends a final {"end": "..."} event after streaming. Defaults to False.
                                    We use this to notify the front end that streaming is done.

    Example:
        await stream_sse_tokens("Hello world!", writer_func, chunk_size=3)
        # Will send chunks: "Hel", "lo ", "wor", "ld!"
    """
    for i in range(0, len(text), chunk_size):
        writer({event_name: text[i : i + chunk_size]})

    if send_stream_end_event:
        writer({"end": "Streaming finished"})
