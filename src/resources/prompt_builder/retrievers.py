from .contextual_prompt_builder import BaseContextRetriever
from typing import List, Dict, Any

class FileContextRetriever(BaseContextRetriever):
    """
    A simple context retriever that reads from a local file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve context from the specified file.

        :param query: The query (ignored in this simple implementation).
        :return: A list containing a single document with the file's content.
        """
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
            return [{
                'source': self.file_path,
                'content': content
            }]
        except FileNotFoundError:
            return [{
                'source': self.file_path,
                'content': f"Error: File not found at {self.file_path}"
            }]
