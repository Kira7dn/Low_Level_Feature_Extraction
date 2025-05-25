from pydantic import BaseModel, HttpUrl
from typing import List


class DesignInfo(BaseModel):
    """Model representing design information from Dribbble."""

    url: HttpUrl
    title: str
    author: str
    image_urls: List[str]
    colors: List[str]
    description: str
    raw_markdown: str
