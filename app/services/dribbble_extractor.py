import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, HttpUrl
from crawl4ai import AsyncWebCrawler

class DesignInfo(BaseModel):
    """Model representing design information from Dribbble."""
    url: HttpUrl
    title: str
    author: str
    image_urls: List[str]
    colors: List[str]
    description: str
    raw_markdown: str

class DribbbleExtractor:
    """Service for extracting design information from Dribbble URLs."""
    
    @classmethod
    async def extract_design_info(cls, url: str) -> DesignInfo:
        """
        Extract design information from a Dribbble URL.
        
        Args:
            url: URL of the Dribbble design page
            
        Returns:
            DesignInfo: Extracted design information
            
        Raises:
            ValueError: If the URL is invalid or extraction fails
        """
        if not url.startswith("https://dribbble.com/shots/"):
            raise ValueError("Invalid Dribbble design URL")
            
        crawler = AsyncWebCrawler(
            headless=True,
            browser="chromium",
            stealth_mode=True,
            verbose=False
        )
        
        try:
            result = await crawler.arun(
                url=url,
                extract_rules={
                    "title": "h1",
                    "author": "a.shot-avatar",
                    "colors": "a.color-chip",
                },
                include_raw_content=True
            )
            
            markdown_content = result.markdown
            
            # Extract basic information
            title = cls._extract_title(markdown_content)
            author = cls._extract_author(markdown_content)
            image_urls = cls._extract_image_urls(markdown_content)
            colors = cls._extract_colors(markdown_content)
            description = cls._extract_description(markdown_content)
            
            return DesignInfo(
                url=url,
                title=title,
                author=author,
                image_urls=image_urls,
                colors=colors,
                description=description,
                raw_markdown=markdown_content
            )
            
        except Exception as e:
            raise ValueError(f"Failed to extract design info: {str(e)}")
    
    @staticmethod
    def _extract_title(markdown: str) -> str:
        """Extract title from markdown content."""
        title_match = re.search(r'<h1[^>]*>(.*?)</h1>', markdown, re.DOTALL)
        return re.sub('<[^<]+?>', '', title_match.group(1)).strip() if title_match else ""
    
    @staticmethod
    def _extract_author(markdown: str) -> str:
        """Extract author from markdown content."""
        author_match = re.search(
            r'<a[^>]*class=["\']shot-avatar[^>]*>.*?<img[^>]*alt=["\'](.*?)["\']', 
            markdown, 
            re.DOTALL
        )
        return author_match.group(1).strip() if author_match else ""
    
    @staticmethod
    def _extract_image_urls(markdown: str) -> List[str]:
        """Extract and clean image URLs from markdown content, excluding avatar images."""
        # Extract from markdown ![alt](url)
        markdown_images = re.findall(r'!\[[^\]]*\](\(https?://[^\s\)]+)', markdown)
        # Extract from HTML <img> tags
        html_images = re.findall(
            r'<img[^>]*src=["\'](https?://[^"\'\s)]+)["\']', 
            markdown, 
            re.IGNORECASE
        )
        
        # Clean and filter URLs
        cleaned_urls = []
        # Patterns to exclude
        exclude_patterns = [
            r'https?://cdn\.dribbble\.com/users/',  # Avatar images
            r'https?://dribbble\.com/assets/icons/',  # Icons
            r'\.(svg|ico)$'  # SVG and ICO files
        ]
        
        for url in markdown_images + html_images:
            # Remove any leading or trailing parentheses, quotes, or whitespace
            cleaned = url.strip().strip('\"\'()')
            
            # Skip unwanted URLs
            if any(re.search(pattern, cleaned, re.IGNORECASE) for pattern in exclude_patterns):
                continue
                
            # Only add if it's a valid URL and passes additional filters
            if cleaned.startswith(('http://', 'https://')):
                cleaned_urls.append(cleaned)
        
        # Remove duplicates while preserving order
        seen = set()
        return [url for url in cleaned_urls if not (url in seen or seen.add(url))]
    
    @staticmethod
    def _extract_colors(markdown: str) -> List[str]:
        """Extract color palette from markdown content."""
        # Extract from data-color attributes
        color_matches = re.findall(
            r'data-color=["\']#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})["\']', 
            markdown
        )
        colors = list({f"#{color.upper()}" for color in color_matches})
        
        # Fallback: Extract from markdown color codes
        if not colors:
            color_matches = re.findall(
                r'\[#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})\]', 
                markdown
            )
            colors = list({f"#{color.upper()}" for color in color_matches})
            
        return colors
    
    @staticmethod
    def _extract_description(markdown: str) -> str:
        """Extract description text from markdown content."""
        # This is a simplified version - you can enhance this further
        # based on your specific needs
        paragraphs = []
        for para in re.split(r'\n\n|\n', markdown):
            para = para.strip()
            if len(para) > 30:
                # Clean up markdown formatting
                clean_para = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', para)  # Remove markdown links
                clean_para = re.sub(r'[#*_`~]', '', clean_para)  # Remove markdown formatting
                clean_para = re.sub(r'\s+', ' ', clean_para).strip()
                if len(clean_para) > 30:
                    paragraphs.append(clean_para)
        
        # Return the first few paragraphs as description
        return "\n\n".join(paragraphs[:3]) if paragraphs else ""
