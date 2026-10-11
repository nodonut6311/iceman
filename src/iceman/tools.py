import base64
import os
import re
import time

import requests
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

fc = Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY"))
vision = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)


@tool
def scrape_markdown(url: str, main_only: bool = True) -> str:
    """Scrape a web page and return its clean markdown content.

    Args:
        url: The page URL to scrape.
        main_only: If True, drop navigation, footers, and sidebars.
    """
    result = fc.scrape(
        url,
        formats=["markdown"],
        only_main_content=main_only,
    )
    return result.markdown or "No content returned."


@tool
def read_email_image(image_url: str) -> str:
    """Read an email address from an image, such as a faculty card's email image.

    Args:
        image_url: Direct URL of the image.
    """
    # Retry on connection resets, which the NUS server produces intermittently.
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(image_url, timeout=20)
            resp.raise_for_status()
            break
        except requests.exceptions.ConnectionError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    mime = resp.headers.get("Content-Type", "image/jpeg")
    b64 = base64.b64encode(resp.content).decode()

    msg = HumanMessage(content=[
        {"type": "text", "text": "This image shows an email address. Reply with only the email address exactly as written, or NONE if there isn't one."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])
    content = vision.invoke([msg]).content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = content.strip()
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    return match.group(0) if match else ""