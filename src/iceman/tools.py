import base64
import os
import re
import time
from typing import Optional
import csv

import requests
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

fc = Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY"))
vision = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, timeout=120, max_retries=0)


def _invoke_with_retry(runnable, input_, attempts: int = 4):
    """Retry on 503 (model overloaded), but fail fast on 429 (quota)."""
    for attempt in range(attempts):
        try:
            return runnable.invoke(input_)
        except Exception as e:
            is_overloaded = "503" in str(e) or "UNAVAILABLE" in str(e)
            if not is_overloaded or attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


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
    for attempt in range(6):
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            break
        except requests.exceptions.ConnectionError:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))

    mime = resp.headers.get("Content-Type", "image/jpeg")
    b64 = base64.b64encode(resp.content).decode()

    msg = HumanMessage(content=[
        {"type": "text", "text": "This image shows an email address. Reply with only the email address exactly as written, or NONE if there isn't one."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])
    content = _invoke_with_retry(vision, [msg]).content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = content.strip()
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    return match.group(0) if match else ""


@tool
def fetch_listing(url: str) -> dict:
    """Fetch a faculty listing page and return its markdown and raw HTML.

    Args:
        url: The faculty page URL.
    """
    result = fc.scrape(url, formats=["markdown", "html"], only_main_content=False)
    return {"markdown": result.markdown or "", "html": result.html or ""}


class FacultyRecord(BaseModel):
    name: str
    title: Optional[str] = None
    dept: Optional[str] = None
    profile_url: Optional[str] = None
    email_text: Optional[str] = Field(
        None,
        description="The email exactly as written on the page, e.g. 'name [AT] iisc.ac.in' or 'name@iisc.ac.in'",
    )
    email_image_url: Optional[str] = Field(
        None,
        description="URL of an image showing the email, if the email is not plain text",
    )
    field_of_work: list[str] = Field(
        default_factory=list,
        description="Research topics, one entry per topic. Empty list if none are shown.",
    )


class FacultyList(BaseModel):
    faculty: list[FacultyRecord]


EXTRACT_PROMPT = """You are extracting faculty members from a university department web page.
Return every faculty member listed on the page. Do not invent people or fields.
Use null for anything not on the page.
For email_text, copy only an email address that appears on the page, with its [AT] or <> markers if present. A username or profile URL is not an email: if no full address is shown, use null.
For email_image_url, use the URL of the image shown in the same block as the email.
For profile_url, use the link to the person's own page.
For field_of_work, return each research topic as a separate item in the list.

PAGE MARKDOWN:
{markdown}
"""


@tool
def extract_faculty(url: str) -> list[dict]:
    """Extract every faculty member from a listing page into a fixed schema.

    Args:
        url: The faculty page URL.
    """
    result = fc.scrape(url, formats=["markdown"], only_main_content=False)
    markdown = result.markdown or ""
    structured = vision.with_structured_output(FacultyList)
    response = _invoke_with_retry(structured, EXTRACT_PROMPT.format(markdown=markdown))
    records = [r.model_dump() for r in response.faculty]
    for r in records:
        t = (r["email_text"] or "").lower()
        if not ("@" in t or "[at]" in t or "(at)" in t):
            r["email_text"] = None
    return records


class ProfileDetails(BaseModel):
    dept: Optional[str] = None
    email_text: Optional[str] = Field(
        None,
        description="An email address shown on this page, as written, with (at), [AT] or <> markers if present. A username or URL is not an email.",
    )
    field_of_work: list[str] = Field(
        default_factory=list,
        description="Research topics, one entry per topic.",
    )


PROFILE_PROMPT = """You are reading one faculty member's profile page.
Extract this person's details. Do not invent information. Use null for anything not on the page.
For field_of_work, return each research topic as a separate item in the list.

PAGE MARKDOWN:
{markdown}
"""


def _has_email(r: dict) -> bool:
    return bool(r.get("email_text") or r.get("email_image_url"))


def _needs_profile(r: dict) -> bool:
    return not _has_email(r) or not r.get("field_of_work")


def _valid_email_text(text: str | None) -> bool:
    t = (text or "").lower()
    return "@" in t or "[at]" in t or "(at)" in t


def enrich_records(records: list[dict], max_profiles: int = 50, delay: float = 2.0) -> list[dict]:
    """Fill missing email or research fields from each person's profile page.

    Only people with a missing email or missing research topics are fetched.
    Values already in the listing record are never overwritten.
    A profile that fails after retries is skipped and the run continues.
    """
    done = 0
    for r in records:
        if not _needs_profile(r) or not r.get("profile_url"):
            continue
        if done >= max_profiles:
            break

        page = None
        for attempt in range(3):
            try:
                page = fc.scrape(r["profile_url"], formats=["markdown"], only_main_content=True)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"skipped {r['name']}: {type(e).__name__}")
                else:
                    time.sleep(2 ** attempt)
        if page is None:
            continue

        structured = vision.with_structured_output(ProfileDetails)
        details = _invoke_with_retry(
            structured, PROFILE_PROMPT.format(markdown=page.markdown or "")
        ).model_dump()

        if not _has_email(r) and _valid_email_text(details["email_text"]):
            r["email_text"] = details["email_text"]
        if not r.get("field_of_work") and details["field_of_work"]:
            r["field_of_work"] = details["field_of_work"]
        if not r.get("dept") and details["dept"]:
            r["dept"] = details["dept"]

        done += 1
        time.sleep(delay)
    return records

def fill_email_from_images(records: list[dict], max_images: int = 10, delay: float = 3.0) -> list[dict]:
    """For records with no text email, read the email from its image.

    Keeps the result only if it contains an @. Stops after max_images attempts.
    Pauses between reads so the server does not block the client.
    Values already present are not changed.
    """
    attempted = 0
    for r in records:
        if r.get("email_text") or not r.get("email_image_url"):
            continue
        if attempted >= max_images:
            break
        if attempted > 0:
            time.sleep(delay)
        attempted += 1
        try:
            address = read_email_image.invoke({"image_url": r["email_image_url"]})
        except Exception as e:
            print(f"image read failed for {r['name']}: {str(e)[:200]}")
            continue
        if address and "@" in address:
            r["email_text"] = address
    print(f"attempted {attempted} image reads")
    return records


CSV_COLUMNS = ["name", "dept", "title", "email", "field_of_work", "profile_url"]


def records_to_rows(records: list[dict]) -> list[dict]:
    """Flatten records into CSV rows with the columns the user asked for."""
    rows = []
    for r in records:
        rows.append({
            "name": r.get("name") or "",
            "dept": r.get("dept") or "",
            "title": r.get("title") or "",
            "email": r.get("email_text") or r.get("email_image_url") or "",
            "field_of_work": "; ".join(r.get("field_of_work") or []),
            "profile_url": r.get("profile_url") or "",
        })
    return rows


@tool
def save_faculty_csv(records: list[dict], path: str = "faculty.csv") -> str:
    """Save faculty records to a CSV file.

    Args:
        records: The faculty records from extract_faculty, enriched and filled.
        path: Where to write the CSV, relative to the project folder.
    """
    rows = records_to_rows(records)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return f"Saved {len(rows)} records to {path}"