import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urlparse
import gc


def extract_dataset_id(url):
    """
    Extract the dataset ID from Open Data URL.

    Arguments:
        url: The URL of the dataset page.

    Returns:
        The dataset ID as a lowercase string.
    """
    m = re.search(r"/([a-z0-9]{4}-[a-z0-9]{4})(?:/|$|\?|#)", url, re.I)
    if not m:
        raise ValueError("Could not extract dataset id from URL")
    return m.group(1).lower()


def extract_host(url):
    """
    Extract host city from Open Data URL.
    """
    host = urlparse(url).netloc
    if not host:
        raise ValueError("Could not extract host from URL")
    return host


def clean_empty_list(val):
    """
    Convert lists that are empty or contain only empty/whitespace strings to None.

    Arguments:
        val: The value to check and clean.

    Returns:
        The original value if it's not an empty list or a list of empty/whitespace strings, otherwise None.
    """
    if isinstance(val, list):
        if all((not str(x).strip()) for x in val):
            return None
    return val


def parse_dataset_html(html):
    """
    Scrape the dataset page HTML to extract rows, columns, and each row description.

    Arguments:
        html: The HTML content of the dataset page.

    Returns:
        A tuple of (rows, columns, each_row_description).
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    rows_val = None
    cols_val = None
    each_row_val = None

    m_rows = re.search(r"Rows\s+([\d,]+\.?\d*[KMBkmb]?)", text, re.I)
    if m_rows:
        raw = m_rows.group(1).replace(",", "")
        suffix = raw[-1].upper() if raw and raw[-1].upper() in "KMB" else ""
        num = float(raw[:-1] if suffix else raw)
        rows_val = int(num * {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1))

    m_cols = re.search(r"Columns\s+(\d+)", text, re.I)
    if m_cols:
        cols_val = m_cols.group(1)

    m_each = re.search(
        r"Each row is a\s+(.*?)(?:\s+Columns\b|\s+Column\b)",
        text,
        re.I,
    )
    if m_each:
        each_row_val = m_each.group(1).strip()

    return rows_val, cols_val, each_row_val


async def fetch_full_dataset_info(url):
    """
    Fetch dataset information from both the API and by scraping the dataset page.

    Arguments:
        url: The URL of the dataset page.

    Returns:
        A DataFrame containing combined information from the API and HTML scraping.
    """
    dataset_id = extract_dataset_id(url)
    host = extract_host(url)

    base = "https://api.us.socrata.com/api/catalog/v1"
    params = {"ids": dataset_id}

    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])
    api_rows = []

    for item in results:
        resource = item.get("resource", {})
        classification = item.get("classification", {})

        api_rows.append({
            "id": resource.get("id", dataset_id),
            "name": resource.get("name"),
            "description": resource.get("description"),
            "agency": resource.get("attribution"),
            "category": classification.get("domain_category"),
            "domain_tags": classification.get("domain_tags"),
            "type": resource.get("type"),
            "columns_field_name": resource.get("columns_field_name"),
            "columns_name": resource.get("columns_name"),
            "columns_description": resource.get("columns_description"),
            "updatedAt": resource.get("updatedAt"),
            "createdAt": resource.get("createdAt"),
            "page_url": item.get("permalink"),
        })

    api_columns = [
        "id",
        "name",
        "description",
        "agency",
        "category",
        "domain_tags",
        "type",
        "columns_field_name",
        "columns_name",
        "columns_description",
        "updatedAt",
        "createdAt",
        "page_url",
    ]
    df_api = pd.DataFrame(api_rows, columns=api_columns)

    for col in ["columns_field_name", "columns_name", "columns_description"]:
        if col in df_api.columns:
            df_api[col] = df_api[col].apply(clean_empty_list)

    if df_api.empty:
        df_api = pd.DataFrame([
            {
                "id": dataset_id,
                "name": None,
                "description": None,
                "agency": None,
                "category": None,
                "domain_tags": None,
                "type": None,
                "columns_field_name": None,
                "columns_name": None,
                "columns_description": None,
                "updatedAt": None,
                "createdAt": None,
                "page_url": f"https://{host}/d/{dataset_id}",
            }
        ])

    page_url = f"https://{host}/d/{dataset_id}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1500)

            html = await page.content()
            rows_val, cols_val, each_row_val = parse_dataset_html(html)

            df_html = pd.DataFrame([
                {
                    "id": dataset_id,
                    "rows": rows_val,
                    "columns": cols_val,
                    "each_row": each_row_val,
                }
            ])

        finally:
            await context.close()
            await browser.close()

    return pd.concat([df_api, df_html], axis=1)
