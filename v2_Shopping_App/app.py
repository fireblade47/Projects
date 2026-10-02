"""Visual shopping assistant for Indian e-commerce sites.

Run with: streamlit run v2_Shopping_App/app.py

Environment variables:
  GROQ_API_KEY or GOOGLE_API_KEY/GEMINI_API_KEY - enables image-to-product recognition
  SERPAPI_API_KEY - enables live Google Shopping results for India
"""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from html import escape
from io import BytesIO
from typing import Any
from urllib.parse import quote_plus

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image


load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(page_title="ShopLens - Prototype Testing", page_icon="🛍️", layout="wide")


@dataclass
class Product:
    title: str
    price: float | None
    currency: str
    merchant: str
    url: str
    product_id: str | None = None
    thumbnail: str | None = None
    rating: float | None = None
    reviews: int | None = None
    delivery: str | None = None


# Retailer → search URL template. {q} is replaced with the URL-encoded query.
RETAILER_SEARCH_URLS: dict[str, str] = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "myntra": "https://www.myntra.com/{q}",
    "croma": "https://www.croma.com/searchB?q={q}%3Arelevance",
    "reliance": "https://www.reliancedigital.in/search?q={q}",
    "vijay sales": "https://www.vijaysales.com/search/{q}",
    "snapdeal": "https://www.snapdeal.com/search?keyword={q}",
    "nykaa": "https://www.nykaa.com/search/result/?q={q}",
    "tatacliq": "https://www.tatacliq.com/search/?searchCategory=all&text={q}",
    "ajio": "https://www.ajio.com/search/?text={q}",
    "meesho": "https://www.meesho.com/search?q={q}",
    "jiomart": "https://www.jiomart.com/search/{q}",
    "bigbasket": "https://www.bigbasket.com/ps/?q={q}",
    "blinkit": "https://blinkit.com/s/?q={q}",
    "zepto": "https://www.zepto.com/search?query={q}",
    "lenskart": "https://www.lenskart.com/search?q={q}",
    "boat": "https://www.boat-lifestyle.com/search?q={q}",
    "decathlon": "https://www.decathlon.in/search?query={q}",
}


def get_secret(name: str) -> str | None:
    names = [name]
    if name == "GEMINI_API_KEY":
        names.append("GOOGLE_API_KEY")
    value = next((os.getenv(key) for key in names if os.getenv(key)), None)
    if value:
        return value
    try:
        return next((st.secrets.get(key) for key in names if st.secrets.get(key)), None)
    except FileNotFoundError:
        return None


def api_error_message(response: requests.Response) -> str:
    try:
        error = response.json().get("error")
        detail = error.get("message") if isinstance(error, dict) else error
    except ValueError:
        detail = None
    return detail or f"API request failed with HTTP {response.status_code}."


def build_retailer_search_url(merchant: str, title: str) -> str:
    """Construct a working retailer search URL for any merchant.

    Google Shopping only returns Google-hosted product links for India, and the
    seller direct_link field is populated inconsistently. Building a search URL
    on the retailer's own domain guarantees a working, retailer-hosted link.
    """
    merchant_lower = (merchant or "").lower()
    template = next(
        (url for name, url in RETAILER_SEARCH_URLS.items() if name in merchant_lower),
        None,
    )
    query = quote_plus(title.strip())
    if template:
        return template.format(q=query)
    # Fallback: Google search scoped to the merchant's name to find the page.
    return f"https://www.google.com/search?q={query}+{quote_plus(merchant)}"


def enrich_with_retailer_links(products: list[Product]) -> list[Product]:
    """Ensure every product has a clickable link pointing to a retailer page.

    If SerpAPI's direct_link succeeded (rare for India), keep it. Otherwise fall
    back to a working retailer-scoped search URL.
    """
    enriched: list[Product] = []
    for product in products:
        url = product.url or ""
        # Treat Google-hosted URLs as unusable for the "go to retailer" flow.
        if not url or "google." in url or url.startswith("https://www.google."):
            url = build_retailer_search_url(product.merchant, product.title)
        enriched.append(replace(product, url=url))
    return enriched


def test_serpapi_connection() -> tuple[bool, str]:
    api_key = get_secret("SERPAPI_API_KEY")
    if not api_key or not api_key.strip():
        return False, "No SERPAPI_API_KEY was found in .env or Streamlit secrets."
    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_shopping", "q": "wireless headphones", "gl": "in",
                "hl": "en", "google_domain": "google.co.in", "api_key": api_key.strip(),
            },
            timeout=30,
        )
        if not response.ok:
            return False, f"Connection failed: {api_error_message(response)}"
        data = response.json()
        if data.get("error"):
            return False, f"Connection failed: {data['error']}"
        count = len(data.get("shopping_results", []))
        return True, f"Connected successfully. SerpAPI returned {count} India shopping listings."
    except requests.RequestException:
        return False, "Connection failed: network request to SerpAPI timed out or could not be completed."


def optimize_image_for_vision(image_bytes: bytes) -> tuple[bytes, str]:
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((1600, 1600))
        output = BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue(), "image/jpeg"


def no_key_result(provider: str) -> dict[str, Any]:
    key_name = "GROQ_API_KEY" if provider == "Groq" else "GOOGLE_API_KEY (or GEMINI_API_KEY)"
    return {
        "search_query": "product from uploaded image",
        "product_type": "Unknown product",
        "brand": None,
        "model": None,
        "attributes": [],
        "confidence": 0.0,
        "note": f"Add {key_name} to identify the image automatically.",
    }


def identify_product_with_groq(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        return no_key_result("Groq")

    prompt = """You are a visual shopping assistant for India. Identify the primary
product in this image. Read visible brand/model text where possible. Return ONLY valid
JSON with these fields: search_query (a precise ecommerce query), product_type, brand
(string or null), model (string or null), attributes (array of 3 short strings),
confidence (number 0 to 1), note (short string). Do not invent a brand or model."""
    optimized_bytes, optimized_mime_type = optimize_image_for_vision(image_bytes)
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:{optimized_mime_type};base64,{base64.b64encode(optimized_bytes).decode()}"
            }},
        ]}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_completion_tokens": 600,
    }
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        json=payload,
        timeout=(10, 90),
    )
    if not response.ok:
        raise RuntimeError(f"Groq could not analyze the image: {api_error_message(response)}")
    text = response.json()["choices"][0]["message"]["content"]
    return json.loads(text)


def identify_product_with_gemini(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return no_key_result("Gemini")
    prompt = """You are a visual shopping assistant for India. Identify the primary
product in this image. Read visible brand/model text where possible. Return ONLY valid
JSON with these fields: search_query (a precise ecommerce query), product_type, brand
(string or null), model (string or null), attributes (array of 3 short strings),
confidence (number 0 to 1), note (short string). Do not invent a brand or model."""
    optimized_bytes, optimized_mime_type = optimize_image_for_vision(image_bytes)
    model = get_secret("GEMINI_MODEL") or "gemini-3.6-flash"
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key.strip()},
        json={
            "contents": [{"parts": [
                {"text": prompt},
                {"inlineData": {
                    "mimeType": optimized_mime_type,
                    "data": base64.b64encode(optimized_bytes).decode(),
                }},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        },
        timeout=(10, 90),
    )
    if not response.ok:
        raise RuntimeError(f"Gemini ({model}) could not analyze the image: {api_error_message(response)}")
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def identify_product(image_bytes: bytes, mime_type: str, provider: str) -> dict[str, Any]:
    if provider == "Gemini":
        return identify_product_with_gemini(image_bytes, mime_type)
    return identify_product_with_groq(image_bytes, mime_type)


def to_number(value: Any) -> float | None:
    try:
        return float(str(value).replace("₹", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def search_live_products(query: str) -> list[Product]:
    api_key = get_secret("SERPAPI_API_KEY")
    if not api_key:
        return []
    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_shopping", "q": query, "gl": "in", "hl": "en",
            "google_domain": "google.co.in", "api_key": api_key,
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"SerpAPI search failed: {api_error_message(response)}")
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"SerpAPI search failed: {payload['error']}")
    products = []
    for item in payload.get("shopping_results", []):
        price = to_number(item.get("extracted_price") or item.get("price"))
        products.append(Product(
            title=item.get("title", "Unnamed product"),
            price=price,
            currency="INR",
            merchant=item.get("source", "Retailer"),
            url=item.get("product_link") or item.get("link") or "",
            product_id=str(item["product_id"]) if item.get("product_id") else None,
            thumbnail=item.get("thumbnail"),
            rating=to_number(item.get("rating")),
            reviews=int(item["reviews"]) if str(item.get("reviews", "")).isdigit() else None,
            delivery=item.get("delivery"),
        ))
    return products


def retailer_search_links(query: str) -> dict[str, str]:
    encoded = quote_plus(query)
    return {
        "Amazon India": f"https://www.amazon.in/s?k={encoded}",
        "Flipkart": f"https://www.flipkart.com/search?q={encoded}",
        "Myntra": f"https://www.myntra.com/{encoded}",
        "Croma": f"https://www.croma.com/searchB?q={encoded}%3Arelevance",
    }


def product_score(product: Product, query: str) -> float:
    terms = set(query.lower().split())
    title_terms = set(product.title.lower().split())
    relevance = len(terms & title_terms) / max(len(terms), 1)
    rating = (product.rating or 0) / 5
    review_bonus = min((product.reviews or 0) / 1000, 1)
    price_bonus = 1 if product.price is not None else 0
    return 0.55 * relevance + 0.25 * rating + 0.12 * review_bonus + 0.08 * price_bonus


def format_price(product: Product) -> str:
    return f"₹{product.price:,.0f}" if product.price is not None else "Price unavailable"


def product_key(p: Product) -> str:
    return p.product_id or f"{p.title}|{p.merchant}"


def add_to_cart(product: Product) -> None:
    """Add a product with a guaranteed-clickable retailer link to the shortlist."""
    cart: list[Product] = st.session_state.cart
    key = product_key(product)
    if not any(product_key(item) == key for item in cart):
        # Ensure the URL always points somewhere useful.
        url = product.url or ""
        if not url or "google." in url:
            url = build_retailer_search_url(product.merchant, product.title)
        cart.append(replace(product, url=url))


if "cart" not in st.session_state:
    st.session_state.cart = []
if "results" not in st.session_state:
    st.session_state.results = []

st.title("🛍️ PriceLens : Compare product prices")
st.caption("Upload a product photo and compare the listings across Indian e-commerce retailers - Prototype testing by Suresh.")

with st.sidebar:
    st.header("Image model")
    vision_provider = st.selectbox("Choose the image model", ["Groq", "Gemini"])
    if vision_provider == "Groq":
        st.caption("Uses GROQ_API_KEY with Qwen vision.")
    else:
        st.caption("Uses GOOGLE_API_KEY or GEMINI_API_KEY with Gemini.")
    if st.button("Test SerpAPI connection"):
        with st.spinner("Checking live India shopping search…"):
            connected, message = test_serpapi_connection()
        if connected:
            st.success(message)
        else:
            st.error(message)
    st.caption("The SerpAPI test performs one small live search and may use an API request.")
    st.divider()
    st.header("Your shortlist")
    if notice := st.session_state.pop("shortlist_notice", None):
        st.info(notice)
    if not st.session_state.cart:
        st.caption("Products you add will appear here.")
    else:
        priced = [item for item in st.session_state.cart if item.price is not None]
        total = sum(item.price for item in priced)
        missing = len(st.session_state.cart) - len(priced)
        label = f"₹{total:,.0f}" + (f" (+{missing} unpriced)" if missing else "")
        st.metric("Estimated total", label)
        for index, item in enumerate(st.session_state.cart):
            cols = st.columns([4, 1])
            # Clickable product title always links to the retailer page.
            link = item.url or build_retailer_search_url(item.merchant, item.title)
            cols[0].markdown(
                "<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\" "
                "style=\"display:block; padding:0.45rem 0; font-weight:600; "
                "text-decoration:none; color:#2563eb;\">"
                "{title} ↗</a>".format(
                    url=escape(link, quote=True),
                    title=escape(item.title[:48]),
                ),
                unsafe_allow_html=True,
            )
            cols[0].caption(f"{format_price(item)} · {item.merchant}")
            if cols[1].button("🗑️", key=f"remove-{index}", help="Remove from shortlist"):
                st.session_state.cart.pop(index)
                st.rerun()
    st.caption("Click a shortlisted product to visit its retailer page. The shortlist is local to this session.")

uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
manual_query = st.text_input("Or refine the product search", placeholder="e.g. Nike Air Max 270 black, UK 9")

if uploaded:
    image_bytes = uploaded.getvalue()
    st.image(Image.open(BytesIO(image_bytes)), caption="Uploaded image", width=360)
    if st.button("Find matching products", type="primary"):
        try:
            with st.spinner("Understanding image and searching Indian stores…"):
                identified = identify_product(image_bytes, uploaded.type or "image/jpeg", vision_provider)
                query = manual_query.strip() or identified["search_query"]
                st.session_state.identified = identified
                st.session_state.query = query
                raw_results = search_live_products(query)
                ranked = sorted(raw_results, key=lambda item: product_score(item, query), reverse=True)
                # Attach a working retailer link to every product before display.
                st.session_state.results = enrich_with_retailer_links(ranked)
        except (requests.RequestException, RuntimeError, KeyError, json.JSONDecodeError) as error:
            st.error(f"Could not complete the search: {error}")

if "query" in st.session_state:
    identified = st.session_state.identified
    st.subheader(f"Results for: {st.session_state.query}")
    fields = [identified.get("product_type", "Unknown"), identified.get("brand"), identified.get("model")]
    st.caption(" · ".join(str(value) for value in fields if value))
    if identified.get("attributes"):
        st.write("Detected details: " + ", ".join(identified["attributes"]))
    if identified.get("note"):
        st.info(identified["note"])

    results: list[Product] = sorted(
        st.session_state.results,
        key=lambda item: product_score(item, st.session_state.query),
        reverse=True,
    )
    if results:
        best = results[0]
        st.success(f"Best match: **{best.title}** — {format_price(best)} at {best.merchant}")
        st.caption(f"Showing top {min(12, len(results))} of {len(results)} listings")

        for index, product in enumerate(results[:12]):
            with st.container(border=True):
                left, middle, right = st.columns([1, 4, 1.6])
                if product.thumbnail:
                    left.image(product.thumbnail, width=100)
                # Clickable title → always goes to a retailer page.
                link = product.url or build_retailer_search_url(product.merchant, product.title)
                middle.markdown(
                    "<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\" "
                    "style=\"font-weight:600; font-size:1.02rem; text-decoration:none; "
                    "color:#2563eb;\">{title} ↗</a>".format(
                        url=escape(link, quote=True),
                        title=escape(product.title),
                    ),
                    unsafe_allow_html=True,
                )
                middle.write(f"{product.merchant} · {format_price(product)}")
                details = []
                if product.rating:
                    details.append(f"⭐ {product.rating:.1f}")
                if product.reviews:
                    details.append(f"{product.reviews:,} reviews")
                if product.delivery:
                    details.append(product.delivery)
                if details:
                    middle.caption(" · ".join(details))

                if right.button("Add to shortlist", key=f"cart-{index}", use_container_width=True):
                    add_to_cart(product)
                    st.rerun()
                right.link_button("Go to product ↗", link, key=f"product-{index}", use_container_width=True)
    else:
        if not get_secret("SERPAPI_API_KEY"):
            st.warning("No SERPAPI_API_KEY was found. Add it to .env to show live products and prices.")
        else:
            st.warning("SerpAPI returned no shopping listings for this query. Try refining the product search or use the retailer searches below.")

    st.subheader("Search retailer sites")
    links = retailer_search_links(st.session_state.query)
    columns = st.columns(len(links))
    for column, (retailer, url) in zip(columns, links.items()):
        column.link_button(f"Search {retailer} ↗", url, use_container_width=True)