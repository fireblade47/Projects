# ShopLens India

An image-to-shopping Streamlit prototype for Indian e-commerce sites.

## Run it

```bash
cd v2_Shopping_App
python -m pip install -r requirements.txt
streamlit run app.py
```

## Configure live search

Add the keys to the repository's existing `.env` file (do not commit it):

```env
GROQ_API_KEY=your-groq-api-key
# Optional: use this when selecting Gemini in the app
GOOGLE_API_KEY=your-google-ai-studio-key
SERPAPI_API_KEY=your-serpapi-key
```

The app automatically loads `.env` at startup. Select **Groq** or **Gemini** in the sidebar to identify the product in the photo; Groq uses `GROQ_API_KEY`, while Gemini uses `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Use **Test SerpAPI connection** to verify live India shopping search (it makes one API request). `SERPAPI_API_KEY` searches Google Shopping restricted to India and returns merchant links, prices, ratings, and images. When a shopper adds a product to the shortlist, the app makes one additional SerpAPI product-offer request to obtain the retailer's direct product URL (for example, Amazon or Flipkart), rather than saving the Google Shopping page. Without either key, the app explains what is unavailable and provides direct Amazon India, Flipkart, Myntra, and Croma search links. Streamlit secrets and shell environment variables also work.

## Checkout design

The in-app cart is a **shortlist**. It links the shopper to the selected retailer for checkout. A true Amazon/Flipkart cart or checkout needs that retailer's approved affiliate/partner APIs, user consent, and any required commercial agreement; it should not be implemented by scraping or automating customer checkout pages.
