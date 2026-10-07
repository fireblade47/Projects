# Blood Work Analyzer

This Streamlit app is the interactive version of `../blood_work_analysis.ipynb`.
It accepts a text-based PDF or plain-text blood report, extracts the report text
locally, asks a selected LLM to classify values using the ranges in that report,
then produces a concise health summary and Indian diet-focused guidance.

## Run it

From the repository root, set one provider key in `.env`:

```env
GROQ_API_KEY=your_key
# or
GOOGLE_API_KEY=your_key
```

Install the project dependencies if needed, then start Streamlit:

```bash
uv sync
uv run streamlit run 2_health_analysis/streamlit_app/app.py
```

The sidebar lets you switch between Groq and Gemini and override the model name.
The sample report is loaded automatically; users can instead paste a report or
upload a UTF-8 `.txt` file or a text-based `.pdf` report. For scanned PDFs with
no selectable text, paste OCR output into the text area.

This application is educational only and should not be used as a substitute for
medical advice, diagnosis, or treatment.
