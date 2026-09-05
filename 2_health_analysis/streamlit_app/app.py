"""Interactive blood-report analysis app based on blood_work_analysis.ipynb."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = APP_DIR.parent / "blood_work.txt"

EXTRACTION_PROMPT = """
You are a medical data extraction assistant.

From the medical report below, extract every reported test value and classify it as
HIGH, LOW, or NORMAL strictly according to the reference range stated in the report.
If a value cannot be classified from the supplied range, state NOT DETERMINED.
Do not make up reference ranges or diagnoses.

Return Markdown only, with this exact structure:
## Test results
| Test name | Value | Status | Reference range |
| --- | --- | --- | --- |

Blood report:
{blood_report}
"""

DIET_PROMPT = """
You are a clinical nutritionist familiar with Indian dietary habits.

Using only the blood-work analysis below, write:
1. A simple, non-alarmist health summary in 4–5 lines.
2. A practical Indian diet plan with exactly these two headings: "Foods to avoid"
   and "Foods to eat more of".

Do not diagnose, prescribe medication, or recommend changing medicines. Mention that
the person should discuss abnormal findings with their clinician.

Blood-work analysis:
{extracted_values}
"""


def get_llm(provider: str, model: str):
    """Create the selected model only after the user requests an analysis."""
    if provider == "Groq":
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Set GROQ_API_KEY in your .env file before using Groq.")
        return ChatGroq(model=model, reasoning_format="parsed")

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Set GOOGLE_API_KEY in your .env file before using Gemini.")
    return ChatGoogleGenerativeAI(model=model)


def response_text(response) -> str:
    """Normalize LangChain text responses across supported providers."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def analyse_report(report: str, provider: str, model: str) -> tuple[str, str]:
    llm = get_llm(provider, model)
    extraction = response_text(llm.invoke(EXTRACTION_PROMPT.format(blood_report=report)))
    diet = response_text(llm.invoke(DIET_PROMPT.format(extracted_values=extraction)))
    return extraction, diet


def load_sample_report() -> str:
    try:
        return DEFAULT_REPORT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Blood Work Analyzer", page_icon="🩺", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            padding-top: 0.85rem;
            padding-bottom: 2rem;
        }
        h1 {
            background: var(--background-color, #ffffff);
            font-size: 1.05rem;
            margin-bottom: 0.1rem;
            padding: 0.35rem 0;
            position: sticky;
            top: 0;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            z-index: 100;
        }
        [data-testid="stCaptionContainer"] {
            margin-bottom: -0.45rem;
        }
        [data-testid="stAlert"] {
            margin: 0.5rem 0 0.75rem;
            padding: 0.35rem 0.65rem;
        }
        [data-testid="stAlert"] p {
            font-size: 0.85rem;
        }
        [data-testid="stTextArea"] textarea {
            min-height: 9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🩺 Blood Work : AI-Powered Diet Recommendation")
    st.caption("Extract lab values from a text report and receive a simple, Indian diet-focused summary.")
    # st.warning(
        # "This tool is for education only and is not medical advice. "
        # "Discuss results, symptoms, and treatment decisions with a qualified clinician."
    # )

    with st.sidebar:
        st.header("Model settings")
        provider = st.selectbox("Provider", ("Groq", "Gemini"))
        default_model = "qwen/qwen3.8-27b" if provider == "Groq" else "gemini-3.5-flash-lite"
        model = st.text_input("Model name", value=default_model)
        st.divider()
        st.caption("Add `GROQ_API_KEY` or `GOOGLE_API_KEY` to a `.env` file in the project root.")

    upload = st.file_uploader("Upload a plain-text blood report", type=["txt"])
    if upload is not None:
        try:
            initial_report = upload.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("Please upload a UTF-8 encoded .txt report.")
            initial_report = ""
    else:
        initial_report = load_sample_report()

    report = st.text_area(
        "Blood report text",
        value=initial_report,
        height=180,
        placeholder="Paste your blood-work report here…",
    )

    if st.button("Analyze report", type="primary", use_container_width=True):
        if not report.strip():
            st.error("Paste or upload a blood report before running the analysis.")
            return
        if not model.strip():
            st.error("Enter a model name.")
            return

        try:
            with st.spinner("Extracting test values and preparing the diet summary…"):
                extracted_values, diet_plan = analyse_report(report, provider, model.strip())
        except Exception as error:
            st.error(f"Analysis could not be completed: {error}")
            return

        st.session_state["extracted_values"] = extracted_values
        st.session_state["diet_plan"] = diet_plan

    if "extracted_values" in st.session_state:
        results_column, summary_column = st.columns(2, gap="medium")
        with results_column:
            # st.subheader("Test results")
            st.markdown(st.session_state["extracted_values"])
        with summary_column:
            st.subheader("Health summary & diet")
            st.markdown(st.session_state["diet_plan"])


if __name__ == "__main__":
    main()
