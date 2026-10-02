import streamlit as st
import cv2
import base64
import tempfile
import os
from datetime import datetime
import pandas as pd
import numpy as np
from groq import Groq
import re
import time
import plotly.express as px

# ===== Page Configuration =====
st.set_page_config(
    page_title="Real-time Shopper Insights: Smart Item Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== Optimized CSS =====
st.markdown("""
    <style>
        .main > div { padding-top: 0rem; padding-bottom: 0rem; }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        h1 {
            margin-top: 0rem !important;
            margin-bottom: 0.2rem !important;
            font-size: 2rem !important;
        }
        .caption-text {
            font-size: 0.9rem !important;
            color: #555 !important;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            margin-top: 0rem !important;
            margin-bottom: 0.3rem !important;
            font-size: 1.1rem !important;
        }
        .element-container { margin-bottom: 0.2rem !important; }
        [data-testid="metric-container"] {
            padding: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
        .stButton button {
            padding: 0.2rem 0.6rem !important;
            font-size: 0.85rem !important;
        }
        .stDataFrame { max-height: 250px !important; }
        .confidence-high { color: #4CAF50; font-weight: bold; }
        .confidence-medium { color: #FF9800; font-weight: bold; }
        .confidence-low { color: #f44336; font-weight: bold; }
        .basket-item {
            padding: 6px;
            margin: 3px 0;
            border-radius: 6px;
            background: #f8f9fa;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Initialize Groq Client (FIXED) =====
@st.cache_resource
def get_groq_client():
    """Initialize and cache Groq client with proper API key handling"""
    # Try multiple ways to get the API key
    api_key = None
    
    # Method 1: Check environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Method 2: Check Streamlit secrets (for cloud deployment)
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
    
    # Method 3: Check .env file (if python-dotenv is installed)
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GROQ_API_KEY")
        except:
            pass
    
    # If no API key found, return None (simulation mode)
    if not api_key:
        st.warning("⚠️ GROQ_API_KEY not found. Using simulation mode.")
        st.info("💡 Set GROQ_API_KEY in environment variables or Streamlit secrets.")
        return None
    
    try:
        client = Groq(api_key=api_key)
        # Test the client with a quick call
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        return None

# Initialize client
client = get_groq_client()

# ===== Debug: Show API Status =====
# if client:
#    st.sidebar.success("✅ Groq API Connected")
# else:
#    st.sidebar.warning("⚠️ Simulation Mode")
#    st.sidebar.caption("Set GROQ_API_KEY for real detection")

# ===== ALLOWED ITEMS =====
ALLOWED_ITEMS = {
    'apple', 'banana', 'orange', 'grape', 'strawberry', 'mango', 'pineapple',
    'tomato', 'potato', 'onion', 'garlic', 'carrot', 'broccoli', 'cucumber',
    'milk', 'cheese', 'yogurt', 'butter', 'cream',
    'bread', 'rice', 'pasta', 'cereal', 'sugar', 'flour', 'oil',
    'chips', 'crackers', 'cookie', 'popcorn', 'pretzel',
    'water', 'juice', 'soda', 'coffee', 'tea',
    'egg', 'chicken', 'beef', 'fish', 'salmon', 'tofu'
}

# ===== CATEGORIES =====
CATEGORIES = {
    'Fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'mango', 'pineapple'],
    'Vegetables': ['tomato', 'potato', 'onion', 'garlic', 'carrot', 'broccoli', 'cucumber'],
    'Dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
    'Bakery': ['bread', 'pasta', 'rice'],
    'Snacks': ['chips', 'crackers', 'cookie', 'popcorn', 'pretzel'],
    'Beverages': ['water', 'juice', 'soda', 'coffee', 'tea']
}

# ===== FILTER WORDS =====
FILTER_WORDS = [
    'she', 'he', 'her', 'him', 'they', 'them', 'person', 'customer',
    'woman', 'man', 'hand', 'arm', 'finger', 'holding', 'reaching',
    'grabbing', 'picking', 'touching', 'carrying', 'looking',
    'store', 'aisle', 'shelf', 'rack', 'display', 'counter', 'cart',
    'basket', 'floor', 'wall', 'ceiling', 'light', 'shadow',
    'right', 'left', 'top', 'bottom', 'center', 'corner',
    'think', 'want', 'analyze', 'image', 'there', 'piece', 'filter',
    'based', 'bag', 'to', 'from', 'with', 'without', 'for', 'by',
    'at', 'on', 'in', 'a', 'an', 'the', 'and', 'or', 'but', 'so'
]

# ===== Helper Functions =====
def extract_frames(video_path, max_frames=3):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        interval = max(1, total_frames // max_frames)
        for i in range(0, total_frames, interval):
            if len(frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    cap.release()
    return frames

def encode_frame_to_base64(frame):
    height, width = frame.shape[:2]
    if height > 320 or width > 320:
        scale = 320 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

def categorize_item(item):
    item_lower = item.lower()
    for category, items in CATEGORIES.items():
        if any(item_lower in i or i in item_lower for i in items):
            return category
    return 'Other'

def is_allowed_item(word):
    word_lower = word.lower().strip()
    if word_lower in FILTER_WORDS or len(word_lower) <= 2:
        return False
    
    if word_lower in ALLOWED_ITEMS:
        return True
    
    if word_lower.endswith('s'):
        if word_lower[:-1] in ALLOWED_ITEMS:
            return True
    if word_lower.endswith('es'):
        if word_lower[:-2] in ALLOWED_ITEMS:
            return True
    
    return False

def clean_item_name(text):
    text_lower = text.lower().strip()
    
    prefixes = [
        'there is ', 'i see ', 'the user ', 'a piece of ', 'items: ', 'item: ',
        'detected: ', 'analysis shows ', 'image shows ', 'frame shows ',
        'perhaps ', 'maybe ', 'likely ', 'possibly ', 'this is '
    ]
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text_lower = text_lower[len(prefix):]
    
    suffixes = [' in hand', ' on shelf', ' in basket', ' on display', ' in cart']
    for suffix in suffixes:
        if suffix in text_lower:
            text_lower = text_lower.split(suffix)[0]
    
    for word in FILTER_WORDS:
        text_lower = text_lower.replace(f' {word} ', ' ')
    
    text_lower = re.sub(r'[^\w\s]', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    
    words = text_lower.split()
    
    for word in words:
        if len(word) > 2 and word not in FILTER_WORDS:
            if is_allowed_item(word):
                return word.title()
    return None

def get_confidence_score(detection_text, item_name):
    text_lower = detection_text.lower()
    score = 0.65
    
    if any(word in text_lower for word in ['definitely', 'clearly', 'obviously', 'certainly']):
        score += 0.2
    elif any(word in text_lower for word in ['likely', 'probably', 'appears', 'seems']):
        score += 0.05
    
    if item_name.lower() in text_lower:
        score += 0.1
    
    return min(1.0, max(0.0, score))

def detect_items_in_frame(frame, model="qwen/qwen3.6-27b"):
    """Detect items in frame using Groq API or simulation"""
    if client is None:
        return simulate_detection()
    
    try:
        base64_image = encode_frame_to_base64(frame)
        
        prompt = """List ONLY product names customers are picking up. One per line. 
        If none, respond with "none".
        Valid: Apple, Milk, Bread, Chips
        Invalid: "She is holding an apple", "Customer reaches for bread"
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=50
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"⚠️ API Error: {str(e)[:50]}. Using simulation.")
        return simulate_detection()

def simulate_detection():
    items = ["Apple", "Banana", "Milk", "Bread", "Chips", "Cheese"]
    num_items = np.random.randint(1, 3)
    return "\n".join(np.random.choice(items, size=num_items, replace=False))

def parse_detections(text, timestamp, frame_num):
    items = []
    if not text or "none" in text.lower():
        return items
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = re.sub(r'^[\d\.\-\*•]+\s*', '', line)
        cleaned = clean_item_name(line)
        
        if cleaned and is_allowed_item(cleaned):
            confidence = get_confidence_score(text, cleaned)
            items.append({
                'Item': cleaned,
                'Frame': frame_num,
                'Timestamp': f"{timestamp:.1f}s",
                'Confidence': confidence,
                'Confidence_Level': 'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.5 else 'Low',
                'Category': categorize_item(cleaned)
            })
    return items

def get_unique_items(results):
    if not results:
        return []
    
    unique = {}
    for item in results:
        name = item['Item']
        if name not in unique:
            unique[name] = {
                'Item': name,
                'First_Detected': item['Timestamp'],
                'Confidence': item['Confidence'],
                'Confidence_Level': item['Confidence_Level'],
                'Category': categorize_item(name)
            }
    
    return sorted(list(unique.values()), key=lambda x: float(x['First_Detected'].rstrip('s')))

def process_video_with_groq(video_bytes, model, max_frames=3):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name
    
    frames = extract_frames(video_path, max_frames)
    os.unlink(video_path)
    
    if not frames:
        return []
    
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, frame in enumerate(frames):
        status_text.text(f"🔍 Frame {idx+1}/{len(frames)}...")
        timestamp = idx * 0.5
        detection_text = detect_items_in_frame(frame, model)
        items = parse_detections(detection_text, timestamp, idx + 1)
        all_results.extend(items)
        progress_bar.progress((idx + 1) / len(frames))
        time.sleep(0.05)
    
    progress_bar.empty()
    status_text.text("✅ Complete!")
    return all_results

# ===== Display Functions =====
def show_shopping_basket_summary(df):
    if df.empty:
        return
    
    st.subheader("🛍️ Basket Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Items", len(df))
    with col2:
        st.metric("Categories", df['Category'].nunique())
    with col3:
        avg_conf = df['Confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.0%}")
    
    items_html = ""
    for _, row in df.iterrows():
        conf_class = "confidence-high" if row['Confidence'] >= 0.8 else "confidence-medium" if row['Confidence'] >= 0.5 else "confidence-low"
        items_html += f'<span class="basket-item" style="display:inline-block;margin:2px;padding:2px 8px;background:#f0f0f0;border-radius:12px;font-size:0.85rem;">{row["Item"]} <span class="{conf_class}">●</span></span>'
    
    st.markdown(items_html, unsafe_allow_html=True)

def show_category_dashboard(df):
    if df.empty:
        return
    
    st.subheader("📊 Categories")
    
    category_counts = df['Category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
    fig.update_traces(textposition='inside', textinfo='percent')
    st.plotly_chart(fig, use_container_width=True)

def show_item_heatmap(df):
    if df.empty or len(df) < 2:
        return
    
    st.subheader("🔥 Top Items")
    
    item_counts = df['Item'].value_counts().head(5)
    
    fig = px.bar(
        x=item_counts.values,
        y=item_counts.index,
        orientation='h',
        color=item_counts.values,
        color_continuous_scale='Viridis',
        height=200
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== Main App =====
st.title("🛒 Real-time Shopper Insights: Smart Item Detection")
st.markdown('<p class="caption-text">Discover Customer Preferences & Shopping Behaviour | Unlock Shopper Intelligence with AI</p>', unsafe_allow_html=True)

# ===== Sidebar =====
with st.sidebar:
    model_option = st.selectbox(
        "🧠 Model",
        ["qwen/qwen3.6-27b", "meta-llama/llama-4-scout-17b-16e-instruct"]
    )
    
    max_frames = st.slider("🎯 Frames", 2, 5, 3)
    
    st.divider()
    st.markdown("### 🔑 API Status")
    
    # Show API status with more detail
    if client:
        st.success("✅ Connected to Groq API")
        st.caption("Using real AI detection")
    else:
        st.warning("⚠️ Simulation Mode")
        st.caption("💡 Set GROQ_API_KEY to enable real detection")
        st.caption("📝 Add to environment or .env file")
    
    st.divider()
    # st.caption("⚡ Optimized for speed")

# ===== Upload =====
uploaded_file = st.file_uploader(
    "📤 Upload video",
    type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
    label_visibility="collapsed"
)

if uploaded_file:
    video_bytes = uploaded_file.read()
    
    col_video, col_results = st.columns([1, 1])
    
    with col_video:
        st.subheader("📹 Video")
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        st.markdown(f"""
            <video width="100%" height="300" controls autoplay loop style="border-radius: 8px;">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)
    
    with col_results:
        st.subheader("📦 Detected Items")
        
        if st.button("🔍 Detect", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                results = process_video_with_groq(video_bytes, model_option, max_frames)
                unique_items = get_unique_items(results)
            
            if unique_items:
                df = pd.DataFrame(unique_items)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Items", len(df))
                with col_b:
                    st.metric("Categories", df['Category'].nunique())
                with col_c:
                    st.metric("Avg Conf", f"{df['Confidence'].mean():.0%}")
                
                display_df = df[['Item', 'First_Detected', 'Confidence_Level', 'Category']]
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=180,
                    column_config={
                        "Item": "Item",
                        "First_Detected": "Time",
                        "Confidence_Level": "Confidence",
                        "Category": "Category"
                    }
                )
                
                show_shopping_basket_summary(df)
                show_category_dashboard(df)
                show_item_heatmap(df)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download",
                    data=csv,
                    file_name=f"items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No items detected")
else:
    st.info("👆 Upload a video to get started")
    
    # Show sample demo
    # if st.button("🎯 Try Demo"):
    #     sample_items = ["Apple", "Banana", "Milk", "Bread", "Chips"]
    #     sample_df = pd.DataFrame({
    #         'Item': sample_items,
    #         'First_Detected': [f"{i*0.5:.1f}s" for i in range(len(sample_items))],
    #         'Confidence': np.random.uniform(0.7, 0.95, len(sample_items)),
    #         'Confidence_Level': ['High'] * len(sample_items),
    #         'Category': [categorize_item(i) for i in sample_items]
    #     })
        
    #     st.success("🎉 Demo Data Generated!")
    #     show_shopping_basket_summary(sample_df)
    #     show_category_dashboard(sample_df)
    #     show_item_heatmap(sample_df)

st.divider()
st.caption("⚡ Built with Groq Vision API | 🔴 Powered by AI · Crafted by Suresh")