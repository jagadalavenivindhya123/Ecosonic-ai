"""
EcoSonic AI - Main Application
Biodiversity Soundscape and Habitat Change Monitor
"""
import streamlit as st
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import APP_TITLE, APP_TAGLINE, APP_VERSION, COLORS, ASSETS_DIR
from database import init_database, seed_demo_data

# Page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(ASSETS_DIR, "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Additional inline styles
st.markdown("""
<style>
    .stApp { background-color: #F9FBF2; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);
        min-width: 260px;
    }
    section[data-testid="stSidebar"] .stRadio > label { color: white !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: white !important;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        margin: 2px 0;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        border: none;
        color: white;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2E7D32, #43A047);
    }
</style>
""", unsafe_allow_html=True)

# Initialize database and seed demo data
init_database()
seed_demo_data()

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h1 style="font-size:1.8rem; margin:0;">🌿 EcoSonic AI</h1>
        <p style="font-size:0.85rem; opacity:0.9; margin-top:0.3rem; font-style:italic;">
            Biodiversity Soundscape Monitor
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Home Dashboard",
            "🎙️ Upload Audio",
            "🤖 AI Classification",
            "🔬 Sound Analysis",
            "📅 Seasonal Monitoring",
            "🌍 Habitat Indicators",
            "📚 Recording History",
            "📄 Reports",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(f"""
    <div style="text-align:center; padding:0.5rem; opacity:0.8;">
        <p style="font-size:0.75rem; margin:0;">Version {APP_VERSION}</p>
        <p style="font-size:0.7rem; margin:0.2rem 0 0 0;">🌱 Protect Biodiversity</p>
    </div>
    """, unsafe_allow_html=True)

# Route to pages
if page == "🏠 Home Dashboard":
    from pages.dashboard import render
    render()
elif page == "🎙️ Upload Audio":
    from pages.upload_audio import render
    render()
elif page == "🤖 AI Classification":
    # AI Classification shows the upload page with emphasis on the classification
    from pages.upload_audio import render
    render()
elif page == "🔬 Sound Analysis":
    from pages.sound_analysis import render
    render()
elif page == "📅 Seasonal Monitoring":
    from pages.seasonal_monitoring import render
    render()
elif page == "🌍 Habitat Indicators":
    from pages.habitat_indicators import render
    render()
elif page == "📚 Recording History":
    from pages.recording_history import render
    render()
elif page == "📄 Reports":
    from pages.reports import render
    render()
elif page == "ℹ️ About Project":
    from pages.about import render
    render()
