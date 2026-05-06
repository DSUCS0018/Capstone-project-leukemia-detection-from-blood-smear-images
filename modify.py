import re

with open("app_v1.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace inject_css
new_css = '''def inject_css():
    css_code = """
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500;700&display=swap');
        
        :root {
          --bg-main: #0A0E1A;
          --bg-card: #111827;
          --bg-elevated: #1A2235;
          --border: #1E2D45;
          --accent: #2563EB;
          --success: #10B981;
          --warning: #F59E0B;
          --danger: #EF4444;
          --text-main: #F1F5F9;
          --text-secondary: #94A3B8;
          --text-muted: #475569;
        }

        html, body, [data-testid="stAppViewContainer"] {
          font-family: 'DM Sans', -apple-system, sans-serif !important;
          background-color: var(--bg-main) !important;
          color: var(--text-main) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            color: var(--text-main) !important;
        }

        [data-testid="stHeader"] {
          background-color: transparent !important;
        }

        [data-testid="stSidebar"] {
          background-color: #0D1526 !important;
          border-right: 1px solid var(--border) !important;
        }

        /* Top Nav */
        .top-nav {
          background: #0D1526;
          border-bottom: 1px solid var(--border);
          padding: 12px 24px;
          margin: -6rem -5rem 2rem -5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          position: sticky;
          top: 0;
          z-index: 999;
        }

        .brand-container {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .brand-logo {
          font-size: 1.5rem;
        }

        .brand-text {
          display: flex;
          flex-direction: column;
        }

        .brand-title {
          font-family: 'DM Sans', sans-serif;
          font-weight: 700;
          font-size: 1.25rem;
          color: white;
          margin: 0;
          letter-spacing: 0.02em;
        }

        .brand-subtitle {
          font-family: 'DM Sans', sans-serif;
          font-size: 0.75rem;
          color: var(--text-secondary);
          text-transform: uppercase;
          margin: 0;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #0D1526 !important;
            border-bottom: 1px solid var(--border) !important;
            gap: 0;
            padding: 0;
            width: 100%;
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
            padding: 14px 24px !important;
            border: none !important;
            background: transparent !important;
            font-weight: 500 !important;
            transition: all 200ms ease;
            border-radius: 0 !important;
            height: auto !important;
            white-space: pre-wrap;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #CBD5E1 !important;
            background: #111827 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #F1F5F9 !important;
            background: #111827 !important;
            border-bottom: 2px solid var(--accent) !important;
            box-shadow: none !important;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] > section {
            background: #111827 !important;
            border: 2px dashed var(--border) !important;
            border-radius: 12px !important;
            padding: 40px !important;
            transition: all 0.2s ease;
        }
        [data-testid="stFileUploader"] > section:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 28px !important;
            font: 600 14px 'DM Sans' !important;
            letter-spacing: 0.03em !important;
            box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
            transition: all 200ms ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
            color: white !important;
        }
        
        [data-testid="stDownloadButton"] > button {
            background: transparent !important;
            border: 1.5px solid var(--accent) !important;
            color: var(--accent) !important;
            box-shadow: none !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
        }
        [data-testid="stDownloadButton"] > button:hover {
            background: rgba(37, 99, 235, 0.1) !important;
            transform: none !important;
            color: var(--accent) !important;
        }

        /* Premium Card / Containers */
        .premium-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 1.5rem;
            transition: all 0.2s ease;
        }
        
        /* KPI Cards */
        .kpi-card-v2 {
            background: #111827;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px 28px;
            position: relative;
            overflow: hidden;
            border-left: 4px solid var(--accent);
        }
        .kpi-label-v2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .kpi-value-v2 {
            font-family: 'DM Mono', monospace;
            font-size: 42px;
            font-weight: 700;
            color: var(--accent);
            line-height: 1;
        }
        .kpi-trend {
            margin-top: 0.75rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--success);
        }

        /* Progress */
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #2563EB, #7C3AED) !important;
            border-radius: 99px !important;
            height: 6px !important;
        }
        [data-testid="stProgress"] > div {
            background-color: #1A2235 !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: var(--border); }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

        /* Expanders */
        [data-testid="stExpander"] {
            background: #0F1929 !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        [data-testid="stExpander"] summary svg {
            fill: var(--accent) !important;
        }

        /* Inputs */
        div[data-testid="stNumberInput"] input, 
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            background-color: #0A0E1A !important;
            color: white !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        div[data-testid="stNumberInput"] input:focus, 
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--accent) !important;
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            background: #111827 !important;
        }
        [data-testid="stDataFrame"] th {
            background-color: #1A2235 !important;
            color: var(--text-secondary) !important;
            text-transform: uppercase !important;
            font-size: 11px !important;
        }

        /* WBC Grid */
        .wbc-grid-v2 { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }
        .wbc-card-v2 { background: #111827; border-radius: 12px; padding: 16px; border: 1px solid var(--border); }
        .wbc-card-v2 img { border-radius: 8px; width: 100%; }
        .wbc-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .wbc-card-id { font-family: 'DM Mono', monospace; color: var(--text-secondary); font-weight: 600; }
        .wbc-badge { padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
        .badge-blast-v2 { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
        .badge-normal-v2 { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
        .wbc-class-title { font-family: 'DM Sans', sans-serif; font-size: 1.125rem; font-weight: 700; color: white; margin-top: 12px; margin-bottom: 4px; }
        .wbc-confidence { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 12px; }
        
        .attr-row { display: flex; justify-content: space-between; padding: 4px 0; font-family: 'DM Mono', monospace; font-size: 0.85rem; }
        .attr-label { color: var(--text-secondary); }
        .attr-value { color: var(--text-main); font-weight: 500; }

        .diag-banner-v2 {
            background: linear-gradient(135deg, #1A0A0A, #2D0F0F);
            border-left: 6px solid var(--danger);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 2rem;
            color: white;
        }
        .diag-title-v2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 11px;
            font-weight: 600;
            color: var(--danger);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }
        .diag-result-v2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    """
    st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)
'''

# Find the start and end of inject_css
start_idx = content.find("def inject_css():")
end_idx = content.find("def render_top_nav():", start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_css + "\n\n" + content[end_idx:]

with open("app_v1.py", "w", encoding="utf-8") as f:
    f.write(content)
