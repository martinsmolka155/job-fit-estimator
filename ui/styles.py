"""Custom CSS injection for corporate professional aesthetic.

Inter typography + JetBrains Mono for technical values + semantic color palette.
"""

from __future__ import annotations


def inject_css() -> str:
    """Return the CSS string for st.markdown(unsafe_allow_html=True)."""
    return """
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* Card pattern */
    .jfe-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .jfe-card-strength {
        border-left: 3px solid #059669;
    }

    .jfe-card-gap {
        border-left: 3px solid #D97706;
    }

    /* Skill chips */
    .jfe-skill-chip {
        display: inline-block;
        background: #F3F4F6;
        color: #374151;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 13px;
        margin: 4px;
    }

    /* Stat cards */
    .jfe-stat-label {
        color: #6B7280;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }

    .jfe-stat-value {
        color: #111827;
        font-size: 32px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Sidebar branding */
    .jfe-logo {
        font-size: 20px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .jfe-tagline {
        font-size: 13px;
        color: #6B7280;
        margin-bottom: 24px;
    }

    /* Streamlit overrides */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
    }

    /* Tab styling */
    [data-baseweb="tab-list"] {
        gap: 24px;
    }

    [data-baseweb="tab"] {
        font-weight: 500;
        color: #6B7280;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: #2563EB;
    }

    /* Footer */
    .jfe-footer {
        margin-top: 48px;
        padding: 16px 0;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-size: 13px;
    }

    .jfe-footer-mono {
        font-family: 'JetBrains Mono', monospace;
        color: #111827;
    }

    /* Welcome screen (no CV uploaded) */
    .jfe-welcome {
        text-align: center;
        padding: 80px 20px;
        max-width: 600px;
        margin: 40px auto;
    }

    .jfe-welcome-headline {
        font-size: 32px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 16px;
    }

    .jfe-welcome-subhead {
        font-size: 16px;
        color: #6B7280;
        line-height: 1.6;
        margin-bottom: 32px;
    }

    .jfe-welcome-note {
        background: #EFF6FF;
        border-left: 3px solid #2563EB;
        padding: 16px 20px;
        border-radius: 4px;
        text-align: left;
        color: #1E40AF;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Salary source + confidence badge pills */
    .jfe-badge-green {
        display: inline-block;
        background: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }

    .jfe-badge-amber {
        display: inline-block;
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }

    .jfe-badge-blue {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #93C5FD;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }

    .jfe-badge-gray {
        display: inline-block;
        background: #F3F4F6;
        color: #374151;
        border: 1px solid #D1D5DB;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }

    .jfe-badge-red {
        display: inline-block;
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
