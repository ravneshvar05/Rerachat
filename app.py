"""
app.py — Streamlit chatbot UI for the RERA Real Estate Project Finder.

Run with:
    streamlit run app.py
"""

import streamlit as st
from config import settings
from logger import logger
from query_parser import parse_query
from search import search
from answer_generator import generate_answer

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🏠",
    layout="centered",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global Typography & App Background ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: radial-gradient(circle at top left, #f8fafc 0%, #e2e8f0 100%);
    }

    /* ── Header ── */
    .chat-header {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        color: #0f172a;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.8);
        transition: transform 0.3s ease;
    }
    .chat-header:hover {
        transform: translateY(-2px);
    }
    .chat-header h1 {
        font-size: 2.6rem;
        margin: 0;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .chat-header p {
        font-size: 1.15rem;
        color: #64748b;
        margin: 0.8rem 0 0;
        font-weight: 500;
    }

    /* ── Animations ── */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.4rem;
        border-radius: 20px 20px 4px 20px;
        margin: 0.5rem 0 0.5rem 4rem;
        font-size: 1.05rem;
        line-height: 1.5;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.25);
        animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .bot-bubble {
        background: white;
        color: #1e293b;
        padding: 1.2rem 1.6rem;
        border-radius: 20px 20px 20px 4px;
        margin: 0.5rem 4rem 0.5rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.04);
        animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* ── Project cards ── */
    .project-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #4f46e5;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        color: #1e293b;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .project-card::after {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.4) 100%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(79, 70, 229, 0.12);
        border-color: #cbd5e1;
    }
    .project-card:hover::after {
        opacity: 1;
    }
    .project-card .proj-title { font-weight: 800; font-size: 1.25rem; color: #0f172a; margin-bottom: 0.4rem; }
    .project-card .proj-sub { color: #64748b; font-size: 1rem; font-weight: 500; }
    .project-card .proj-badges { margin-top: 1.2rem; display: flex; flex-wrap: wrap; gap: 8px; }
    .badge {
        background: #f1f5f9;
        color: #334155;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    .project-card:hover .badge {
        background: #eef2ff;
        color: #4f46e5;
        border-color: #c7d2fe;
    }

    /* ── Input area ── */
    .stTextInput > div > div > input {
        border-radius: 30px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.85rem 1.5rem !important;
        font-size: 1.05rem !important;
        background: white !important;
        color: #1e293b !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.15) !important;
    }

    /* ── Streamlit Buttons (including suggestions/Run) ── */
    .stButton > button {
        border-radius: 30px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: white !important;
        color: #4f46e5 !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03) !important;
    }
    .stButton > button:hover {
        background-color: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.2) !important;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
        box-shadow: 0 2px 5px rgba(79, 70, 229, 0.1) !important;
    }
    .stButton > button p, .stButton > button span {
        color: inherit !important;
    }

    /* Hide Streamlit branding */
    header, #MainMenu, footer { visibility: hidden !important; }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────

def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []          # list of {role, content, results?}
    if "groq_history" not in st.session_state:
        st.session_state.groq_history = []      # list of {role, content} for LLM


init_state()


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="chat-header">
    <h1>🏠 RERA Project Finder</h1>
    <p>Tell me what kind of home you're looking for — I'll find the best matches.</p>
</div>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def render_project_card(r: dict) -> str:
    title = f"{r.get('project_name', 'Project')} by {r.get('developer_name', '')}"
    
    # Base project location
    sub = f"{r.get('city', '')}"
    if r.get('neighbourhood'):
        sub += f" · {r.get('neighbourhood')}"

    # Project amenities badges
    badges = []
    if r.get("has_clubhouse"):     badges.append("🏛️ Clubhouse")
    if r.get("has_pool"):          badges.append("🏊 Pool")
    if r.get("has_park"):          badges.append("🌳 Park")
    if r.get("has_sports_courts"): badges.append("🎾 Sports")

    badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    
    # Render unit variations
    units_html = ""
    units = r.get("matching_units", [])
    if units:
        units_html += '<div style="margin-top: 1rem; border-top: 1px solid #f1f5f9; padding-top: 0.8rem;">'
        units_html += '<div style="font-size: 0.85rem; font-weight: 700; color: #64748b; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;">Matching Configurations</div>'
        
        for u in units:
            unit_desc = f"<b>{u.get('bhk')} BHK {u.get('property_type')}</b>"
            if u.get("super_builtup_sqft"):
                unit_desc += f" · {u['super_builtup_sqft']} sqft"
            elif u.get("carpet_area_sqft"):
                unit_desc += f" · Carpet: {u['carpet_area_sqft']} sqft"
                
            unit_badges = []
            if u.get("has_pooja_room"):   unit_badges.append("🛕 Pooja")
            if u.get("has_terrace"):      unit_badges.append("🌿 Terrace")
            if u.get("has_garden"):       unit_badges.append("🪴 Garden")
            if u.get("has_study_room"):   unit_badges.append("📚 Study")
            if u.get("has_gym"):          unit_badges.append("💪 Gym")
            
            unit_badges_str = " · ".join(unit_badges)
            if unit_badges_str:
                unit_desc += f" <span style='color: #8b5cf6; font-size: 0.8rem; margin-left: 6px;'>({unit_badges_str})</span>"
                
            units_html += f'<div style="font-size: 0.9rem; margin-bottom: 4px; color: #334155;">✓ {unit_desc}</div>'

        units_html += '</div>'

    return f"""
<div class="project-card">
    <div class="proj-title">{title}</div>
    <div class="proj-sub">{sub}</div>
    <div class="proj-badges">{badge_html}</div>
    {units_html}
</div>"""


def display_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]
    results = msg.get("results", [])

    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{content}</div>', unsafe_allow_html=True)
        if results:
            for r in results:
                st.markdown(render_project_card(r), unsafe_allow_html=True)


# ─── Chat History Display ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    display_message(msg)


# ─── Welcome / Suggestions (only shown when chat is empty) ───────────────────

if not st.session_state.messages:
    st.markdown("""
<div class="bot-bubble">
    👋 Hello! I can help you find the perfect home from our real estate projects.<br><br>
    You can start by telling me:<br>
    • How many bedrooms (BHK) you need<br>
    • What type of property (apartment, villa, row house)<br>
    • Which city you're looking in<br>
    • Any special requirements (pooja room, garden, gym, etc.)
</div>
""", unsafe_allow_html=True)

    suggestions = [
        "3 BHK Villa in Ahmedabad",
        "2 BHK Apartment with gym",
        "1 BHK affordable apartment",
        "Villa with pooja room and garden",
    ]
    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        if col.button(s, use_container_width=True):
            st.session_state["_chip_input"] = s
            st.rerun()


# ─── Input ────────────────────────────────────────────────────────────────────

# Check if a suggestion chip was clicked
pending_chip = st.session_state.pop("_chip_input", None)

user_input = st.chat_input(placeholder="Describe the home you're looking for...")

# Determine the final query to process
active_query = None
if user_input and user_input.strip():
    active_query = user_input.strip()
elif pending_chip:
    active_query = pending_chip

# ─── Process Input ────────────────────────────────────────────────────────────

if active_query:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": active_query})
    st.session_state.groq_history.append({"role": "user", "content": active_query})

    # Instantly display the user's message bubble before processing starts
    display_message({"role": "user", "content": active_query})

    with st.spinner("Searching projects..."):
        try:
            # Step 1: Parse query
            parsed = parse_query(active_query, st.session_state.groq_history)

            if parsed.needs_clarification:
                # Ask for more info
                bot_reply = parsed.clarification_question
                results = []
                logger.info(f"Clarification needed: {bot_reply}")
            else:
                # Step 2: Search
                results = search(parsed, top_k=5)

                # Step 3: Generate answer
                bot_reply = generate_answer(
                    user_query=active_query,
                    results=results,
                    conversation_history=st.session_state.groq_history,
                )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            bot_reply = "Something went wrong on my end. Please try again!"
            results = []

    # Save bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "results": results,
    })
    st.session_state.groq_history.append({"role": "assistant", "content": bot_reply})

    st.rerun()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Search Tips")
    st.markdown("""
- Mention **BHK** (e.g., *3 BHK*)
- Specify **type** (Apartment / Villa / Row House)
- Mention **city** name
- Add features: *pooja room, gym, garden, terrace*
""")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.groq_history = []
        st.rerun()
    st.divider()
    st.caption("Powered by Groq + ChromaDB + Streamlit")
