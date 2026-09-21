
import streamlit as st
import requests, os, pathlib, pandas as pd, ast

# Paths
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
USER_ITEM_CSV = DATA_DIR / "user_item_dl.csv"
RULES_CSV = DATA_DIR / "rules.csv"

# Load data
products_df = pd.read_csv(PRODUCTS_CSV) if PRODUCTS_CSV.exists() else None
user_item_df = pd.read_csv(USER_ITEM_CSV) if USER_ITEM_CSV.exists() else None
rules_df = pd.read_csv(RULES_CSV) if RULES_CSV.exists() else None

if user_item_df is None or rules_df is None:
    st.error("Missing user_item_dl.csv or rules.csv in data/ folder.")
    st.stop()

# Build antecedent item set
ante_set: set[str] = set()
for row in rules_df["antecedent"].astype(str):
    try:
        ante_set.update([i.strip().lower() for i in ast.literal_eval(row)])
    except Exception:
        ante_set.add(row.strip().lower())

# Config & CSS
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config("Hybrid Recommender System", "💼", layout="centered")
css_path = pathlib.Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Helpers
def lookup_meta(name: str):
    if products_df is None:
        return {"name": name, "price": None, "category": None}
    row = products_df[products_df.item_id.str.lower() == name.lower()]
    return {"name": name, "price": row.iloc[0].price, "category": row.iloc[0].category} if not row.empty else {"name": name, "price": None, "category": None}

# Initialize Context Cart (replaces old "Cart")
if "context_cart" not in st.session_state:
    st.session_state["context_cart"] = []

def add_to_context(item_name):
    if item_name and item_name not in st.session_state["context_cart"]:
        st.session_state["context_cart"].append(item_name)
        st.toast(f"Added '{item_name}' to context!", icon="✅")

def remove_from_context(item_name):
    if item_name in st.session_state["context_cart"]:
        st.session_state["context_cart"].remove(item_name)
        st.rerun() # Rerun to update the display immediately

def clear_context():
    st.session_state["context_cart"] = []
    st.rerun()

def render_block(title, items):
    st.markdown(f"### {title}")
    if not items:
        st.info("No recommendations available.")
        return
        
    # Render grid
    cols = st.columns(3)
    for idx, it in enumerate(items):
        if isinstance(it, dict):
            name = it.get("item", "")
            score = it.get("score", None)
            # Removed reason display as requested
        else:
            name = it
            score = None
            
        meta = lookup_meta(name)
        with cols[idx % 3]:
            # Card Container
            with st.container(border=True):
                st.markdown(f"**{meta['name']}**")
                
                if meta['price']:
                    st.markdown(f"<span style='color:green;font-weight:bold'>${meta['price']:.2f}</span>", unsafe_allow_html=True)
                
                if score is not None:
                    st.caption(f"Score: {score}")

# User list
user_rank = user_item_df["user_id"].value_counts().rename_axis("user_id").reset_index(name="count")
all_user_ids = user_rank.user_id.astype(int).tolist()
# Prepend Fake User for Cold Start
all_user_ids.insert(0, 00000)


# UI
st.markdown("<h2>Hybrid Recommender System</h2>", unsafe_allow_html=True)
col_u, col_i = st.columns(2)

with col_u:
    sel_user = st.selectbox("Select User", all_user_ids)
    if sel_user == 0:
        st.write("*(Cold-Start User - No History)*")
    else:
        cnt = user_rank[user_rank.user_id == sel_user]["count"].iat[0]
        st.write(f"Total items purchased: **{cnt}**")

# Context Construction Logic
st.markdown("### Build Your Context (Cart)")

# 1. Determine base pool of items
if sel_user == 0:
    # User 00000: All items from products.csv
    base_pool_df = products_df if products_df is not None else pd.DataFrame(columns=["item_id", "category"])
else:
    # Existing User: Items in User History AND Antecedent Set
    user_items_list = user_item_df[user_item_df.user_id == sel_user]["item_id"].unique().tolist()
    valid_items = [i for i in user_items_list if i.lower().strip() in ante_set]
    # Filter products_df to only these items
    if products_df is not None:
        base_pool_df = products_df[products_df['item_id'].isin(valid_items)]
    else:
        base_pool_df = pd.DataFrame(columns=["item_id", "category"])

# 2. Category Filter
all_categories = sorted(base_pool_df['category'].dropna().unique().tolist())
all_categories.insert(0, "All Categories")
sel_category = st.selectbox("Filter by Category", all_categories)

# 3. Filter Items by Category
if sel_category != "All Categories":
    filtered_items = sorted(base_pool_df[base_pool_df['category'] == sel_category]['item_id'].unique().tolist())
else:
    filtered_items = sorted(base_pool_df['item_id'].unique().tolist())

# 4. Item Selection Widget
col_sel, col_add = st.columns([3, 1])
with col_sel:
    selected_item_to_add = st.selectbox("Select Product", options=[""] + filtered_items, format_func=lambda x: "Select an item..." if x == "" else x)

with col_add:
    st.write("") # Spacer
    st.write("")
    if st.button("Add to Context"):
        if selected_item_to_add:
            add_to_context(selected_item_to_add)
        else:
            st.warning("Select an item first.")

# 5. Display Current Context (Old Cart style but for Context)
if st.session_state["context_cart"]:
    st.write(" **Current Context Items:**")
    
    # Use container to list them
    with st.container(border=True):
        for c_item in st.session_state["context_cart"]:
            c1, c2 = st.columns([8, 1])
            with c1: 
                st.write(f"• {c_item}")
            with c2:
                if st.button("x", key=f"rem_{c_item}", help="Remove from context"):
                    remove_from_context(c_item)
        
        if st.button("Clear Context"):
            clear_context()
else:
    st.info("No items selected for context yet.")

chosen = st.session_state["context_cart"]
rule_items = filtered_items # For compatibility with old logic checks if needed, but we rely on 'chosen' mostly

k = st.slider("Top-K Recommendations", 1, 10, 3)

# --- Logic to handle state and display ---

if "show_recs" not in st.session_state:
    st.session_state["show_recs"] = False

# 1. Show "Trending" ONLY if we are NOT showing specific recommendations
if not st.session_state["show_recs"]:
    st.markdown("### 🔥 Top Selling Products This Week")
    try:
        r_pop = requests.get(f"{API_URL}/recommend/popular", params={"top_k": 4}, timeout=5)
        pop_items = r_pop.json().get("results", [])
        if pop_items:
            cols = st.columns(len(pop_items))
            for idx, col in enumerate(cols):
                item = pop_items[idx]
                with col:
                    meta = lookup_meta(item['item'])
                    # Simple card styled with internal CSS or inline
                    st.markdown(f"""
                    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 8px; text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div style="font-weight: bold; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">
                            {meta['name']}
                        </div>
                        <div>
                            <span style="color: #2e7d32; font-weight: bold;">${meta['price'] if meta['price'] else 'N/A'}</span>
                            <br>
                            <small style="color: #666;">Sold: {item['score']}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        # Silently fail or show small warning
        pass

st.markdown("---")


can_generate = len(chosen) > 0 or (sel_user != 0) 
if st.button("Generate Recommendations", disabled=not can_generate):
    st.session_state["show_recs"] = True

# 3. IF we should show recommendations
if st.session_state["show_recs"]:
    # Hybrid Controls
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        use_auto = st.checkbox("🤖 Auto-Alpha Mode", value=True, help="Let the system decide the best weight based on User History and Context.")
    
    alpha = None
    if not use_auto:
        with col_c2:
            alpha = st.slider("Weight (Alpha)", 0.0, 1.0, 0.5, 0.1, help="1.0 = Trust AI (User History), 0.0 = Trust Rules (Context)")
    else:
        with col_c2:
            st.info("System will calculate Alpha automatically...")

    with st.spinner("Fetching hybrid recommendations..."):
        hybrid_recs = []
        auto_stats = {}
        alpha_used = alpha
        
        # Use the LAST added item as the primary context for now, or send all?
        # The backend endpoint currently takes 'item' (singular). 
        # Ideally we would update backend to take list, but for now we follow established pattern 
        # or just pick the last one as "most recent context".
        context_item = chosen[-1] if chosen else ""

        try:
            # Prepare params
            params = {"user_id": sel_user, "item": context_item, "top_k": k}
            if alpha is not None:
                params["alpha"] = alpha
            
            # Call Hybrid API
            r = requests.get(f"{API_URL}/recommend/hybrid", params=params, timeout=30)
            data = r.json()
            hybrid_recs = data.get("results", [])
            alpha_used = data.get("alpha_used", alpha)
            auto_stats = data.get("auto_stats", {})
            
        except Exception as e:
            st.error(f"Hybrid Recommendation Error: {e}")

    # Render Results
    if st.button("Clear Results"):
        st.session_state["show_recs"] = False
        st.rerun()

    # Display Auto-Alpha Explanation
    if use_auto and auto_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calculated Alpha", f"{alpha_used}")
        c2.metric("User History", f"{auto_stats.get('n_interactions', 0)} txns")
        c3.metric("Profile Confidence", f"{auto_stats.get('C_user', 0)}")
        c4.metric("Context Align", f"{auto_stats.get('S_align', 0)}")
        
        with st.expander("ℹ️ How was Alpha calculated?"):
            st.write(f"""
            **Formula:** `Alpha = 0.2 + 0.6 * C_user + 0.2 * S_align`
            * **User Confidence (C_user):** Measures how much we know about this user.
            * **Context Alignment (S_align):** Measures if the current item matches the user's usual taste.
            """)

    if chosen:
        st.markdown(f"**active Context used:** *{chosen[-1]}* (from {len(chosen)} items in selection)")
    
    # Display as a single unified block
    render_block("🌟 Hybrid Recommendations (AI + Rules)", hybrid_recs)
