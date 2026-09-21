import time
import logging
from pathlib import Path
from fastapi import FastAPI, Request, Response

from backend.logger import setup_logging
from backend.recommender_fp import FPGrowthRecommender
from backend.recommender_dl import DLRecommender

# setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid Recommender API",
    version="1.0.0",
    description="A hybrid recommendation system using FP-Growth Algorithms and Deep Learning (NCF)"
)

# Load FP-Growth rules
fp_rec = FPGrowthRecommender(Path("data/rules.csv"))

# Load Deep Learning model safely
try:
    dl_rec = DLRecommender(Path("models/ncf_model.pt"))
except Exception as e:
    logger.warning(f"Could not load DL model: {e}")
    dl_rec = None  # fallback if model missing

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    logger.info("%s %s | %d | %.2f ms", request.method, request.url.path, response.status_code, duration)
    return response

@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

@app.get("/recommend/by-item")
def rec_by_item(item: str, top_k: int = 5):
    logger.info(f"by-item request: item={item}, top_k={top_k}")
    recs = fp_rec.recommend(item, top_k)
    return {"input": item, "type": "item-based", "results": recs}

@app.get("/recommend/by-user")
def rec_by_user(user_id: int, top_k: int = 5):
    logger.info(f"by-user request: user_id={user_id}, top_k={top_k}")
    if dl_rec is None:
        return {"error": "Deep Learning model not loaded"}
    recs = dl_rec.recommend(user_id, top_k)
    return {"input": user_id, "type": "user-based", "results": recs}

# --- Popularity Based ---
from backend.recommender_popular import PopularityRecommender

# Load Popularity model
try:
    # Using user_item_dl.csv is better because it represents individual user-item interactions
    pop_rec = PopularityRecommender(Path("data/user_item_dl.csv"))
except Exception as e:
    logger.warning(f"Could not load Popularity model: {e}")
    pop_rec = None

@app.get("/recommend/popular")
def rec_popular(top_k: int = 10):
    logger.info(f"popular request: top_k={top_k}")
    if pop_rec is None:
        return {"error": "Popularity model not loaded"}
    
    recs = pop_rec.recommend(top_k)
    return {"type": "popularity-based", "results": recs}

import numpy as np
import math

# ... (Previous imports)

def cosine_similarity(v1, v2):
    try:
        if v1 is None or v2 is None:
            return 0.0
        # Ensure flat arrays
        v1 = np.array(v1).flatten()
        v2 = np.array(v2).flatten()
        
        if v1.shape != v2.shape:
             return 0.0
             
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2)) # Ensure float return
    except Exception:
        return 0.0

def compute_alpha(n_interactions, context_items, user_emb, context_emb):
    try:
        # C_user: User Confidence based on history length
        # Cap at 50 interactions as "mature" profile
        # Handle count=0 case
        n = max(0, n_interactions)
        C_user = min(1.0, math.log1p(n) / math.log1p(50))
        
        # S_align: Alignment between User History and Current Context
        S_align = cosine_similarity(user_emb, context_emb)

        # Formula: Baseline + History Confidence + Alignment
        alpha = 0.2 + (0.6 * C_user) + (0.2 * S_align)
        
        # Clamp
        alpha = max(0.1, min(alpha, 0.9))

        return round(float(alpha), 2), round(float(C_user), 2), round(float(S_align), 2)
    except Exception as e:
        # Fallback values if math fails
        return 0.5, 0.0, 0.0

@app.get("/recommend/hybrid")
def rec_hybrid(user_id: int, item: str, top_k: int = 5, alpha: float = None):
    """
    Hybrid Recommender with Dynamic Alpha and Category Fallback.
    If 'alpha' is provided query param, use it. Otherwise compute automatically.
    If FP-Growth has no rules for the item, fallback to popular items in same category.
    """
    # Clean input
    item = item.strip() if item else ""

    # 1. Gather Metadata for Auto-Alpha
    auto_stats = {}
    final_alpha = alpha

    if final_alpha is None:
        try:
            # Calculate Auto Alpha
            n_interactions = 0
            if pop_rec:
                n_interactions = pop_rec.get_user_interaction_count(user_id)
            
            u_emb = None
            c_emb = None
            
            if dl_rec:
                u_emb = dl_rec.get_user_embedding(user_id)
                if item:
                    c_emb = dl_rec.get_item_embedding(item)
            
            computed_alpha, c_user, s_align = compute_alpha(n_interactions, [item] if item else [], u_emb, c_emb)
            final_alpha = computed_alpha
            
            auto_stats = {
                "n_interactions": int(n_interactions),
                "C_user": c_user,
                "S_align": s_align,
                "computed_alpha": final_alpha
            }
        except Exception as e:
            logger.error(f"Auto-Alpha Error: {e}")
            final_alpha = 0.5 # Safe Fallback
            auto_stats = {"error": str(e)}

    logger.info(f"hybrid request: user_id={user_id}, item={item}, alpha={final_alpha}, stats={auto_stats}")
    
    # 1. Get NCF candidates
    ncf_list = []
    if dl_rec:
        ncf_list = dl_rec.recommend(user_id, top_k=20)
        
    # 2. Get FP-Growth candidates
    fp_list = fp_rec.recommend(item, top_k=20)

    # 3. FALLBACK: If no FP-Growth rules for this item, use category-based popularity
    fallback_used = False
    if not fp_list and item:
        try:
            import pandas as pd
            products_path = Path("data/products.csv")
            if products_path.exists():
                products_df = pd.read_csv(products_path)
                # Normalize item_id for comparison
                products_df['item_id_lower'] = products_df['item_id'].str.strip().str.lower()
                
                # Find category of the selected item
                item_row = products_df[products_df['item_id_lower'] == item.strip().lower()]
                if not item_row.empty:
                    item_category = item_row.iloc[0]['category']
                    logger.info(f"✨ FALLBACK: No FP rules for '{item}'. Category: '{item_category}'")
                    
                    # Get popular items filtered by category
                    if pop_rec:
                        category_items = pop_rec.recommend_by_category(item_category, top_k=10)
                        logger.info(f"Found {len(category_items)} popular items in '{item_category}'")
                        
                        # Format as fp_list style with MUCH higher scores
                        # Since we're falling back, we want these to have significant weight
                        # Use score directly (popularity count) normalized to 0-1 range by max
                        if category_items:
                            max_score = max(x["score"] for x in category_items)
                            fp_list = [
                                {
                                    "item": x["item"], 
                                    "score": (x["score"] / max_score) * 0.9  # Normalize to 0-0.9 range
                                } 
                                for x in category_items
                            ]
                            fallback_used = True
                            auto_stats["fallback"] = f"Category: {item_category}"
                            logger.info(f"Fallback scores: {[(x['item'][:30], round(x['score'], 3)) for x in fp_list[:3]]}")
                else:
                    logger.warning(f"Item '{item}' not found in products.csv")
        except Exception as e:
            logger.warning(f"Category fallback error: {e}")

    # 4. Merge and Re-score
    candidates = {}
    for x in ncf_list:
        name = x["item"]
        candidates.setdefault(name, {"ncf": 0.0, "fp": 0.0})
        candidates[name]["ncf"] = x["score"]

    for x in fp_list:
        name = x["item"]
        candidates.setdefault(name, {"ncf": 0.0, "fp": 0.0})
        candidates[name]["fp"] = x["score"]

    final_list = []
    beta = 1.0 - final_alpha
    
    for name, scores in candidates.items():
        final_score = (final_alpha * scores["ncf"]) + (beta * scores["fp"])
        sources = []
        if scores["ncf"] > 0: sources.append("User Interest")
        if scores["fp"] > 0: 
            if fallback_used:
                sources.append("Popular in Category")
            else:
                sources.append("Related Item")
        
        final_list.append({
            "item": name,
            "score": round(final_score, 4),
            "reason": ", ".join(sources)
        })

    final_list.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "user_id": user_id,
        "context_item": item,
        "type": "hybrid",
        "alpha_used": final_alpha,
        "auto_stats": auto_stats,
        "fallback_used": fallback_used,
        "results": final_list[:top_k]
    }


