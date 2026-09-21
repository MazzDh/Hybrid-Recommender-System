from pathlib import Path
from typing import List, Dict, Union
import pandas as pd

class PopularityRecommender:
    def __init__(self, data_path: Union[str, Path]):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load data
        # Expecting a CSV with 'item_id' column or similar
        df = pd.read_csv(data_path)
        
        # Check if item_id column exists
        if "item_id" not in df.columns:
             # Fallback: maybe it's transaction data with 'items' column?
             if "items" in df.columns:
                 # If it's transaction string specific (like transaction_fpgrowth), 
                 # we might need to explode. But let's assume we use user_item_dl.csv 
                 # which has user_id, item_id, rating/interaction
                 raise ValueError("CSV must have 'item_id' column for popularity counting")

        self.df = df

        # Count frequencies
        # Group by item_id and count
        self.popular_df = df["item_id"].value_counts().reset_index()
        self.popular_df.columns = ["item", "score"]
        
        # Keep only top items to save memory/time, e.g., top 100
        self.popular_df = self.popular_df.head(100)
        
        # Load products for category mapping
        try:
            products_path = Path("data/products.csv")
            if products_path.exists():
                self.products_df = pd.read_csv(products_path)
            else:
                self.products_df = None
        except Exception:
            self.products_df = None

    def get_user_interaction_count(self, user_id: int) -> int:
        if "user_id" not in self.df.columns:
            return 0
        return len(self.df[self.df["user_id"] == user_id])


    def recommend(self, top_k: int = 10) -> List[Dict[str, float]]:
        # Return top K items
        # Ensure we don't request more than available
        k = min(top_k, len(self.popular_df))
        
        subset = self.popular_df.head(k)
        
        results = []
        for _, row in subset.iterrows():
            results.append({
                "item": str(row["item"]),
                "score": int(row["score"])
            })
            
        return results
    
    
    def recommend_by_category(self, category: str, top_k: int = 10) -> List[Dict[str, Union[str, int]]]:
        """
        Return popular items filtered by category.
        """
        if self.products_df is None:
            # Fallback to regular popular if no category info
            return self.recommend(top_k)
        
        # Normalize both for comparison
        category_normalized = category.strip().lower()
        
        # Add normalized column if not exists
        if 'item_id_lower' not in self.products_df.columns:
            self.products_df['item_id_lower'] = self.products_df['item_id'].str.strip().str.lower()
        if 'category_lower' not in self.products_df.columns:
            self.products_df['category_lower'] = self.products_df['category'].str.strip().str.lower()
        
        # Normalize popular_df item names for matching
        popular_with_lower = self.popular_df.copy()
        popular_with_lower['item_lower'] = popular_with_lower['item'].str.strip().str.lower()
        
        # Merge popular_df with products_df to get categories
        merged = popular_with_lower.merge(
            self.products_df[['item_id_lower', 'category_lower']], 
            left_on='item_lower', 
            right_on='item_id_lower', 
            how='left'
        )
        
        # Filter by category (case-insensitive)
        category_items = merged[merged['category_lower'] == category_normalized]
        
        # Sort by score and take top_k
        category_items = category_items.sort_values('score', ascending=False).head(top_k)
        
        results = []
        for _, row in category_items.iterrows():
            results.append({
                "item": str(row["item"]),  # Use original case
                "score": int(row["score"])
            })
        
        return results


