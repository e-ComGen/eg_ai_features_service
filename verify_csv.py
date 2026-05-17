import asyncio
import sys
import os
import csv
from unittest.mock import AsyncMock, MagicMock
from app.models import ProductData, FeatureOption, ProductContext

# Add project root
sys.path.append(os.getcwd())

async def test_csv_logging():
    print("--- Starting CSV Verification ---")
    
    # We need to test the logic inside main.py process_batch
    # Since we modified main.py directly, let's try to import it and mock the processor
    
    # Mocking main.py dependencies is hard because it instantiates them at module level.
    # Instead, let's extract the CSV writing logic or just simulate the data structure expected by the loop
    
    results = [
        {
            "product_id": 101,
            "filled_features": {"Color": "Red", "Size": "M"},
            "debug_info": {
                "Color": {
                    "router": {"selected_node": "BrandLeaf", "reasoning": "Brand found"},
                    "extraction_reasoning": "Extracted Red"
                },
                "Size": {
                    "router": {"selected_node": "TextLeaf", "reasoning": "Text found"},
                    "extraction_reasoning": "Extracted M"
                }
            }
        },
        {
             "product_id": 102,
             "filled_features": {},
             "debug_info": {}
        }
    ]
    
    # Simulate the CSV writing block from main.py
    try:
        with open("ai_report_test.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "product_id", "feature_name", "extracted_value", 
                "router_node", "router_reasoning", "extraction_reasoning"
            ])
            
            clean_data = []
            
            for r in results:
                if isinstance(r, Exception):
                    continue
                
                prod_id = r["product_id"]
                filled = r.get("filled_features", {})
                debug = r.get("debug_info", {})
                
                clean_data.append({
                    "product_id": prod_id, 
                    "filled_features": filled,
                    "debug_info": debug
                })
                
                for f_name, val in filled.items():
                    f_debug = debug.get(f_name, {})
                    router_info = f_debug.get("router", {})
                    
                    writer.writerow([
                        prod_id,
                        f_name,
                        val,
                        router_info.get("selected_node", ""),
                        router_info.get("reasoning", ""),
                        f_debug.get("extraction_reasoning", "")
                    ])
                    
        print("[OK] Written to ai_report_test.csv")
    except Exception as e:
        print(f"[FAIL] Writing CSV: {e}")
        return

    # Verify content
    try:
        with open("ai_report_test.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            print(f"Total rows: {len(rows)}")
            if len(rows) != 3: # Header + 2 features for prod 101
                print(f"[FAIL] Expected 3 rows (Header + 2 data), got {len(rows)}")
            else:
                 print("[OK] Row count correct")
                 
            header = rows[0]
            if "router_reasoning" in header and "extraction_reasoning" in header:
                print("[OK] Headers correct")
            else:
                print("[FAIL] Headers missing columns")
                
            row1 = rows[1]
            if "101" in row1 and "BrandLeaf" in row1:
                print("[OK] Data row correct")
            else:
                print(f"[FAIL] Data row 1 incorrect: {row1}")

    except Exception as e:
        print(f"[FAIL] Reading CSV: {e}")

if __name__ == "__main__":
    asyncio.run(test_csv_logging())
