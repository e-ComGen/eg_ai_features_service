import asyncio
import sys
import os
import csv
from unittest.mock import AsyncMock, MagicMock

# Add project root
sys.path.append(os.getcwd())

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Mock dependencies
sys.modules['app.database'] = MagicMock()
sys.modules['app.security'] = MagicMock()

# Mock MatcherService to avoid loading SentenceTransformer and printing unicode
sys.modules['app.services.matcher'] = MagicMock()
# Also mock the class specifically if needed, but module mock should cover it if done early enough
# However, app.main imports MatcherService directly. 
# Better to mock app.services.matcher.MatcherService

from app.main import process_batch
from app.models import BatchPayload, ProductData, ProductContext

async def test_complete_reporting():
    print("--- Starting Complete Reporting Verification ---")
    
    # Setup Data
    payload = BatchPayload(
        client_id=1,
        products=[
            ProductData(
                id=777,
                name="Complete Report Test",
                category_id=1,
                context=ProductContext()
            )
        ],
        schemas={
            "1": {"Color": {"type": "text"}, "Size": {"type": "text"}} 
        }
    )
    
    # Mock processor
    import app.main
    
    async def mock_process(product, schema, client_id):
        return {
            "product_id": product.id,
            "filled_features": {"Color": "Red"}, # Size is missing from filled!
            "debug_info": {
                "Color": {
                    "router": {"selected_node": "BrandLeaf", "reasoning": "Found"},
                    "extraction_reasoning": "Extracted Red"
                },
                "Size": {
                    "router": {},
                    "extraction_reasoning": "Skipped: Already exists" # Or Cached Empty
                }
            }
        }
    
    app.main.processor = MagicMock()
    app.main.processor.process_product = AsyncMock(side_effect=mock_process)
    
    # Run
    await process_batch(payload)
    
    # Verify CSV
    if os.path.exists("ai_service_output.csv"):
        with open("ai_service_output.csv", "r", encoding="utf-8") as f:
            content = f.read()
            print("--- CSV Content ---")
            print(content)
            
            # Check for Size (which was empty/skipped)
            if "777,Size,,," in content.replace("\n", "").replace("\r", "") or "Skipped" in content:
                 print("[OK] Skipped item found in CSV")
            else:
                 print("[FAIL] Skipped item missing from CSV")
    else:
        print("[FAIL] CSV not found")

if __name__ == "__main__":
    asyncio.run(test_complete_reporting())
