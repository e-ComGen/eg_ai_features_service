import asyncio
import sys
import os
import csv
from unittest.mock import AsyncMock, MagicMock

# Add project root
sys.path.append(os.getcwd())

# Mock dependencies BEFORE importing app.main
sys.modules['app.database'] = MagicMock()
sys.modules['app.security'] = MagicMock()

# Now import app.main
from app.main import process_batch
from app.models import BatchPayload, ProductData, ProductContext

async def test_main_csv_generation():
    print("--- Starting Integration Verification ---")
    
    # Setup Data
    payload = BatchPayload(
        client_id=1,
        products=[
            ProductData(
                id=555,
                name="Integration Test Product",
                category_id=1,
                context=ProductContext()
            )
        ],
        schemas={
            "1": {"Color": {"type": "text"}} # Minimal schema
        }
    )
    
    # Mock the internal processor in app.main
    import app.main
    
    async def mock_process(product, schema, client_id):
        return {
            "product_id": product.id,
            "filled_features": {"Color": "Blue"},
            "debug_info": {
                "Color": {
                    "router": {"selected_node": "MockNode", "reasoning": "MockReason"},
                    "extraction_reasoning": "MockExtraction"
                }
            }
        }
    
    app.main.processor = MagicMock()
    app.main.processor.process_product = AsyncMock(side_effect=mock_process)
    
    # Run the endpoint function
    print("Calling process_batch...")
    response = await process_batch(payload)
    
    print(f"Response status: {response.get('status')}")
    
    # Check if CSV exists
    if os.path.exists("ai_service_output.csv"):
        print("[OK] ai_service_output.csv created")
        
        with open("ai_service_output.csv", "r", encoding="utf-8") as f:
            content = f.read()
            print("--- CSV Content ---")
            print(content)
            
            # Check content (allowing for line ending differences)
            if "555,Color,Blue,MockNode,MockReason,MockExtraction" in content.replace("\n", "").replace("\r", ""):
                 print("[OK] Content verified")
            else:
                 print("[FAIL] Content mismatch")
    else:
        print("[FAIL] ai_service_output.csv NOT found")

if __name__ == "__main__":
    asyncio.run(test_main_csv_generation())
