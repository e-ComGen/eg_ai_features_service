import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock

# Add project root
sys.path.append(os.getcwd())

from app.services.job_processor import JobProcessor
from app.models import ProductData, FeatureOption, ProductContext

async def test_cache_flag():
    print("--- Starting Cache Flag Verification ---")

    # 1. Mock Dependencies
    mock_pipeline = MagicMock()
    mock_pipeline.extract_feature = AsyncMock(return_value={
        "value": "AI Value",
        "tokens": 10,
        "router_debug": {},
        "extraction_reasoning": "AI extracted"
    })

    mock_db_cache = MagicMock()
    mock_db_cache.get_cached_value = AsyncMock(return_value="Cached Value")
    mock_db_cache.set_cached_value = AsyncMock()

    mock_matcher = MagicMock()
    mock_matcher.find_best_match = MagicMock(return_value=None)

    # Global semaphore
    semaphore = asyncio.Semaphore(10)

    # Instantiate Processor
    processor = JobProcessor(mock_pipeline, mock_db_cache, mock_matcher, semaphore)

    # Test Data
    product = ProductData(
        id=1,
        category_id=10,
        name="Test Product",
        context=ProductContext()
    )
    schema = {
        "Color": FeatureOption(type="text")
    }
    client_id = 999

    # --- Test Case 1: use_cache=True (Default) ---
    print("\n[Test 1] use_cache=True...")
    # We expect it to return "Cached Value" and NOT call pipeline
    result = await processor.process_product(product, schema, client_id, use_cache=True)
    
    filled = result["filled_features"]
    print(f"Result: {filled}")
    
    if filled.get("Color") == "Cached Value":
        print("[OK] Returned cached value")
    else:
        print(f"[FAIL] Expected 'Cached Value', got '{filled.get('Color')}'")

    # Verify calls
    if mock_db_cache.get_cached_value.called:
        print("[OK] Cache was queried")
    else:
        print("[FAIL] Cache was NOT queried")
        
    if not mock_pipeline.extract_feature.called:
        print("[OK] Pipeline was NOT called")
    else:
        print("[FAIL] Pipeline WAS called unexpectedly")

    # --- Test Case 2: use_cache=False ---
    print("\n[Test 2] use_cache=False...")
    # Reset mocks
    mock_db_cache.get_cached_value.reset_mock()
    mock_pipeline.extract_feature.reset_mock()
    
    # We expect it to IGNORE "Cached Value" (even if we mock it to return something)
    # and return "AI Value" from pipeline
    
    result = await processor.process_product(product, schema, client_id, use_cache=False)
    
    filled = result["filled_features"]
    print(f"Result: {filled}")

    if filled.get("Color") == "AI Value":
        print("[OK] Returned AI value")
    else:
        print(f"[FAIL] Expected 'AI Value', got '{filled.get('Color')}'")

    # Verify calls
    if not mock_db_cache.get_cached_value.called:
        print("[OK] Cache was NOT queried")
    else:
        print("[FAIL] Cache WAS queried unexpectedly")
        
    if mock_pipeline.extract_feature.called:
        print("[OK] Pipeline was called")
    else:
        print("[FAIL] Pipeline was NOT called")

if __name__ == "__main__":
    asyncio.run(test_cache_flag())
