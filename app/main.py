import asyncio
from fastapi import FastAPI, Depends
from .security import verify_internal_token
from .models import BatchPayload
from .database import init_db
from .services.db_cache import DatabaseCacheManager
from .services.llm_manager import OpenAIManager
from .services.job_processor import JobProcessor
from .services.ai_pipeline import AiFeaturePipeline
from .services.matcher import MatcherService
from .config import OPENAI_API_KEY, WEB_SEARCH_MODEL, WEB_SEARCH_MAX_CONCURRENT

app = FastAPI(title="AI Worker")

GLOBAL_LIMIT = 50
global_semaphore = asyncio.Semaphore(GLOBAL_LIMIT)

db_cache = DatabaseCacheManager()
llm_manager = OpenAIManager(api_key=OPENAI_API_KEY)
pipeline = AiFeaturePipeline(
    llm_manager,
    web_search_model=WEB_SEARCH_MODEL,
    web_search_max_concurrent=WEB_SEARCH_MAX_CONCURRENT,
)
matcher = MatcherService(None)
processor = JobProcessor(pipeline, db_cache, matcher, global_semaphore)


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.post("/process-batch", dependencies=[Depends(verify_internal_token)])
async def process_batch(payload: BatchPayload):
    print(f"⚙️ Worker: Start processing {len(payload.products)} items...")

    async def safe_task(prod, sch):
        try:
            return await processor.process_product(
                prod, sch, payload.client_id,
                use_cache=payload.use_cache,
                research_mode=payload.research_mode
            )
        except Exception as e:
            print(f"🔥 ERROR Product {prod.id}: {e}")
            return {"product_id": prod.id, "error": str(e), "filled_features": {}}

    tasks = []
    for product in payload.products:
        cat_id = str(product.category_id)
        schema = payload.schemas.get(cat_id) or payload.schemas.get(int(cat_id))
        if schema:
            tasks.append(safe_task(product, schema))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    import csv
    import json
    import os
    clean_data = []

    csv_file = "ai_service_output.csv"
    file_exists = os.path.isfile(csv_file)

    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(csv_file) == 0:
                writer.writerow([
                    "product_id", "feature_name", "extracted_value",
                    "router_node", "router_reasoning", "extraction_reasoning",
                    "deduced_context", "judge_data", "source", "source_urls"
                ])

            for r in results:
                if isinstance(r, Exception):
                    continue

                prod_id = r.get("product_id")
                if not prod_id: continue

                filled = r.get("filled_features", {})
                debug = r.get("debug_info", {})

                clean_data.append({
                    "product_id": prod_id,
                    "filled_features": filled,
                    "debug_info": debug
                })

                all_features = sorted(debug.keys())
                for f_name in all_features:
                    val = filled.get(f_name, "")
                    f_debug = debug.get(f_name, {})
                    router_info = f_debug.get("router", {})

                    # Вытаскиваем данные судьи и превращаем в строку для CSV
                    judge_info = f_debug.get("judge_data")
                    judge_str = json.dumps(judge_info, ensure_ascii=False) if judge_info else ""

                    src_urls = f_debug.get("source_urls") or []
                    writer.writerow([
                        prod_id,
                        f_name,
                        str(val),
                        router_info.get("selected_node", ""),
                        router_info.get("reasoning", ""),
                        f_debug.get("extraction_reasoning", ""),
                        f_debug.get("deduced_context", ""),
                        judge_str,
                        f_debug.get("source", "") or "",
                        ";".join(src_urls)
                    ])

    except Exception as e:
        print(f"⚠️ Failed to write CSV report: {e}")

    return {"status": "success", "data": clean_data}