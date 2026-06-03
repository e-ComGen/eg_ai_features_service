import asyncio
import logging
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, Depends, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from .security import verify_internal_token
from .models import BatchPayload
from .database import init_db
from .services.db_cache import DatabaseCacheManager
from .services.llm_manager import OpenAIManager
from .services.job_processor import JobProcessor
from .services.ai_pipeline import AiFeaturePipeline
from .services.matcher import MatcherService
from .services.enrichment import VisionProducer, WebSearchProducer
from .services.enrichment.sources.icecat_source import IceCatSource
from .services.providers.factory import get_main_manager, get_vision_provider, get_web_search_client
from .config import OPENAI_API_KEY, WEB_SEARCH_MODEL, WEB_SEARCH_MAX_CONCURRENT, PROVIDER_MAIN
from .services.excel.wb_excel import WbExcelReader, WbExcelWriter
from .services.excel.ozon_excel import OzonExcelReader, OzonExcelWriter
from .services.enrichment.pipeline_adapter import PipelineAdapter

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Worker")

GLOBAL_LIMIT = 50
global_semaphore = asyncio.Semaphore(GLOBAL_LIMIT)

db_cache = DatabaseCacheManager()

# OpenAIManager is always constructed — it owns the raw AsyncOpenAI client used
# by TreeRouter (beta.chat.completions.parse) and as openai fallback.
openai_manager = OpenAIManager(api_key=OPENAI_API_KEY)

# Main LLM manager: config-driven (DeepSeek / OpenRouter / OpenAI fallback).
# Returns StructuredLlmManager (new providers) or OpenAIManager (PROVIDER_MAIN=openai).
main_llm_manager = get_main_manager(openai_manager=openai_manager)
logger.info("Using main LLM provider: %s", PROVIDER_MAIN)

# AiFeaturePipeline still receives the manager via its constructor.
# TreeRouter always uses openai_manager.client (beta.parse) — not swapped here.
pipeline = AiFeaturePipeline(
    main_llm_manager,
    web_search_model=WEB_SEARCH_MODEL,
    web_search_max_concurrent=WEB_SEARCH_MAX_CONCURRENT,
)

# Enrichment producers wired to config-selected providers.
vision_producer = VisionProducer()        # auto-detects PROVIDER_VISION from config
websearch_producer = WebSearchProducer()  # auto-detects PROVIDER_WEB_SEARCH from config

# IceCat brand-verified source: free Open tier covers ~200 major brands
# (ASUS, Lenovo, Samsung, Sony, Bosch, Philips, Apple, HP, MSI, Dell, ...).
# Niche brands return 403 → graceful skip, no exception. Construction never
# fails; missing ICECAT_EMAIL/TOKEN just means every request returns [].
icecat_source = IceCatSource()

matcher = MatcherService(None)
processor = JobProcessor(
    pipeline,
    db_cache,
    matcher,
    global_semaphore,
    vision_producer=vision_producer,
    websearch_producer=websearch_producer,
    icecat_source=icecat_source,
)


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
                research_mode=payload.research_mode,
                options=payload.options,
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


# ---------------------------------------------------------------------------
# Stage 2: Excel bulk processing + Single product endpoint
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("/tmp/eg_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry (MVP — для production нужен Redis)
_jobs: dict[str, dict] = {}


@app.post("/fill-single")
async def fill_single_product(
    name: str,
    description: str = "",
    category_id: int = 0,
    brand: str | None = None,
    image_urls: list[str] | None = None,
    source_urls: list[str] | None = None,
    marketplace: str = "default",
):
    """Заполнить характеристики ОДНОГО товара (для quick demo).

    targets_raw пустой — для single product caller может расширить endpoint позже,
    передавая targets в payload или взяв стандартный набор из category dictionary.
    """
    adapter = PipelineAdapter()
    targets_raw: list[dict] = []  # TODO: принять targets в payload или взять из category dictionary

    values = await adapter.run(
        product_id=0,
        product_name=name,
        product_description=description,
        category_id=category_id,
        brand=brand,
        image_urls=image_urls or [],
        source_urls=source_urls or [],
        targets_raw=targets_raw,
        marketplace=marketplace,
    )
    return {
        "filled_attributes": [
            {
                "attribute_id": v.attribute_id,
                "value": v.value,
                "confidence": v.confidence,
                "source": v.source.value,
                "evidence": v.evidence,
            }
            for v in values
        ]
    }


@app.post("/excel/upload")
async def upload_excel(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    marketplace: str = "wb",  # 'wb' or 'ozon'
):
    """Принять Excel файл, запустить фоновую обработку, вернуть job_id."""
    job_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{job_id}_input.xlsx"
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _jobs[job_id] = {"status": "queued", "progress": 0, "total": 0, "marketplace": marketplace}
    background_tasks.add_task(_process_excel_job, job_id, saved_path, marketplace)
    return {
        "job_id": job_id,
        "status_url": f"/excel/status/{job_id}",
        "download_url": f"/excel/download/{job_id}",
    }


@app.get("/excel/status/{job_id}")
async def excel_status(job_id: str):
    """Получить статус обработки Excel-файла."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@app.get("/excel/download/{job_id}")
async def excel_download(job_id: str):
    """Скачать обработанный Excel-файл."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    if _jobs[job_id]["status"] != "done":
        raise HTTPException(425, "Not ready yet")
    output_path = UPLOAD_DIR / f"{job_id}_output.xlsx"
    return FileResponse(str(output_path), filename=f"filled_{job_id[:8]}.xlsx")


async def _process_excel_job(job_id: str, input_path: Path, marketplace: str) -> None:
    """Background job processor для Excel bulk enrichment."""
    try:
        if marketplace == "wb":
            reader: WbExcelReader | OzonExcelReader = WbExcelReader(input_path)
            writer: WbExcelWriter | OzonExcelWriter = WbExcelWriter(input_path)
        elif marketplace == "ozon":
            reader = OzonExcelReader(input_path)
            writer = OzonExcelWriter(input_path)
        else:
            raise ValueError(f"Unknown marketplace: {marketplace}")

        products = reader.read_products()
        targets_raw = (
            reader.get_target_attributes(products)
            if hasattr(reader, "get_target_attributes")
            else []
        )

        _jobs[job_id]["total"] = len(products)
        _jobs[job_id]["status"] = "processing"

        adapter = PipelineAdapter()
        for i, product in enumerate(products):
            try:
                values = await adapter.run(
                    product_id=product.get("row_index", i),
                    product_name=product.get("name") or "",
                    product_description=product.get("description"),
                    category_id=0,  # TODO: extract from sheet if possible
                    brand=product.get("brand"),
                    targets_raw=targets_raw,
                    marketplace=marketplace,
                )
                ai_filled: dict[str, str] = {}
                for v in values:
                    # Map attribute_id → column name
                    target = next((t for t in targets_raw if t["id"] == v.attribute_id), None)
                    if target:
                        ai_filled[target["name"]] = str(v.value)
                product["ai_filled"] = ai_filled
            except Exception as exc:
                product["ai_filled"] = {}
                product["error"] = str(exc)
            _jobs[job_id]["progress"] = i + 1

        output_path = UPLOAD_DIR / f"{job_id}_output.xlsx"
        writer.write_filled(output_path, products)
        _jobs[job_id]["status"] = "done"
    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)