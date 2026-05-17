import json
import os
from typing import Optional, Any


class CacheManager:
    def __init__(self, cache_file='llm_product_cache.json'):
        self.cache_file = cache_file
        self.local_cache = self._load_cache()
        print(f"📦 Cache loaded from {self.cache_file}. Records: {len(self.local_cache)}")

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.local_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Cache save error: {e}")

    def _generate_id_key(self, product_id: int, feature_name: str) -> str:
        # SIMPLE AND STRICT: product_id + feature_name
        # Example: "pid_1055:Color"
        return f"pid_{product_id}:{feature_name.strip().lower()}"

    def get_feature_value(self, product_id: int, feature_name: str) -> Optional[str]:
        key = self._generate_id_key(product_id, feature_name)
        return self.local_cache.get(key)

    def set_feature_value(self, product_id: int, feature_name: str, value: str):
        key = self._generate_id_key(product_id, feature_name)

        # Only save if value is different (optional optimization)
        if self.local_cache.get(key) != value:
            self.local_cache[key] = value
            self._save_cache()