import pickle
import os
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch


class MatcherService:
    def __init__(self, cache_manager):
        # cache_manager принимаем для совместимости (Dependency Injection),
        # но для векторов используем свой собственный файл.

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Loading embedding model on: {self.device.upper()}...")

        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # Файл для хранения векторов (замена Redis)
        self.vector_file = 'vectors.pkl'
        self.vector_cache = self._load_vectors()

        print(f"✅ Model loaded. Vector cache size: {len(self.vector_cache)}")

    def _load_vectors(self):
        if os.path.exists(self.vector_file):
            try:
                with open(self.vector_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def _save_vectors(self):
        # Сохраняем кеш на диск
        try:
            with open(self.vector_file, 'wb') as f:
                pickle.dump(self.vector_cache, f)
        except Exception as e:
            print(f"⚠️ Failed to save vector cache: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        # 1. Если вектор уже есть в памяти - отдаем
        if text in self.vector_cache:
            return self.vector_cache[text]

        # 2. Если нет - считаем на GPU/CPU
        vector = self.model.encode(text, convert_to_numpy=True)

        # 3. Сохраняем в память
        self.vector_cache[text] = vector

        # 4. Сохраняем на диск (для надежности можно делать это реже, но пока пишем сразу)
        self._save_vectors()

        return vector

    def find_best_match(self, target: str, options: list[str]) -> str | None:
        if not options or not target:
            return None

        # Игнорируем мусор
        target_clean = target.lower().strip().strip(".,;:\"'")
        if target_clean in ['unknown', 'n/a', 'not specified', 'none', 'null']:
            return None

        is_ram_debug = "8gb" in target_clean or "8 gb" in target_clean

        target_nospace = target_clean.replace(" ", "").replace("-", "")

        for opt in options:
            opt_nospace = opt.lower().replace(" ", "").replace("-", "")

            # Показываем сравнение для RAM
            if is_ram_debug:
                is_match = (target_nospace == opt_nospace)

            if target_nospace == opt_nospace:
                return opt


        best_lev_score = 0
        best_lev_option = None

        for opt in options:
            score = fuzz.ratio(target_clean, opt.lower())
            if score > best_lev_score:
                best_lev_score = score
                best_lev_option = opt

        threshold = 85 if len(target_clean) < 5 else 90

        if best_lev_score >= threshold:
            return best_lev_option

        # --- ЭТАП 2: Вектора (Смысл) ---
        target_vec = self.get_embedding(target)
        option_vecs = [self.get_embedding(opt) for opt in options]

        cosine_scores = util.cos_sim(target_vec, np.array(option_vecs))[0]
        best_vec_idx = int(np.argmax(cosine_scores))
        best_vec_score = float(cosine_scores[best_vec_idx])

        candidate = options[best_vec_idx]

        if best_vec_score >= 0.60:
            return candidate

        print(f"\n💀 [MATCHING FAILED] -------------------------------")
        print(f"   📥 AI Output:    '{target}'")
        print(f"   🧹 Cleaned:      '{target_clean}'")
        print(f"   🧩 Normalized:   '{target_nospace}'")
        print(f"   📊 Best Fuzzy:   {best_lev_score}% vs '{best_lev_option}' (Need {threshold}%)")
        print(f"   🧠 Best Vector:  {best_vec_score:.4f} vs '{candidate}' (Need 0.60)")
        print(f"   📋 All Options:  {options}")
        print(f"----------------------------------------------------\n")
        return None