import atexit
import pickle
import os
import re
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch

# Regex for Fix 3: pure numeric codes (6–14 digits), e.g. TNVED codes
_NUMERIC_CODE_RE = re.compile(r"^\d{6,14}$")

# Морфология (RU): прилагательное-кандидат → существительное-основа словаря Ozon.
# Лемматизация + сопоставление общей основы (корня), без хардкода словарей синонимов.
try:
    import pymorphy3
    _MORPH = pymorphy3.MorphAnalyzer()
except Exception:  # pragma: no cover - морфология опциональна
    _MORPH = None

# Стоп-токены, которые не несут смысловой нагрузки при сопоставлении лемм.
_LEMMA_STOPWORDS = {
    "и", "или", "для", "из", "с", "со", "на", "в", "не", "по", "the", "a", "of",
}
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


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
        # Fix 1: count new entries since last disk flush; persist every N encodes
        self._new_entries_since_save = 0
        self._SAVE_INTERVAL = 500
        atexit.register(self._save_vectors)

        # Кэш лемм по строкам и кэш лемматизированных allowed-списков по id(list).
        # Критично для перфоманса: один и тот же enum не лемматизируем повторно.
        self._lemma_token_cache: dict[str, list[str]] = {}
        self._options_lemma_cache: dict[int, list[list[str]]] = {}

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

        # Fix 1: flush to disk every SAVE_INTERVAL new entries (not on every encode)
        self._new_entries_since_save += 1
        if self._new_entries_since_save >= self._SAVE_INTERVAL:
            self._save_vectors()
            self._new_entries_since_save = 0

        return vector

    def _lemmatize_tokens(self, text: str) -> list[str]:
        """Лемматизировать значимые токены строки к нормальной форме.

        Кэшируется по исходной строке. Прилагательные нормализуются к своей
        нормальной форме (pymorphy не превращает прил.→сущ., поэтому связка
        прил.↔основа-сущ. делается позже через общую основу/корень).
        """
        if _MORPH is None:
            return []
        if text in self._lemma_token_cache:
            return self._lemma_token_cache[text]
        lemmas: list[str] = []
        for tok in _TOKEN_RE.findall(text.lower()):
            if tok in _LEMMA_STOPWORDS or len(tok) < 3:
                continue
            try:
                lemmas.append(_MORPH.parse(tok)[0].normal_form)
            except Exception:
                lemmas.append(tok)
        self._lemma_token_cache[text] = lemmas
        return lemmas

    @staticmethod
    def _stem_affinity(a: str, b: str) -> float:
        """Близость двух лемм по общей основе (корню).

        Возвращает 0..1. Высокое значение означает однокоренные слова
        (прил. ↔ сущ.): `хлопковый`↔`хлопок`, `эластичный`↔`эластан`.
        Требует длинной общей приставки-корня — разные корни дают ~0.
        """
        if not a or not b:
            return 0.0
        common = 0
        for x, y in zip(a, b):
            if x == y:
                common += 1
            else:
                break
        shorter = min(len(a), len(b))
        if shorter == 0:
            return 0.0
        # Минимум 4 общих символа корня — иначе считаем совпадение случайным
        # (`всесезонный` vs `демисезон` имеют общий хвост, но разный корень → 0).
        if common < 4:
            return 0.0
        return common / shorter

    def _lemma_fallback_match(self, target: str, options: list[str]) -> str | None:
        """Морфологический fallback: связать прил.-кандидат с сущ.-основой словаря.

        Лемматизирует target и каждый allowed (с кэшем на список options),
        считает близость по общей основе, принимает ТОЛЬКО однозначного
        победителя с заметным отрывом от второго места. Иначе None.
        """
        if _MORPH is None:
            return None
        target_lemmas = self._lemmatize_tokens(target)
        if not target_lemmas:
            return None

        cache_key = id(options)
        options_lemmas = self._options_lemma_cache.get(cache_key)
        if options_lemmas is None or len(options_lemmas) != len(options):
            options_lemmas = [self._lemmatize_tokens(opt) for opt in options]
            self._options_lemma_cache[cache_key] = options_lemmas

        STEM_THRESHOLD = 0.60     # минимальная доля общего корня
        MARGIN = 0.15             # отрыв лучшего от второго — гард от неоднозначности

        scores: list[float] = []
        for opt_lemmas in options_lemmas:
            if not opt_lemmas:
                scores.append(0.0)
                continue
            # Лучшая пара токенов (любой токен target ↔ любой токен allowed).
            best_pair = 0.0
            for tl in target_lemmas:
                for ol in opt_lemmas:
                    if tl == ol:
                        best_pair = max(best_pair, 1.0)
                    else:
                        best_pair = max(best_pair, self._stem_affinity(tl, ol))
            scores.append(best_pair)

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        if best_score < STEM_THRESHOLD:
            return None

        # Гард от ложных совпадений: второй по близости не должен быть рядом.
        second = 0.0
        for i, s in enumerate(scores):
            if i != best_idx and s > second:
                second = s
        if best_score - second < MARGIN:
            return None

        return options[best_idx]

    def find_best_match(self, target: str, options: list[str]) -> str | None:
        if not options or not target:
            return None

        # Игнорируем мусор
        target_clean = target.lower().strip().strip(".,;:\"'")
        if target_clean in ['unknown', 'n/a', 'not specified', 'none', 'null']:
            return None

        # Fix 3: short-circuit numeric-code prefix match (TNVED and similar).
        # Handles both exact matches ("6204620000" vs "6204620000 - Брюки женские")
        # and granularity mismatches ("6109100010" vs "6109100000 - Футболки..."):
        # strip non-digits from both sides; match when one is a prefix of the other
        # (min 6 digits). Zero effect on text attributes — only for pure-digit targets.
        if _NUMERIC_CODE_RE.match(target_clean):
            target_digits = re.sub(r"\D", "", target_clean)
            _HS8_LEVEL = 8
            _MIN_DIGITS = 6
            for opt in options:
                # Extract code portion: everything before " - " separator handles both
                # "6109100000 - desc" and spaced "6109 10 000 0 - desc" formats.
                opt_code_part = opt.split(" - ")[0] if " - " in opt else opt
                opt_digits = re.sub(r"\D", "", opt_code_part)
                if len(opt_digits) < _MIN_DIGITS:
                    continue
                # Compare at HS-8 subheading granularity (first 8 digits) so that
                # "6109100010" (EAEU national subposition) matches "6109100000"
                # (Ozon HS-8 base code). Falls back to full prefix for shorter codes.
                cmp_len = min(len(target_digits), len(opt_digits), _HS8_LEVEL)
                if cmp_len >= _MIN_DIGITS and target_digits[:cmp_len] == opt_digits[:cmp_len]:
                    return opt

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

        # Fix 2: batch-encode only options not yet in vector_cache
        missing = [opt for opt in options if opt not in self.vector_cache]
        if missing:
            batch_vecs = self.model.encode(missing, convert_to_numpy=True, batch_size=256)
            for text, vec in zip(missing, batch_vecs):
                self.vector_cache[text] = vec
            # Periodic flush accounting
            self._new_entries_since_save += len(missing)
            if self._new_entries_since_save >= self._SAVE_INTERVAL:
                self._save_vectors()
                self._new_entries_since_save = 0

        option_vecs = [self.vector_cache[opt] for opt in options]

        cosine_scores = util.cos_sim(target_vec, np.array(option_vecs))[0]
        best_vec_idx = int(np.argmax(cosine_scores))
        best_vec_score = float(cosine_scores[best_vec_idx])

        candidate = options[best_vec_idx]

        if best_vec_score >= 0.60:
            return candidate

        # --- ЭТАП 3: Морфология (прил.-кандидат → сущ.-основа словаря) ---
        # Спасает RU-прилагательные от источников: `хлопковый`→`Хлопок`,
        # `эластичный`→`Эластан`. Только однозначные совпадения по общей основе.
        lemma_match = self._lemma_fallback_match(target, options)
        if lemma_match is not None:
            return lemma_match

        print(f"\n💀 [MATCHING FAILED] -------------------------------")
        print(f"   📥 AI Output:    '{target}'")
        print(f"   🧹 Cleaned:      '{target_clean}'")
        print(f"   🧩 Normalized:   '{target_nospace}'")
        print(f"   📊 Best Fuzzy:   {best_lev_score}% vs '{best_lev_option}' (Need {threshold}%)")
        print(f"   🧠 Best Vector:  {best_vec_score:.4f} vs '{candidate}' (Need 0.60)")
        print(f"   📋 All Options:  {options}")
        print(f"----------------------------------------------------\n")
        return None