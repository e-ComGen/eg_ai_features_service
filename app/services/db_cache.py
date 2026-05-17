import hashlib
import re
from sqlalchemy import Column, Integer, String, UniqueConstraint, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import Base


# --- 1. МОДЕЛЬ ТАБЛИЦЫ ---
class CachedFeature(Base):
    __tablename__ = "cached_features"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, index=True)  # Разделяем клиентов
    product_id = Column(Integer, index=True)  # ID товара
    feature_name = Column(String, index=True)  # Название фичи
    input_hash = Column(String)  # Хэш "грязного" текста (описания)
    ai_value = Column(String)  # Ответ GPT

    # Уникальность: у одного клиента один товар может иметь одну характеристику
    __table_args__ = (
        UniqueConstraint('client_id', 'product_id', 'feature_name', name='uix_cache_entry'),
    )


# --- 2. МЕНЕДЖЕР КЕША ---
class DatabaseCacheManager:
    def __init__(self):
        # Словарь замен (как мы обсуждали раньше)
        self.UNIT_MAP = {
            r'\"': ' inch ', r'”': ' inch ', r'inch': ' inch ', r'дюйм': ' inch '
        }

    def _calculate_hash(self, text: str) -> str:
        """
        Создает 'слепок' описания.
        Если тут изменится хоть цифра - хэш будет другой.
        """
        if not text: return "empty"

        # 1. Нормализация (убираем HTML, приводим к нижнему регистру)
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)  # убираем теги

        # 2. Канонизация единиц (дюймы и т.д.)
        for pattern, replacement in self.UNIT_MAP.items():
            text = re.sub(pattern, replacement, text)

        # 3. Чистка от лишних символов (оставляем буквы и цифры)
        text = re.sub(r'[^\w\s]', '', text)

        # 4. Сортировка слов (чтобы "Red iPhone" == "iPhone Red")
        words = text.split()
        words.sort()
        clean_text = "".join(words)

        # 5. MD5 Хэш
        return hashlib.md5(clean_text.encode()).hexdigest()

    async def get_cached_value(
            self,
            session: AsyncSession,
            client_id: int,
            product_id: int,
            feature_name: str,
            current_description: str
    ) -> str | None:
        """
        Возвращает значение ТОЛЬКО если:
        1. Запись есть.
        2. Описание товара (current_description) НЕ ИЗМЕНИЛОСЬ с прошлого раза.
        """
        # Считаем хэш того, что пришло сейчас
        current_hash = self._calculate_hash(current_description)

        # Ищем в базе
        query = select(CachedFeature).where(
            CachedFeature.client_id == client_id,
            CachedFeature.product_id == product_id,
            CachedFeature.feature_name == feature_name
        )
        result = await session.execute(query)
        record = result.scalar_one_or_none()

        if not record:
            return None  # Кеша нет вообще

        # ГЛАВНАЯ ПРОВЕРКА: Изменился ли товар?
        if record.input_hash != current_hash:
            print(f"🔄 Cache Stale for Product {product_id}: Description changed!")
            return None  # Товар изменился, кеш протух, надо перегенерировать

        print(f"✅ DB Cache HIT for Product {product_id} ({feature_name})")
        return record.ai_value

    async def set_cached_value(
            self,
            session: AsyncSession,
            client_id: int,
            product_id: int,
            feature_name: str,
            current_description: str,
            value: str
    ):
        if not value: return

        current_hash = self._calculate_hash(current_description)

        # Пытаемся найти существующую запись
        query = select(CachedFeature).where(
            CachedFeature.client_id == client_id,
            CachedFeature.product_id == product_id,
            CachedFeature.feature_name == feature_name
        )
        result = await session.execute(query)
        record = result.scalar_one_or_none()

        if record:
            # Обновляем (если поменялось описание или значение)
            record.input_hash = current_hash
            record.ai_value = value
        else:
            # Создаем новую
            new_record = CachedFeature(
                client_id=client_id,
                product_id=product_id,
                feature_name=feature_name,
                input_hash=current_hash,
                ai_value=value
            )
            session.add(new_record)

        await session.commit()