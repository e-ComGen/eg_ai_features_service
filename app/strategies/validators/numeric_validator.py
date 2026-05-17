from fractions import Fraction
from typing import Any


class NumericValidator:
    """
    Утилита для детерминированной проверки наличия числового значения в тексте,
    учитывая математические эквиваленты (запятые, нули, дроби).
    """

    @staticmethod
    def generate_equivalents(value: Any) -> set[str]:
        equivalents = set()
        val_str = str(value).strip()

        try:
            # Превращаем в float (обрабатывая запятые, если вдруг они пришли)
            val_float = float(val_str.replace(',', '.'))

            # 1. Базовые форматы: 3.4 и 3,4 (убираем лишние нули через формат %g)
            clean_str = f"{val_float:g}"
            equivalents.add(clean_str)
            equivalents.add(clean_str.replace('.', ','))

            # 2. Если это целое число (например 5.0 -> 5)
            if val_float.is_integer():
                equivalents.add(str(int(val_float)))
            else:
                # 3. Магия дробей (Fraction)
                # limit_denominator(100) спасает от микро-погрешностей float
                frac = Fraction(val_float).limit_denominator(100)

                # Формат неправильной дроби (например, 5/4)
                equivalents.add(f"{frac.numerator}/{frac.denominator}")

                # Формат смешанной дроби (например, 1 1/4)
                if val_float > 1:
                    whole = int(val_float)
                    remainder = val_float - whole
                    if remainder > 0:
                        rem_frac = Fraction(remainder).limit_denominator(100)
                        equivalents.add(f"{whole} {rem_frac.numerator}/{rem_frac.denominator}")
                        equivalents.add(f"{whole}-{rem_frac.numerator}/{rem_frac.denominator}")

        except ValueError:
            # Если вообще не число (например "N/A"), просто возвращаем как есть
            equivalents.add(val_str)

        return equivalents

    @classmethod
    def is_value_in_text(cls, extracted_value: Any, text: str) -> bool:
        """
        Проверяет, есть ли хотя бы одно легитимное представление числа в исходном тексте.
        """
        equivalents = cls.generate_equivalents(extracted_value)

        # Для удобства поиска приводим текст к нижнему регистру
        text_lower = text.lower()

        for eq in equivalents:
            if eq in text_lower:
                return True

        return False