from ..base import BaseStrategyNode

class RootStrategy(BaseStrategyNode):
    name = "ROOT"
    description = "Start here."

    @classmethod
    def get_layer_logic(cls) -> str:
        return ""
    # У него нет инструкции, значит это Ветка.
    # Детей он найдет сам: всех, кто наследуется от RootStrategy.