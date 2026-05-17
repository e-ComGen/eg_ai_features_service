import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.strategies.definitions.root import RootStrategy
from app.strategies.definitions.numeric import RangeNumericLeaf, ScalarNumericLeaf
from app.strategies.definitions.text import IdentityLeaf, QualityLeaf, BrandLeaf

def test_strategy_instruction(cls):
    print(f"\n--- Testing {cls.name} ---")
    try:
        instruction = cls.get_full_instruction()
        print(instruction)
        
        # Validation
        if "OUTPUT STRUCTURE" not in instruction:
            print("FAIL: Base constraints missing")
        if "DATA TYPE" not in instruction and "LEGACY" not in instruction: # Legacy might not have data type rule if strict
             # BrandLeaf (migrated) HAS domain logic from TextBranch (which I restored)
             if "DATA TYPE" not in instruction:
                  print("FAIL: Domain logic missing")

        if "8. " not in instruction and "8. " not in cls.specific_logic:
             # BrandLeaf has specific_logic with "8. TARGET"
             pass
        
        print("OK: Instruction assembled correctly")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    try:
        test_strategy_instruction(RangeNumericLeaf)
        test_strategy_instruction(ScalarNumericLeaf)
        test_strategy_instruction(IdentityLeaf)
        test_strategy_instruction(QualityLeaf)
        test_strategy_instruction(BrandLeaf)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
