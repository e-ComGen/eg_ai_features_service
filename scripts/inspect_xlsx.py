"""Quick inspector for the importer's output Excel — dumps sheets/cells/colors."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from openpyxl import load_workbook

wb = load_workbook(Path(__file__).resolve().parent / "test_psu_output.xlsx")
print(f"Sheets: {wb.sheetnames}\n")

for sn in wb.sheetnames:
    ws = wb[sn]
    print(f"=" * 70)
    print(f"SHEET: {sn}  ({ws.max_row} rows × {ws.max_column} cols)")
    print(f"=" * 70)

    if sn == "Результаты":
        # Show header
        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        print(f"COLUMNS ({len(headers)}):")
        for i, h in enumerate(headers, 1):
            print(f"  [{i:2d}] {str(h)[:55]}")
        print()
        # Show data row by row with color
        print(f"DATA (rows 2-{ws.max_row}):")
        for row_idx in range(2, ws.max_row + 1):
            print(f"\n--- Row {row_idx} ---")
            for col_idx in range(1, ws.max_column + 1):
                c = ws.cell(row=row_idx, column=col_idx)
                header = headers[col_idx - 1]
                val = str(c.value) if c.value is not None else "(empty)"
                # Fill color (skip default)
                fill = c.fill.start_color.rgb if c.fill and c.fill.start_color else None
                color_label = ""
                if fill and isinstance(fill, str) and fill not in ("00000000", "FFFFFFFF", None):
                    if "C6EFCE" in fill.upper() or fill.upper().startswith("FFC6"):
                        color_label = " [GREEN]"
                    elif "FFEB" in fill.upper() or fill.upper().startswith("FFFFEB"):
                        color_label = " [YELLOW]"
                    elif "FFC7" in fill.upper() or "FF9999" in fill.upper() or "FFCCCC" in fill.upper():
                        color_label = " [RED]"
                    else:
                        color_label = f" [color={fill}]"
                print(f"  {str(header)[:30]:30s} = {val[:60]}{color_label}")

    elif sn == "Чек-лист":
        # Show all
        for row_idx in range(1, min(ws.max_row + 1, 30)):
            row = [str(ws.cell(row=row_idx, column=c).value or "") for c in range(1, ws.max_column + 1)]
            print("  " + " | ".join(s[:30] for s in row))
        if ws.max_row > 30:
            print(f"  ... {ws.max_row - 30} more rows")

    elif sn == "Сводка":
        for row_idx in range(1, ws.max_row + 1):
            row = [str(ws.cell(row=row_idx, column=c).value or "") for c in range(1, ws.max_column + 1)]
            print("  " + " | ".join(s[:60] for s in row))
    print()
