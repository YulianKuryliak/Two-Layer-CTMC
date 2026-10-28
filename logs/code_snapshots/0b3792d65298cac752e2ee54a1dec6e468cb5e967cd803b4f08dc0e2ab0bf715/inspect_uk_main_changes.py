from pathlib import Path
import difflib
import importlib.util
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('uk_builder', root / 'extrascripts/analysis/build_smpt_final_uk_sections_1_5.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
paper = root / 'paper/SMPT paper'
baseline = module.render((paper / 'final_en_sections_1_3_20261009/user_sections_1_3_UK_snapshot.tex').read_text(encoding='utf-8-sig'))
main = (paper / 'main-SMPT_UK.tex').read_text(encoding='utf-8-sig')
current = main[main.index(r'\section{Вступ}'):main.index(r'\section{Що насправді')]
print(''.join(difflib.unified_diff(baseline.splitlines(True), current.splitlines(True))))
