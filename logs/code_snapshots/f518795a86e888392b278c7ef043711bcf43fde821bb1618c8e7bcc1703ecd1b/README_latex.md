# Локальне візуальне редагування статті

Відкрийте цей проєкт у VS Code через WSL. Після встановлення розширення виконайте `Developer: Reload Window` у палітрі команд (`Ctrl+Shift+P`).

Редаговані файли:
- `paper/SMPT paper/readability_revision_parallel_black_20261008/sections_1_3_EN.tex`
- `paper/SMPT paper/readability_revision_parallel_black_20261008/sections_1_3_UK.tex`

Ці два файли відкриваються у LaTeX Visual Editor. Якщо відкрився код, виконайте `LaTeX: Open Visual Editor` у палітрі команд. Кнопка ока перемикає візуальний редактор і код. Складні команди або TikZ можуть потребувати редагування у коді.

Збереження (`Ctrl+S`) запускає LuaLaTeX через LaTeX Workshop. Для ручної збірки: `Ctrl+Alt+B`. Для PDF: команда `LaTeX Workshop: View LaTeX PDF file`. Директиви `% !TeX root` направляють збірку до відповідної мовної статті. Секції підключено через `\input`, тому зміни потрапляють у PDF без попередньої генерації оболонки.

Для оновлення двомовного PDF запустіть `Tasks: Run Task` → `LaTeX: rebuild bilingual article (EN + UK)`. Або з термінала WSL у корені проєкту:

```bash
.venv/bin/python extrascripts/analysis/build_smpt_readability_parallel.py
```

Результат: `paper/SMPT paper/claude-version_merged_bilingual_UK_EN_plain_sections_1_3_black.pdf`.

Звичайне збереження оновлює PDF відповідної мови; двомовний PDF оновлюється окремою задачею. Генератор перевіряє збереження поточних посилань, міток, шести формул та рисунка. Якщо свідомо змінювати їх, перевірки генератора також потребуватимуть узгодження.

Якщо двомовний PDF відкритий у Windows-програмі, яка блокує запис, новий результат зберігається поруч із суфіксом `_updated.pdf`.
