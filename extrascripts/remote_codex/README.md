# ChatGPT web → private tunnel → Codex у WSL

Цей міст дає ChatGPT Work інструменти для передавання завдань локальному Codex.
Windows-застосунок, OpenClaw, публічний HTTP-сервер і відкритий вхідний порт не потрібні.
Міст використовує наявний вхід Codex через ChatGPT. Окремий runtime API key потрібен
саме для транспорту Secure MCP Tunnel; наявність підписки ChatGPT не замінює цей ключ.

## Що вже підготовлено

- `bridge.py`: stdio MCP сервер без сторонніх Python залежностей.
- `manage.py`: встановлення офіційного Linux `tunnel-client` з перевіркою SHA256,
  повторна перевірка бінарника перед використанням, приховане введення ключа
  та запуск через його штатний managed runtime.
- `test_bridge.py`: автономна перевірка протоколу, виконання, скасування й повторів
  на імітаторі Codex; не викликає модель і не запускає симуляції.

Схема: браузер ChatGPT → приватний Secure MCP Tunnel OpenAI → `tunnel-client`
у WSL → `bridge.py` → `codex exec` у цьому проєкті.

## Завершити підключення

1. У вебверсії ChatGPT перевірити Plugins / Apps → **Add custom MCP server**.
   Доступ до власних MCP серверів залежить від акаунта та прав workspace.
   Якщо цього пункту немає, цей шлях поки недоступний для акаунта; локальна
   установка не може додати відсутній пункт інтерфейсу.
2. Відкрити [Platform Tunnels](https://platform.openai.com/settings/organization/tunnels),
   створити/вибрати приватний тунель і пов'язати його з потрібним ChatGPT workspace.
   Для створення потрібні Tunnels Read + Manage; для користування — Read + Use.
3. Створити окремий **runtime key** із потрібними tunnel permissions у
   [Platform API keys](https://platform.openai.com/settings/organization/api-keys).
   Не використовувати admin key. Не надсилати ключ у чат.
4. У терміналі WSL, з кореня проєкту, виконати одну команду:

   ```bash
   python3 extrascripts/remote_codex/manage.py connect
   ```

   Ввести tunnel ID і ключ у приховане поле. Ключ зберігається лише у
   `~/.local/share/codex-web-bridge/runtime-key`, права `0600`, каталог `0700`.
   У профіль передається посилання `file:...`, не сам ключ. Це дозволяє
   штатному managed runtime працювати після закриття термінала.
   Повторний запуск використовує збережений ключ. Для ротації замінити вміст
   цього приватного файла локально, зберігши права, й перепідключити runtime.
5. Перевірити фактичні `process_running`, `healthy`, `ready` у звіті команди.
   Сам факт запуску процесу не означає готовність тунелю.
6. У ChatGPT Plugins → Add custom MCP server вибрати **Connection: Tunnel**,
   той самий tunnel ID, назву **WSL Codex**. Для stdio сервера — **No authentication**:
   доступ уже контролює приватний тунель і його organization/workspace permissions.
   Не публікувати цей плагін і не надавати доступ стороннім workspace.
7. Установити створений плагін, відкрити новий Work чат, вибрати його через `@`
   і попросити: **«Виклич codex_info і покажи шлях проєкту у WSL»**.
   Лише успішний виклик із ChatGPT підтверджує наскрізне підключення.

Після цього з телефона/іншого комп'ютера відкрити той самий ChatGPT workspace
у браузері та використовувати цей плагін у Work. Комп'ютер і WSL мають працювати
та мати мережу. Автоматичний старт WSL після перезавантаження Windows не налаштовано.

## Як давати завдання

> Через WSL Codex перевір, чи існують усі файли рисунків статті. Нічого не змінюй
> і не запускай симуляцій. Дочекайся результату завдання.

> Через WSL Codex виправ описану помилку в [файлі]. Дозволяю редагування проєкту.
> Не запускай симуляцій. Покажи змінені файли й результати перевірки.

Інструменти: `codex_info`, `codex_submit`, `codex_status`, `codex_recent`, `codex_cancel`.
`codex_submit` приймає новий UUID `request_id`; повтор того самого запиту
з тим самим ID не запускає нове завдання. Після прийняття треба опитувати
`codex_status`. Один активний bridge job на проєкт; паралельні термінальні
сесії Codex цей міст не блокує, тому не давати їм одночасно редагувати ті самі файли.
Для продовження завершеного bridge job передати `previous_job_id`.
Це окрема сесія Codex, а не автоматичне під'єднання до поточного термінального чату.

За замовчуванням окремий permission profile розширює `:read-only`;
для дозволених правок — `:workspace`.
`danger-full-access` не використовується. Інтерактивні approvals недоступні:
Codex запускається з `--ask-for-approval never`, тож за межами дозволених дій
може завершити завдання з поясненням обмеження. У профілі заборонено читання
runtime-key, каталогів налаштувань/стану tunnel-client і `~/.codex/auth.json`
із команд агента. Перевірка harmless canary обох профілів виконує справжній
OS sandbox без запуску моделі. Інші файли домашнього каталогу не приховані.
Запуск використовує `--ignore-user-config`, щоб старий `sandbox_mode` не вимкнув
ці обмеження: вхід ChatGPT і AGENTS.md зберігаються, користувацькі налаштування
моделі та локальних інтеграцій у ці jobs автоматично не переносяться.
Jobs успадковують лише базові змінні середовища; сторонні API keys, tokens,
shell initialization overrides і proxy settings автоматично не передаються.
Доступ до самого плагіна має бути приватним.
Успішний exit code означає завершення агента: зміст `answer` може містити блокер.

Ліміт одного завдання — 1 година; логи обмежені 20 MiB на файл, відповідь MCP
повертає лише хвости. Повні локальні матеріали — у
`~/.local/share/codex-web-bridge/jobs/<job_id>/`.
Скасування завершує процес Codex; воно не скасовує вже зроблені зміни файлів.
Після перезапуску WSL перервані роботи не повторюються автоматично.
Правила AGENTS.md, журналювання коду й заборона неузгоджених симуляцій чинні
також для завдань через міст; це інструкції агента, не окремий блокувальник команд.

## Керування та перевірка

```bash
python3 extrascripts/remote_codex/manage.py status
python3 extrascripts/remote_codex/manage.py stop
python3 extrascripts/remote_codex/manage.py check
python3 extrascripts/remote_codex/bridge.py info
```

`stop` зупиняє тунель. Уже прийнятий job незалежний від його stdio процесу;
для зупинки самого job слід спочатку викликати `codex_cancel` і дочекатися
статусу `cancelled`. Через розрив тунелю завдання не запускається повторно.

## Перевірені джерела та межі готовності

- [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
- [Власний MCP сервер у ChatGPT](https://developers.openai.com/api/docs/guides/custom-mcp-server)
- [Codex non-interactive CLI](https://learn.chatgpt.com/docs/cli/reference)
- [Permission profiles](https://learn.chatgpt.com/docs/permissions)

Підготовка 2026-10-17: офіційний `tunnel-client v0.0.16` установлено тільки
у WSL (`~/.local/bin/tunnel-client`), SHA256 архіву перевірено за digest офіційного
GitHub release. Дев'ять автономних перевірок пройдено, включно з локальним
in-memory тунелем офіційного клієнта й OS sandbox canary.
Окремий короткий запуск справжнього Codex через job bridge завершився відповіддю
`WSL_BRIDGE_READY` без виклику інструментів чи зміни файлів.
Ключів не створено/не отримано, tunnel ID не налаштований,
наскрізного виклику з браузера ще немає. Попередній `codex remote-control`
є окремим механізмом; цей міст не залежить від нього.
