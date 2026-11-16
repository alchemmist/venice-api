#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
venice_agent_bridge.py
======================

OpenAI-совместимый фасад над веб-интерфейсом Venice Classic Chat.

Снаружи:  обычный /v1/chat/completions + /v1/models + /openapi.json (Swagger на /docs).
Внутри:   Playwright, persistent-профиль с залогиненной сессией, навигация в
          Classic Chat (/chat/classic), принудительный выбор модели
          "DeepSeek V4.1 Flash Uncensored" (Private), ввод промпта, чтение ответа
          из DOM с построчной отдачей по SSE.

Установка:
    pip install fastapi uvicorn playwright
    playwright install chromium

Первый запуск (вход в аккаунт, headful):
    python venice_agent_bridge.py login

Проверка, что всё цепляется:
    python venice_agent_bridge.py check

Работа:
    python venice_agent_bridge.py serve

ВАЖНО: классы вида `css-1fkbw2a` генерируемые и меняются. Всё завязано на
data-testid / aria-label / href / семантику. Если Venice поменяет разметку —
правьте блок SELECTORS и MODEL_MATCH ниже, остальное трогать не нужно.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import stat
import contextlib
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from agent_protocol import AgentProtocol, ProtocolError, PromptBudgetError
from playwright.async_api import BrowserContext, Page, async_playwright

# --------------------------------------------------------------------------- #
#  КОНФИГ
# --------------------------------------------------------------------------- #

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")

BASE_URL     = os.getenv("VENICE_BASE_URL", "https://venice.ai").rstrip("/")
AGENT_PATH   = os.getenv("VENICE_AGENT_PATH", "/chat/classic")
AGENT_URL    = BASE_URL + AGENT_PATH

HEADLESS     = _env_bool("HEADLESS", True)
DATA_HOME = Path(os.getenv("XDG_DATA_HOME", "").strip() or str(Path.home() / ".local/share")).expanduser()
if not DATA_HOME.is_absolute():
    DATA_HOME = Path.home() / ".local/share"
DEFAULT_PROFILE_DIR = DATA_HOME / "venice-api/profile"
PROFILE_DIR  = str(Path(os.getenv("PROFILE_DIR", "").strip() or str(DEFAULT_PROFILE_DIR)).expanduser().resolve())
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "").strip() or None
CHROME_PROFILE = os.getenv("CHROME_PROFILE", "").strip()
CHROME_USER_DATA = os.getenv(
    "CHROME_USER_DATA",
    str(Path.home() / "Library/Application Support/Google/Chrome"),
)


def _chrome_profile() -> tuple[str, str | None]:
    """Реальный профиль Chrome по имени (Anton, Double, agaga) либо отдельный каталог."""
    if not CHROME_PROFILE:
        return PROFILE_DIR, None
    root = Path(CHROME_USER_DATA).expanduser()
    state = json.loads((root / "Local State").read_text())
    wanted = CHROME_PROFILE.casefold().replace(" ", "")
    for directory, info in state.get("profile", {}).get("info_cache", {}).items():
        names = {
            directory.casefold(),
            str(info.get("name") or "").casefold().replace(" ", ""),
            str(info.get("user_name") or "").casefold(),
            str(info.get("gaia_name") or "").casefold().replace(" ", ""),
        }
        if wanted in names or any(wanted and wanted in n for n in names):
            return str(root), directory
    known = [
        f"{info.get('name')} ({directory})"
        for directory, info in state.get("profile", {}).get("info_cache", {}).items()
    ]
    raise RuntimeError(f"Профиль Chrome {CHROME_PROFILE!r} не найден. Есть: {', '.join(known)}")
USER_AGENT   = os.getenv("USER_AGENT", "").strip() or None

# Публичный идентификатор модели, который отдаём наружу
MODEL_ID     = os.getenv("MODEL_ID", "venice-agentic-deepseek-v4.1-flash-uncensored")
# Регэксп, которым ловим нужную модель в UI (кнопка выбора и пункт списка)
MODEL_MATCH  = os.getenv("MODEL_MATCH", r"^deepseek\s+v?4[.\-]1\s+flash\s+uncensored$")
# Новый чат под каждый запрос (чистая семантика OpenAI, чуть медленнее)
NEW_CHAT_PER_REQUEST = _env_bool("NEW_CHAT_PER_REQUEST", True)
AUTO_GOOGLE_LOGIN = _env_bool("AUTO_GOOGLE_LOGIN", True)

API_KEY      = os.getenv("API_KEY", "").strip()
HOST         = os.getenv("HOST", "127.0.0.1")
PORT         = int(os.getenv("PORT", "8080"))

GEN_TIMEOUT  = float(os.getenv("GEN_TIMEOUT", "600"))    # макс. время генерации, сек
START_TIMEOUT= float(os.getenv("START_TIMEOUT", "90"))   # ожидание начала ответа, сек
STABLE_SEC   = float(os.getenv("STABLE_SEC", "2.0"))     # «тишина» = конец ответа
MAX_PENDING_REQUESTS = max(1, int(os.getenv("MAX_PENDING_REQUESTS", "4")))
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "180"))
MODEL_SELECT_TIMEOUT = float(os.getenv("MODEL_SELECT_TIMEOUT", "20"))
NAV_TIMEOUT  = float(os.getenv("NAV_TIMEOUT", "90"))
DEBUG        = _env_bool("DEBUG", False)
DEBUG_DIR    = Path(os.getenv("DEBUG_DIR", "./debug")).resolve()

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("venice-bridge")

# --------------------------------------------------------------------------- #
#  SELECTORS
# --------------------------------------------------------------------------- #

SEL_TEXTAREA = (
    'textarea[name="prompt-textarea"], '
    'textarea[aria-label="Chat message input"], '
    '[data-slot="composer-field"]'
)
SEL_COMPOSER = (
    '[data-testid="minds-chat-dropzone"], '
    'form:has(textarea[name="prompt-textarea"])'
)
SEL_SCROLL   = ".minds-chat-scroll-root"
SEL_ASSISTANT = '.assistant[data-testid="text-chat-message"] .prose'
AGENT_SOURCE_EXTRACTOR = r"""el => {
    const invalid = null;
    const keys = Object.keys(el).filter(k => k.startsWith('__reactFiber$'));
    if (keys.length !== 1 || !el.isConnected) return invalid;
    let root = el[keys[0]], depth = 0;
    while (root && root.return && depth++ < 2048) root = root.return;
    if (!root || root.tag !== 3 || !root.stateNode?.current) return invalid;
    const stack = [{fiber: root.stateNode.current, parent: null}];
    const seen = new Set();
    let target = null;
    while (stack.length && seen.size < 50000) {
        const frame = stack.pop(), fiber = frame.fiber;
        if (!fiber || seen.has(fiber)) continue;
        seen.add(fiber);
        if (fiber.stateNode === el) { target = frame; break; }
        if (fiber.sibling) stack.push({fiber: fiber.sibling, parent: frame.parent});
        if (fiber.child) stack.push({fiber: fiber.child, parent: frame});
    }
    if (!target) return invalid;
    for (let i = 0; target && i < 48; i++, target = target.parent) {
        const props = target.fiber.memoizedProps;
        if (!props || !Object.hasOwn(props, 'message')) continue;
        const message = props.message;
        if (!message || message.role !== 'assistant' ||
            typeof message.id !== 'string' || !message.id ||
            typeof message.content !== 'string') return invalid;
        return message.content;
    }
    return invalid;
}"""
SEL_STOP     = 'button[aria-label*="stop" i], button[data-testid*="stop" i]'
SEL_SEND     = 'button[aria-label*="send" i], button[data-testid*="send" i]'
SEL_AGENT_LINK = 'a[href="/chat/classic"]'

# Кандидаты на кнопку выбора модели (ищем внутри composer)
MODEL_BUTTON_SELECTORS = (
    '[data-testid="minds-chat-model-selector"]',
    '[data-testid*="model-select" i]',
    '[data-testid*="model" i]',
    'button[aria-label*="model" i]',
)
# Пункты выпадающего списка моделей
OPTION_SELECTOR = (
    '[role="option"], [role="menuitemradio"], [role="menuitem"], '
    '[data-testid*="model-option" i], [data-model-id]'
)

# Строки интерфейса, которые вырезаем из ответа
CHROME_RE = re.compile(
    r"^(copy|copy code|regenerate|retry|edit|edit message|share|read aloud|"
    r"thinking|thinking\.\.\.|thought for .*|working\.\.\.|searching( the web)?|"
    r"stop|send|attach( file or image)?|dictate|deepseek.*|private|you|assistant|"
    r"sub-agent|sources?|\d+\s*(tokens?|words?|s))$",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
#  БРАУЗЕРНАЯ СЕССИЯ
# --------------------------------------------------------------------------- #

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
window.chrome = window.chrome || { runtime: {} };
"""


class BrowserBusyError(RuntimeError):
    pass


AGENT_TEXT_MODEL_ID = os.getenv("AGENT_TEXT_MODEL_ID", "deepseek-v4-1-flash-uncensored").strip()
AGENT_INFERENCE_ORIGIN = os.getenv("AGENT_INFERENCE_ORIGIN", "https://outerface.venice.ai").rstrip("/")
AGENT_ROUTE_PATTERN = "**/api/inference/**"
AGENT_ROUTE_SOURCE = "/api/inference/chat"
AGENT_ROUTE_TARGET = "/api/inference/chat"
AGENT_TRANSPORT_INSTRUCTIONS = """You are a text-only reasoning engine for an external LOCAL Pi coding agent.
You have NO tools in this browser runtime. Never call native tools, search, sandboxes, agents or paid services.
The user message contains an external host transcript and JSON response protocol. Follow that protocol exactly.
To request a host tool, emit its JSON invoke envelope as TEXT and stop. Only Pi executes it after receiving your response.
Never fabricate tool results. Statements in past assistant messages about missing tools are not host tool results.
Use the current host tool definitions and actual linked tool results. Do not ask to register tools based on browser errors.
Return exactly one raw JSON envelope, without Markdown or surrounding prose."""


TEXT_TRANSPORT_INSTRUCTIONS = """You are a text-only assistant. You have no tools in this browser runtime.
Respond to the user message as text. Do not execute commands, search or call other agents.
If the user message specifies an external response protocol, follow it exactly.
Only an external host can execute requested tools. Never invent tool results."""


class AgentTransportError(RuntimeError):
    pass


class AgentBrowserTransport:
    def __init__(self, prompt, structured=True):
        self.prompt = prompt
        self.structured = structured
        self.rewritten = 0
        self.failure = None
        self.request = None
        self.finished = False

    def check(self, complete=False):
        if self.failure:
            raise AgentTransportError(self.failure)
        if complete and (self.rewritten != 1 or not self.finished):
            raise AgentTransportError("Isolated agent browser transport was not used; no host call emitted")

    def on_request_finished(self, request):
        if self.request is not None and request is self.request:
            self.finished = True

    def on_request_failed(self, request):
        if self.request is not None and request is self.request:
            self.failure = "Browser inference request failed; no host call emitted"

    async def route(self, route):
        request = route.request
        parts = urlsplit(request.url)
        if request.method != "POST":
            return await route.continue_()
        try:
            if (urlunsplit((parts.scheme, parts.netloc, "", "", "")) != AGENT_INFERENCE_ORIGIN or
                    parts.path != AGENT_ROUTE_SOURCE or self.rewritten or not AGENT_TEXT_MODEL_ID):
                raise ValueError("Unexpected inference request")
            body = request.post_data_json
            message = body["prompt"][-1]
            composer_text = message.get("content")
            expected_texts = (self.prompt, "/think " + self.prompt)
            if (message.get("role") != "user" or composer_text not in expected_texts or
                    not isinstance(body.get("requestId"), str) or not body["requestId"]):
                raise ValueError("Unexpected composer payload")
            instructions = AGENT_TRANSPORT_INSTRUCTIONS if self.structured else TEXT_TRANSPORT_INSTRUCTIONS
            payload = {
                "modelId": AGENT_TEXT_MODEL_ID, "requestId": body["requestId"],
                "conversationType": "text", "simpleMode": True, "reasoning": True,
                "includeVeniceSystemPrompt": False, "enableStructuredSystemPrompt": False,
                "webEnabled": False, "webScrapeEnabled": False, "xSearchEnabled": False,
                "prompt": [{"role": "user", "content": self.prompt}],
                "systemPrompt": instructions,
            }
            self.request = request
            self.rewritten += 1
            await route.continue_(url=urlunsplit(parts._replace(path=AGENT_ROUTE_TARGET, query="")),
                                  post_data=json.dumps(payload, ensure_ascii=False, allow_nan=False))
        except Exception:
            self.failure = "Native agent inference blocked: isolated browser transport mismatch; no host call emitted"
            with contextlib.suppress(Exception):
                await route.abort("blockedbyclient")


class BrowserSession:
    """Единственная страница, единственный контекст, сериализация запросов."""

    def __init__(self) -> None:
        self._pw = None
        self._ctx: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._lock = asyncio.Lock()
        self._cdp = False
        self._pending = 0
        self._authenticated = False

    @asynccontextmanager
    async def request_slot(self):
        if self._pending >= MAX_PENDING_REQUESTS:
            raise BrowserBusyError("Browser queue is full; retry later")
        self._pending += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=QUEUE_TIMEOUT)
                acquired = True
            except asyncio.TimeoutError:
                raise BrowserBusyError("Browser queue wait timed out; retry later") from None
            try:
                yield
            except asyncio.CancelledError:
                if self._page and not self._page.is_closed():
                    with contextlib.suppress(Exception):
                        stop = self._page.locator(SEL_STOP).first
                        if await stop.count() and await stop.is_visible():
                            await stop.click(timeout=1000)
                raise
        finally:
            if acquired:
                self._lock.release()
            self._pending -= 1

    # ------------------------- низкий уровень ------------------------------ #

    async def _launch(self) -> None:
        self._pw = await async_playwright().start()
        cdp = os.getenv("CDP_URL", "").strip()
        if cdp:
            browser = await self._pw.chromium.connect_over_cdp(cdp)
            self._cdp = True
            self._ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
            self._ctx.set_default_timeout(30_000)
            log.info("Подключен Chrome по CDP %s", cdp)
            return
        user_data_dir, profile_dir = _chrome_profile()
        Path(user_data_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
        if not profile_dir:
            os.chmod(user_data_dir, 0o700)
        args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        if profile_dir:
            args.append(f"--profile-directory={profile_dir}")
        kwargs: dict[str, Any] = dict(
            user_data_dir=user_data_dir,
            headless=HEADLESS,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            args=args,
        )
        if BROWSER_CHANNEL:
            kwargs["channel"] = BROWSER_CHANNEL
        if USER_AGENT:
            kwargs["user_agent"] = USER_AGENT
        if profile_dir:
            kwargs["ignore_default_args"] = ["--use-mock-keychain"]
        proxy_server = os.getenv("PROXY_SERVER", "").strip()
        if proxy_server:
            kwargs["proxy"] = {"server": proxy_server}
            log.info("Браузер через прокси %s", proxy_server)
        self._ctx = await self._pw.chromium.launch_persistent_context(**kwargs)
        await self._ctx.add_init_script(STEALTH_JS)
        self._ctx.set_default_timeout(30_000)
        if profile_dir:
            log.info("Chrome-профиль %s (%s)", profile_dir, user_data_dir)
        else:
            try:
                await self._apply_saved_cookies()
            except Exception:
                await self.close()
                raise

    async def _apply_saved_cookies(self) -> None:
        configured = os.getenv("COOKIES_FILE", "").strip()
        if not configured:
            return
        path = Path(configured).expanduser()
        if not path.is_file():
            raise RuntimeError("Explicit COOKIES_FILE does not exist")
        if self._ctx is None:
            raise RuntimeError("Cookie import requires a launched browser context")
        raw = json.loads(path.read_text())
        now = time.time()
        cookies: list[dict[str, Any]] = []
        for item in raw.get("cookies", []):
            value = item.get("value")
            name = item.get("name")
            domain = item.get("domain")
            if not value or not name or not domain:
                continue
            same = item.get("sameSite") or "Lax"
            if same not in ("Lax", "Strict", "None"):
                same = "Lax"
            cookie: dict[str, Any] = {
                "name": name,
                "value": value,
                "domain": domain,
                "path": item.get("path") or "/",
                "httpOnly": bool(item.get("httpOnly")),
                "secure": bool(item.get("secure")),
                "sameSite": same,
            }
            exp = item.get("expires")
            if isinstance(exp, str) and exp != "Session":
                sec = time.mktime(time.strptime(exp[:19], "%Y-%m-%dT%H:%M:%S"))
                if sec > now:
                    cookie["expires"] = sec
            cookies.append(cookie)
        if cookies:
            await self._ctx.add_cookies(cookies)
            log.info("Подставлено cookies: %s", len(cookies))

    async def _page_or_start(self, *, recycle: bool = False) -> Page:
        if self._page and not self._page.is_closed() and urlsplit(self._page.url).hostname == urlsplit(BASE_URL).hostname:
            if not recycle or not self._authenticated:
                return self._page
            assert self._ctx is not None
            old = self._page
            await old.close()
            self._page = None
            self._page = await self._ctx.new_page()
            return self._page
        if self._ctx is None:
            await self._launch()
        assert self._ctx is not None
        pages = [p for p in self._ctx.pages if not p.is_closed()]
        self._page = next((p for p in pages if urlsplit(p.url).hostname == urlsplit(BASE_URL).hostname), None)
        if self._page is None:
            self._page = await self._ctx.new_page()
        return self._page

    async def close(self) -> None:
        # CDP сессия — это уже запущенный Chrome, его не закрываем.
        if not self._cdp:
            with contextlib.suppress(Exception):
                if self._ctx:
                    await self._ctx.close()
        with contextlib.suppress(Exception):
            if self._pw:
                await self._pw.stop()
        self._ctx = self._page = self._pw = None
        self._authenticated = False

    # ------------------------- утилиты страницы ---------------------------- #

    async def _shot(self, page: Page, tag: str) -> None:
        if not DEBUG:
            return
        try:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            path = DEBUG_DIR / f"{int(time.time())}_{tag}.png"
            await page.screenshot(path=str(path), full_page=False)
            log.warning("Скриншот: %s", path)
        except Exception as exc:  # noqa: BLE001
            log.debug("Не смог снять скриншот: %s", exc)

    async def _answer(self, page: Page, baseline_count: int, structured: bool = False) -> str:
        messages = page.locator(SEL_ASSISTANT)
        if await messages.count() <= baseline_count:
            return ""
        source = await messages.last.evaluate(AGENT_SOURCE_EXTRACTOR)
        return source if isinstance(source, str) else None

    async def _is_generating(self, page: Page) -> bool:
        try:
            stop = page.locator(SEL_STOP).first
            if await stop.count() and await stop.is_visible():
                return True
            send = page.locator(SEL_SEND).first
            if await send.count():
                return await send.is_disabled()
        except Exception:  # noqa: BLE001
            pass
        return False

    async def _dismiss_overlays(self, page: Page) -> None:
        for sel in (
            'button[aria-label="Dismiss"]',
            'button:has-text("Accept")',
            'button:has-text("Got it")',
            'button:has-text("Allow all")',
        ):
            loc = page.locator(sel).first
            try:
                if await loc.count() and await loc.is_visible():
                    await loc.click(timeout=1500)
                    await page.wait_for_timeout(200)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------- подготовка ---------------------------------- #

    async def _ensure_login(self, page: Page) -> None:
        # Reuse the persistent Google session; never enter passwords or MFA.
        await page.wait_for_selector(
            '[data-testid="account-menu"], button:has-text("Venice Guest"), button[aria-label="Continue with Google"]',
            timeout=NAV_TIMEOUT * 1000,
        )
        account = page.locator('[data-testid="account-menu"]')
        text = await account.inner_text() if await account.count() else ""
        if text.strip() and "Venice Guest" not in text:
            self._authenticated = True
            return
        self._authenticated = False
        if not AUTO_GOOGLE_LOGIN:
            raise RuntimeError("Venice login required")
        await page.goto(BASE_URL + "/sign-in", wait_until="domcontentloaded", timeout=NAV_TIMEOUT * 1000)
        await page.get_by_role("button", name="Continue with Google", exact=True).click(timeout=30_000)
        # A remembered single Google account normally returns without interaction.
        # Do not choose identities or accept new consent blindly.
        try:
            await page.wait_for_url(lambda url: urlsplit(str(url)).hostname == urlsplit(BASE_URL).hostname
                                    and "/sign-in" not in urlsplit(str(url)).path,
                                    timeout=45_000)
            await page.wait_for_function("""() => {
                const el = document.querySelector('[data-testid="account-menu"]');
                return el && el.innerText.trim() && !el.innerText.includes("Venice Guest");
            }""", timeout=30_000)
        except Exception:
            raise RuntimeError("Google session requires operator verification; passwords/MFA are not automated") from None
        await page.goto(AGENT_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT * 1000)
        self._authenticated = True

    async def _new_chat(self, page: Page) -> None:
        await page.goto(AGENT_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT * 1000)
        await self._ensure_login(page)
        await page.wait_for_function(
            """() => {
                const text = document.body && document.body.innerText || '';
                return location.pathname === '/chat/classic' && !text.trim().startsWith('Loading');
            }""",
            timeout=NAV_TIMEOUT * 1000,
        )
        await page.wait_for_selector(SEL_TEXTAREA, state="visible", timeout=NAV_TIMEOUT * 1000)
        await self._dismiss_overlays(page)
        await page.wait_for_timeout(1200)

    async def _ensure_agent_view(self, page: Page) -> None:
        """Гарантирует неагентный Classic Chat, без внутреннего tool runtime."""
        if urlsplit(page.url).path == AGENT_PATH:
            return
        await self._new_chat(page)

    async def _model_button(self, page: Page):
        comp = page.locator(SEL_COMPOSER).first
        scope = comp if await comp.count() else page

        for sel in MODEL_BUTTON_SELECTORS:
            loc = scope.locator(sel).first
            try:
                if await loc.count() and await loc.is_visible():
                    return loc
            except Exception:  # noqa: BLE001
                continue

        # Фолбэк: ищем кнопку с текстом, похожим на название модели / бейдж Private
        try:
            btns = scope.locator("button")
            for i in range(await btns.count()):
                b = btns.nth(i)
                try:
                    if not await b.is_visible():
                        continue
                    text = (await b.inner_text()).strip()
                except Exception:  # noqa: BLE001
                    continue
                if text and (re.search(MODEL_MATCH, text, re.I) or text.strip() in ("Agent", "Auto") or "private" in text.lower()):
                    return b
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _wait_model_button(self, page: Page):
        deadline = time.monotonic() + MODEL_SELECT_TIMEOUT
        while True:
            button = await self._model_button(page)
            if button is not None:
                return button
            if time.monotonic() >= deadline:
                raise RuntimeError("Cannot verify model selector; refusing to use an unknown model")
            await asyncio.sleep(0.25)

    async def _ensure_model(self, page: Page) -> None:
        # Hydration can render the composer before its model selector. Wait
        # before any prompt is submitted; never retry an already sent prompt.
        btn = await self._wait_model_button(page)

        current = ""
        with contextlib.suppress(Exception):
            current = (await btn.inner_text()).strip()
        if current and re.search(MODEL_MATCH, current, re.I):
            log.debug("Модель уже нужная: %s", current.replace("\n", " / "))
            return

        log.info("Переключаю модель. Сейчас: %r", current.replace("\n", " / "))
        await btn.click()
        await page.wait_for_timeout(500)

        # Если в попапе есть поиск — фильтруем
        for ssel in ('input[type="search"]', 'input[placeholder*="earch" i]'):
            s = page.locator(ssel).first
            with contextlib.suppress(Exception):
                if await s.count() and await s.is_visible():
                    await s.fill("deepseek")
                    await page.wait_for_timeout(700)
                    break

        opt = page.get_by_text(re.compile(MODEL_MATCH, re.I)).last
        await opt.wait_for(state="visible", timeout=20_000)
        await opt.click()
        await page.wait_for_timeout(500)

        # Проверка
        btn2 = await self._wait_model_button(page)
        if btn2 is not None:
            with contextlib.suppress(Exception):
                text = (await btn2.inner_text()).strip()
                if text and re.search(MODEL_MATCH, text, re.I):
                    log.info("Модель выбрана: %s", text.replace("\n", " / "))
                    return
        raise RuntimeError(
            "Не удалось подтвердить выбор модели. Проверьте MODEL_MATCH / селекторы "
            "(запустите `python venice_agent_bridge.py check` с DEBUG=1)."
        )

    async def _prepare(self, page: Page) -> None:
        if NEW_CHAT_PER_REQUEST:
            await self._new_chat(page)
        else:
            await self._ensure_agent_view(page)
        await page.wait_for_selector(SEL_TEXTAREA, state="visible", timeout=NAV_TIMEOUT * 1000)
        await self._ensure_agent_view(page)
        await self._ensure_model(page)

    # ------------------------- основной цикл ------------------------------- #

    async def complete(
        self,
        prompt: str,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        structured: bool = False,
    ) -> str:
        async with self.request_slot():
            page = await self._page_or_start(recycle=NEW_CHAT_PER_REQUEST)
            try:
                await self._prepare(page)
            except Exception:
                await self._shot(page, "prepare_fail")
                raise

            transport = AgentBrowserTransport(prompt, structured)
            page.on("requestfinished", transport.on_request_finished)
            page.on("requestfailed", transport.on_request_failed)
            try:
                await page.route(AGENT_ROUTE_PATTERN, transport.route)
                answer = await self._generate_on_page(page, prompt, structured, transport)
                transport.check(complete=True)
            finally:
                page.remove_listener("requestfinished", transport.on_request_finished)
                page.remove_listener("requestfailed", transport.on_request_failed)
                await page.unroute(AGENT_ROUTE_PATTERN, transport.route)
            if on_delta:
                await on_delta(answer)
            return answer

    async def _generate_on_page(self, page, prompt, structured, transport):
        baseline_count = await page.locator(SEL_ASSISTANT).count()

        ta = page.locator(SEL_TEXTAREA).first
        await ta.click()
        with contextlib.suppress(Exception):
            await ta.fill("")
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(150)
        await page.keyboard.press("Enter")

        # --- ждём начала ответа ---
        started_at = time.monotonic()
        answer = ""
        while time.monotonic() - started_at < START_TIMEOUT:
            if transport:
                transport.check()
            await asyncio.sleep(0.3)
            answer = await self._answer(page, baseline_count, structured)
            if (answer and answer.strip()) or transport.request is not None or await self._is_generating(page):
                break
        else:
            await self._shot(page, "start_timeout")
            raise RuntimeError("Модель не начала отвечать (START_TIMEOUT)")

        # --- читаем поток ---
        deadline = time.monotonic() + GEN_TIMEOUT
        last_change = time.monotonic()
        finished_seen = None
        # The web UI can rewrite partial Markdown. Buffer until completion
        # rather than emitting an irreversible, corrupted SSE prefix.

        while time.monotonic() < deadline:
            if transport:
                transport.check()
            await asyncio.sleep(0.25)
            current = await self._answer(page, baseline_count, structured)
            if transport.finished and finished_seen is None:
                finished_seen = time.monotonic()

            if current != answer:
                answer = current
                last_change = time.monotonic()
                continue

            idle = time.monotonic() - last_change
            settle_window = max(STABLE_SEC, 5) if current is None else STABLE_SEC
            if (idle >= STABLE_SEC and finished_seen is not None
                    and time.monotonic() - finished_seen >= settle_window
                    and not await self._is_generating(page)):
                break
        else:
            raise RuntimeError("Generation deadline exceeded; refusing to report a normal completion")

        await self._shot(page, "done")
        if answer is None:
            raise AgentTransportError("Assistant source extraction failed after completed response; no tool call emitted")
        if not answer.strip():
            raise RuntimeError("Model returned no assistant content")
        answer = answer.strip()
        return answer


def _clean_answer(raw: str, baseline: str, prompt: str) -> str:
    """Вырезает из ленты эхо промпта и служебные строки UI."""
    if len(raw) <= len(baseline):
        return ""
    tail = raw[len(baseline):] if raw.startswith(baseline) else raw[len(baseline):]

    words = prompt.strip().split()[:14]
    if words:
        pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
        m = pattern.search(tail)
        if m:
            tail = tail[m.end():]

    lines = [ln for ln in tail.split("\n") if not CHROME_RE.match(ln.strip())]
    return "\n".join(lines).lstrip("\n")


SESSION = BrowserSession()

# --------------------------------------------------------------------------- #
#  API-МОДЕЛИ
# --------------------------------------------------------------------------- #

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "user"
    content: Any = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Any = None
    parallel_tool_calls: Optional[bool] = None


def _flatten(messages: list[ChatMessage]) -> str:
    prepared: list[tuple[str, str]] = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            buf: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("text"):
                        buf.append(str(part["text"]))
                elif part is not None:
                    buf.append(str(part))
            content = "\n".join(buf)
        text = (content or "").strip()
        if text:
            prepared.append(((m.role or "user").lower(), text))

    if not prepared:
        return ""
    if len(prepared) == 1 and prepared[0][0] == "user":
        return prepared[0][1]

    labels = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT", "tool": "TOOL"}
    blocks = [f"{labels.get(r, r.upper())}:\n{t}" for r, t in prepared]
    blocks.append("ASSISTANT:")
    return "\n\n".join(blocks)


def _record_rejection(text: str, category: str, request_id: str, attempt: int) -> Optional[str]:
    configured = os.getenv("AGENT_REJECTION_DIR", "/tmp/venice-agent-rejections").strip()
    if not configured:
        return None
    try:
        root = Path(configured)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.geteuid():
            raise OSError("Unsafe diagnostic directory")
        raw = text.encode("utf-8", errors="surrogatepass")
        identity = uuid.uuid4().hex
        path = root / ("rejection-" + identity + ".json")
        record = {"id": identity, "request_id": request_id, "attempt": attempt,
                  "category": category, "utf8_bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest(), "truncated": len(raw) > 65536,
                  "raw_utf8_base64": base64.b64encode(raw[:65536]).decode("ascii")}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(record, output)
        paths = sorted(root.glob("rejection-*.json"), key=lambda item: item.stat().st_mtime_ns)
        for old in paths[:-8]:
            old.unlink()
        log.warning("Rejected agent response id=%s category=%s attempt=%s", identity, category, attempt)
        return identity
    except Exception:
        log.warning("Protected rejection diagnostic unavailable; raw response not logged")
        return None


def _usage(prompt: str, completion: str) -> dict[str, int]:
    pt = max(1, len(prompt) // 4)
    ct = max(1, len(completion) // 4)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


async def _auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    if authorization.split(None, 1)[1].strip() != API_KEY:
        raise HTTPException(401, "Invalid API key")


# --------------------------------------------------------------------------- #
#  ПРИЛОЖЕНИЕ
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Сервис на http://%s:%s | модель: %s | agentic: %s", HOST, PORT, MODEL_ID, AGENT_URL)
    async def maintain_login():
        while True:
            async with SESSION._lock:
                try:
                    page = await SESSION._page_or_start()
                    if urlsplit(page.url).hostname != urlsplit(BASE_URL).hostname:
                        await page.goto(AGENT_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT * 1000)
                    await SESSION._ensure_login(page)
                except Exception:
                    log.warning("Browser authentication is not ready; retrying without password/MFA automation")
                    if SESSION._page is None or SESSION._page.is_closed():
                        await SESSION.close()
            await asyncio.sleep(15)

    task = asyncio.create_task(maintain_login()) if os.getenv("CDP_URL", "").strip() else None
    try:
        yield
    finally:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await SESSION.close()


app = FastAPI(
    title="Venice Agentic Bridge",
    version="1.0.0",
    description=(
        "OpenAI-совместимый шлюз к веб-интерфейсу Venice Classic Chat. "
        "Внутри — Playwright, жёстко зафиксированный Classic Chat и модель "
        "DeepSeek V4.1 Flash Uncensored (Private)."
    ),
    lifespan=lifespan,
)


@app.get("/", summary="Краткая информация о сервисе")
async def root() -> dict[str, Any]:
    return {
        "service": "venice-agentic-bridge",
        "model": MODEL_ID,
        "agent_url": AGENT_URL,
        "streaming": "buffered_sse",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/healthz", "/docs", "/openapi.json"],
    }


@app.get("/healthz", summary="Состояние браузерной сессии")
async def healthz() -> JSONResponse:
    def result(authenticated: bool, busy: bool = False) -> JSONResponse:
        return JSONResponse({"ok": authenticated, "authenticated": authenticated, "busy": busy},
                            status_code=200 if authenticated else 503)
    if SESSION._lock.locked():
        return result(SESSION._authenticated, busy=True)
    async with SESSION._lock:
        page = SESSION._page
        if page is None or page.is_closed():
            SESSION._authenticated = False
            return result(False)
        try:
            text = await asyncio.wait_for(
                page.locator('[data-testid="account-menu"]').inner_text(timeout=1500), 2)
            SESSION._authenticated = bool(text.strip() and "Venice Guest" not in text)
        except Exception:
            SESSION._authenticated = False
        return result(SESSION._authenticated)


@app.get("/v1/models", summary="Список доступных моделей", dependencies=[Depends(_auth)])
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "venice-webui",
                "root": MODEL_ID,
                "permission": [],
            }
        ],
    }


@app.post(
    "/v1/chat/completions",
    summary="Чат-комплишн (stream и non-stream)",
    response_model=None,
    dependencies=[Depends(_auth)],
)
async def chat_completions(req: ChatCompletionRequest):
    if req.model != MODEL_ID:
        raise HTTPException(400, "Unsupported model")
    protocol = None
    try:
        if req.tools or any(m.tool_calls or m.role == "tool" for m in req.messages):
            protocol = AgentProtocol(req.tools or [], req.tool_choice,
                                     [m.model_dump(exclude_none=True) for m in req.messages])
            prompt = protocol.prompt()
        else:
            if req.tool_choice not in (None, "auto", "none"):
                raise ProtocolError("Tool choice requires tools")
            prompt = _flatten(req.messages)
    except PromptBudgetError as exc:
        return JSONResponse(status_code=400, content={"error": exc.as_openai_error()})
    except Exception:
        raise HTTPException(400, "Invalid or unsupported agent request (tools, choice, history, schema or budget)") from None

    def decode(text):
        return protocol.decode(text) if protocol else ({"role": "assistant", "content": text}, "stop")
    if not prompt.strip():
        raise HTTPException(400, "messages пусты")

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    async def generate_decoded(on_delta=None, on_recovery=None):
        input_text = prompt
        outputs = []
        recovery = None
        for attempt in range(2):
            text = await SESSION.complete(input_text, **({"on_delta": on_delta} if on_delta else {}),
                                          **({"structured": True} if protocol else {}))
            outputs.append(text)
            try:
                message, finish = decode(text)
            except ProtocolError as exc:
                category = str(exc)
                diagnostic_id = _record_rejection(text, category, cid, attempt + 1)
                exc.diagnostic_id = diagnostic_id
                exc.category = category
                if not protocol or attempt == 1:
                    raise
                try:
                    input_text = protocol.prompt(recovery=True)
                except PromptBudgetError:
                    raise exc from None
                recovery = {"attempts": 1, "diagnostic_id": diagnostic_id, "category": category}
                notice = f"[bridge: protocol regeneration 1/1; diagnostic_id={diagnostic_id or 'unavailable'}; rejected reply executed NO host tools]\n"
                if on_recovery:
                    await on_recovery(notice)
                continue
            if recovery:
                message["bridge_protocol_recovery"] = recovery
                if not on_recovery:
                    message["content"] = notice + (message.get("content") or "")
            used_prompt = prompt + input_text if recovery else prompt
            return message, finish, _usage(used_prompt, "".join(outputs))
        raise AssertionError("Unreachable regeneration state")

    if not req.stream:
        try:
            message, finish, usage = await generate_decoded()
        except AgentTransportError as exc:
            raise HTTPException(502, str(exc)) from None
        except ProtocolError as exc:
            ref = getattr(exc, "diagnostic_id", None)
            raise HTTPException(502, "Invalid assistant agent envelope; no tool call emitted" +
                                (f"; diagnostic_id={ref}" if ref else "")) from None
        except BrowserBusyError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from None
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка генерации")
            raise HTTPException(502, f"Ошибка браузерной сессии: {exc}") from exc

        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": req.model or MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        }

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def worker() -> None:
        async def on_delta(delta: str) -> None:
            if not protocol:
                await queue.put(("delta", delta))

        try:
            async def on_recovery(notice):
                await queue.put(("notice", notice))
            await queue.put(("final", await generate_decoded(on_delta, on_recovery)))
        except AgentTransportError as exc:
            await queue.put(("error", {"message": str(exc), "type": "upstream_agent_transport_error"}))
        except ProtocolError as exc:
            await queue.put(("error", {"message": "Invalid assistant agent envelope; no tool call emitted", "type": "upstream_protocol_error",
                                       "diagnostic_id": getattr(exc, "diagnostic_id", None),
                                       "category": getattr(exc, "category", "validation")}))
        except BrowserBusyError as exc:
            await queue.put(("error", {"message": str(exc), "type": "rate_limit_error", "code": 429}))
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка генерации (stream)")
            await queue.put(("error", {"message": str(exc), "type": "upstream_error"}))

    def sse(obj: dict[str, Any]) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def chunk(delta: dict[str, Any], finish: Optional[str] = None, usage=None) -> str:
        event = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model or MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        if usage is not None:
            event["usage"] = usage
        return sse(event)

    async def streamer():
        task = asyncio.create_task(worker())
        try:
            yield chunk({"role": "assistant", "content": ""})
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if kind in ("delta", "notice"):
                    yield chunk({"content": payload})
                elif kind == "final":
                    message, finish, usage = payload
                    if message.get("tool_calls"):
                        yield chunk({"tool_calls": [dict(call, index=i) for i, call in enumerate(message["tool_calls"])]})
                    elif protocol:
                        yield chunk({"content": message["content"]})
                    yield chunk({}, finish, usage if protocol else None)
                    yield "data: [DONE]\n\n"
                    break
                else:
                    err = sse({"error": payload})
                    yield err
                    yield "data: [DONE]\n\n"
                    break
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        streamer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

async def _cmd_login() -> None:
    await SESSION._launch()
    page = await SESSION._page_or_start()
    await page.goto(AGENT_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    body = (await page.locator("body").inner_text())[:500]
    print("\nБраузер открыт, cookies подставлены автоматически.")
    print("URL:", page.url)
    print(body.replace("\n", " | ")[:400])
    print(f"Профиль: {PROFILE_DIR}. Окно остаётся открытым.\n")
    await asyncio.Event().wait()


async def _cmd_check() -> None:
    global HEADLESS
    page = await SESSION._page_or_start()
    await SESSION._new_chat(page)
    await SESSION._ensure_agent_view(page)
    print("URL:", page.url)
    btn = await SESSION._model_button(page)
    print("Кнопка модели:", "найдена" if btn else "НЕ НАЙДЕНА")
    if btn:
        print("Текст кнопки:", (await btn.inner_text()).strip().replace("\n", " / "))
    await SESSION._ensure_model(page)
    print("После выбора:", (await (await SESSION._model_button(page)).inner_text()).strip().replace("\n", " / "))
    print("\nПробный запрос...")
    out = await SESSION.complete("Ответь одним словом: работает?")
    print("Ответ модели:", out[:400])
    await SESSION.close()


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "login":
        asyncio.run(_cmd_login())
    elif cmd == "check":
        asyncio.run(_cmd_check())
    elif cmd == "serve":
        uvicorn.run(app, host=HOST, port=PORT, log_level="debug" if DEBUG else "info")
    else:
        print("Использование: python venice_agent_bridge.py [login|check|serve]")


if __name__ == "__main__":
    main()
