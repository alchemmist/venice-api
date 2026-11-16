"""Isolated connectivity probe. Does not import cookies, log in, or run inference.

Requires explicit proxy and a dedicated profile. Never point it at a live profile.
Output deliberately excludes page contents, cookies, headers and tokenized URLs.
"""
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import async_playwright


async def main():
    proxy = os.environ.get("PROXY_SERVER", "")
    parsed = urlsplit(proxy)
    if parsed.scheme not in {"http", "socks5"} or not parsed.hostname:
        raise SystemExit("An explicit http:// or socks5:// PROXY_SERVER is required")
    if parsed.username or parsed.password:
        raise SystemExit("Proxy credentials in URLs are not supported")
    profile = Path(os.environ.get("PROFILE_DIR", "/data/profile")).resolve()
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            str(profile), channel="chrome", headless=False,
            proxy={"server": proxy}, chromium_sandbox=True,
            viewport={"width": 1440, "height": 900},
        )
        try:
            page = await context.new_page()
            response = await page.goto("https://venice.ai", wait_until="domcontentloaded", timeout=45000)
            print(json.dumps({
                "browser": context.browser.version if context.browser else "persistent",
                "http_status": response.status if response else None,
                "final_origin": urlsplit(page.url).netloc,
                "note": "Connectivity only; login and inference are NOT verified",
            }))
        finally:
            await context.close()


if __name__ == "__main__":
    asyncio.run(main())
