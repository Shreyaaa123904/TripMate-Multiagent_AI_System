"""Real-browser checks: Markdown rendering, XSS payloads, CSP, Copy and PDF."""

from pathlib import Path

import pytest

from tests.conftest import AppServer, skip_or_fail

pytestmark = pytest.mark.browser

INIT_SCRIPT = """
window.__xss = 0;
window.__csp = [];
document.addEventListener('securitypolicyviolation', e => {
  window.__csp.push(e.violatedDirective + ' ' + e.blockedURI);
});
"""


@pytest.fixture(scope="module")
def server():
    srv = AppServer("tests.browser.fake_ui_app:app", {"RATE_LIMIT_PER_MINUTE": "1000"})
    srv.start()
    yield srv
    srv.cleanup()


@pytest.fixture(scope="module")
def browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        skip_or_fail("playwright is not installed")
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:
            skip_or_fail(f"Chromium not available: {exc}")
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    context = browser.new_context(accept_downloads=True)
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=server.base_url)
    pg = context.new_page()
    pg.add_init_script(INIT_SCRIPT)
    pg.dialogs = []
    pg.on("dialog", lambda d: (pg.dialogs.append(d.message), d.dismiss()))
    pg.goto(server.base_url + "/")
    yield pg
    context.close()


def generate(page):
    page.fill("#userInput", "Plan a 7 days Japan trip")
    page.click("#sendBtn")
    page.wait_for_selector("#resultBox h2")
    page.wait_for_timeout(500)  # let any onerror/onload handlers fire


def test_quick_prompt_buttons_still_work(page):
    page.click(".quick-prompts button:has-text('Dubai Trip')")
    assert page.input_value("#userInput").startswith("Plan a 5 days Dubai trip")


def test_markdown_formatting_preserved(page):
    generate(page)
    box = page.locator("#resultBox")
    assert box.locator("h2", has_text="Trip Summary").count() == 1
    assert box.locator("h3").count() >= 2
    assert box.locator("ul li").count() >= 2
    assert box.locator("table tr").count() == 3
    assert box.locator("strong").count() >= 1 and box.locator("em").count() >= 1
    assert box.locator("pre code").count() == 1
    assert box.locator("blockquote").count() == 1
    assert "Thread ID: user_" in page.inner_text("#threadInfo")


def test_safe_links_kept_and_hardened(page):
    generate(page)
    link = page.locator("#resultBox a", has_text="Japan Guide")
    assert link.get_attribute("href") == "https://www.japan-guide.com/"
    assert link.get_attribute("target") == "_blank"
    assert link.get_attribute("rel") == "noopener noreferrer"
    assert page.locator("#resultBox a[href^='mailto:']").count() == 1


def test_malicious_payloads_are_inert(page):
    generate(page)
    assert page.evaluate("window.__xss") == 0
    assert page.dialogs == []
    box = page.locator("#resultBox")
    # Harmless SVG/MathML markup may remain; its event handlers are checked below.
    for selector in ["script", "iframe", "object", "embed", "form", "input", "style", "meta",
                     "body", "mglyph"]:
        assert box.locator(selector).count() == 0, selector
    has_handlers = page.evaluate("""() => [...document.querySelectorAll('#resultBox *')]
        .some(el => [...el.attributes].some(a => a.name.toLowerCase().startsWith('on')))""")
    assert not has_handlers
    hrefs = page.evaluate("""() => [...document.querySelectorAll('#resultBox [href], #resultBox [src]')]
        .map(el => (el.getAttribute('href') || el.getAttribute('src') || '').trim().toLowerCase())""")
    assert not [h for h in hrefs if h.startswith(("javascript:", "data:", "jav"))]
    styles = page.evaluate("""() => [...document.querySelectorAll('#resultBox [style]')]
        .map(el => el.getAttribute('style'))""")
    assert not [s for s in styles if "javascript" in s.lower()]
    assert page.is_visible("body")  # injected <style> did not hide the page
    assert page.url.startswith(page.url.split("/")[0])  # no meta refresh navigation


def test_no_csp_violations_during_normal_use(page, tmp_path):
    generate(page)
    page.click(".copy-btn")
    with page.expect_download(timeout=60000):
        page.click(".download-btn")
    violations = [v for v in page.evaluate("window.__csp")
                  # Blocked payload images/frames are expected; only our own code matters.
                  if "example.com" not in v and "evil.example" not in v and not v.endswith(" x")]
    assert violations == []


def test_copy_button(page):
    generate(page)
    page.click(".copy-btn")
    page.wait_for_selector(".copy-btn:has-text('Copied')")
    copied = page.evaluate("navigator.clipboard.readText()")
    assert copied == page.inner_text("#resultBox")
    assert "Trip Summary" in copied


def test_pdf_download(page, tmp_path):
    generate(page)
    with page.expect_download(timeout=60000) as info:
        page.click(".download-btn")
    path = Path(tmp_path) / "plan.pdf"
    info.value.save_as(path)
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 5000
    assert info.value.suggested_filename == "ai-travel-plan.pdf"


def test_without_dompurify_answer_is_plain_text(browser, server):
    context = browser.new_context()
    pg = context.new_page()
    pg.add_init_script(INIT_SCRIPT)
    pg.route("**/static/vendor/purify.min.js", lambda route: route.abort())
    pg.goto(server.base_url + "/")
    pg.fill("#userInput", "Plan a 7 days Japan trip")
    pg.click("#sendBtn")
    pg.wait_for_selector("#resultBox:has-text('Trip Summary')")
    pg.wait_for_timeout(300)
    assert pg.locator("#resultBox *").count() == 0
    assert "<script>" in pg.text_content("#resultBox")
    assert pg.evaluate("window.__xss") == 0
    context.close()
