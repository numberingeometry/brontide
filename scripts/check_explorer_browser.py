"""Exercise atlas controls against the committed data in a real browser.

Run a local static server at the repository root, then pass its URL:
    python scripts/check_explorer_browser.py --url http://127.0.0.1:8000/

Requires Playwright and Chromium. This is a browser acceptance check, separate
from the CPU-only scientific verification job.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def expected_count(record: dict, months: list[str], start: int, end: int) -> int:
    return sum(value for value, month in zip(record["raw_counts"], months)
               if start <= int(month[:4]) <= end)


def assert_count(page, expected: int) -> None:
    page.wait_for_function(
        "expected => Number(document.getElementById('stat-one').title.replaceAll(',', '')) === expected",
        arg=expected,
    )


def slider(page, element_id: str, value: int) -> None:
    page.locator(f"#{element_id}").evaluate(
        "(el, value) => {el.value = value; el.dispatchEvent(new Event('input', {bubbles: true}));}",
        str(value),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="URL serving the repository root")
    parser.add_argument("--chromium", default=shutil.which("chromium"))
    args = parser.parse_args()
    data = json.loads((ROOT / "artifacts/explorer/regions.json").read_text())
    microdata = json.loads((ROOT / "artifacts/explorer/microregions.json").read_text())
    manifest = json.loads((ROOT / "artifacts/explorer/manifest.json").read_text())
    sample_years = (ROOT / "artifacts/explorer/year.bin").read_bytes()
    # Pick a real named territory with several microregions.
    region = next(r for r in data["regions"] if r["name"] == "Quantum Many-Body Dynamics")
    micro = next(r for r in microdata["regions"] if r["superregion_id"] == region["id"])
    years = [int(month[:4]) for month in data["months"]]
    start, end = 2017, 2019
    year0 = manifest["sample"]["year0"]
    errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chromium, headless=True,
                                     args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000},
                                      reduced_motion="reduce")
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("Boolean(window.__BRONTIDE__)")
        assert_count(page, sum(data["monthly_totals"]))
        assert page.locator("#topic-list button").count() == len(data["regions"])

        page.locator("#topic-search").fill("Quantum Many-Body")
        page.locator(f'#topic-list button[data-region="{region["id"]}"]').click()
        assert_count(page, region["series_n_papers"])
        assert page.locator("#chart-unit").inner_text() == "% of arXiv"
        assert page.locator("#forecast-legend").is_visible()
        title_link = page.locator("#representative-title a")
        query = parse_qs(urlparse(title_link.get_attribute("href")).query)
        assert query["query"] == [region["representatives"][0]["title"]]
        assert query["searchtype"] == ["title"]

        slider(page, "year-from", start - year0)
        slider(page, "year-to", end - year0)
        assert_count(page, expected_count(region, data["months"], start, end))
        denominator = sum(n for n, year in zip(data["monthly_totals"], years) if start <= year <= end)
        expected_share = expected_count(region, data["months"], start, end) / denominator * 100
        displayed_share = float(page.locator("#stat-two").inner_text().rstrip("%"))
        assert abs(displayed_share - expected_share) <= 0.0051
        expected_visible = sum(start - year0 <= year <= end - year0 for year in sample_years)
        assert page.evaluate("window.__BRONTIDE__.getVisiblePointCount()") == expected_visible

        page.locator("#compare-toggle").check()
        assert page.evaluate("window.__BRONTIDE__.getVisiblePointCount()") == sum(
            year <= end - year0 for year in sample_years)
        assert_count(page, expected_count(region, data["months"], start, end))
        endpoint_means = []
        for year in (start, end):
            shares = [n / total * 100 for n, total, y in
                      zip(region["raw_counts"], data["monthly_totals"], years) if y == year]
            endpoint_means.append(sum(shares) / len(shares))
        displayed_delta = float(page.locator("#comparison-note strong").inner_text().split()[0])
        assert abs(displayed_delta - (endpoint_means[1] - endpoint_means[0])) <= 0.0051

        page.locator("#topic-search").fill("")
        page.locator("#topic-sort").select_option("change")

        def observed_change(record: dict) -> float:
            means = []
            for year in (start, end):
                values = [n / total * 100 for n, total, y in
                          zip(record["raw_counts"], data["monthly_totals"], years) if y == year]
                means.append(sum(values) / len(values))
            return means[1] - means[0]

        expected_order = [r["id"] for r in sorted(data["regions"], key=observed_change, reverse=True)]
        displayed_order = page.locator("#topic-list button").evaluate_all(
            "buttons => buttons.map(button => Number(button.dataset.region))")
        assert displayed_order == expected_order

        page.locator("#subarea-toggle").click()
        page.locator(f'#microregion-select option[value="{micro["id"]}"]').wait_for(state="attached")
        page.locator("#microregion-select").select_option(str(micro["id"]))
        assert_count(page, expected_count(micro, data["months"], start, end))
        assert not page.locator("#representative").is_visible()
        assert page.evaluate("window.__BRONTIDE__.getState().selected") == region["id"]

        page.locator('[data-view="map"]').click()
        page.locator("#share-view").click()
        saved_url = page.url
        before = page.evaluate("window.__BRONTIDE__.getState()")
        page.goto(saved_url, wait_until="networkidle")
        page.wait_for_function("Boolean(window.__BRONTIDE__)")
        after = page.evaluate("window.__BRONTIDE__.getState()")
        for key in ("mode", "from", "to", "selected", "micro", "compare"):
            assert before[key] == after[key], (key, before[key], after[key])
        assert_count(page, expected_count(micro, data["months"], start, end))

        # In-page navigation must restore a new microregion even with its panel open.
        other_micro = next(r for r in microdata["regions"]
                           if r["superregion_id"] == region["id"] and r["id"] != micro["id"])
        page.evaluate(
            "p => { const h = new URLSearchParams(location.hash.slice(1)); "
            "h.set('micro', p); location.hash = h.toString(); }", str(other_micro["id"]),
        )
        page.wait_for_function("id => window.__BRONTIDE__.getState().micro === id",
                               arg=other_micro["id"])
        assert_count(page, expected_count(other_micro, data["months"], start, end))

        page.locator("#clear-selection").click()
        assert_count(page, denominator)
        assert not page.locator("#forecast-legend").is_visible()
        assert page.locator("#chart-unit").inner_text() == "papers"

        # Open the on-demand territory browser before choosing another field.
        page.locator("#topic-search").focus()
        # A tour changes the parent territory; stale microregion options must not survive.
        page.locator(f'#topic-list button[data-region="{region["id"]}"]').click()
        page.locator("#subarea-toggle").click()
        page.locator(f'#microregion-select option[value="{micro["id"]}"]').wait_for(state="attached")
        page.locator("#story-start").click()
        assert not page.locator("#subarea-content").is_visible()
        assert page.evaluate("window.__BRONTIDE__.getState().micro") is None
        page.locator("#story-next").click()
        parent = page.evaluate("window.__BRONTIDE__.getState().selected")
        page.locator("#subarea-toggle").click()
        page.wait_for_function("document.querySelectorAll('#microregion-select option').length > 1")
        options = page.locator("#microregion-select option").evaluate_all(
            "options => options.filter(option => option.value !== '').map(option => Number(option.value))")
        assert set(options) == {r["id"] for r in microdata["regions"] if r["superregion_id"] == parent}
        page.evaluate("location.hash = 'view=map&from=2017&to=2019&region=3'")
        page.wait_for_function("window.__BRONTIDE__.getState().selected === 3")
        assert not page.locator("#subarea-content").is_visible()

        # Shared URLs are untrusted inputs. Malformed years must keep a usable map.
        page.goto(args.url.split("#", 1)[0] + "#from=bad&to=2019.5", wait_until="networkidle")
        page.wait_for_function("Boolean(window.__BRONTIDE__)")
        recovered = page.evaluate("window.__BRONTIDE__.getState()")
        for key in ("from", "to"):
            assert math.isfinite(recovered[key]) and recovered[key] == int(recovered[key])
        assert 0 <= recovered["from"] <= recovered["to"] <= manifest["sample"]["year1"] - year0
        recovered_count = sum(n for n, year in zip(data["monthly_totals"], years)
                              if year0 + recovered["from"] <= year <= year0 + recovered["to"])
        assert_count(page, recovered_count)
        assert not errors, errors
        browser.close()
    print("PASS: exact window counts and shares, annual sample filtering, comparison math, "
          "microregion statistics, representative links, shared-view restoration, "
          "and malformed-URL recovery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
