"""Exercise atlas controls against the committed data in a real browser.

Run a local static server at the repository root, then pass its URL:
    python scripts/check_explorer_browser.py --url http://127.0.0.1:8000/

Requires Playwright and Chromium. This is a browser acceptance check, separate
from the CPU-only scientific verification job.
"""
from __future__ import annotations

import argparse
import base64
import re
import tempfile
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


def check_currents(page, browser, url: str, data: dict, errors: list[str]) -> None:
    """Check rendered composition against independent sums of the frozen counts."""
    page.goto(url.split("#", 1)[0], wait_until="networkidle")
    page.wait_for_function("Boolean(window.__BRONTIDE__)")
    assert page.evaluate("window.__BRONTIDE__.getState().mode") == "currents"
    assert not page.evaluate("window.__BRONTIDE__.getState().playing")
    model = page.evaluate("window.__BRONTIDE__.getCurrents()")
    records = {r["id"]: r for r in data["regions"]}
    assert model["months"] == data["months"]
    assert model["order"] == sorted(records)
    assert model["windowMonths"] == 12 and model["forecast"] is False
    assert page.locator("#currents-interaction path").count() == len(records)
    for month, date in enumerate(data["months"]):
        first = max(0, month - 11)
        denominator = sum(data["monthly_totals"][first:month + 1])
        assert model["windowStarts"][month] == data["months"][first]
        assert model["totals"][month] == denominator
        assert abs(sum(row[month] for row in model["shares"]) - 100) < 1e-10
        for row, region_id in enumerate(model["order"]):
            count = sum(records[region_id]["raw_counts"][first:month + 1])
            assert model["counts"][row][month] == count
            assert abs(model["shares"][row][month] - count / denominator * 100) < 1e-11
    # Check actual SVG hit geometry, not only the exported composition model.
    paths = page.locator("#currents-interaction path").evaluate_all("nodes => nodes.map(n => n.getAttribute('d'))")
    for row, path in enumerate(paths):
        coordinates = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path)]
        drawn_thickness = coordinates[-1] - coordinates[1]
        expected_thickness = model["layout"]["thickness"] * model["shares"][row][0] / 100
        assert abs(drawn_thickness - expected_thickness) < 0.021

    # Keyboard exploration exposes the available early-2007 windows and raw months.
    first_id = model["order"][0]
    ribbon = page.locator(f'[data-current-region="{first_id}"]')
    ribbon.focus()
    page.keyboard.press("Home")
    assert "2007-01 — 2007-01 · 1 month" in page.locator("#map-tooltip").inner_text(), page.locator("#map-tooltip").inner_text()
    page.keyboard.press("ArrowRight")
    text = page.locator("#map-tooltip").inner_text()
    assert "2007-01 — 2007-02 · 2 months" in text
    assert "Raw month" in text and "Ribbon" in text
    page.keyboard.press("ArrowDown")
    assert page.locator(f'[data-current-region="{model["order"][1]}"]').evaluate("el => el === document.activeElement")
    page.keyboard.press("Enter")
    assert page.evaluate("window.__BRONTIDE__.getState().selected") == model["order"][1]
    page.locator("#clear-selection").click()
    page.locator("#theme-summary").focus()

    # Hit a real ribbon at a known observed month and check both displayed statistics.
    region_id = max(records, key=lambda i: records[i]["n_papers"])
    row = model["order"].index(region_id)
    month = 120
    layout = model["layout"]
    bounds = page.locator("#currents-interaction").bounding_box()
    x = bounds["x"] + layout["x"] + month / (len(data["months"]) - 1) * layout["width"]
    y = bounds["y"] + layout["centers"][month] + (sum(values[month] for values in model["shares"][:row]) + model["shares"][row][month] / 2 - 50) / 100 * layout["thickness"]
    page.mouse.move(x, y)
    page.locator("#map-tooltip").wait_for(state="visible")
    text = page.locator("#map-tooltip").inner_text()
    record = records[region_id]
    assert record["name"] in text
    assert f'{model["shares"][row][month]:.2f}%' in text
    assert f'{record["raw_counts"][month] / data["monthly_totals"][month] * 100:.2f}%' in text
    assert f'{record["raw_counts"][month]:,} of {data["monthly_totals"][month]:,} papers' in text
    page.mouse.move(5, 5)

    # Theme choice preserves the scientific state and affects actual pixels and PNGs.
    previous = page.evaluate("window.__BRONTIDE__.getState()")
    rendered = []
    backgrounds = {"paper": [252, 250, 245], "sage": [231, 235, 212], "night": [16, 29, 32]}
    with tempfile.TemporaryDirectory() as directory:
        for theme, expected_rgb in backgrounds.items():
            page.locator("#theme-summary").click()
            page.locator(f'button[data-theme="{theme}"]').click()
            page.wait_for_timeout(80)
            assert page.evaluate("document.documentElement.dataset.theme") == theme
            assert page.evaluate("localStorage.getItem('brontide-theme')") == theme
            after = page.evaluate("window.__BRONTIDE__.getState()")
            for key in ("mode", "from", "to", "selected", "micro", "compare", "camera", "observation"):
                assert previous[key] == after[key]
            rendered.append(page.locator("#currents-canvas").evaluate("c => c.toDataURL()"))
            with page.expect_download() as download:
                page.locator("#export-image").click()
            output = Path(directory) / f"{theme}.png"
            download.value.save_as(output)
            encoded = base64.b64encode(output.read_bytes()).decode()
            pixel = page.evaluate("""async encoded => {
                const im = new Image(); im.src = 'data:image/png;base64,' + encoded;
                await im.decode(); const c = document.createElement('canvas'); c.width = c.height = 1;
                const ctx = c.getContext('2d'); ctx.drawImage(im, 0, 0);
                return Array.from(ctx.getImageData(0, 0, 1, 1).data).slice(0, 3);
            }""", encoded)
            assert pixel == expected_rgb, (theme, pixel)
    assert len(set(rendered)) == 3
    page.reload(wait_until="networkidle")
    page.wait_for_function("Boolean(window.__BRONTIDE__)")
    assert page.evaluate("window.__BRONTIDE__.getState().theme") == "night"
    for mode in ("space", "terrain", "map", "currents"):
        page.locator(f'[data-view="{mode}"]').click()
        assert page.evaluate("window.__BRONTIDE__.getState().mode") == mode
    slider(page, "year-to", 2013 - 2007)
    assert page.evaluate("window.__BRONTIDE__.getCurrents().endpoint") == data["months"].index("2013-12")
    assert page.evaluate("window.__BRONTIDE__.getCurrents().order") == model["order"]
    page.locator("#share-view").click()
    shared = page.url
    page.goto(shared, wait_until="networkidle")
    page.wait_for_function("Boolean(window.__BRONTIDE__)")
    assert page.evaluate("window.__BRONTIDE__.getState().mode") == "currents"
    assert page.evaluate("window.__BRONTIDE__.getState().to") == 6

    # Touch selection and the complete temporal view also work without WebGL.
    fallback = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True,
                                   is_mobile=True, reduced_motion="reduce")
    mobile = fallback.new_page()
    mobile.on("pageerror", lambda exc: errors.append(str(exc)))
    mobile.add_init_script("""const getContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(type, ...args) {
            return type.includes('webgl') ? null : getContext.call(this, type, ...args);
        };""")
    mobile.goto(url.split("#", 1)[0], wait_until="networkidle")
    mobile.wait_for_function("Boolean(window.__BRONTIDE__)")
    assert mobile.evaluate("window.__BRONTIDE__.getState().webgl") is False
    assert mobile.evaluate("window.__BRONTIDE__.getState().mode") == "currents"
    assert mobile.locator('[data-view="space"]').is_disabled()
    assert mobile.locator('[data-view="terrain"]').is_disabled()
    assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
    mobile_model = mobile.evaluate("window.__BRONTIDE__.getCurrents()")
    layout = mobile_model["layout"]
    bounds = mobile.locator("#currents-interaction").bounding_box()
    x = bounds["x"] + layout["x"] + month / (len(data["months"]) - 1) * layout["width"]
    y = bounds["y"] + layout["centers"][month] + (sum(values[month] for values in model["shares"][:row]) + model["shares"][row][month] / 2 - 50) / 100 * layout["thickness"]
    mobile.touchscreen.tap(x, y)
    assert mobile.evaluate("window.__BRONTIDE__.getState().selected") == region_id
    observation_text = mobile.locator("#current-observation").inner_text()
    assert "Raw month" in observation_text and "Ribbon" in observation_text
    assert f'{record["raw_counts"][month]:,} of {data["monthly_totals"][month]:,} papers' in observation_text
    assert mobile.evaluate("window.__BRONTIDE__.getState().observation") == month
    mobile.locator("#share-view").click()
    assert "at=2017-01" in mobile.url
    mobile.reload(wait_until="networkidle")
    mobile.wait_for_function("Boolean(window.__BRONTIDE__)")
    assert mobile.evaluate("window.__BRONTIDE__.getState().observation") == month
    assert mobile.locator("#current-observation").inner_text() == observation_text
    mobile.locator("#share-view").click(trial=True)
    mobile.locator("#export-image").click(trial=True)
    mobile.locator("#clear-selection").click()
    mobile.locator('[data-view="map"]').click()
    assert mobile.evaluate("window.__BRONTIDE__.getState().mode") == "map"
    fallback.close()


def png_info(page, path: Path) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return page.evaluate("""async encoded => {
        const im = new Image(); im.src = 'data:image/png;base64,' + encoded;
        await im.decode(); const c = document.createElement('canvas'); c.width = c.height = 1;
        const ctx = c.getContext('2d'); ctx.drawImage(im, 0, 0);
        return {width: im.naturalWidth, height: im.naturalHeight,
                pixel: Array.from(ctx.getImageData(0, 0, 1, 1).data).slice(0, 3)};
    }""", encoded)


def open_pulse(page, url: str, at: str | None = None) -> None:
    page.goto(url.split("#", 1)[0] + "#view=pulse" + (f"&at={at}" if at else ""), wait_until="networkidle")
    page.wait_for_function("Boolean(window.__BRONTIDE__) && window.__BRONTIDE__.getPulse() !== null")
    page.wait_for_timeout(120)


def check_pulse(page, browser, url: str, errors: list[str]) -> None:
    pulse = json.loads((ROOT / "artifacts/explorer/pulse.json").read_text())
    regions = {r["id"]: r for r in json.loads((ROOT / "artifacts/explorer/regions.json").read_text())["regions"]}
    months = pulse["months"]
    page.goto(url.split("#", 1)[0], wait_until="networkidle")
    page.wait_for_function("Boolean(window.__BRONTIDE__)")
    page.locator('[data-view="pulse"]').click()
    page.wait_for_function("window.__BRONTIDE__.getPulse() !== null")
    assert page.evaluate("window.__BRONTIDE__.getState().mode") == "pulse"
    assert page.locator("#pulse-chart path").count() >= 5
    state = page.evaluate("window.__BRONTIDE__.getPulse()")
    assert state["month"] == months[-1] and not state["pinned"]
    legend = page.locator("#pulse-chart").text_content()
    for key in ("territories", "microregions", "within"):
        assert f"{pulse[key]['angle_deg'][-1]:.1f}°" in legend, key
    # The chart is a keyboard slider whose value is the pinned or dock month.
    chart = page.locator("#pulse-chart")
    assert chart.get_attribute("role") == "slider"
    assert chart.get_attribute("aria-valuemax") == str(len(months) - 1)
    assert chart.get_attribute("aria-valuenow") == str(len(months) - 1)
    # Movers match the published leaders, in order, with their real names.
    names = page.locator("#pulse-movers .pulse-name").all_inner_texts()
    assert names == [regions[k]["name"] for k, _, _ in pulse["territories"]["leaders"][-1][:5]]
    # The dock year moves the cursor to the last observed month of that year.
    slider(page, "year-to", 2013 - 2007)
    assert page.evaluate("window.__BRONTIDE__.getPulse().month") == "2013-12"
    # Keyboard: Home pins the first comparison and records it in the URL.
    chart.focus()
    page.keyboard.press("Home")
    assert page.evaluate("window.__BRONTIDE__.getPulse()")["index"] == 0
    page.keyboard.press("PageDown")
    assert page.evaluate("window.__BRONTIDE__.getPulse().month") == months[12]
    page.wait_for_function("v => document.getElementById('pulse-chart').getAttribute('aria-valuenow') === v", arg="12")
    page.wait_for_function(f"location.hash.includes('at={months[12]}')")
    # Click, then arrows: each press moves one month although the pointer stays on the chart.
    box = chart.bounding_box()
    page.mouse.click(box["x"] + box["width"] * .55, box["y"] + box["height"] * .7)
    clicked = page.evaluate("window.__BRONTIDE__.getPulse()")
    assert clicked["pinned"]
    for step in (1, 2, 3):
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(40)
        now = page.evaluate("window.__BRONTIDE__.getPulse()")
        assert now["index"] == clicked["index"] + step, (step, now, clicked)
    page.wait_for_function("v => document.getElementById('pulse-chart').getAttribute('aria-valuenow') === v",
                           arg=str(clicked["index"] + 3))
    assert "inside fields" in chart.get_attribute("aria-valuetext")
    page.mouse.move(5, 5)
    # Leaving Pulse drops the pin; returning shows the dock month, matching the URL.
    page.locator('[data-view="map"]').click()
    page.locator('[data-view="pulse"]').click()
    page.wait_for_timeout(80)
    assert not page.evaluate("window.__BRONTIDE__.getPulse().pinned")
    page.wait_for_function("location.hash.includes('view=pulse') && !location.hash.includes('at=')")
    # A hash change without a month also clears the pin.
    open_pulse(page, url, "2012-03")
    assert page.evaluate("window.__BRONTIDE__.getPulse().month") == "2012-03"
    page.evaluate("location.hash = 'view=pulse&from=2007&to=2026'")
    page.wait_for_timeout(150)
    assert not page.evaluate("window.__BRONTIDE__.getPulse().pinned")
    # A well-formed month outside the series is dropped, not echoed into the URL.
    open_pulse(page, url, "2007-05")
    assert page.evaluate("window.__BRONTIDE__.getState().pulseAt") is None
    page.wait_for_function("location.hash.includes('view=pulse') && !location.hash.includes('at=')")
    # Shared links restore the pinned comparison.
    open_pulse(page, url, "2017-09")
    assert page.evaluate("window.__BRONTIDE__.getPulse().month") == "2017-09"
    i = months.index("2017-09")
    lead = pulse["territories"]["leaders"][i][0][0]
    page.locator("#pulse-movers button").first.click()
    assert page.evaluate("window.__BRONTIDE__.getState().selected") == lead
    # Exports use one fixed layout and each theme's own paper color.
    backgrounds = {"night": [16, 29, 32], "paper": [252, 250, 245]}
    sizes = set()
    with tempfile.TemporaryDirectory() as directory:
        for theme, rgb in backgrounds.items():
            page.locator("#theme-summary").click()
            page.locator(f'button[data-theme="{theme}"]').click()
            assert page.evaluate("document.documentElement.dataset.theme") == theme
            with page.expect_download() as download:
                page.locator("#export-image").click()
            assert download.value.suggested_filename == "brontide-pulse-2017-09.png"
            output = Path(directory) / f"pulse-{theme}.png"
            download.value.save_as(output)
            info = png_info(page, output)
            assert info["pixel"] == rgb, (theme, info)
            sizes.add((info["width"], info["height"]))
        # Medium and phone widths: the headline readouts are not covered, and a phone
        # export has the same layout as a desktop one.
        for width, height in ((768, 900), (390, 844)):
            context = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce")
            small = context.new_page()
            small.on("pageerror", lambda exc: errors.append(str(exc)))
            open_pulse(small, url, "2017-09")
            hits = small.evaluate("""() => {
                const out = [];
                for (const el of document.querySelectorAll('#pulse-chart text')) {
                    if (!/across 64 fields|including 1,024|inside fields|HOW FAR/.test(el.textContent)) continue;
                    const b = el.getBoundingClientRect();
                    for (const fx of [.1, .5, .9]) {
                        const hit = document.elementFromPoint(b.left + b.width * fx, b.top + b.height / 2);
                        out.push([el.textContent, Boolean(hit && hit.closest('#pulse-chart'))]);
                    }
                    if (b.right > innerWidth - 8) out.push([el.textContent + ' (overflows)', false]);
                }
                return out;
            }""")
            assert len(hits) >= 12 and all(ok for _, ok in hits), (width, [h for h in hits if not h[1]])
            if width == 390:
                with small.expect_download() as download:
                    small.locator("#export-image").click()
                output = Path(directory) / "pulse-phone.png"
                download.value.save_as(output)
                sizes.add(tuple(png_info(small, output)[k] for k in ("width", "height")))
            else:
                # With a field open, the inspector and the visible pulse panels do not overlap.
                small.locator("#pulse-movers button").first.click()
                small.wait_for_timeout(150)
                overlap = small.evaluate("""() => {
                    const ins = document.getElementById('inspector').getBoundingClientRect();
                    return ['pulse-chart', 'pulse-compass', 'pulse-movers'].filter((id) => {
                        const b = document.getElementById(id).getBoundingClientRect();
                        return b.width > 0 && b.right > ins.left && b.left < ins.right && b.bottom > ins.top && b.top < ins.bottom;
                    });
                }""")
                assert not overlap, overlap
            context.close()
    assert len(sizes) == 1, sizes
    assert not errors, errors


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
        check_currents(page, browser, args.url, data, errors)
        check_pulse(page, browser, args.url, errors)
        assert not errors, errors
        browser.close()
    print("PASS: exact window counts and shares, annual sample filtering, comparison math, "
          "microregion statistics, representative links, shared-view restoration, "
          "malformed-URL recovery, exact trailing-year composition, raw hover values, "
          "keyboard/touch access, theme persistence, themed PNG export, 2D fallback, and pulse "
          "readouts, slider value, movers, click-then-keyboard, pin and URL sync, medium and "
          "phone layout, fixed-layout pulse export, dock and shared-link restoration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
