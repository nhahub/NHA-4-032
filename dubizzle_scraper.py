"""
Dubizzle Egypt — Bulk Property Scraper
=======================================
Handles 100 k+ URLs with:
  • Async I/O  (aiohttp, configurable concurrency)
  • Auto-resume  (CSV checkpoint — restart any time after interruption)
  • Incremental Excel writing  (saves every 500 rows; safe to Ctrl+C)
  • Rate-limit detection  (backs off automatically on 429 / 403)

SETUP (one-time):
    pip install aiohttp beautifulsoup4 openpyxl lxml

RUN:
    python dubizzle_scraper.py                   # full run  (~107 k URLs)
    python dubizzle_scraper.py --limit 500       # quick test
    python dubizzle_scraper.py --workers 15      # faster (if good connection)
    python dubizzle_scraper.py --resume          # continue after interruption

FILES:
    Input      : URLs.xlsx           (same folder as this script)
    Output     : dubizzle_listings.xlsx
    Checkpoint : .checkpoint.csv     (auto-created; delete to restart from scratch)

EXPECTED RUNTIME (rough guide):
    107,000 URLs @ 10 workers  ≈  4–6 hours
    107,000 URLs @ 20 workers  ≈  2–3 hours
"""

import asyncio, aiohttp, argparse, csv, json, os, re, sys, time
from pathlib import Path
from bs4 import BeautifulSoup
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE      = "URLs.xlsx"
OUTPUT_FILE     = "dubizzle_listings.xlsx"
CHECKPOINT_FILE = ".checkpoint.csv"
DEFAULT_WORKERS = 40
BATCH_SIZE      = 1000          # write to Excel every N rows
TIMEOUT_SEC     = 12
MAX_RETRIES     = 2

FIELDS = [
    "URL", "Property Type", "Built-Up Area (m²)", "Price (EGP)",
    "Ownership", "Bedrooms", "Bathrooms", "Furnished",
    "Payment Option", "Down Payment", "Completion Status",
    "Compound", "District", "City", "Governorate",
    "Status",
]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": "https://www.dubizzle.com.eg/",
}

# ── HTML parser ───────────────────────────────────────────────────────────────

def clean(text) -> str | None:
    return " ".join(str(text).split()) if text else None

def after_label(label: str, text: str) -> str | None:
    m = re.search(rf"(?i){re.escape(label)}\s*\n\s*(.+)", text)
    return clean(m.group(1)) if m else None

def parse_html(html: str, url: str) -> dict:
    row = {f: None for f in FIELDS}
    row["URL"]    = url
    row["Status"] = "OK"

    soup = BeautifulSoup(html, "lxml")
    body = soup.get_text("\n", strip=True)

    if len(body) < 300 or "Access Denied" in body or "Just a moment" in body:
        row["Status"] = "BLOCKED"
        return row

    # ── Price ────────────────────────────────────────────────────────────────
    for tag in soup.find_all(string=re.compile(r"EGP\s*[\d,]+")):
        m = re.search(r"EGP\s*([\d,]+)", str(tag))
        if m:
            row["Price (EGP)"] = m.group(1).replace(",", "")
            break

    # ── Location: ONLY from breadcrumb links (/properties/ or /realestate/ hrefs)
    # This avoids the site navigation menu polluting compound/district fields.
    loc = []
    for a in soup.find_all("a", href=re.compile(r"/(properties|realestate)/", re.I)):
        txt = a.get_text(strip=True)
        m = re.search(r"for (?:Sale|Rent) in (.+)", txt, re.I)
        if m:
            val = m.group(1).strip()
            if val and val not in loc:
                loc.append(val)

    # Dubizzle breadcrumb order: Governorate → City → District → Compound
    # But many listings skip District and go straight to Compound.
    # Heuristic: if the last crumb contains "Compound" or ends with known
    # compound keywords, treat it as Compound; otherwise it may be a district.
    if len(loc) >= 1:
        row["Governorate"] = loc[0]
    if len(loc) >= 2:
        row["City"] = loc[1]
    if len(loc) == 3:
        # Could be district-only OR compound-only — put in both so nothing is lost
        last = loc[2]
        if re.search(r"compound|residence|heights|park|hills|city|view|square|gardens|"
                     r"mall|tower|village|villas|estate|lake|bay|coast|gate|zone",
                     last, re.I):
            row["Compound"] = last
        else:
            row["District"] = last
    if len(loc) >= 4:
        row["District"] = loc[2]
        row["Compound"] = loc[3]

    # ── JSON-LD (may give cleaner city/region data) ───────────────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            addr = data.get("address", {}) if isinstance(data, dict) else {}
            if isinstance(addr, dict):
                row["City"]        = row["City"]        or clean(addr.get("addressLocality"))
                row["Governorate"] = row["Governorate"] or clean(addr.get("addressRegion"))
        except Exception:
            pass

    # ── Subtitle line below ad title: "CompoundName, DistrictName"
    # Only run if we still don't have a compound, and only match short clean strings.
    if not row["Compound"]:
        # The subtitle sits near the top of the page, before heavy nav/footer content.
        # Limit search to first 8000 chars of raw HTML to avoid footer noise.
        top_soup = BeautifulSoup(html[:8000], "lxml")
        for tag in top_soup.find_all(["p", "span", "a"]):
            txt = tag.get_text(strip=True)
            # Must be short, no nav keywords, and mention a known location pattern
            if (8 < len(txt) < 80
                    and not re.search(r"sale|rent|search|login|signup|motor|fashion|"
                                      r"electronics|pets|jobs|service|for sale|for rent",
                                      txt, re.I)
                    and re.search(r"compound|settlement|district|maadi|nasr|zamalek|"
                                  r"heliopolis|october|zayed|cairo|giza|rehab|madinaty",
                                  txt, re.I)):
                parts = [p.strip() for p in txt.split(",")]
                if parts[0] and len(parts[0]) > 3:
                    row["Compound"] = parts[0]
                    if len(parts) >= 2 and not row["District"] and len(parts[1]) > 3:
                        row["District"] = parts[1]
                    break

    # ── Structured detail panel ───────────────────────────────────────────────
    # Restrict to panel section only (before the Description heading) so that
    # agent description prose cannot pollute structured fields like Down Payment.
    desc_pos = re.search(r"\nDescription\s*\n", body, re.I)
    panel    = body[: desc_pos.start()] if desc_pos else body[:3000]

    row["Property Type"]     = after_label("Type",              panel)
    row["Ownership"]         = after_label("Ownership",         panel)
    row["Furnished"]         = after_label("Furnished",         panel)
    row["Payment Option"]    = after_label("Payment Option",    panel)
    row["Completion Status"] = after_label("Completion status", panel)

    # Down Payment: valid formats are "EGP X,XXX,XXX", "X%", or "0".
    # Also sanity-check: DP must be <= price to reject description noise.
    dp_m = re.search(
        r"Down Payment\s*\n\s*((?:EGP\s*)?[\d,]+(?:\s*EGP)?|[\d.]+\s*%|0)",
        panel, re.I
    )
    if dp_m:
        dp_val = clean(dp_m.group(1))
        try:
            price  = int(row.get("Price (EGP)") or 0)
            dp_num = int(re.sub(r"[^\d]", "", dp_val))
            if price == 0 or dp_num <= price:
                row["Down Payment"] = dp_val
        except Exception:
            row["Down Payment"] = dp_val

    m = re.search(r"Area\s*\(m.?\)\s*\n?\s*([\d,]+)", panel, re.I)
    if m:
        row["Built-Up Area (m²)"] = m.group(1).replace(",", "")

    m = re.search(r"Bedrooms?\s*\n?\s*(\d+)", panel, re.I)
    if m:
        row["Bedrooms"] = int(m.group(1))

    m = re.search(r"Bathrooms?\s*\n?\s*(\d+)", panel, re.I)
    if m:
        row["Bathrooms"] = int(m.group(1))

    return row


# ── Async fetch with retry + backoff ─────────────────────────────────────────

async def fetch(session: aiohttp.ClientSession, url: str,
                sem: asyncio.Semaphore, backoff: list) -> dict:
    async with sem:
        if backoff[0] > 0:
            await asyncio.sleep(backoff[0])

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.get(
                    url, headers=BROWSER_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                    allow_redirects=True,
                ) as resp:
                    if resp.status == 429:
                        wait = 30 + attempt * 30
                        backoff[0] = max(backoff[0], wait / 2)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status in (403, 503) and attempt < MAX_RETRIES:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    backoff[0] = max(0.0, backoff[0] - 0.05)
                    html = await resp.text(errors="replace")
                    row  = parse_html(html, url)
                    if resp.status != 200:
                        row["Status"] = f"HTTP {resp.status}"
                    return row
            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(3)
                    continue
                return {f: None for f in FIELDS} | {"URL": url, "Status": "TIMEOUT"}
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(3)
                    continue
                return {f: None for f in FIELDS} | {"URL": url, "Status": f"ERR:{exc}"}

        return {f: None for f in FIELDS} | {"URL": url, "Status": "FAILED"}


# ── Excel helpers ─────────────────────────────────────────────────────────────

COL_WIDTHS = {
    "URL": 55, "Property Type": 16, "Built-Up Area (m²)": 16, "Price (EGP)": 16,
    "Ownership": 14, "Bedrooms": 11, "Bathrooms": 11, "Furnished": 11,
    "Payment Option": 16, "Down Payment": 16, "Completion Status": 18,
    "Compound": 32, "District": 24, "City": 20, "Governorate": 18, "Status": 14,
}
_thin      = Side(style="thin", color="CCCCCC")
_BORDER    = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)
_ALT_FILL  = PatternFill("solid", fgColor="EDF2FF")
_ERR_FILL  = PatternFill("solid", fgColor="FFE0E0")

def init_workbook() -> tuple:
    wb = Workbook()
    ws = wb.active
    ws.title = "Property Listings"
    hdr_fill = PatternFill("solid", fgColor="1F3864")
    hdr_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    for col, field in enumerate(FIELDS, 1):
        c = ws.cell(row=1, column=col, value=field)
        c.fill = hdr_fill; c.font = hdr_font; c.border = _BORDER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[c.column_letter].width = COL_WIDTHS.get(field, 16)
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "B2"
    return wb, ws

def write_batch(ws, rows: list[dict], start_row: int):
    for r_idx, row in enumerate(rows, start_row):
        bad  = str(row.get("Status", "")).upper() not in ("OK", "")
        fill = _ERR_FILL if bad else (_ALT_FILL if r_idx % 2 == 0 else None)
        for c_idx, field in enumerate(FIELDS, 1):
            c = ws.cell(row=r_idx, column=c_idx, value=row.get(field))
            if fill:
                c.fill = fill
            c.alignment = Alignment(
                horizontal="left" if c_idx == 1 else "center",
                vertical="center"
            )
            c.border = _BORDER


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_done_urls() -> set:
    done = set()
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("URL"):
                    done.add(row["URL"])
    return done

def open_checkpoint() -> tuple:
    exists = Path(CHECKPOINT_FILE).exists()
    fh = open(CHECKPOINT_FILE, "a", newline="", encoding="utf-8")
    w  = csv.DictWriter(fh, fieldnames=FIELDS)
    if not exists:
        w.writeheader()
    return fh, w


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args):
    if not Path(INPUT_FILE).exists():
        sys.exit(f"ERROR: '{INPUT_FILE}' not found in {os.getcwd()}")

    print("Loading URLs …", flush=True)
    wb_in = load_workbook(INPUT_FILE, read_only=True)
    ws_in = wb_in.active
    all_urls = [
        str(r[0]).strip()
        for r in ws_in.iter_rows(min_row=2, values_only=True)
        if r[0] and str(r[0]).startswith("http")
    ]
    wb_in.close()

    if args.limit:
        all_urls = all_urls[:args.limit]

    # Resume
    done_urls = load_done_urls() if args.resume else set()
    todo = [u for u in all_urls if u not in done_urls]

    print(f"  Total  : {len(all_urls):>10,}")
    print(f"  Done   : {len(done_urls):>10,}")
    print(f"  To do  : {len(todo):>10,}")
    print(f"  Workers: {args.workers}")
    print()

    if not todo:
        print("All URLs already processed. Delete .checkpoint.csv to restart.")
        return

    # Build output workbook (pre-load existing checkpoint rows when resuming)
    wb_out, ws_out = init_workbook()
    next_row = 2
    if args.resume and Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, newline="", encoding="utf-8") as f:
            prev = list(csv.DictReader(f))
        write_batch(ws_out, prev, start_row=2)
        next_row += len(prev)
        wb_out.save(OUTPUT_FILE)

    cp_fh, cp_writer = open_checkpoint()

    sem     = asyncio.Semaphore(args.workers)
    backoff = [0.0]
    conn    = aiohttp.TCPConnector(ssl=False, limit=args.workers + 5, ttl_dns_cache=300)

    t0        = time.time()
    done      = 0
    ok_count  = 0
    err_count = 0
    batch: list[dict] = []

    print(f"Scraping {len(todo):,} URLs …")
    print(f"(Excel saved every {BATCH_SIZE} rows — safe to Ctrl+C and resume)\n")

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [fetch(session, u, sem, backoff) for u in todo]

        for coro in asyncio.as_completed(tasks):
            row = await coro
            done += 1
            (ok_count if row.get("Status") == "OK" else err_count).__class__   # no-op type check
            if row.get("Status") == "OK":
                ok_count += 1
            else:
                err_count += 1

            batch.append(row)
            cp_writer.writerow(row)
            cp_fh.flush()

            # Batch write to Excel
            if len(batch) >= BATCH_SIZE or done == len(todo):
                write_batch(ws_out, batch, start_row=next_row)
                next_row += len(batch)
                batch = []
                ws_out.auto_filter.ref = (
                    f"A1:{ws_out.cell(1, len(FIELDS)).column_letter}1"
                )
                wb_out.save(OUTPUT_FILE)

            # Progress line
            if done % 500 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed else 1
                eta_s   = (len(todo) - done) / rate
                print(
                    f"  {done:7,}/{len(todo):,}  "
                    f"OK:{ok_count:,}  Err:{err_count:,}  "
                    f"{rate:.1f} req/s  "
                    f"ETA {int(eta_s//3600)}h {int(eta_s%3600//60):02d}m",
                    flush=True,
                )

    cp_fh.close()
    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"  Completed in {elapsed/3600:.1f} h ({elapsed/60:.0f} min)")
    print(f"  OK      : {ok_count:,}")
    print(f"  Errors  : {err_count:,}")
    print(f"  Saved → {OUTPUT_FILE}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit",   type=int, default=None, help="Process first N URLs")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers (default 10)")
    p.add_argument("--resume",  action="store_true", help="Skip already-scraped URLs (reads .checkpoint.csv)")
    asyncio.run(main(p.parse_args()))
