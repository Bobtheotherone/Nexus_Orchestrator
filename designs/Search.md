

# SNAP Manuals Search Engine for the State of Alaska

## Problem Statement

The State of Alaska's Supplemental Nutrition Assistance Program (SNAP) maintains a collection of policy manuals that caseworkers, administrators, and the public need to reference. Currently, finding specific policy guidance requires manually browsing through lengthy documents. There is no unified, full-text search capability across all SNAP manuals, leading to slow lookups, inconsistent policy application, and frustration for users who need quick answers.

## Context & Background

- **SNAP (formerly Food Stamps)** is administered in Alaska by the Division of Public Assistance (DPA) under the Department of Health.
- Alaska's SNAP manuals cover eligibility, income/resource limits, work requirements, certification periods, and special provisions (e.g., rural/tribal considerations unique to Alaska).
- Manuals are typically published as PDFs or HTML pages on state government websites, with periodic revisions tracked by section and effective date.
- No existing open-source tool specifically targets Alaska SNAP manual search. General-purpose search engines (Google) index some pages but lack structure-aware, section-level search.
- Key constraints:
  - Manuals may be in PDF, HTML, or Word format.
  - Content changes with each policy update cycle — the system must support re-ingestion.
  - Users range from caseworkers with moderate technical skill to members of the public.
  - The solution should be deployable on modest infrastructure (single server or small cloud instance).

## Proposed Solution

Build a self-contained search application with three layers:

1. **Ingestion Pipeline** — Crawl/download Alaska SNAP manuals, extract text, split into searchable sections, and store in a search index.
2. **Search Engine** — A full-text search index (using Tantivy or SQLite FTS5) that supports ranked keyword search, phrase matching, and filtering by manual/chapter/revision date.
3. **Web Interface** — A lightweight web UI where users type queries and receive ranked results with highlighted snippets and direct links to the source section.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  AK DPA     │────▶│  Ingestion   │────▶│  Search      │────▶│  Web UI   │
│  Manuals    │     │  Pipeline    │     │  Index       │     │  (FastAPI)│
│  (PDF/HTML) │     │              │     │  (FTS5)      │     │           │
└─────────────┘     └──────────────┘     └──────────────┘     └───────────┘
```

## Implementation Details

### 1. Ingestion Pipeline

```
src/
  snap_search/
    __init__.py
    ingest/
      __init__.py
      crawler.py        # Download manuals from AK DPA website
      extractor.py       # PDF/HTML → plain text
      chunker.py         # Split text into sections
      loader.py          # Insert sections into search index
    search/
      __init__.py
      index.py           # FTS5 index management
      query.py           # Query parsing and execution
    web/
      __init__.py
      app.py             # FastAPI application
      templates/         # Jinja2 HTML templates
    config.py
    models.py
```

**Crawler (`crawler.py`)**

```python
import httpx
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup

ALASKA_DPA_BASE = "https://dpaweb.hss.state.ak.us/manuals/"

class ManualCrawler:
    def __init__(self, base_url: str = ALASKA_DPA_BASE, output_dir: Path = Path("data/raw")):
        self.base_url = base_url
        self.output_dir = output_dir
        self.client = httpx.Client(follow_redirects=True, timeout=30.0)

    def discover_manuals(self) -> list[str]:
        """Scrape the manual index page for PDF/HTML links."""
        resp = self.client.get(self.base_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if href.endswith((".pdf", ".htm", ".html")):
                links.append(urljoin(self.base_url, href))
        return links

    def download(self, url: str) -> Path:
        """Download a single manual file to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        dest = self.output_dir / filename
        if dest.exists():
            return dest
        resp = self.client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest

    def crawl_all(self) -> list[Path]:
        urls = self.discover_manuals()
        return [self.download(url) for url in urls]
```

**Extractor (`extractor.py`)**

```python
import pdfplumber
from bs4 import BeautifulSoup
from pathlib import Path

def extract_text(filepath: Path) -> str:
    """Extract plain text from PDF or HTML files."""
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(filepath)
    elif suffix in (".html", ".htm"):
        return _extract_html(filepath)
    raise ValueError(f"Unsupported file type: {suffix}")

def _extract_pdf(filepath: Path) -> str:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)

def _extract_html(filepath: Path) -> str:
    soup = BeautifulSoup(filepath.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)
```

**Chunker (`chunker.py`)**

```python
import re
from dataclasses import dataclass

@dataclass
class Section:
    manual_name: str
    section_id: str
    title: str
    body: str
    source_file: str

# Matches patterns like "Section 602-1" or "4.2.3 Eligibility"
SECTION_PATTERN = re.compile(
    r"^((?:Section\s+)?[\d][\d.\-]*[\d]?)\s+(.+)$", re.MULTILINE
)

def chunk_into_sections(text: str, manual_name: str, source_file: str) -> list[Section]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        # Treat entire document as one section
        return [Section(manual_name, "1", manual_name, text.strip(), source_file)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append(Section(
            manual_name=manual_name,
            section_id=match.group(1).strip(),
            title=match.group(2).strip(),
            body=body,
            source_file=source_file,
        ))
    return sections
```

### 2. Search Index (SQLite FTS5)

```python
# search/index.py
import sqlite3
from pathlib import Path
from snap_search.ingest.chunker import Section

DB_PATH = Path("data/snap_search.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def create_index(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            manual_name TEXT NOT NULL,
            section_id TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            source_file TEXT NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
            title,
            body,
            content='sections',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS sections_ai AFTER INSERT ON sections BEGIN
            INSERT INTO sections_fts(rowid, title, body)
            VALUES (new.id, new.title, new.body);
        END;

        CREATE TRIGGER IF NOT EXISTS sections_ad AFTER DELETE ON sections BEGIN
            INSERT INTO sections_fts(sections_fts, rowid, title, body)
            VALUES ('delete', old.id, old.title, old.body);
        END;
    """)

def insert_section(conn: sqlite3.Connection, section: Section):
    conn.execute(
        "INSERT INTO sections (manual_name, section_id, title, body, source_file) "
        "VALUES (?, ?, ?, ?, ?)",
        (section.manual_name, section.section_id, section.title,
         section.body, section.source_file),
    )
```

### 3. Query Engine

```python
# search/query.py
from dataclasses import dataclass
import sqlite3

@dataclass
class SearchResult:
    manual_name: str
    section_id: str
    title: str
    snippet: str
    source_file: str
    rank: float

def search(conn: sqlite3.Connection, query: str, manual_filter: str | None = None,
           limit: int = 20) -> list[SearchResult]:
    sql = """
        SELECT s.manual_name, s.section_id, s.title,
               snippet(sections_fts, 1, '<mark>', '</mark>', '…', 48) AS snippet,
               s.source_file,
               rank
        FROM sections_fts
        JOIN sections s ON s.id = sections_fts.rowid
        WHERE sections_fts MATCH ?
    """
    params: list = [query]

    if manual_filter:
        sql += " AND s.manual_name = ?"
        params.append(manual_filter)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    cursor = conn.execute(sql, params)
    return [
        SearchResult(
            manual_name=row[0], section_id=row[1], title=row[2],
            snippet=row[3], source_file=row[4], rank=row[5],
        )
        for row in cursor.fetchall()
    ]

def list_manuals(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute("SELECT DISTINCT manual_name FROM sections ORDER BY manual_name")
    return [row[0] for row in cursor.fetchall()]
```

### 4. Web Interface

```python
# web/app.py
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from snap_search.search.index import get_connection
from snap_search.search.query import search, list_manuals

app = FastAPI(title="Alaska SNAP Manual Search")
templates = Jinja2Templates(directory="src/snap_search/web/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    conn = get_connection()
    manuals = list_manuals(conn)
    return templates.TemplateResponse("index.html", {
        "request": request, "manuals": manuals, "results": None, "query": "",
    })

@app.get("/search", response_class=HTMLResponse)
async def search_manuals(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    manual: str | None = Query(None),
):
    conn = get_connection()
    manuals = list_manuals(conn)
    results = search(conn, q, manual_filter=manual or None)
    return templates.TemplateResponse("index.html", {
        "request": request, "manuals": manuals,
        "results": results, "query": q,
    })
```

### 5. CLI Entrypoint

```python
# cli.py
import click
from pathlib import Path
from snap_search.ingest.crawler import ManualCrawler
from snap_search.ingest.extractor import extract_text
from snap_search.ingest.chunker import chunk_into_sections
from snap_search.search.index import get_connection, create_index, insert_section

@click.group()
def cli():
    """Alaska SNAP Manual Search Tool"""

@cli.command()
@click.option("--source-dir", type=Path, default=Path("data/raw"))
def ingest(source_dir: Path):
    """Crawl, extract, and index all SNAP manuals."""
    crawler = ManualCrawler(output_dir=source_dir)
    click.echo("Crawling manuals...")
    files = crawler.crawl_all()
    click.echo(f"Downloaded {len(files)} files.")

    conn = get_connection()
    create_index(conn)

    total_sections = 0
    for f in files:
        text = extract_text(f)
        manual_name = f.stem.replace("_", " ").title()
        sections = chunk_into_sections(text, manual_name, str(f))
        for section in sections:
            insert_section(conn, section)
        total_sections += len(sections)

    conn.commit()
    click.echo(f"Indexed {total_sections} sections from {len(files)} manuals.")

@cli.command()
@click.argument("query")
def find(query: str):
    """Search manuals from the command line."""
    conn = get_connection()
    from snap_search.search.query import search
    results = search(conn, query)
    for r in results:
        click.echo(f"[{r.manual_name} § {r.section_id}] {r.title}")
        click.echo(f"  {r.snippet}\n")

@cli.command()
def serve():
    """Start the web search interface."""
    import uvicorn
    uvicorn.run("snap_search.web.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    cli()
```

## API / Interface Design

### CLI Commands

| Command | Description |
|---|---|
| `python -m snap_search ingest` | Crawl and index all SNAP manuals |
| `python -m snap_search find "income eligibility"` | Search from the terminal |
| `python -m snap_search serve` | Launch the web UI on port 8000 |

### HTTP Endpoints

| Method | Path | Parameters | Description |
|---|---|---|---|
| `GET` | `/` | — | Home page with search box |
| `GET` | `/search` | `q` (required), `manual` (optional) | Full-text search, returns HTML |
| `GET` | `/api/search` | `q`, `manual`, `limit` | JSON API (future) |
| `GET` | `/api/manuals` | — | List indexed manuals (future) |

### Query Syntax (FTS5)

- Simple keywords: `income eligibility`
- Phrase match: `"gross income"`
- Boolean: `income AND NOT self-employment`
- Prefix: `elig*`
- Column filter: `title:certification`

## Data Model Changes

Single SQLite database (`data/snap_search.db`):

```sql
-- Core table
sections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    manual_name   TEXT NOT NULL,      -- e.g. "Alaska SNAP Eligibility Manual"
    section_id    TEXT NOT NULL,      -- e.g. "602-1"
    title         TEXT NOT NULL,      -- section heading
    body          TEXT NOT NULL,      -- full section text
    source_file   TEXT NOT NULL,      -- path to original file
    ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 virtual table (auto-synced via triggers)
sections_fts (title, body);

-- Metadata tracking table for re-ingestion
ingestion_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url    TEXT NOT NULL,
    file_hash     TEXT NOT NULL,      -- SHA-256 of downloaded file
    ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The `ingestion_log` table allows the pipeline to skip files that haven't changed between runs.

## Security Considerations

| Concern | Mitigation |
|---|---|
| **SQL Injection** | All queries use parameterized statements. FTS5 `MATCH` input is parameterized, not interpolated. |
| **XSS in snippets** | FTS5 `snippet()` output uses `<mark>` tags. All other template output is auto-escaped by Jinja2. Snippet markers are whitelisted in the template with `\|safe` only for the `<mark>` tags. |
| **Crawling scope** | Crawler is restricted to the configured `base_url` domain. URL join prevents directory traversal. |
| **Denial of Service** | Query length capped at 200 characters. Result limit capped at 100. Rate limiting via reverse proxy (nginx) in production. |
| **Data sensitivity** | SNAP manuals are public documents. No PII is stored. No authentication required for read-only search. |
| **Dependency supply chain** | Pin all dependencies with hashes in `requirements.txt`. Minimal dependency set (FastAPI, pdfplumber, httpx, BeautifulSoup4). |

## Testing Strategy

### Unit Tests

```python
# tests/test_chunker.py
def test_chunk_splits_on_section_headers():
    text = "Section 602-1 Eligibility\nSome text.\nSection 602-2 Income\nMore text."
    sections = chunk_into_sections(text, "Test Manual", "test.pdf")
    assert len(sections) == 2
    assert sections[0].section_id == "602-1"
    assert sections[1].title == "Income"

def test_chunk_handles_no_headers():
    text = "Just a plain paragraph of text with no section markers."
    sections = chunk_into_sections(text, "Test", "test.pdf")
    assert len(sections) == 1

# tests/test_query.py
def test_search_returns_ranked_results(indexed_db):
    results = search(indexed_db, "income eligibility")
    assert len(results) > 0
    assert all(r.rank <= 0 for r in results)  # FTS5 rank is negative, closer to 0 = better

def test_search_manual_filter(indexed_db):
    results = search(indexed_db, "income", manual_filter="Eligibility Manual")
    assert all(r.manual_name == "Eligibility Manual" for r in results)

def test_search_empty_query_rejected():
    # FastAPI validation rejects empty q parameter
    ...
```

### Integration Tests

- **Ingest → Search round-trip**: Feed a known test PDF through the full pipeline, then verify expected sections are findable.
- **Web endpoint tests**: Use FastAPI `TestClient` to assert HTTP 200 on `/` and `/search?q=test`, check HTML contains result markup.
- **Re-ingestion idempotency**: Run ingest twice on the same files; verify no duplicate sections.

### Edge Cases

- PDFs with scanned images (no extractable text) — log a warning, skip gracefully.
- Manuals with non-standard section numbering — fallback to paragraph-level chunking.
- Very large sections (>10,000 words) — split at paragraph boundaries to keep snippets meaningful.
- Unicode characters in Alaska Native place names — ensure `unicode61` tokenizer handles correctly.

## Rollout Plan

### Phase 1 — MVP (Weeks 1–2)
- Manual file ingestion (drop PDFs into `data/raw/`, run `ingest`).
- CLI search (`find` command).
- SQLite FTS5 index.
- Basic unit and integration tests.

### Phase 2 — Web Interface (Weeks 3–4)
- FastAPI web UI with search box, results page, manual filter dropdown.
- Snippet highlighting.
- Deploy behind nginx on a single server or small cloud VM.

### Phase 3 — Automated Crawling (Weeks 5–6)
- Scheduled crawl of the Alaska DPA website (cron job or systemd timer).
- Change detection via file hashing — only re-index changed documents.
- Ingestion log for audit trail.

### Phase 4 — Enhancements (Future)
- JSON REST API for programmatic access.
- Search analytics (popular queries, zero-result queries).
- Optional semantic search via sentence embeddings for natural-language queries.
- Bookmarking / citation export for caseworkers.

### Rollback Strategy
- SQLite database can be deleted and fully rebuilt from source files at any time (`ingest` is idempotent after clearing the DB).
- Web application is stateless — revert to any prior commit and restart.
- No external service dependencies to coordinate.

## Open Questions

1. **Source URL stability** — Does the Alaska DPA website have a stable URL structure for manuals, or do paths change with redesigns? Need to verify and potentially maintain a manual URL registry.
2. **Manual formats** — Are all manuals available as text-based PDFs, or are some scanned images requiring OCR (Tesseract)?
3. **Update frequency** — How often are SNAP manuals revised? This determines crawl schedule and whether version diffing is worthwhile.
4. **Tribal provisions** — Alaska has unique tribal SNAP provisions. Are these in separate documents or embedded within the main manuals?
5. **Accessibility requirements** — If this tool will be publicly available, it must meet Section 508 / WCAG 2.1 AA standards. Need to confirm scope of audience.
6. **Hosting constraints** — Will this run on state infrastructure (with potential procurement/approval requirements) or on an independent server?
7. **Scope expansion** — Should this eventually cover other DPA program manuals (Medicaid, TANF, Adult Public Assistance), or remain SNAP-only?
