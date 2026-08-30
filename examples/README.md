# Examples

`backend/wiki/` starts empty on a fresh clone — pages are created as raw
documents are ingested, not shipped pre-populated (see `PRIVACY.md`). This
directory has a small sample document you can use to try the raw→wiki
ingestion pipeline immediately, without needing to supply your own real
document first.

## Try it

1. Copy the sample document into `backend/raw/`:

   ```bash
   cp examples/raw/sample-article.md backend/raw/
   ```

2. Start Localist:

   ```bash
   ./start_localist.sh
   ```

3. In the sidebar's expandable Files browser, select `sample-article.md`
   under Raw and click **Ingest to wiki**. This sends `raw_path` in the
   task context, which routes through `Planner`'s raw-file/ingest priority
   rule to `WikiAgent`, which reads the document and produces a structured
   wiki page.

4. Check `backend/wiki/` — a new page should now exist, following the OKF
   frontmatter convention documented in `backend/SCHEMA.md`.

There's no canned "expected output" page here — the resulting wiki page is
model-generated, so its exact content depends on the model you're running.
