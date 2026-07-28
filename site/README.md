# Interactive handbook

A static, dependency-free website generated from this repository's `README.md` and `src/` tree.
Eleven HTML pages: an index that acts as the table of contents, plus one page per architectural layer.

Each concept page follows the same structure:

1. **The problem it solves** — why the layer exists at all
2. **The production code** — the real files from this repo, annotated
3. **A toy example** — a runnable simulation of the concept, in the browser
4. **Failure modes** — a table of what breaks without it

## Opening it

```bash
# Straight from disk — everything works over file://
xdg-open site/index.html      # or: open site/index.html

# Or serve it, if you prefer
python3 -m http.server -d site 8080
```

No build step, no bundler, no network access at runtime. Every simulation runs locally in JavaScript.

## Contents

| Page | Layer | Interactive pieces |
|---|---|---|
| `index.html` | Overview + repo map | Request-lifecycle stepper |
| `01-modular-codebase.html` | Dependencies, settings, containers | Settings resolver · compose topology explorer |
| `02-data-persistence.html` | SQLModel entities, DTOs | DTO leak inspector · password validator · state reducer |
| `03-security.html` | Rate limits, sanitisation, JWT, trimming | Rate limiter · sanitiser · JWT workbench · context trimmer |
| `04-service-layer.html` | Pooling, retries, fallback | Connection-pool simulator · retry/fallback simulator |
| `05-agent-architecture.html` | LangGraph, memory, tools | Graph stepper · prompt renderer · memory demo |
| `06-api-gateway.html` | Auth, DI, SSE | Dependency resolver · SSE stream player |
| `07-observability.html` | Metrics, middleware, logs | Histogram explorer · middleware onion · log formatter |
| `08-devops.html` | Docker, Makefile, CI | Layer-cache simulator · entrypoint check · CI walkthrough |
| `09-evaluation.html` | LLM-as-a-judge | Judge playground · eval cost planner · regression trend |
| `10-stress-testing.html` | Load testing | Load simulator · PromQL explorer · capacity planner |

## Files

```
site/
├── index.html              entry point / table of contents
├── 01-…10-*.html           one page per layer
└── assets/
    ├── styles.css          light + dark theme, responsive
    └── site.js             nav, theme toggle, syntax highlighting, shared widget helpers
```

Navigation, the sidebar and the prev/next pager are generated from the `PAGES` array at the top of
`assets/site.js` — add a page there and it appears everywhere.
