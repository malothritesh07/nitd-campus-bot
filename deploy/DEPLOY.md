# Deploying

One service. FastAPI serves both the API and the widget, so there is no separate
frontend to host — splitting it onto Vercel would add CORS config, a second
deployment and an API base URL to manage, in exchange for nothing. Revisit that
only if the widget ever becomes a real SPA with a build step.

```
   Browser
      │
      ▼
 ┌──────────────────────┐
 │  one container       │
 │  FastAPI  + static/  │   serves  /  and  /api/*
 └──────────────────────┘
      │              │
      ▼              ▼
 MongoDB Atlas    Groq API
 (free tier)      (syllabus only)
```

---

## Do these first

**1. Rotate the secrets.** The Atlas password and Groq key have been shared in
plaintext. Deploying means pasting them into another dashboard — do that with
fresh values, not the old ones.

**2. Set `SERVER_PEPPER` to a real random string.** Once, before any shop codes
are issued. Changing it later invalidates every code already handed out.

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

**3. Open the Atlas network allowlist.** Atlas currently permits your laptop's
IP. A cloud host has different, rotating addresses, so connections will time out
with a confusing error. In Atlas → **Network Access** → add `0.0.0.0/0`.

This is the single most common first-deploy failure. The database is still
protected by username and password; the allowlist is a second layer that a
dynamic-IP host cannot satisfy.

---

## Choosing a host

The deciding constraint: PyTorch plus sentence-transformers holds roughly
300–500 MB of RAM, and the image is around 1.5 GB.

| Host | Free tier | Verdict |
|---|---|---|
| **Hugging Face Spaces** | Docker, generous RAM, no card | Best free fit; config is in this folder |
| **Railway** | trial credits, then ~$5/mo | Excellent DX; worth it once the college uses it |
| **Render** | 512 MB RAM | Risky — torch may OOM. Sleeps after ~15 min idle |
| **Fly.io** | small VMs, card required | Works, more configuration |

One thing already helps: `get_model()` is lazy, so the embedding model only
loads when a prose query arrives. The 94% of queries that never need it leave
idle memory low.

---

## Hugging Face Spaces

### Manual, once

1. **Create the Space** — huggingface.co/new-space → SDK **Docker** → name it
   `nitd-campus-bot`.

2. **Copy the build files to the Space repo root.** Spaces reads the metadata
   block at the top of `README.md`, so the Space needs its own:

   ```bash
   git clone https://huggingface.co/spaces/<you>/nitd-campus-bot hf-space
   cp -r <this-repo>/* hf-space/
   cp <this-repo>/deploy/huggingface/README.md   hf-space/README.md
   cp <this-repo>/deploy/huggingface/Dockerfile  hf-space/Dockerfile
   cd hf-space && git add -A && git commit -m "Initial" && git push
   ```

3. **Add secrets** — Space → Settings → *Variables and secrets*:

   | Name | Value |
   |---|---|
   | `MONGO_URI` | `mongodb+srv://…` (rotated) |
   | `DB_NAME` | `nitd_campus` |
   | `SERVER_PEPPER` | the random string from above |
   | `GROQ_API_KEY` | `gsk_…` (rotated) — optional |
   | `EMBED_OFFLINE` | `1` |
   | `LANGSMITH_TRACING` | `false` |

4. **Seed the database once**, from your laptop, pointed at the same cluster:

   ```bash
   python seed_config.py
   python sync.py
   python seed_shops.py        # prints the owner codes — save them
   ```

   The deployed app reads the same Atlas cluster, so this only needs doing once
   and never on the host.

5. **Wait for the build.** First one takes 5–10 minutes; the image is large
   because torch and the embedding model are baked in.

### Automatic afterwards

`.github/workflows/sync-to-huggingface.yml` mirrors every push to `main` into
the Space. Add three GitHub secrets and it runs itself:

`HF_TOKEN` (a write token from huggingface.co/settings/tokens), `HF_USERNAME`,
`HF_SPACE`.

---

## Railway

Simpler, but not free beyond the trial credits.

1. railway.app → **New Project** → *Deploy from GitHub repo*
2. It detects the root `Dockerfile` automatically
3. Add the same environment variables under **Variables**
4. Add `PORT=8000`, or let Railway inject `$PORT` and change the start command to
   `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Seed the database from your laptop as above

---

## After it is live

**Lock CORS.** `ALLOWED_ORIGINS` defaults to `*`, which is right for a standalone
demo. Once the widget is embedded on the college site, set it to that origin:

```
ALLOWED_ORIGINS=https://nitdelhi.ac.in,https://www.nitdelhi.ac.in
```

**Warn about the cold start.** Free tiers sleep. The first request after idle
takes 30–60 seconds, and someone who does not expect that assumes it is broken.
Put the warning next to the link.

**Update the README's Live Link section** with the URL.

**Embedding on the college page** needs no code change — the widget calls its own
origin, so it works as an iframe or by copying `static/index.html`'s widget
markup and pointing `API` at the deployed URL.

---

## Checking it worked

```bash
curl https://<your-space>.hf.space/api/rag/stats     # corpus counts
curl https://<your-space>.hf.space/api/ops           # cache + rate limits
```

If `/api/rag/stats` returns zeros, the app is running but the database was never
seeded. If it hangs or errors on connection, the Atlas allowlist is the cause.
