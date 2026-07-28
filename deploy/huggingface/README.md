---
title: NIT Delhi Campus Bot
emoji: 🎓
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Student help chatbot that refuses to guess — 94% of answers never touch an LLM
---

# NIT Delhi Campus Bot

A student help chatbot for NIT Delhi: fees, labs, faculty, syllabus, admission
checklists and live campus shop status.

**The design decision worth knowing:** faculty records in the source data are
near-identical sentences with one name swapped — measured pairwise similarity
between *different people* is **0.975**. An embedding model cannot separate them,
so semantic search returns the wrong professor. The same holds for room numbers
and rupee amounts: their meaning is exact, not semantic.

So retrieval is classified by behaviour, not topic. Fees, labs, faculty,
admission and shop status are MongoDB lookups rendered by template — **94% of
queries never reach a language model**, and a figure cannot be reworded or
invented. Only syllabus prose uses Groq, for phrasing.

Every answer carries a badge naming the path that produced it.

## Try

| Ask | What it shows |
|---|---|
| `btech 3rd sem josaa fee below 1 lakh day scholar` | exact figure + source PDF |
| `AIRIL lab capacity` | room-level lookup |
| `dr haleem` | typo-tolerant name match |
| `4th semester fee` | **refuses** — that semester isn't published |
| `total fee for all 4 years` | **refuses** — it never does arithmetic |
| `which lab has a quantum computer` | **refuses** — no such equipment |

The refusals are the interesting part.

## Notes

- First request after idle may take ~30s while the Space wakes.
- Running on a reduced sample corpus. The full corpus is six departments.

**Source:** https://github.com/malothritesh07/nitd-campus-bot
