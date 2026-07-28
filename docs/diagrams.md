# Diagrams (source)

Kept here so the README blocks and this file stay in sync.

## 1. System architecture

```
                                  User
                                    │
                                    ▼
                          Chat Widget  (vanilla JS)
                          8 category chips · no build step
                                    │
                                    ▼
                            FastAPI   /api/chat
                                    │
                                    ▼
                            Query Processor
                 typo tolerance · slot carry-over · pronoun resolution
                                    │
                                    ▼
                            Category Router
                       chip selection, or inferred from text
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
      EXACT PATHS                                    SEMANTIC PATH
   no vectors · no LLM                             syllabus · about
            │                                               │
            ▼                                               ▼
      Metadata Filter                               Scoped Prefilter
  program · semester · route                       programme + semester
  category · residence · room no                  applied BEFORE ranking
            │                                               │
            │                                               ▼
            │                                        Hybrid Retrieval
            │                                  ┌───────────────────────┐
            │                                  │  BM25      sparse     │
            │                                  │  Cosine    dense 384d │
            │                                  │  RRF       k = 60     │
            │                                  └───────────────────────┘
            │                                               │
            │                                               ▼
            │                                       Parent Expansion
            │                                        module → course
            │                                               │
            ▼                                               ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                       MongoDB Atlas                           │
    │  fee_rows · chunks · fee_docs · admission_docs · shop_status  │
    │  config · ingest_log · shop_audit · feedback · link_only      │
    └───────────────────────────────────────────────────────────────┘
            │                                               │
            ▼                                               ▼
     Template Renderer                              Context Builder
   numbers copied verbatim                        capped at 2 500 chars
   nothing is generated                                     │
            │                                               ▼
            │                                     Groq · Llama 3.1 8B
            │                                        phrasing only
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                   Answer  +  Method Badge  +  Source Link
                                    │
                                    ▼
                                  User

   config collection ─────────────► thresholds · prompts · lexicon · UI
   (re-read every 60s, no redeploy)

   LangSmith ◄───────────────────── traces retrieval AND generation
```

## 2. Ingestion / sync

```
        Source files
   JSON · JSONL · scraped PDFs
              │
              ▼
         corpus.py
   build desired state, no DB writes
   content_hash on every record
              │
              ▼
          sync.py
   diff desired  vs  what is live
              │
   ┌──────────┼──────────┬───────────┐
   ▼          ▼          ▼           ▼
  NEW      CHANGED   UNCHANGED    MISSING
   │          │          │           │
   ▼          ▼          ▼           ▼
 embed     re-embed    touch      ARCHIVE
 locally   locally   last_verified  status="archived"
   │          │       no re-embed   not deleted
   └──────────┴──────────┴───────────┘
              │
              ▼
        MongoDB Atlas
              │
              ▼
         ingest_log
   who changed what, and when
```

## 3. Retrieval decision path

```
                Query + category + carried slots
                              │
                              ▼
                         Route class
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
  STRUCTURED                                     PROSE
        │                                           │
        ▼                                           ▼
  Indexed find                              Prefilter by
  IXSCAN ~60 ms                          programme + semester
        │                                           │
        ▼                                           ▼
   How many rows?                          BM25  +  dense cosine
        │                                           │
   ┌────┼────┐                                      ▼
   │    │    │                                Fuse with RRF
   ▼    ▼    ▼                                      │
 one  many  none                                    ▼
   │    │    │                                Parent expansion
   │    │    │                                      │
   ▼    ▼    ▼                                      ▼
Answer Ask  REFUSE                          cosine ≥ 0.30 ?
       for  and state                          │        │
     missing real                             no       yes
      slot  coverage                           │        │
                                               ▼        ▼
                                            REFUSE    Groq
                                                    context-only
```
