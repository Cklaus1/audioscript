# Speaker Identity System — Specification

**Version:** 1.0
**Date:** 2026-03-25
**Status:** Approved
**Guiding Principle:** Link first. Name second. Confirm last.

---

## 1. Product Boundary

### AudioScript owns (fast operational layer)
- Transcription + diarization
- Per-call speaker segmentation
- Cross-call speaker linking (stable voice identity)
- Calendar join (timestamp + attendees)
- Lightweight identity resolution (DB match + calendar + basic transcript)
- Unknown-speaker rollups + review queue generation

### DeepScript owns (deeper reasoning layer)
- Heavier identity inference (LLM-based)
- Cross-call behavioral analysis
- Importance ranking + strategic relevance
- Relationship/role deduction
- Ambiguous-speaker review workflows
- Cross-project speaker intelligence

**The rule:** AudioScript produces trustworthy structure. DeepScript produces richer interpretation.

---

## 2. Core Objects

### CallRecord
```json
{
  "call_id": "call_2026_03_25_001",
  "source": "zoom|meet|phone|upload",
  "title": "Weekly vendor sync",
  "start_time": "2026-03-25T14:00:00-04:00",
  "end_time": "2026-03-25T14:52:00-04:00",
  "timezone": "America/New_York",
  "calendar_event_id": "gcal_evt_123",
  "calendar_series_id": "series_abc",
  "organizer": "me@company.com",
  "attendees": [
    {"name": "Chris", "email": "chris@company.com"},
    {"name": "Dana", "email": "dana@vendor.com"}
  ],
  "transcript_id": "tx_001",
  "importance_score": 0.77
}
```

### SpeakerOccurrence
A speaker as they appear in one call.
```json
{
  "occurrence_id": "occ_001_spk_02",
  "call_id": "call_2026_03_25_001",
  "local_label": "SPEAKER_01",
  "speaker_cluster_id": "spk_a91f",
  "display_name": null,
  "identity_status": "unknown",
  "resolution_source": null,
  "resolution_confidence": 0.0,
  "total_speaking_seconds": 812,
  "segments": [
    {"start": 12.1, "end": 18.4, "text": "Thanks everyone for joining..."}
  ]
}
```

### SpeakerIdentity
Stable identity across calls, even if unnamed.
```json
{
  "speaker_cluster_id": "spk_a91f",
  "canonical_name": null,
  "aliases": [],
  "status": "unknown|candidate|probable|confirmed",
  "embedding_centroid": [0.12, -0.88, 0.03],
  "sample_count": 9,
  "first_seen": "2026-02-12T10:00:00-05:00",
  "last_seen": "2026-03-25T14:52:00-04:00",
  "total_calls": 7,
  "total_speaking_seconds": 4832,
  "typical_co_speakers": ["spk_me", "spk_31bc"],
  "candidate_people": [
    {
      "name": "Chris",
      "email": "chris@company.com",
      "score": 0.81,
      "evidence": ["calendar_overlap", "transcript_self_reference"]
    }
  ]
}
```

### SpeakerEvidence
Every identity decision should be explainable.
```json
{
  "evidence_id": "ev_001",
  "speaker_cluster_id": "spk_a91f",
  "call_id": "call_2026_03_25_001",
  "type": "db_match|calendar_overlap|llm_inference|user_confirmation|manual_merge",
  "score": 0.87,
  "summary": "Voice matched enrolled Chris profile",
  "details": {"cosine_similarity": 0.92},
  "created_at": "2026-03-25T15:00:00-04:00"
}
```

### SpeakerReviewItem
```json
{
  "review_item_id": "rev_001",
  "speaker_cluster_id": "spk_a91f",
  "priority_score": 0.88,
  "reason": "frequent_unknown_speaker",
  "summary": "Unknown speaker in 7 calls, 93 min. Likely Chris from calendar.",
  "candidate_names": ["Chris", "Dan"],
  "recommended_action": "ask_user",
  "status": "open"
}
```

---

## 3. Resolution States

| State | Meaning |
|-------|---------|
| `unknown` | Linked voice cluster, no name candidate |
| `candidate` | Has name candidates (0.60-0.79 confidence) |
| `probable` | Strong candidate (0.80-0.91 confidence) |
| `confirmed` | Verified identity (0.92+ or user-confirmed) |

**Key design choice:** A speaker can be **confirmed with no name** but stable internal identity. `spk_a91f` appearing in 12 calls is valuable even unnamed.

---

## 4. Resolution Pipeline (AudioScript sync)

```
Stage A: Diarization
  audio → speaker segments with local labels (SPEAKER_00, SPEAKER_01)

Stage B: Embedding extraction
  per-speaker centroid from representative segments

Stage C: Existing speaker DB match
  cosine similarity against known SpeakerIdentity records
  high match → attach existing speaker_cluster_id
  medium match → attach as tentative candidate
  no match → create new unknown speaker_cluster_id

Stage D: Cross-call stitching
  unknown voice matches unknown from prior calls → merge into same cluster
  no name required — just stable linkage

Stage E: Calendar join
  attach event metadata, attendees, recurring series
  creates identity candidates (not finalizations)

Stage F: Transcript/entity inference (lightweight)
  "Hey Chris", "This is Dana from Acme", self-introductions
  creates candidate evidence, not direct writes

Stage G: Confidence scoring
  aggregate: embedding + calendar + transcript + recurrence + prior confirmations

Stage H: Decision
  auto-confirm if strong enough (see §6)
  mark probable if decent but not safe
  leave unknown if weak
  create review item if valuable or ambiguous
```

---

## 5. Decision Order (trust ladder)

1. **Match known DB identity** (highest trust)
2. **Match known unknown-cluster identity** (preserves cross-call linkage)
3. **Calendar-based candidate generation** (structured metadata)
4. **Transcript-based inference** (useful but lower trust)
5. **User confirmation** (highest authority for final naming)

---

## 6. Confidence Model

### Embedding match thresholds
| Score | Classification |
|-------|---------------|
| >= 0.90 | Strong match |
| 0.80 - 0.89 | Moderate match |
| 0.70 - 0.79 | Weak candidate |
| < 0.70 | No match |

### Final identity confidence bands
| Score | Action |
|-------|--------|
| >= 0.92 | Auto-confirm |
| 0.80 - 0.91 | Probable (auto-link, don't rename publicly) |
| 0.60 - 0.79 | Candidate only |
| < 0.60 | Unknown |

### Hard rules — only auto-confirm a name if:
- Previously confirmed speaker DB match with strong embedding score, OR
- Repeated calendar + embedding consistency across multiple calls, OR
- Explicit user confirmation

**Never auto-confirm from LLM inference alone.**

---

## 7. Speaker DB Design

### Two modes
**A. Confirmed identities** — trusted named records (me, Chris, John from Acme)
**B. Stable unknown identities** — linked across calls but unnamed (spk_a91f, spk_31bc)

The DB is not just "named people." It is the long-term memory of recurring voices.

### Store more than embeddings
```
speaker_cluster_id     — stable internal identity
canonical_name         — optional human label
aliases                — alternative names
status                 — unknown | candidate | probable | confirmed
embedding_centroid     — voice fingerprint
sample_count           — number of enrollment samples
first_seen / last_seen — temporal range
total_calls            — appearance count
total_speaking_seconds — total voice time
typical_co_speakers    — who they're usually with
organizations          — domains/companies seen with
confirmed_vs_inferred  — trust level
enrollment_source      — db_match | calendar | llm | user_confirmed
```

---

## 8. Unknown Speaker Review Queue

**This is a major feature, not a debug tool.**

### Generate review items for:
- Unknown speaker appears in many calls
- Speaker appears in high-importance calls
- Likely identity has strong but not sufficient evidence
- Multiple unknown clusters may be the same person
- One cluster may be polluted by bad diarization

### Priority formula
```
priority =
  0.35 * normalized_total_minutes +
  0.25 * normalized_call_count +
  0.20 * call_importance +
  0.20 * resolvability_score
```

### Unknown speaker summary table
```
Speaker     Calls  Minutes  Last Seen  Candidates       Topics           Priority
spk_a91f    7      93       Mar 25     Chris (0.81)     hiring, planning  0.88
spk_31bc    4      41       Mar 19     Dana (0.67)      vendor sync       0.66
spk_51de    2      6        Mar 02     none             small talk        0.12
```

---

## 9. DB Update Rules

### Auto-update allowed
- Add occurrence to existing confirmed speaker on strong match
- Add occurrence to existing unknown cluster on strong match
- Update centroids incrementally
- Add non-final candidate evidence

### Auto-update NOT allowed
- Renaming from weak inference
- Merging clusters from weak evidence
- Replacing confirmed identity with a new guess
- Writing LLM-only guesses as confirmed identity

---

## 10. Merge / Split Operations

### Merge
Two clusters are likely same person → combine occurrences, recompute centroid, retain history

### Split
One cluster got polluted → separate by embedding outliers, create new cluster, preserve audit

Both operations must be explicit and reversible.

---

## 11. Database Schemas

```sql
-- Stable speaker identities (named or unnamed)
CREATE TABLE speaker_identities (
  id TEXT PRIMARY KEY,                    -- spk_a91f
  canonical_name TEXT NULL,
  status TEXT NOT NULL,                   -- unknown|candidate|probable|confirmed
  embedding_centroid JSON NOT NULL,
  sample_count INTEGER NOT NULL,
  first_seen TIMESTAMP NOT NULL,
  last_seen TIMESTAMP NOT NULL,
  total_calls INTEGER NOT NULL,
  total_speaking_seconds INTEGER NOT NULL,
  created_from TEXT NOT NULL,
  updated_at TIMESTAMP NOT NULL
);

-- Per-call speaker appearances
CREATE TABLE speaker_occurrences (
  id TEXT PRIMARY KEY,
  call_id TEXT NOT NULL,
  speaker_cluster_id TEXT NOT NULL,
  local_label TEXT NOT NULL,              -- SPEAKER_00
  display_name TEXT NULL,
  resolution_source TEXT NULL,
  resolution_confidence REAL NOT NULL,
  total_speaking_seconds INTEGER NOT NULL
);

-- Evidence trail for identity decisions
CREATE TABLE speaker_evidence (
  id TEXT PRIMARY KEY,
  speaker_cluster_id TEXT NOT NULL,
  call_id TEXT NULL,
  type TEXT NOT NULL,                     -- db_match|calendar_overlap|llm_inference|user_confirmation
  score REAL NOT NULL,
  summary TEXT NOT NULL,
  details JSON,
  created_at TIMESTAMP NOT NULL
);

-- Name candidates for unknown speakers
CREATE TABLE speaker_candidates (
  id TEXT PRIMARY KEY,
  speaker_cluster_id TEXT NOT NULL,
  candidate_name TEXT NOT NULL,
  candidate_email TEXT NULL,
  score REAL NOT NULL,
  source TEXT NOT NULL,
  evidence JSON
);

-- Review queue for ambiguous speakers
CREATE TABLE speaker_review_queue (
  id TEXT PRIMARY KEY,
  speaker_cluster_id TEXT NOT NULL,
  priority_score REAL NOT NULL,
  reason TEXT NOT NULL,
  summary TEXT NOT NULL,
  status TEXT NOT NULL,                   -- open|resolved|dismissed
  created_at TIMESTAMP NOT NULL
);
```

---

## 12. Service Architecture

| Service | Responsibility | Layer |
|---------|---------------|-------|
| `SpeakerDatabase` | Store/fetch identities, embedding match, create clusters, merge/split | AudioScript |
| `SpeakerResolutionEngine` | Full resolution pipeline (stages A-H), confidence scoring | AudioScript |
| `CalendarJoiner` | Map call timestamp → calendar event, fetch attendees | AudioScript |
| `UnknownSpeakerReporter` | Summarize unresolved speakers, sort by priority, create review queue | AudioScript |
| `SpeakerInferenceEngine` | Deep identity reasoning, relationship analysis, importance scoring | DeepScript |

---

## 13. Implementation Phases

### Phase 1 (AudioScript v0.3)
- Speaker DB match in sync pipeline
- Stable unknown speaker IDs (spk_xxxx)
- Occurrence storage per call
- Calendar timestamp storage
- Update existing SpeakerDatabase class

### Phase 2
- Calendar join (event → attendees → candidates)
- Candidate generation from calendar + transcript
- Confidence scoring (embedding + calendar + transcript)
- Review queue generation

### Phase 3
- Post-hoc labeling flow (CLI: `audioscript speakers label spk_a91f "Chris"`)
- Merge/split tools
- Unknown speaker summary report
- Enrollment workflow improvements

### Phase 4 (DeepScript)
- LLM-based identity reasoning
- Relationship graph
- Org/person-role inference
- Strategic meeting analysis
- Cross-project speaker intelligence

---

## 14. Call Output Format (with speaker identity)

```json
{
  "call_id": "call_2026_03_25_001",
  "speakers": [
    {
      "local_label": "SPEAKER_00",
      "speaker_cluster_id": "spk_me",
      "display_name": "You",
      "status": "confirmed",
      "resolution_source": "db_match",
      "confidence": 0.99
    },
    {
      "local_label": "SPEAKER_01",
      "speaker_cluster_id": "spk_a91f",
      "display_name": null,
      "status": "probable",
      "resolution_source": "calendar_plus_transcript",
      "confidence": 0.84,
      "candidate_names": [
        {"name": "Chris", "score": 0.84}
      ]
    }
  ]
}
```

---

## 15. LLM Usage Policy

### Use LLM for
- Extracting candidate names from transcript
- Interpreting self-introductions
- Role inference
- Summarizing unknown speakers
- Explaining why a speaker might be someone

### Do NOT use LLM as final source of truth

**Good:** "Likely Dana because speaker says 'from Acme' and Dana is the only external attendee."
**Bad:** "Model thinks it sounds like Dana."
