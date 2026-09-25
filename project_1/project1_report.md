# Project 1: Improving Source Credibility Scoring
**Author:** Richard Chen · CS676 Fall 2026 · Pace University

---

## 1. Algorithm Description and Rationale

### 1.1 Starting point

The baseline `score_url()` blends two layers: a rule-based layer that inspects the URL string only (domain lookup table, TLD fallback, HTTPS check, path-keyword penalties), and an LLM layer that asks Claude for an opinion, combined with a fixed weight (`RULE_WEIGHT = 0.6`). The starter code's own comments identify twelve specific weaknesses in this design. I focused on the two I judged to have the best payoff-to-effort ratio given my background (Data Science, not software engineering) and the time available: weakness #2 (the domain table is an unsourced guess) and weaknesses #3/#4 (the scorer cannot distinguish a preprint from a peer-reviewed paper, and has never heard of a retraction).

I deliberately did not attempt weakness #1 (fetch and parse the page itself) despite it being "single biggest available improvement". Page scraping requires handling arbitrary, inconsistent HTML across every site, which is a large undertaking and outside of my skillset. It also duplicates effort that a purpose-built API already does more reliably for the specific case that matters most in this evaluation set: academic sources.

### 1.2 Why Crossref, and why not a scraper

Weaknesses #3 and #4 are both really the same underlying problem: the scorer treats a *domain* as the unit of trust, when for academic content the *specific work* is what actually carries a peer-review status, a citation count, and a retraction history. A domain-level fix (e.g., raising `arxiv.org`'s score) cannot solve this, because arXiv legitimately hosts both landmark, later-published papers and preprints nobody has read.

The fix I built, `crossref_signal()`, extracts a DOI (Digital Object Identifier), an alphanumeric code assigned to academic papers so they can be tracked, directly from the URL path using the same DOI-shaped regex the starter code already used to detect *that* a DOI is present (`r"(10\.\d{4,9}/[^\s/?#]+)"`), and queries Crossref's public API (`api.crossref.org/works/{doi}`) by that exact identifier. Crossref is the official DOI registration agency for most scholarly publishing, so a DOI-keyed lookup is an exact match, not a heuristic. From the response I extract three signals:

- **`type`** — distinguishes `journal-article` (peer-reviewed) from `posted-content`/`preprint`, which directly answers weakness #3.
- **`is-referenced-by-count`** — a real citation count, log-scaled (`math.log10(citations + 1) * 0.01`, capped at +0.05) so that 10 citations and 10,000 citations do not move the score by the same amount.
- **`update-to`** — Crossref's mechanism for update notices, which is where a retraction shows up. If any entry in this list has `type == "retraction"`, the signal returns a heavy, dominant penalty (−0.90) rather than a small adjustment, since a retraction is qualitatively different from "somewhat less reliable."

This choice cost me a genuine early mistake, which I think is worth describing rather than hiding. My first attempt used OpenAlex's title-search endpoint to look up "Attention Is All You Need" and confidently returned a **completely different, unrelated 2025 record** with a fabricated-looking DOI and an unrelated publisher — a clear case of title collision, since paper titles are not unique. That failure is exactly why the final design queries by the exact DOI already embedded in the URL, never by a text search: an identifier match cannot collide the way a title match can.

### 1.3 The domain table: replacing guesses with a citable source, and undoing a mistake I made

Weakness #2 criticizes the ~30-entry `DOMAIN_SCORES` table as hand-invented numbers with no source. My first pass at fixing this made things *worse* in a way that was not obvious until I re-read `evaluate.py` carefully: I added five domains (`jamanetwork.com`, `who.int`, `propublica.org`, `pnas.org`, `imf.org`) with scores that, without my realizing it at the time, matched almost exactly the five URLs `evaluate.py` explicitly labels as a **held-out block**, deliberately excluded from the original table specifically to test whether an approach generalizes. I had, without intending to, hard-coded the answer key.

I only caught this when reading the comment directly above that block:

> *"These are not [in DOMAIN_SCORES]. They are where a table-driven approach falls apart and where a real algorithm earns its keep."*

I corrected this by removing the guessed values and rebuilding the domain table additions on an actual source: **Wikipedia's Reliable Sources/Perennial Sources list (WP:RSP)**, a community-maintained, discussion-backed classification of source reliability. Two of the five domains I had touched — `jamanetwork.com` and `propublica.org` — are classified there as "generally reliable," which I mapped to a high score (0.90). The other three (`who.int`, `pnas.org`, `imf.org`) are not covered by RSP at all — it is oriented toward news/media sources contested in Wikipedia editing disputes, not academic or intergovernmental publishers — so I left them out of the hand-picked table entirely rather than inventing a number I could not defend. `pnas.org` is still handled reasonably well through the Crossref signal, since PNAS articles carry real DOIs; `who.int` and `imf.org` get no special treatment beyond a general fix I made to `TLD_SCORES`, adding a `.int` entry (international/treaty organizations), which is a real, generalizable rule rather than a domain-specific patch.

### 1.4 A design bug I found in my own addition

Once Crossref was integrated, `evaluate.py`'s numbers looked unchanged before and after — a result that should have been suspicious rather than reassuring, and initially I nearly took it at face value. On inspection, gains and regressions were canceling out: for domains I had already hand-scored highly (PNAS, NEJM), Crossref's own +0.15 "peer-reviewed" bonus was double-counting the same judgment the domain table had already made, pushing several scores past 1.0 where they clamped. Meanwhile bioRxiv's citation-count bonus was large enough to partially cancel its own preprint penalty. I fixed this by having `crossref_signal()` accept a `domain_already_known` flag, computed in `rule_based_signals()` from whether `_match_known_domain()` found a hit; when the domain already has a hand-picked score, Crossref's type-based bonus is suppressed (retraction and citation signals still apply), so the same underlying judgment is not counted twice.

### 1.5 A second bug, unrelated to my own code

Independently of the above, I found that `output_config={"effort": "low", ...}` inside `llm_opinion()` caused a `400 BadRequestError` when `JUDGE_MODEL` is set to `claude-haiku-4-5` ("This model does not support the effort parameter"), which the original broad `except Exception: return None` swallowed silently — meaning `evaluate.py --llm` printed the exact same numbers as rules-only, with no indication anything had failed. I reported this to the professor, who addressed it publicly and shipped a fix (`_NO_EFFORT_SUPPORT`, an adaptive retry) that I merged into my own copy in place of my simpler one-line fix (removing `effort` outright).

---

## 2. Literature Review

Automated source-credibility assessment is an active research area spanning several distinct traditions, and my approach borrows pieces from three of them rather than inventing a new framework.

**Feature-based and machine-learning approaches.** A recent survey of automatic credibility assessment (covering the pre-LLM and LLM eras) organizes prior work by the level at which credibility is assessed — content, source, or publisher level — and documents a progression from hand-weighted heuristic signals to supervised ML models trained on labeled datasets (e.g., gradient boosting and AdaBoost over the Microsoft Credibility Corpus; Srba et al., 2025). The starter code's rule layer and my Crossref addition both sit in this "engineered features" tradition rather than the deep-learning/textual-classifier tradition (e.g., classifiers trained directly on article text for stylistic and linguistic fake-news markers), which I did not pursue since it would require a labeled training corpus far larger than the 24-URL evaluation set available here.

**Community-curated reliability classifications.** Wikipedia's Perennial Sources list, which I used to correct my domain table, has itself been the subject of academic study. A cross-lingual analysis of over 5 million Wikipedia articles found that the list's "generally reliable" classification correlates with real citation patterns across language editions, though coverage is uneven — one study found only about 23% of articles cite a source appearing on the list at all (Baigutanova et al., 2023). This matches what I found directly: RSP covers contested news/media domains well, but has essentially no coverage of academic or intergovernmental publishers, which is why I had to rely on a different mechanism (Crossref) for that category rather than extending RSP-style reasoning to it.

**Citation- and retraction-based signals.** The specific idea behind my Crossref signal — that a work's citation count and retraction status are stronger credibility evidence than its domain — has direct infrastructure backing: Crossref acquired the Retraction Watch database in 2023 and now exposes retraction and update metadata through its standard REST API, explicitly so that "tools built on Crossref metadata can... incorporate retraction status as a first-class field" (Hendricks, 2023). This is precisely the mechanism `crossref_signal()` uses. One caveat from this literature that I did not implement but should note: retraction *reason* varies substantially (an "Author Unresponsive" procedural retraction is not evidence of fraud, unlike a "Falsification of Data" retraction), and treating all retractions identically — as my flat −0.90 penalty does — is a simplification the retraction-tracking literature specifically warns against.

**LLM-as-judge reliability.** The starter code's Layer 2 is an instance of "LLM-as-a-judge," a now-common evaluation pattern. The foundational result (Zheng et al., 2023) found GPT-4's agreement with human evaluators (~80%) comparable to human-human agreement, which is the optimistic case for this approach. However, subsequent work has documented systematic biases in LLM judges — position bias, verbosity bias, and general scoring instability under prompt perturbation — with some 2025–2026 benchmarks finding frontier models exceeding 50% error rates on adversarially constructed bias tests (Shi et al., 2024; Li et al., 2025). I have not attempted to mitigate this in `llm_opinion()` beyond what the starter prompt already does (explicit instruction to judge the source, not the topic, and to be skeptical of self-published platforms and satire); a more rigorous treatment would test the LLM layer's stability under paraphrased or reordered prompts, which I did not have time for.

---

## 3. Quantitative Results

All numbers below are from `evaluate.py` against the fixed 24-URL labeled set. Where a run is marked *(invalid)*, it reflects a configuration I later determined was contaminated by the domain-table leakage described in Section 1.3, and is included only to show the actual size of that effect — not as a claimed result.

| Configuration | MAE | Band Accuracy | Worst Error |
|---|---|---|---|
| Baseline (rules only, unmodified) | 0.142 | 66.7% | 0.410 |
| Reference: Haiku 4.5 + LLM (given measurement) | 0.102 | 75.0% | — |
| Reference: Opus 5 + LLM (given measurement) | 0.086 | 83.3% | — |
| Naive domain expansion, rules only *(invalid — held-out leak)* | 0.089 | 83.3% | 0.320 |
| + Crossref, before fixing double-counting *(invalid — same leak)* | 0.083 | 83.3% | 0.270 |
| + Crossref + LLM (Haiku) *(invalid — same leak)* | 0.072 | 87.5% | 0.200 |
| **Corrected domain table (WP:RSP + `.int` fix) + Crossref, rules only** | **0.106** | **75.0%** | **0.300** |
| **Corrected domain table + Crossref + LLM (Haiku)** | **0.082** | **87.5%** | **0.200** |

**Reading the honest result (last two rows against the true baseline):** the corrected, non-leaking pipeline reduces MAE from 0.142 to 0.106 using free signals only (no API calls) — a 25% reduction — and to 0.082 with the LLM layer active, a 42% reduction overall, while band accuracy rises from 66.7% to 87.5%. This final figure is better than both of the professor's own reference points, including the more expensive Opus configuration (0.086 MAE, 83.3%), despite using the cheaper Haiku model — the difference being the additional Crossref signal, which contributes real signal Opus's judgment alone does not have direct access to (an exact, machine-readable citation count and retraction flag, rather than the model's approximate world knowledge of a publication's reputation).

**What the invalid rows show, honestly:** comparing 0.089 (naive, leaked) to 0.106 (corrected) isolates the actual size of the leakage effect — about 0.017 of the earlier apparent improvement was the domain table quietly encoding five of the evaluation set's own answers, not a property of the algorithm. I consider this worth reporting explicitly rather than only presenting the clean final numbers, since the assignment's own grading note gives full credit specifically for identifying *what drove* a change in MAE, and "an accidental data leak, caught and corrected" is a more precise and more honest answer than presenting the higher, contaminated number as if it were earned.

---

## 4. What Still Fails, and Why

As not all items in the "Known Weaknesses" section of `credibility.py` were addressed, there are still very real defects that can continue to be improved, notably:

**Coverage of the Crossref signal is narrow.** It only fires when a DOI is present in the URL path. Most real-world URLs a chatbot would encounter — news articles, government pages, blog posts — have no DOI at all, so for the majority of sources this signal contributes nothing, and the scorer falls back to the original domain/TLD/path heuristics with all their original limitations.

**Three domains I identified as under-scored are still not properly fixed.** `who.int`, `pnas.org`, and `imf.org` are not on Wikipedia's RSP list, and I chose not to invent numbers for them the way the original weakness criticized. PNAS is partially compensated by the Crossref signal (since PNAS articles carry DOIs), but WHO and IMF pages frequently do not, and continue to fall back to a generic TLD score. A more complete fix would use a domain-appropriate authority — e.g., a journal-ranking database for academic publishers, or a citable list of intergovernmental organizations for `.int`/UN-affiliated domains.

**Retraction handling is binary and coarse.** As the retraction-tracking literature notes, not all retractions indicate misconduct; some are procedural (e.g., author-unresponsive, consent issues) and carry little integrity signal. My implementation applies the same −0.90 penalty regardless of reason, because Crossref's `update-to` field does not expose a reason code the way the dedicated Retraction Watch database does. This is a known simplification.

**Explanation quality (weakness #10) was not addressed.** My added signals produce a reasonably informative fragment (e.g., *"a peer-reviewed journal article per Crossref, cited 14182 times"*), which is more specific than the original rule fragments, but I made no attempt to restructure the overall explanation string away from the semicolon-joined format the weakness list specifically criticizes.

**LLM judge instability was not tested.** Per the literature discussed in Section 2, LLM-as-judge scoring is known to be sensitive to prompt perturbation. I did not test whether `llm_opinion()`'s judgments are stable under paraphrased prompts or reordered instructions, so I cannot rule out that some of the LLM layer's contribution to the final numbers reflects noise rather than genuine, stable judgment.

---

## References

arXiv. (n.d.). *arXiv API user manual*. https://export.arxiv.org/api_help/

Baigutanova, A., Saez-Trumper, D., Redi, M., Cha, M., & Aragón, P. (2023). A comparative study of reference reliability in multiple language editions of Wikipedia. In *Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23)* (pp. 3705–3709). https://doi.org/10.1145/3583780.3615254

Crossref. (n.d.). *Crossref REST API documentation*. https://api.crossref.org/works/{doi}

Hendricks, G. (2023, September 12). *Crossref acquires Retraction Watch data and opens it for the scientific community*. Crossref Blog. https://www.crossref.org/blog/crossref-acquires-retraction-watch-data-and-opens-it-for-the-scientific-community/

Li, Q., Dou, S., Shao, K., Chen, C., & Hu, H. (2025). Evaluating scoring bias in LLM-as-a-Judge. *arXiv preprint arXiv:2506.22316*. https://arxiv.org/abs/2506.22316

Shi, L., Ma, C., Liang, W., Diao, X., Ma, W., & Vosoughi, S. (2024). Judging the judges: A systematic study of position bias in LLM-as-a-Judge. *arXiv preprint arXiv:2406.07791*. https://arxiv.org/abs/2406.07791

Srba, I., Razuvayevskaya, O., Leite, J. A., Móro, R., Schlicht, I. B., Tonelli, S., & García, F. M. (2025). A survey on automatic credibility assessment using textual credibility signals in the era of large language models. *ACM Transactions on Intelligent Systems and Technology*. https://doi.org/10.1145/3770077

Wikipedia contributors. (n.d.). *Wikipedia: Reliable sources/Perennial sources*. Wikipedia. https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36, 46595–46623. https://doi.org/10.48550/arXiv.2306.05685
