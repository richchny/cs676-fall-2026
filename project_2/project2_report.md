# Project 2: PersonaForge
**Author:** Richard Chen · CS676 Fall 2026 · Pace University
**Model used for all generated personas, conversations, and judge verdicts:** `claude-haiku-4-5`

# PersonaForge: Technical Report

---

## 1. Summary and scope

PersonaForge is a Python package in which a persona is a markdown file, an agent is that file plus a model, and a conversation is agents taking turns. The skeleton shipped with a regex-based intent router and strict round-robin conversations. I chose to go deep on two of the five suggested extensions, and to harden the package around them:

| Scope item | README extension | Status |
|---|---|---|
| **Tool-use orchestrator** replacing the regex router | #1 (headline task) | Done, run live |
| **Observer / AI judge** with an `evaluate_conversation` tool | #4 | Done, run live |
| **Robustness**: per-turn failure handling, unique names and filenames, transcripts kept out of the persona folder | supports Part 3 | Done |
| **Conversation quality fixes**: scene framing, narration filter, persona roster in the prompt | emerged from live failures | Done |

**Deliberately out of scope:** agent memory (#2) and smarter turn-taking (#3). I chose depth over breadth: each of the four changes above is backed by a live failure, a fix, a regression test, and in most cases a saved transcript. The remaining skeleton limitations are listed in section 9.

**Stated scope for "feature completeness":** a terminal agent where the user can phrase requests freely; Claude chooses and chains tools to create personas, list them, run conversations, and evaluate them; failures degrade gracefully; `--offline` still works for free with no key.

---

## 2. How to run it

```bash
uv sync
uv run pytest                 # 84 tests, offline, no API key
uv run main.py --offline      # whole app, scripted replies, free
uv run main.py                # live, needs ANTHROPIC_API_KEY in .env
```

A fresh clone into `/tmp` was used to confirm that `uv sync`, `uv run pytest`, and `uv run main.py --offline` all work from what is on GitHub, with no `.env`.

Example live session (natural phrasing the old router could not handle):

```
you › create persona new employee at same F500 company as marcus webb fresh out of college
you › have both converse about 2027 open enrollment for health insurance, Jade does not
      understand what a HDHP is. converse for 6 turns
you › evaluate the conversation
```

---

## 3. Design write-up

### 3.1 Architecture

```
you → cli.py → Orchestrator.handle()
                 ├─ model supports tools (ClaudeLLM)  → _handle_with_tools loop
                 │                                        └─ execute_tool() → persona / conversation / observer code
                 └─ no tool support (DemoLLM, offline) → original regex router
```

All model calls go through one method family in `llm.py`, which is why everything is testable with fakes.

| Module | Role | Added or changed |
|---|---|---|
| `llm.py` | `ClaudeLLM`, `ScriptedLLM`, `DemoLLM` behind one interface | Added `ToolCall`, `ToolResponse`, `complete_with_tools`, `ScriptedToolLLM`; default model set to Haiku |
| `tools.py` (new) | Tool schemas plus `execute_tool` dispatcher | Four tools: `create_persona`, `list_personas`, `start_conversation`, `evaluate_conversation` |
| `orchestrator.py` | The loop that lets Claude choose tools | Added `_handle_with_tools`, roster injection, round cap |
| `observer.py` (new) | LLM judge, tolerant JSON parser, transcript loader | New |
| `conversation.py` | Turn loop and transcripts | Per-turn failure handling, scene framing, unique timestamped transcript names |
| `persona.py` | Persona files | Non-overwriting save, `find_personas`, unique names, author prompt keeps user-given names |
| `agent.py` | Persona plus model | Stricter narration rule, narration filter |

### 3.2 Tool-use orchestrator

**Problem.** The regex router could not handle "create a skeptical data scientist named Ada", could not chain steps, could not ask clarifying questions, and hard-coded six turns.

**Design.** The user's message goes to Claude together with four tool definitions. While `stop_reason == "tool_use"`, the loop runs each requested tool, appends one `tool_result` per `tool_use` id in a single user message, and calls Claude again. The model's own reply, including `tool_use` blocks, is echoed back into the history unchanged, because the API requires it. The loop is capped at `MAX_TOOL_ROUNDS = 8` so a confused model cannot spin forever.

**Key decisions and why:**

- **A new method rather than changing `complete()`.** `complete()` returns a string, and a tool call is not a string. Adding `complete_with_tools()` returning small dataclasses of my own (`ToolCall`, `ToolResponse`) left every existing caller and test untouched and kept Anthropic's types out of the rest of the package.
- **Tool errors are returned as text, never raised.** `execute_tool` catches everything and returns `"Error: ..."`. Claude can read "'marcus' is ambiguous; it matches Marcus Chen, Marcus Webb" and ask the user which one. The cost of this choice is that a coding mistake inside a tool also surfaces as an error string; this happened once (a missing `Path` import) and was caught because the tests assert on exact result text.
- **Hard limits live in code, not in the prompt.** `MAX_TURNS = 20` caps what the model can request, since each turn is a billed call. The `evaluate_conversation` tool strips directory parts from any filename the model supplies, so it cannot read outside the transcripts folder.
- **Regex router kept as the offline fallback.** `hasattr(self.llm, "complete_with_tools")` selects the path. `--offline` therefore works on a clean clone with no key. The fallback path received the same ambiguity and duplicate-participant checks as the tool path, so the "talks to himself" bug is fixed there too.
- **Roster injection.** On every round the system prompt includes the list of personas that currently exist (see 3.6).

### 3.3 Observer agent (AI judge)

**Problem.** I was finding failures by reading transcripts by hand: characters saying "roleplay", narrating actions, drifting off topic. I wanted that check automated, and the course requires an AI judge.

**Design.** `judge_conversation(conversation, llm)` sends the transcript and a rubric to Claude and asks for JSON: per speaker `in_character` and `stage_directions`, plus conversation-level `on_topic` and a short note. It returns a `Verdict`.

**Key decisions:**

- **Tolerant parsing.** This applies the lesson from the frontmatter bug in the README: be strict about the fields needed, generous about everything else. `_extract_json` handles code fences and chatter around the JSON; `parse_verdict` never raises.
- **`Optional[bool]`, not `bool`.** "The judge did not say" (`None`) is different from "no problem" (`False`). A garbled reply produces `parsed=False`, and the tool reports "could not produce a readable verdict" instead of "no problems flagged". A failed judge must never look like a pass.
- **No wasted calls.** An empty conversation is refused without calling the model.
- **Loader for saved transcripts.** `load_transcript` rebuilds a `Conversation` from a saved `.md` file, so old runs can be judged and compared. `latest_transcript` picks the most recently modified file.
- **Exposed as a tool.** "Was that conversation any good?" runs the judge on the latest transcript, or on a named file.

### 3.4 Robustness changes

| Failure found | Fix | Regression test |
|---|---|---|
| Two personas with the same name overwrote each other on disk | `save_persona` picks a free filename (`name-2.md`), with an explicit `overwrite=True` escape hatch | `test_save_does_not_overwrite_existing`, `test_save_can_overwrite_on_request` |
| Duplicate names break `PersonaAgent`, which identifies its own lines by name | `generate_persona` tells the model which names are taken, then `_unique_name` guarantees uniqueness in code | three `generate_persona` tests |
| Different names can produce the same slug (`Maria O'Hara` vs `Maria O Hara`) | The filename check stays as a second layer: names protect conversation identity, filenames protect data on disk | covered by the save tests |
| `conversation.md` was read as a persona called "Unnamed", and each conversation overwrote the last | Transcripts go to `transcripts/` with `YYYY-MM-DD_HHMMSS_topic.md` names | `test_transcripts_do_not_overwrite_each_other`, `test_transcripts_are_not_loaded_as_personas` |
| One failing turn lost the whole conversation | `run_conversation` catches the failure, keeps earlier lines, records `Conversation.error`, and the saved transcript ends with `Stopped early after N turns: ...` | three failure tests |
| Empty model replies were recorded as blank turns | Treated as a failure that stops the conversation | `test_empty_reply_stops_the_conversation` |
| Ambiguous lookups silently picked the first match | `find_personas` returns all matches; tools report the ambiguity | ambiguity tests in `test_tools.py` and `test_conversation.py` |

**Design choice: stop on failure rather than skip the turn.** Skipping would break the alternation, so the next agent would see a transcript ending in its own line, which is the same shape that produced the empty turns in the Marcus run. Stopping with a partial transcript is simpler and more honest. The SDK already retries transient errors, so a failure that reaches this code is usually worth stopping on.

### 3.5 Scene framing

The topic used to reach the first speaker as a bare noun phrase (`Scene: a difficult patient`). The model treated it as an under-specified user message and asked for clarification. It is now an instruction: "You are in a conversation. The subject is: {topic}. Begin speaking in character now; do not ask for clarification or describe the setup." Measured effect in section 6.

### 3.6 Roster injection

Restarting the app emptied the orchestrator's history, so Claude did not know Ada Chen and Elena Vasquez already existed. It created Ada Rothstein and Elena Okonkwo, which made "Ada" ambiguous. The earlier good runs had partly depended on Claude happening to call `list_personas` first. The fix was to supply the facts in code: the current roster is appended to the system prompt on every round, and the prompt tells Claude not to recreate a persona that already exists.

### 3.7 Narration handling

Prompting alone did not stop narration (section 6). The package now asks in the prompt and enforces in code: `_strip_stage_directions` removes multi-word asterisk actions such as `*pauses, jaw tightens*` while keeping single-word emphasis such as `*understood*`.

---

## 4. Annotated transcript

**File:** `transcripts/clean_espresso_run.md` (original name `2026-11-09_224312_dialing-in-an-espresso-shot.md`). Personas: Marcus Webb (barista and shop owner) and Derek Holloway (home enthusiast). Prompt: *"have Marcus and Derek converse about dialing in an espresso shot for 6 turns"*. Model: `claude-haiku-4-5`.

**What worked, with evidence from the transcript:**

1. **Tool chain.** The user asked "evaluate that conversation"; Claude selected `evaluate_conversation` without being told the tool name and judged the latest file.
2. **Turn control.** The request said 6 turns; exactly 6 were produced. The regex router would have used its hard-coded default.
3. **Character voice is distinct.** Marcus is terse and prescriptive ("Pick a grind setting and pull at least three shots before you change anything"). Derek is hesitant and self-correcting ("I've been kind of... tweaking it after every shot").
4. **Persistent topic.** Every turn stays on grind, tamping, and consistency. The scene framing held: no "what's the setup?" opening.
5. **Conversational memory within the transcript.** Marcus's advice in turn 3 (adjust only deliberately) is picked up by Derek in turn 4 ("lock in a grind setting and do three shots") and again in turn 6.
6. **No narration.** I read all six turns: no asterisk actions and no unmarked body description. The single `*thinking*` in turn 6 is word-level emphasis, which the filter is designed to keep.
7. **Judge and human agree:** "no problems flagged".

**What the transcript shows less well (limits of my own rubric):**

- Derek's last two turns largely restate Marcus's advice ("Okay, so I'm not supposed to be tweaking after every shot, I get it") before asking a new question. The judge's rubric has no repetition check, so "no problems flagged" does not mean "no problems".
- This run predates the narration filter and the stricter prompt rule, so it shows a low-emotion topic that did not trigger narration, not evidence that narration is solved (see section 6).

A second clean run, `transcripts/clean_hr_enrollment_run.md` (original `2026-11-12_220502_2027-open-enrollment-for-health-insuranc.md`), was produced after all fixes. It used free phrasing ("create persona new employee at same F500 company as marcus webb fresh out of college"), created personas in two steps, ran exactly 6 turns on an HDHP/HSA topic, and was judged clean. I agreed with the judge. One in-world detail the rubric cannot catch: Marcus offers to "send you a link to the benefits portal", a promise a character cannot keep.

---

## 5. Evidence index

All transcripts are in `transcripts/`. Original timestamped names are kept here for traceability.

| File | Original name | What it shows |
|---|---|---|
| `before_regex_router_marcus.md` | *(pasted from a terminal session; not generated as a file)* | Regex router picks Marcus Chen twice: six turns, one confused reply, five empty turns, no warning |
| `before_scene_fix.md` | `2026-11-08_225025_a-difficult-patient.md` | Bare-topic scene: roleplay talk, topic abandoned, narration |
| `before_scene_fix_hospital_staffing.md` | `2026-11-08_231930_hospital-staffing.md` | Ada opens "I'm ready to roleplay as Ada Chen"; Elena narrates in the third person without asterisks |
| `robustness_missing_key.md` | `2026-11-08_232343_hospital-staffing.md` | Failure on turn 1 (missing key): zero turns, clean stop, transcript still saved |
| `robustness_partial_outage.md` | `2026-11-08_232441_hospital-staffing.md` | Two real turns, then a simulated outage on turn 3; partial transcript preserved |
| `after_scene_fix.md` | `2026-11-08_232836_a-difficult-patient.md` | Scene fix applied: in character and on topic; narration remains |
| `prompt_rule_only_run1.md` | `2026-11-09_224801_a-difficult-patient.md` | Stricter prompt rule only: narration persists |
| `prompt_rule_only_run2.md` | `2026-11-09_225051_a-difficult-patient.md` | Same, heaviest narration of any run |
| `after_filter_run1.md` | `2026-11-09_230330_a-difficult-patient.md` | Narration filter plus roster fix: no asterisk narration; one filter artifact (section 7) |
| `clean_espresso_run.md` | `2026-11-09_224312_dialing-in-an-espresso-shot.md` | Annotated clean run (section 4) |
| `clean_hr_enrollment_run.md` | `2026-11-12_220502_2027-open-enrollment-for-health-insuranc.md` | Final clean run with free phrasing |

Provenance notes: the Marcus file is a verbatim terminal capture, not a transcript the system saved. The two robustness files and the early staffing file were produced by short experiments (a missing key, and a script that simulated an outage after two real calls), and any author notes pasted into them are annotations, not system output.

---

## 6. Measured before/after results

The judge was used as the instrument, and I read each transcript myself.

### 6.1 Scene framing fix

| Run | Character breaks | Narration | On topic |
|---|---|---|---|
| Before (`before_scene_fix`) | Ada refuses to roleplay; Elena says "ready to roleplay" (judge flagged only Ada) | Elena | No: agents invent a hospital efficiency study |
| Before (`before_scene_fix_hospital_staffing`) | Ada: "I'm ready to roleplay as Ada Chen" | Elena (unmarked, third person) | Partly |
| After (`after_scene_fix`) | None | Elena | Yes |

The fix removed the character breaks and the topic drift. It did not touch narration, which confirmed narration has a separate cause.

### 6.2 Narration

| Condition | Runs | Elena narrated | Notes |
|---|---|---|---|
| Original rule ("no stage directions") | 1 (`after_scene_fix`) | 1 of 1 | |
| Stricter prompt rule only | 2 | **2 of 2** | The rule was verified live in the system prompt; the model ignored it |
| Prompt rule plus code filter | 1 ICU run (`after_filter_run1`) | 0 of 1 asterisk actions | Residual unmarked narration is not caught by the filter |
| Prompt rule plus filter, non-ICU topic | 1 (`clean_hr_enrollment_run`) | 0 of 1 | |

Narration appeared only for Elena (an emotionally loaded ICU nurse), never for Ada or the espresso and HR personas. My working hypothesis is that emotional personas invite acting. It is a hypothesis, not a finding.

**Sample-size honesty.** These are single runs per condition on a stochastic model, so none of this is a statistical result. Two of two failures under the prompt-only condition, and zero narration under the filter, are consistent with the claim that prompts are requests and code is a guarantee. They do not prove it. The judge also reads the already-filtered transcript, so after the filter it measures what the filter misses, not what the model wrote. I did not log raw pre-filter replies, which would be the cleaner experiment.

### 6.3 Regex router versus tool use

| Behaviour | Before (regex router) | After (tool use) |
|---|---|---|
| "marcus and marcus" | Same persona twice; 5 of 6 turns empty; no warning | Ambiguity reported; Claude asks which one |
| "create a skeptical data scientist named Ada" | Fell through to a canned stub reply | Persona created |
| "...for 8 turns" | Hard-coded 6 | Exactly 8 |
| Multi-step requests | First matching pattern only | Chained tools in one request |

---

## 7. Failure analysis

Each entry lists what happened, why, what I did, and what remains.

**F1. Regex router talked a persona to himself.** `find_persona("marcus")` returned the first of two Marcuses both times. Identical names then made each agent treat every line as its own, so the model saw a transcript ending in an assistant turn and returned empty text. Five blank turns were printed with no warning; six calls bought almost nothing. *Fix:* ambiguity and same-persona checks, plus empty-reply detection. *Evidence:* `before_regex_router_marcus.md`.

**F2. Agents break character when the scene is vague.** Across three early runs, agents said "roleplay", asked what the scene was, or offered numbered menus of scenarios. Cause: bare-topic scene turn. *Fix:* instruction-style scene. Not fully eliminated in principle, since it depends on the model following instructions.

**F3. Narration survives the prompt rule.** Two of two runs under the stricter rule still contained `*pauses*`, `*leans against the counter*`, and so on. *Fix:* code-level filter. *Limits:* unmarked narration ("She pauses, looking at Ada directly") passes through, and the judge can miss it too.

**F4. The narration filter corrupted dialogue.** In `after_filter_run1.md`, Ada's line reads "Has anyone actually *askedhe* believes the risks are": the pattern paired the closing mark of one emphasis span with the opening mark of the next and deleted the text between. This was found by reading the transcript, and the judge did not notice it. It was subsequently fixed by requiring the opening mark to follow a non-word character and the closing mark to precede one, with a regression test using the exact corrupted line. The saved transcript predates the fix and is kept unedited as evidence. The remaining limitation is that any regex over free text can misfire on shapes I have not seen.

**F5. Stateless orchestrator created duplicates.** After a restart Claude did not know existing personas and created new ones, making a name ambiguous (two wasted calls and two stray files). *Fix:* roster injected into the system prompt on every round. The orchestrator still has no memory of what it did in earlier sessions; the roster supplies the state that matters.

**F6. User-specified names were dropped.** "Elena" came back as "Marisa Kowalski"; "Ada" became "Priya Desai", then "Ada Chen". The tool layer passed the user's words through correctly; the loss happened inside `generate_persona`, whose author prompt asked for "a realistic full name". Moving the rule into prose ("use exactly that name; add a surname only if none was given") fixed the first-name case but still adds invented surnames. Identifying *which layer* lost the information was the useful part.

**F7. Claude's final summaries sometimes misdescribe tool results.** Examples: "I've created Ada" when the tool created Priya Desai; "the system already created Marcus Webb" when `create_persona` had just run; a gloss about "analytical skepticism" that was not in any result; unprompted advice to "remind Elena to stick to pure dialogue", which the app cannot act on. These are harmless here but would matter if the summary drove decisions. The events line printed from the actual tool result is the reliable record.

**F8. The judge has false negatives.** In `before_scene_fix`, the judge flagged Ada but described Elena as staying in character even though she says "I'm here in character, ready to roleplay". It also missed the "askedhe" corruption (F4) and cannot see repetition or in-world impossibilities (Derek's repeated restatements; Marcus promising a link). The judge is the same model family as the generator, so it may be lenient toward its own style; I did not test a stronger judge model.

**F9. Favourite names.** Haiku produced "Marcus Webb" in three separate sessions. The avoid-list only covers names currently on disk, so after clearing `temp/` it has nothing to avoid. This supports keeping the code-level uniqueness guard.

**F10. Cost.** Calls per action (live mode):

| Action | Calls |
|---|---|
| Create a persona | 1 (persona writing) + orchestrator rounds (usually 2: choose tool, summarise) |
| Conversation of N turns | N + orchestrator rounds (usually 2) |
| Evaluate a conversation | 1 (judge) + orchestrator rounds (usually 2) |

A six-turn conversation after two persona creations is roughly 6 + 2 + (2 x 3) = 14 calls on Haiku. Two further costs: every turn resends the full transcript (agent task 2 in the skeleton, unchanged), and `tool_history` grows without limit within a session. `MAX_TURNS` and `MAX_TOOL_ROUNDS` bound the worst case. I used Haiku throughout while iterating; the default was changed from Opus for this reason.

**F11. Provenance gap.** The first "before" run (Marcus) was not saved as a file; it was captured from the terminal afterwards and is labelled as such. The regex path is now bypassed for the live model, so that run cannot be regenerated without disabling the `hasattr` check.

---

## 8. Test coverage

**84 tests**, all offline and free, run in well under a second. Every model call in the tests goes through a scripted fake.

| File | Tests | What it protects |
|---|---|---|
| `test_agent.py` | 10 | Persona reaches the prompt; transcript is rewritten from the agent's point of view; narration rule is present; filter strips narration, keeps single-word emphasis, and does not eat text between emphasis marks |
| `test_conversation.py` | 27 | Turn loop; transcripts (unique names, not read as personas); regex-path routing and ambiguity checks; tool-loop routing (plain answer, tool result carries the right id, errors fed back, runaway loop stopped); per-turn failure handling; roster reaches the prompt |
| `test_llm.py` | 7 | Offline stubs and key-free construction |
| `test_observer.py` | 8 | Clean, fenced, and chatty judge replies; garbage reply does not raise; missing fields are unknown, not false; no call for an empty conversation |
| `test_persona.py` | 18 | Slugs, round-trip, parser regressions seen in live runs, non-overwriting save, unique names, avoid-list reaches the prompt |
| `test_tools.py` | 14 | Fake model replays in order; every tool has a schema; each tool's happy path and errors; turn cap; the evaluate tool cannot escape the transcripts folder and never turns an unreadable verdict into a pass |

**Method.** Where possible a bug was reproduced as a failing test before the fix. For example, the duplicate-save test failed against the old `save_persona` and passed after. Two of my own tests caught real mistakes during development (a missing import surfaced as an `Error: NameError` string, and the offline-path ambiguity test initially failed because the edit had not landed).

**Not covered by tests:** real-model behaviour (character, narration, name handling). Those are covered by the saved transcripts and the judge, not by assertions. The narration filter's behaviour on unseen shapes, and prompt-level behaviour generally, remain empirical.

---

## 9. Limitations and remaining work

- **Skeleton limitations I did not address:** no agent memory across conversations; round-robin turn-taking with a fixed turn count; no mid-conversation persona edits; `find_persona` substring matching remains for the legacy lookup (the new `find_personas` reports ambiguity instead).
- **The regex fallback is a deliberate trade.** Offline mode keeps the old router so the project runs free with no key. It now has the same ambiguity protection but not the flexibility of tool use.
- **Single-sample evidence.** Section 6 reports one to three runs per condition. A proper evaluation would run each condition many times and log raw pre-filter replies.
- **Judge validity.** Same model family as the generator, and spot-checks found false negatives. A stronger judge model, or a second judge, would be the next experiment.
- **Persona folder is relative to the working directory** (`temp/` by default), so launching from another folder shows an empty roster. `--persona-dir` overrides it.
- **No public deployment** (the optional bonus) was attempted.

---

## 10. Rubric map

| Rubric item | Where |
|---|---|
| Clean `uv sync && uv run main.py` | Section 2 (fresh-clone test) |
| Substantive extension | Sections 3.2, 3.3 |
| Own tests, passing | Section 8 |
| Coherent package structure | Section 3.1; logic lives in `tools.py`, `observer.py`, `orchestrator.py`, not `main.py` |
| Design write-up | Section 3 |
| Saved transcript with commentary | Sections 4 and 5 |
| Honest failure analysis | Sections 6 and 7 |
| Test coverage | Section 8 |
| Feature completeness against stated scope | Section 1 |
| Code quality | Typed signatures, docstrings stating intent, no logic in `main.py`, errors that say what to do next |
| Robustness | Section 3.4 and failure entries F1, F5, F11 |
| Novelty | Observer agent that automates the failure checks found by hand, run on the project's own transcripts to produce before/after measurements (section 6) |
| Live demo | `uv run main.py` with the sessions in section 2 |