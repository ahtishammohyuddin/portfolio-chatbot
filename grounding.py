"""
grounding.py - check that a model's answer cites the passages it was given.

What this does: the model is asked to reply with JSON naming the numbered
passages it used and quoting them. We then check, with plain code, that every
cited quote really appears word for word in a passage retrieved for THIS
question. If anything is off, the visitor gets the usual safe refusal.

What this does NOT do:
  * It does not check that the answer follows from the quotes. A model can
    quote something true and still say something the quote does not support.
    This is quote verification, not fact verification.
  * It does not stop prompt injection. If a passage contains instructions, the
    model can follow them and still quote them, and the quote is valid.

Standard library only. No Streamlit, no network, no model calls: the model is
passed in as a function, so everything here can be tested offline.

How a quote is matched to a passage (the policy; see "quote matching" below):
  * both sides are normalised the same way: NFKC, case-folded, whitespace
    collapsed, punctuation kept;
  * ONE LINE: a quote must lie inside a single line of the passage, so two list
    items or two lines of a wrapped paragraph can never be merged into one quote;
  * WHOLE WORDS: a quote cut out of the middle of a word is refused;
  * MEANINGFUL: at least two real words (letters or digits), so punctuation-only
    or single-letter "quotes" are refused;
  * what visitors see is the passage's own wording, never the model's text.

GROUNDING_MODE (see parse_mode): "off" keeps the original answer path exactly;
"on" uses the validated path. Anything other than "on" means "off".
"""

import json
import re
import unicodedata
from dataclasses import dataclass

from rag import NO_ANSWER, SYSTEM_PROMPT, build_prompt

# ---------------------------------------------------------------- settings
# Easy to adjust. The tests import these, so they follow any change here.

MIN_QUOTE_CHARS = 12      # a quote must be at least this long once normalised...
MIN_QUOTE_WORDS = 2       # ...and have at least this many words made of letters/digits...
MIN_WORD_CHARS = 2        # ...of at least this many characters each (so "a b c d" is not a quote)
MAX_ANSWER_CHARS = 1200   # longer answers are refused
MAX_CITATIONS = 6         # more citations than this are refused
MAX_RAW_CHARS = 20_000    # model output longer than this is not even parsed

MODE_ON = "on"
MODE_OFF = "off"

# Texts shown to the visitor. Unchanged from the original app.py wording.
RETRIEVAL_ONLY_TEXT = "Closest matches from his notes are shown below."
UNAVAILABLE_TEXT = "Sorry, the AI service is unavailable right now. Please try again shortly."

# Why an answer was refused. Fixed identifiers only: they never carry passage
# text or model text, so they are safe to keep in tests and logs.
R_MALFORMED = "malformed_json"
R_WRONG_TYPES = "wrong_types"
R_MODEL_REFUSED = "model_refused"
R_BLANK = "blank_answer"
R_TOO_LONG = "answer_too_long"
R_NO_CITATIONS = "no_citations"
R_TOO_MANY = "too_many_citations"
R_BAD_REF = "bad_reference"
R_SHORT_QUOTE = "quote_too_short"
R_NOT_FOUND = "quote_not_found"
R_PARTIAL = "quote_not_whole_words"
R_REFUSAL_WORDING = "refusal_wording"
R_BAD_UNICODE = "invalid_unicode"
R_INTERNAL = "internal_error"
REASONS = (R_MALFORMED, R_WRONG_TYPES, R_MODEL_REFUSED, R_BLANK, R_TOO_LONG, R_NO_CITATIONS,
           R_TOO_MANY, R_BAD_REF, R_SHORT_QUOTE, R_NOT_FOUND, R_PARTIAL, R_REFUSAL_WORDING, R_BAD_UNICODE, R_INTERNAL)

# ------------------------------------------------------------------ prompt

GROUNDING_RULES = """\

These rules replace the earlier instruction about how to word a refusal. Never write a \
refusal sentence (such as "I don't have that information") in "answer". If the passages \
do not contain the answer, set "answered" to false and leave "answer" empty: the app \
writes the refusal. "answered": true is only for answers that state facts from the \
passages and cite them.

Reply with one JSON object and nothing else, in exactly this shape:
{"answered": true, "answer": "<your short answer>", \
"citations": [{"ref": 1, "quote": "<words copied exactly from passage [1]>"}]}
- "ref" is the number of the passage in CONTEXT that the quote comes from.
- Each quote must be copied exactly, word for word, from ONE line of that passage's \
text (never join two lines or list items), starting and ending on whole words, \
at least a few words long. Never quote the passage's heading line.
- Cite at least one passage for every answer.
- If the passages do not contain the answer, reply \
{"answered": false, "answer": "", "citations": []}.
"""

# The original prompt is kept as is; the grounded one is that prompt plus the rules.
GROUNDED_SYSTEM_PROMPT = SYSTEM_PROMPT + GROUNDING_RULES


# --------------------------------------------------------------- structures


@dataclass(frozen=True)
class Citation:
    """A citation as the model wrote it (not yet checked)."""
    ref: int
    quote: str


@dataclass(frozen=True)
class GroundedAnswer:
    """The model's reply, parsed and type-checked but not yet validated."""
    answered: bool
    text: str
    citations: tuple


@dataclass(frozen=True)
class VerifiedCitation:
    """A citation whose quote was found in the passage it names.

    quote is the PASSAGE's own wording for the matched span (original case,
    spacing and characters), not the text the model wrote.
    """
    ref: int
    heading: str
    quote: str


@dataclass(frozen=True)
class Validated:
    ok: bool
    text: str                 # the answer if ok, otherwise NO_ANSWER
    citations: tuple = ()     # VerifiedCitation items, only when ok
    reason: str = None        # one of REASONS when not ok


@dataclass(frozen=True)
class Outcome:
    """What the app should show for one question."""
    text: str
    citations: tuple = ()
    plain: bool = False       # True: show text as plain text, never as Markdown/HTML
    reason: str = None
    kind: str = ""


class GroundingError(Exception):
    """Carries a reason code only. Never put passage or model text in here."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------- normalising


def normalize(text: str) -> str:
    """Unicode NFKC, case-folded, whitespace collapsed to single spaces.

    Punctuation is kept on purpose: "Python, SQL" and "Python SQL" differ.
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _is_valid_utf8(text: str) -> bool:
    """True if text can be encoded as strict UTF-8.

    JSON can carry half of a UTF-16 pair (an escape such as \\ud800), which Python keeps as
    a str that cannot be encoded. Drawing such text, or keeping it in the session,
    could crash the page, so it is refused before it can be accepted or shown.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _body(chunk) -> str:
    """The passage text without its heading line (chunk.text is 'heading\\nbody')."""
    return chunk.text.partition("\n")[2]


# ------------------------------------------------------- refusal wording

# An "answered" reply must not be a refusal in disguise: a refusal sentence plus any
# real quote would otherwise pass validation and be shown as a sourced answer.
# The rule is deliberately narrow so that factual answers are not refused for
# containing a negation: only the START of the answer is checked, against fixed
# phrases of refusal wording. Limits: English only, start of the answer only, fixed
# phrases only. A refusal worded differently, or placed after a factual first
# sentence, is not detected.
_REFUSAL_STARTS = (
    "i don't have", "i do not have", "i don't know", "i do not know",
    "i have no information", "i have no record", "i can't", "i cannot",
    "i couldn't find", "i could not find", "i am not able", "i'm not able",
    "i am unable", "i'm unable", "i am not sure", "i'm not sure",
    "i'm sorry", "i am sorry", "sorry,", "sorry.",
    "there is no information", "there's no information",
    "no information about", "no information on", "no information is",
    "no information available", "no information in", "no information regarding",
    "the notes do not", "the notes don't", "his notes do not", "his notes don't",
    "the passages do not", "the passages don't", "nothing in the notes",
    "nothing in his notes", "nothing in the passages",
    "this is not mentioned", "that is not mentioned", "that's not mentioned",
    "it is not mentioned", "it's not mentioned",
    "it's best to contact", "it is best to contact", "please contact",
)


def looks_like_refusal(text: str) -> bool:
    """True if the answer OPENS with refusal wording (see _REFUSAL_STARTS)."""
    t = unicodedata.normalize("NFKC", text).casefold()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = " ".join(t.split()).lstrip(" \"'([{*_-.,:;!?>\u201c\u201d")  # leading quotes, brackets, Markdown
    return t.startswith(_REFUSAL_STARTS)


# ----------------------------------------------------------------- parsing

_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(.*)\r?\n```\Z", re.DOTALL | re.IGNORECASE)


def parse_grounded_answer(raw) -> GroundedAnswer:
    """Turn model output into a GroundedAnswer or raise GroundingError(reason).

    Accepts one JSON object, optionally wrapped in a Markdown code fence.
    Nothing is repaired: prose around the JSON is malformed output.
    """
    if not isinstance(raw, str) or len(raw) > MAX_RAW_CHARS:
        raise GroundingError(R_MALFORMED)
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        raise GroundingError(R_MALFORMED) from None
    if not isinstance(data, dict):
        raise GroundingError(R_MALFORMED)

    answered = data.get("answered")
    if not isinstance(answered, bool):
        raise GroundingError(R_WRONG_TYPES)
    if not answered:  # an explicit refusal needs nothing else
        return GroundedAnswer(False, "", ())

    answer, citations = data.get("answer"), data.get("citations")
    if not isinstance(answer, str) or not isinstance(citations, list):
        raise GroundingError(R_WRONG_TYPES)
    parsed = []
    for item in citations:
        if not isinstance(item, dict):
            raise GroundingError(R_WRONG_TYPES)
        ref, quote = item.get("ref"), item.get("quote")
        if isinstance(ref, bool) or not isinstance(ref, int) or not isinstance(quote, str):
            raise GroundingError(R_WRONG_TYPES)
        parsed.append(Citation(ref, quote))
    return GroundedAnswer(True, answer, tuple(parsed))


# ----------------------------------------------------------- quote matching
#
# Policy (all deterministic):
#   1. NORMALISE both sides the same way (normalize(): NFKC, case-folded,
#      whitespace collapsed, punctuation kept).
#   2. ONE LINE: the quote must lie inside a single line of the passage body. It
#      is never matched across a line break, so two list items, or two lines of a
#      wrapped paragraph, cannot be merged into one quote. (If the notes wrap
#      paragraphs over several lines, a quote that crosses a wrap is refused;
#      relax this knowingly.) A line break inside the MODEL's quote is only
#      whitespace and is fine.
#   3. WHOLE WORDS: if the quote starts (ends) with a letter or digit, the passage
#      character just before (after) it must not be a letter or digit. A quote cut
#      from the middle of a word is refused. Edges that are punctuation are not
#      checked, and a hyphen counts as a word break.
#   4. MEANINGFUL: at least MIN_QUOTE_CHARS normalised characters and at least
#      MIN_QUOTE_WORDS words (runs of letters/digits, any script) of
#      MIN_WORD_CHARS or more characters. Punctuation-only and single-letter
#      "quotes" are refused.
#   5. SHOWN: visitors see the passage's own wording for the matched span.
# Limits: text written without spaces (such as Chinese) has no words by this
# definition and is refused. The span is found with a per-character-cluster
# normaliser that agrees with normalize() on ordinary text; where they could
# differ (normalisation that composes characters across a combining-mark
# boundary) the quote simply fails to match, which is the safe direction.

_WORD = re.compile(r"[^\W_]+")  # runs of letters and digits
_FOUND, _MISSING, _PARTIAL = "found", "missing", "partial"


def _meaningful_words(normalized_quote: str) -> list:
    return [w for w in _WORD.findall(normalized_quote) if len(w) >= MIN_WORD_CHARS]


def _normalize_with_spans(text: str):
    """normalize(text) plus, for each output character, the (start, end) in text of
    the character cluster (a character and its combining marks) it came from.

    Returns (normalized, spans) with len(spans) == len(normalized). Used to point a
    match back at the passage's original wording.
    """
    out, spans = [], []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.category(text[j]).startswith("M"):
            j += 1
        for ch in unicodedata.normalize("NFKC", text[i:j]).casefold():
            if ch.isspace():
                if out and out[-1] != " ":
                    out.append(" ")
                    spans.append((i, j))
            else:
                out.append(ch)
                spans.append((i, j))
        i = j
    if out and out[-1] == " ":
        out.pop()
        spans.pop()
    return "".join(out), spans


def _on_word_boundaries(line: str, start: int, end: int) -> bool:
    if line[start].isalnum() and start > 0 and line[start - 1].isalnum():
        return False
    if line[end - 1].isalnum() and end < len(line) and line[end].isalnum():
        return False
    return True


def _locate(quote: str, chunk):
    """Find a normalised quote on one line of a passage body.

    Returns (status, passage_wording, key): status is "found", "missing" (not on
    any single line) or "partial" (only inside words); passage_wording is the
    original text of the matched span; key identifies the span, for de-duplication.
    """
    cut_from_a_word = False
    for index, line in enumerate(_body(chunk).splitlines()):
        norm, spans = _normalize_with_spans(line)
        at = norm.find(quote)
        while at != -1:
            end = at + len(quote)
            if _on_word_boundaries(norm, at, end):
                first, last = spans[at][0], spans[end - 1][1]
                return _FOUND, line[first:last].strip(), (index, first, last)
            cut_from_a_word = True
            at = norm.find(quote, at + 1)
    return (_PARTIAL if cut_from_a_word else _MISSING), None, None


# -------------------------------------------------------------- validation


def _refuse(reason: str) -> Validated:
    return Validated(ok=False, text=NO_ANSWER, citations=(), reason=reason)


def _verify(citation: Citation, results):
    """Return (reason, None) if the citation is not acceptable, else (None, (wording, key))."""
    if not 1 <= citation.ref <= len(results):  # only passages retrieved for THIS question
        return R_BAD_REF, None
    quote = normalize(citation.quote)
    if len(quote) < MIN_QUOTE_CHARS or len(_meaningful_words(quote)) < MIN_QUOTE_WORDS:
        return R_SHORT_QUOTE, None
    status, wording, key = _locate(quote, results[citation.ref - 1][0])
    if status == _MISSING:
        return R_NOT_FOUND, None
    if status == _PARTIAL:
        return R_PARTIAL, None
    return None, (wording, (citation.ref,) + key)


def validate(answer: GroundedAnswer, results, *, strict: bool = True) -> Validated:
    """Check a parsed answer against the passages retrieved for this question.

    results is the same (chunk, score) list that was numbered in the prompt, so
    a citation's ref is a 1-based position in it.

    strict=True: one bad citation refuses the whole answer. strict=False: bad
    citations are dropped, and at least one good one must remain.
    """
    if not answer.answered:
        return _refuse(R_MODEL_REFUSED)
    # Before anything else, and regardless of `strict`: the answer and every quote must be
    # valid UTF-8. A reply with a broken string is not trusted at all.
    if not _is_valid_utf8(answer.text) or not all(_is_valid_utf8(c.quote) for c in answer.citations):
        return _refuse(R_BAD_UNICODE)
    text = answer.text.strip()
    if not text:
        return _refuse(R_BLANK)
    if len(text) > MAX_ANSWER_CHARS:
        return _refuse(R_TOO_LONG)
    if looks_like_refusal(text):  # a refusal must come as answered=false, never as an answer
        return _refuse(R_REFUSAL_WORDING)
    if not answer.citations:
        return _refuse(R_NO_CITATIONS)
    if len(answer.citations) > MAX_CITATIONS:
        return _refuse(R_TOO_MANY)

    verified, seen, last_problem = [], set(), None
    for citation in answer.citations:
        problem, found = _verify(citation, results)
        if problem:
            if strict:
                return _refuse(problem)
            last_problem = problem
            continue
        wording, key = found
        if key in seen:  # the same span of the same passage, however the model wrote it
            continue
        seen.add(key)
        verified.append(VerifiedCitation(citation.ref, results[citation.ref - 1][0].heading, wording))
    if not verified:
        return _refuse(last_problem or R_NO_CITATIONS)
    return Validated(ok=True, text=text, citations=tuple(verified))


def validate_response(raw, results, *, strict: bool = True) -> Validated:
    """parse_grounded_answer + validate. Never raises for bad model output."""
    try:
        answer = parse_grounded_answer(raw)
    except GroundingError as err:
        return _refuse(err.reason)
    return validate(answer, results, strict=strict)


# ------------------------------------------------------------ orchestration


def parse_mode(value=None) -> str:
    """'on' (any case, surrounding spaces ok) turns grounding on; anything else is off."""
    if isinstance(value, str) and value.strip().lower() == MODE_ON:
        return MODE_ON
    return MODE_OFF


def answer_from_results(question, results, ask_fn, *, has_key, mode=MODE_OFF) -> Outcome:
    """Decide what to show for one question, given the retrieved passages.

    ask_fn(prompt, system_prompt) -> str is the model call, passed in so this
    can be tested without Gemini. Behaviour that is the same in both modes:
    no passages -> the standard refusal, no model call; no API key ->
    retrieval-only message, no model call; model exception -> "unavailable".

    mode "off": the original path, byte for byte (original prompt, reply shown as is).
    mode "on":  grounded prompt, reply validated; any failure -> the standard refusal.
    """
    if not results:
        return Outcome(NO_ANSWER, kind="no_passages")
    if not has_key:
        return Outcome(RETRIEVAL_ONLY_TEXT, kind="retrieval_only")

    # build_prompt sits inside the same try as the model call, as in the original
    # app.py: any failure on the way to the model is "unavailable", never a crash.
    if mode != MODE_ON:
        try:
            text = ask_fn(build_prompt(question, results), SYSTEM_PROMPT) or NO_ANSWER
        except Exception:
            return Outcome(UNAVAILABLE_TEXT, kind="unavailable")
        return Outcome(text, kind="answered_ungrounded")

    try:
        raw = ask_fn(build_prompt(question, results), GROUNDED_SYSTEM_PROMPT)
    except Exception:
        return Outcome(UNAVAILABLE_TEXT, kind="unavailable")
    try:
        checked = validate_response(raw, results)
    except Exception:  # a bug here must fail closed, never show unchecked text
        checked = _refuse(R_INTERNAL)
    if checked.ok:
        return Outcome(checked.text, checked.citations, plain=True, kind="answered_grounded")
    return Outcome(NO_ANSWER, reason=checked.reason, kind="refused")


def citation_lines(citations) -> list:
    """One display line per verified citation, e.g. [1] Education: "quote"."""
    return [f'[{c.ref}] {c.heading}: "{c.quote}"' for c in citations]
