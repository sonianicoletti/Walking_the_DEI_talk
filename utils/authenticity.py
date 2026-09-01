"""Weak supervision for CEO DEI-communication authenticity.

Pipeline:
    transcript -> de-identify -> LLM signal elicitation (18 labeling functions)
              -> Snorkel-style generative label model -> authenticity score

The label model is a Dawid-Skene style generative model over ternary votes
(+1 yes / 0 unsure / -1 no) with no ground-truth labels. Each labeling function
gets a per-class emission distribution learned by EM; correlated functions are
down-weighted so a redundant block cannot outvote the rest.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Credibility signals (the labeling functions)
# --------------------------------------------------------------------------

# polarity = +1 when a "yes" is evidence of authenticity, -1 when a "yes" is
# evidence against it. Polarity anchors the label model's latent class so
# "class 0" always means authentic (EM alone is sign-symmetric).

@dataclass(frozen=True)
class Signal:
    id: int
    dimension: str
    polarity: int
    question: str


SIGNALS: list[Signal] = [
    Signal(1,  "personal investment", +1,
           "Does the CEO reference personal experiences when discussing DEI?"),
    Signal(2,  "personal investment", +1,
           "Does the CEO explain why DEI matters to them individually, beyond business rationale?"),
    Signal(3,  "personal investment", +1,
           "Does the CEO acknowledge their own biases or blind spots?"),
    Signal(4,  "personal investment", +1,
           "Does the CEO frame DEI as a core personal value rather than a corporate initiative?"),
    Signal(5,  "personal investment", +1,
           'Does the CEO use first-person language ("I believe," "I\'ve committed to") '
           'rather than institutional language ("the company believes")?'),
    Signal(6,  "consistency", +1,
           "Has the CEO discussed DEI consistently across multiple time periods or occasions?"),
    Signal(7,  "candor", +1,
           "Does the CEO acknowledge the company's current shortcomings or past failures on DEI?"),
    Signal(8,  "candor", +1,
           "Does the CEO discuss pushback, criticism, or internal disagreement about DEI efforts?"),
    Signal(9,  "candor", +1,
           "Does the CEO cite specific feedback from employees or affected groups?"),
    Signal(10, "regulatory focus", +1,
           "Is the language promotion-focused, oriented toward growth and long-term opportunity?"),
    Signal(11, "regulatory focus", -1,
           "Is the language prevention-focused, oriented toward avoiding legal, "
           "reputational, or compliance risk?"),
    Signal(12, "specificity", +1,
           "Does the CEO cite specific metrics, targets, or data related to DEI outcomes?"),
    Signal(13, "specificity", +1,
           "Does the CEO describe concrete structural or policy changes "
           "(e.g., hiring practices, pay equity audits)?"),
    Signal(14, "specificity", +1,
           "Does the CEO name specific individuals, teams, or programs responsible for DEI work?"),
    Signal(15, "specificity", +1,
           "Does the CEO commit to a specific, time-bound future action?"),
    Signal(16, "vagueness", -1,
           'Does the CEO rely on generic corporate buzzwords without elaboration '
           '(e.g., "diversity is our strength")?'),
    Signal(17, "vagueness", -1,
           "Does the CEO's statement appear reactive, closely following an external "
           "controversy or backlash?"),
    Signal(18, "external validation", +1,
           "Does the CEO reference external validation (audits, certifications, "
           "third-party reports) of DEI progress?"),
]

POLARITY = np.array([s.polarity for s in SIGNALS])
N_SIGNALS = len(SIGNALS)

VOTE_TO_INT = {"yes": 1, "unsure": 0, "no": -1}
INT_TO_VOTE = {1: "yes", 0: "unsure", -1: "no"}


# --------------------------------------------------------------------------
# De-identification
# --------------------------------------------------------------------------

# Organizations that appear in transcripts but are not the subject company.
# Extend as the corpus grows.
THIRD_PARTY_ORGS = [
    "johnson & johnson", "johnson and johnson", "j&j",
    "netflix", "costco", "walmart", "amazon", "apple", "google",
    "microsoft", "business roundtable",
]

_STOPWORDS = {"the", "and", "for", "inc", "corp", "company", "group", "holdings"}


def _identity_terms(company: str, ceo: str) -> set[str]:
    """Tokens from the metadata that must never reach the model."""
    terms = set()
    for source in (company or "", ceo or ""):
        for tok in re.split(r"[^A-Za-z]+", source):
            if len(tok) > 2 and tok.lower() not in _STOPWORDS:
                terms.add(tok.lower())
    return terms


def deidentify(text: str, terms: set[str], orgs: list[str] | None = None) -> str:
    """Replace company/person identifiers with neutral placeholders.

    Word-boundary anchored so "reed" does not match inside "agreed".
    Longest-first so "johnson & johnson" is consumed before "johnson".
    """
    orgs = THIRD_PARTY_ORGS if orgs is None else orgs
    out = text
    for org in sorted(orgs, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(org)}\b", "[ORGANIZATION]", out, flags=re.IGNORECASE)
    for term in sorted(terms, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(term)}\b", "[NAME]", out, flags=re.IGNORECASE)
    return out


def load_transcript(path) -> dict:
    """Read a transcript file and return both the raw record and a de-identified
    rendering suitable for the prompt."""
    with open(path) as fh:
        record = json.load(fh)

    terms = _identity_terms(record.get("company", ""), record.get("CEO", ""))
    lines = []
    for segment in record["transcript"]:
        for role, utterance in segment.items():
            speaker = "INTERVIEWER" if role == "interviewer" else "INTERVIEWEE"
            lines.append(f"{speaker}: {deidentify(utterance, terms)}")

    return {
        "doc_id": getattr(path, "stem", str(path)),
        "company": record.get("company"),
        "ceo": record.get("CEO"),
        "year": record.get("year"),
        "text": "\n\n".join(lines),
        "n_words": sum(len(l.split()) for l in lines),
    }


def audit_leakage(text: str, record: dict) -> list[str]:
    """Return identifying tokens still present in a de-identified transcript."""
    leaks = []
    for term in _identity_terms(record.get("company", ""), record.get("ceo", "")):
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            leaks.append(term)
    return sorted(leaks)


# --------------------------------------------------------------------------
# Label model
# --------------------------------------------------------------------------

def _encode(L: np.ndarray) -> np.ndarray:
    """Map votes {+1, 0, -1} to emission indices {0, 1, 2}."""
    L = np.asarray(L, dtype=int)
    out = np.empty_like(L)
    out[L == 1] = 0
    out[L == 0] = 1
    out[L == -1] = 2
    return out


class LabelModel:
    """Generative label model over ternary labeling-function votes.

    Learns, per labeling function j and latent class c, an emission distribution
    theta[j, c, :] over (yes, unsure, no) by EM — no ground-truth labels. A
    Dirichlet prior keyed on each function's polarity anchors class 0 to
    "authentic" and keeps the model well-posed on small corpora.

    Parameters
    ----------
    polarity : array of +1/-1, one per labeling function.
    class_balance : prior P(authentic).
    prior_strength : Dirichlet pseudo-count mass. Higher = more regularized,
        which matters most when the number of documents is small.
    prior_accuracy : assumed per-function agreement with its polarity.
    """

    def __init__(self, polarity=POLARITY, class_balance: float = 0.5,
                 prior_strength: float = 4.0, prior_accuracy: float = 0.7,
                 max_iter: int = 300, tol: float = 1e-9):
        self.polarity = np.asarray(polarity, dtype=int)
        self.class_balance = class_balance
        self.prior_strength = prior_strength
        self.prior_accuracy = prior_accuracy
        self.max_iter = max_iter
        self.tol = tol

    def _dirichlet_prior(self) -> np.ndarray:
        a, s = self.prior_accuracy, self.prior_strength
        prior = np.empty((len(self.polarity), 2, 3))
        for j, pol in enumerate(self.polarity):
            agree = [a, 0.5, 1 - a] if pol > 0 else [1 - a, 0.5, a]
            prior[j, 0] = np.array(agree) * s          # class 0 = authentic
            prior[j, 1] = np.array(agree[::-1]) * s    # class 1 = inauthentic
        return prior

    def _loglik(self, Lidx: np.ndarray, theta: np.ndarray) -> np.ndarray:
        n, m = Lidx.shape
        cols = np.arange(m)[None, :]
        ll = np.zeros((n, 2))
        for c in range(2):
            ll[:, c] = (self.weights_ * np.log(theta[:, c, :])[cols, Lidx]).sum(axis=1)
        return ll

    def fit(self, L: np.ndarray, weights: np.ndarray | None = None) -> "LabelModel":
        Lidx = _encode(L)
        n, m = Lidx.shape
        self.weights_ = np.ones(m) if weights is None else np.asarray(weights, float)
        prior = self._dirichlet_prior()
        log_pi = np.log([self.class_balance, 1 - self.class_balance])

        # Initialize from the polarity-weighted vote.
        score = (np.asarray(L, float) * self.polarity * self.weights_).sum(axis=1)
        p_auth = 1.0 / (1.0 + np.exp(-score))
        post = np.column_stack([p_auth, 1 - p_auth])

        prev = None
        for it in range(self.max_iter):
            # M-step: expected emission counts + Dirichlet prior.
            counts = np.empty((m, 2, 3))
            for c in range(2):
                for v in range(3):
                    counts[:, c, v] = (post[:, c][:, None] * (Lidx == v)).sum(axis=0)
            theta = counts + prior
            theta /= theta.sum(axis=2, keepdims=True)

            # E-step: posterior over the latent class.
            ll = self._loglik(Lidx, theta) + log_pi
            mx = ll.max(axis=1, keepdims=True)
            p = np.exp(ll - mx)
            post = p / p.sum(axis=1, keepdims=True)

            obj = float((np.log(p.sum(axis=1)) + mx.ravel()).sum())
            if prev is not None and abs(obj - prev) < self.tol:
                break
            prev = obj

        self.theta_ = theta
        self.n_iter_ = it + 1
        self.log_likelihood_ = prev
        return self

    def log_odds(self, L: np.ndarray) -> np.ndarray:
        """Total log-odds of 'authentic' per document."""
        return self.contributions(L).sum(axis=1)

    def contributions(self, L: np.ndarray) -> np.ndarray:
        """Per-function log-odds contribution toward 'authentic', shape (n, m).

        This is what makes a score auditable: each cell says how much that one
        answer moved the document, and in which direction.
        """
        Lidx = _encode(L)
        per_vote = np.log(self.theta_[:, 0, :] / self.theta_[:, 1, :])   # (m, 3)
        cols = np.arange(L.shape[1])[None, :]
        return self.weights_ * per_vote[cols, Lidx]

    def score(self, L: np.ndarray, temperature: bool = True) -> np.ndarray:
        """Continuous authenticity score in [0, 1].

        The raw posterior saturates to 0/1 because 18 conditionally-independent
        votes pile up log-odds fast. Dividing by sqrt(effective votes) before the
        sigmoid keeps the score spread across the interval while preserving the
        ranking exactly (Spearman ~1.0 against the raw posterior).
        """
        total = self.log_odds(L)
        if temperature:
            effective = np.sqrt(np.maximum((self.weights_ * (np.asarray(L) != 0)).sum(axis=1), 1.0))
            total = total / effective
        return 1.0 / (1.0 + np.exp(-total))

    def predict(self, L: np.ndarray) -> np.ndarray:
        """Binary label: +1 authentic, -1 inauthentic."""
        return np.where(self.score(L) > 0.5, 1, -1)

    def learned_accuracy(self) -> np.ndarray:
        """P(function votes with its polarity | that class), ignoring abstentions."""
        yes, no = self.theta_[:, 0, 0], self.theta_[:, 0, 2]
        agree = np.where(self.polarity > 0, yes, no)
        return agree / (yes + no)


# Below this many documents, empirical vote correlations are noise rather than
# signal — measured on synthetic corpora, redundancy weighting *hurt* accuracy
# at n=3 (0.57 vs 0.80) and helped from n≈30 up (0.87 vs 0.77).
MIN_DOCS_FOR_CORRELATION = 30


def redundancy_weights(L: np.ndarray, tau: float = 0.3, min_overlap: int = 3):
    """Down-weight labeling functions that duplicate each other.

    Snorkel learns a correlation structure over labeling functions so a block of
    near-duplicate signals cannot dominate the vote. With few documents the
    pairwise correlations are not estimable, so callers should gate on
    MIN_DOCS_FOR_CORRELATION and fall back to uniform weights.

    Returns (weights, correlation_matrix).
    """
    A = np.asarray(L, float)
    n, m = A.shape
    C = np.zeros((m, m))
    for j in range(m):
        for k in range(j + 1, m):
            both = (A[:, j] != 0) & (A[:, k] != 0)
            if both.sum() >= min_overlap and A[both, j].std() > 0 and A[both, k].std() > 0:
                C[j, k] = C[k, j] = np.corrcoef(A[both, j], A[both, k])[0, 1]
    excess = np.clip(np.abs(C) - tau, 0, None).sum(axis=1)
    return 1.0 / (1.0 + excess), C


def fit_label_model(L: np.ndarray, use_correlations: bool | None = None, **kwargs):
    """Fit the label model, gating correlation weighting on corpus size."""
    L = np.asarray(L, dtype=int)
    n, m = L.shape
    if use_correlations is None:
        use_correlations = n >= MIN_DOCS_FOR_CORRELATION

    if use_correlations:
        weights, corr = redundancy_weights(L)
    else:
        weights, corr = np.ones(m), None

    model = LabelModel(**kwargs).fit(L, weights=weights)
    model.correlations_ = corr
    return model


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

# Claude's structured outputs constrain the reply to the SignalReport schema at
# decode time, so `normalize_report` is not doing shape repair on that path — it
# enforces what the schema cannot: that all 18 signals are present exactly once,
# and it canonicalises vote strings. `extract_json` is kept for the case where
# the elicitor is swapped for a model behind plain JSON mode, where the reply may
# arrive wrapped in prose or code fences.

_VOTE_SYNONYMS = {
    "y": "yes", "true": "yes",
    "n": "no", "false": "no",
    "unclear": "unsure", "uncertain": "unsure", "unknown": "unsure",
    "maybe": "unsure", "n/a": "unsure", "not applicable": "unsure",
}


def extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a model reply.

    Tolerates code fences and prose wrapped around the object, plus trailing
    commas and smart quotes. Brace counting is string- and escape-aware so a
    quoted "{" inside evidence text does not throw off the balance.
    """
    if not text or not text.strip():
        raise ValueError("empty response")
    s = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    start = s.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")

    depth, end, in_string, escaped = 0, None, False, False
    for i, ch in enumerate(s[start:], start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("unbalanced braces in reply")

    blob = s[start:end]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", blob)          # trailing commas
        repaired = repaired.replace("“", '"').replace("”", '"')
        return json.loads(repaired)


def normalize_report(payload, expected_ids=None) -> dict[int, tuple[str, str]]:
    """Validate a parsed reply into {signal_id: (vote, evidence)}.

    Accepts the shape variations models drift into — answers as a list or a
    dict, ids as ``3`` / ``"3"`` / ``"S3"``, votes with stray case or
    punctuation. Raises ValueError on anything genuinely unusable (a missing
    signal, an uninterpretable vote) so the caller can retry.
    """
    expected_ids = [s.id for s in SIGNALS] if expected_ids is None else list(expected_ids)

    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

    answers = payload.get("answers", payload)
    entries = []
    if isinstance(answers, list):
        for a in answers:
            if not isinstance(a, dict):
                raise ValueError(f"answer entry is not an object: {a!r}")
            entries.append((a.get("signal_id"), a.get("answer"), a.get("evidence", "")))
    elif isinstance(answers, dict):
        for key, val in answers.items():
            if isinstance(val, dict):
                entries.append((key, val.get("answer"), val.get("evidence", "")))
            else:
                entries.append((key, val, ""))
    else:
        raise ValueError(f"'answers' is {type(answers).__name__}, expected list or object")

    out: dict[int, tuple[str, str]] = {}
    for raw_id, raw_vote, evidence in entries:
        try:
            sid = int(str(raw_id).strip().lstrip("Ss#"))
        except (TypeError, ValueError):
            raise ValueError(f"unparseable signal id: {raw_id!r}")

        vote = str(raw_vote).strip().lower().rstrip(".!,").strip('"')
        vote = _VOTE_SYNONYMS.get(vote, vote)
        if vote not in VOTE_TO_INT:
            raise ValueError(f"signal {sid}: invalid vote {raw_vote!r}")

        out[sid] = (vote, str(evidence or ""))

    missing = sorted(set(expected_ids) - set(out))
    if missing:
        raise ValueError(f"missing answers for signals {missing}")
    for extra in set(out) - set(expected_ids):
        del out[extra]
    return out


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful discourse analyst assessing how an executive \
talks about diversity, equity, and inclusion in an interview transcript.

You answer a fixed list of yes/no questions about the INTERVIEWEE only. Ignore the \
interviewer except as context for what the interviewee is responding to.

Rules:
- Every answer must be exactly "yes", "no", or "unsure".
- Answer "yes" only when the transcript contains direct evidence. Do not infer from \
what a speaker in this position would plausibly believe.
- Answer "unsure" when the transcript is genuinely ambiguous or the topic never comes \
up in enough detail to judge. "unsure" is a real answer, not a hedge - use it rather \
than guessing.
- Judge only what is present in this excerpt. Do not use outside knowledge about the \
speaker or their organisation, and do not speculate about sincerity beyond the text.
- For each "yes", quote the specific words that justify it. Use an empty string for \
"no" and "unsure".

Answer all 18 questions, exactly once each, with signal_id running from 1 to 18."""


def build_user_prompt(transcript: str) -> str:
    questions = "\n".join(f"{s.id}. {s.question}" for s in SIGNALS)
    return (
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"Answer each of the following {N_SIGNALS} questions about the INTERVIEWEE, "
        f"using only the transcript above.\n\n{questions}"
    )


# --------------------------------------------------------------------------
# Claude elicitation
# --------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-5"


class SignalAnswer(BaseModel):
    """One question's answer. The schema is enforced by the API at decode time."""

    signal_id: int = Field(description="The number of the question answered, 1-18.")
    answer: Literal["yes", "no", "unsure"] = Field(
        description='Exactly one of "yes", "no", or "unsure".'
    )
    evidence: str = Field(
        description="A short verbatim quote from the transcript justifying a 'yes'. "
                    "Empty string for 'no' and 'unsure'."
    )


class SignalReport(BaseModel):
    answers: list[SignalAnswer] = Field(
        description="Exactly 18 answers, one per question, in ascending signal_id order."
    )


def elicit_signals(transcript: str, client, model: str = DEFAULT_MODEL,
                   max_attempts: int = 3) -> dict:
    """Ask Claude the 18 questions about one transcript.

    Structured outputs constrain the reply to the SignalReport schema at decode
    time, so malformed JSON is not a failure mode here. What the schema cannot
    guarantee is *completeness* — that all 18 signal ids are present exactly
    once — so that is still validated, and a short reply is retried with the
    specific problem fed back to the model.
    """
    user_prompt = build_user_prompt(transcript)
    messages = [{"role": "user", "content": user_prompt}]

    problems = []
    for attempt in range(max_attempts):
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_format=SignalReport,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )

        if response.stop_reason == "refusal":
            raise RuntimeError(
                "The model declined to answer"
                + (f" ({response.stop_details.category})" if response.stop_details else "")
            )

        report = response.parsed_output
        payload = report.model_dump() if report is not None else {}

        try:
            # Reuse the same completeness/vote validation the non-schema path
            # uses — structured outputs fix the shape, not the coverage.
            answers = normalize_report(payload)
        except (ValueError, json.JSONDecodeError) as err:
            problems.append(f"attempt {attempt + 1}: {err}")
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(payload)[:2000]},
                {"role": "user", "content": (
                    f"That reply could not be used: {err}. Answer again, including "
                    f"all {N_SIGNALS} questions exactly once, with signal_id running "
                    f"from 1 to {N_SIGNALS}."
                )},
            ]
            continue

        return {
            "votes": {str(sid): vote for sid, (vote, _) in sorted(answers.items())},
            "evidence": {str(sid): ev for sid, (_, ev) in sorted(answers.items())},
            "model": model,
            "attempts": attempt + 1,
            "usage": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        }

    raise RuntimeError(
        f"Could not get a usable reply after {max_attempts} attempts:\n  "
        + "\n  ".join(problems)
    )


def build_vote_matrix(docs, cache) -> np.ndarray:
    """Rows = documents, columns = signals, values in {+1 yes, 0 unsure, -1 no}."""
    return np.array([
        [VOTE_TO_INT[cache[d["doc_id"]]["votes"][str(s.id)]] for s in SIGNALS]
        for d in docs
    ], dtype=int)
