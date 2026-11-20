"""Milestones 1, 2 and 4 (data, mining, evaluation) plus the recommenders the CLI uses.

    M1  employees -> time-ordered sequences          generate_employees / load_events_csv / build_sequences
    M2  frequent patterns                            mine_itemsets / mine_transitions
    M3  language model                               app/lm.py (CareerLM), driven by train_models()
    M4  evaluation                                   evaluate()
    M5  command line                                 app/cli.py
"""
from __future__ import annotations

import csv
import itertools
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .lm import CareerLM, Vocab
from .ontology import DEPARTMENTS, ENTRY, GRAPH, all_tokens, category, valid_next

# =====================================================================================
# Milestone 1: skills and roles as sequences over time
# =====================================================================================

@dataclass
class Employee:
    """Raw HR record: who, which department, and dated events (an acquired skill or a new role)."""
    id: str
    department: str
    events: list[tuple[str, str]]          # (YYYY-MM, token) in any order


@dataclass
class Sequence:
    """One employee's history as a time-ordered list of distinct tokens."""
    id: str
    department: str
    tokens: list[str]
    months: list[int]                      # months since the employee's first event, aligned with tokens

    @property
    def seniority(self) -> str:
        return seniority_at(self.months[-1] if self.months else 0)


def seniority_at(months_since_first_event: int) -> str:
    """Seniority is derived from time in the record, so it is also known for a partial history."""
    return "junior" if months_since_first_event < 18 else "mid" if months_since_first_event < 48 else "senior"


def _month_number(ym: str) -> int:
    y, m = ym.split("-")[:2]
    return int(y) * 12 + int(m) - 1


def build_sequences(employees: list[Employee]) -> list[Sequence]:
    """Sort each employee's events by date, keep only the first time a token appears, drop one-event histories."""
    out = []
    for e in employees:
        seen, toks, mons = set(), [], []
        for ym, tok in sorted(e.events, key=lambda ev: (_month_number(ev[0]), ev[1])):
            if tok in seen:
                continue
            seen.add(tok)
            toks.append(tok)
            mons.append(_month_number(ym))
        if len(toks) >= 2:
            out.append(Sequence(e.id, e.department, toks, [m - mons[0] for m in mons]))
    return out


def load_events_csv(path: str) -> list[Employee]:
    """Read an HR export with columns employee_id, department, date, item (item is `skill:X` or `role:Y`;
    a bare name is treated as a skill)."""
    by_id: dict[str, Employee] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tok = row["item"].strip()
            if ":" not in tok:
                tok = f"skill:{tok}"
            e = by_id.setdefault(row["employee_id"], Employee(row["employee_id"], row["department"].strip(), []))
            e.events.append((row["date"].strip()[:7], tok))
    return list(by_id.values())


def save_events_csv(employees: list[Employee], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["employee_id", "department", "date", "item"])
        for e in employees:
            for ym, tok in e.events:
                w.writerow([e.id, e.department, ym, tok])


def generate_employees(n: int, seed: int = 0, variant: str = "company") -> list[Employee]:
    """Synthetic histories sampled from the ontology graph (there is no real HR export in this project).

    variant="company"  : follows the graph closely (5% off-graph noise)  - the organisation we care about.
    variant="industry" : flattened weights and 30% off-graph noise       - a broader, messier outside corpus
                         used for pretraining."""
    rng = random.Random(seed)
    noise, flatten = (0.05, 1.0) if variant == "company" else (0.30, 0.5)
    people = []
    for i in range(n):
        dept = rng.choices(DEPARTMENTS, weights=[3, 3, 2, 2])[0]
        graph, pool = GRAPH[dept], sorted({t for a, o in GRAPH[dept].items() for t in (a, *o)})
        entry = ENTRY[dept]
        cur = rng.choices(list(entry), weights=list(entry.values()))[0]
        held, target_len = [cur], rng.randint(4, 11)
        while len(held) < target_len:
            nxt = None
            if rng.random() < noise:
                nxt = rng.choice([t for t in pool if t not in held] or [None])
            else:
                # successors of the latest token; if exhausted, fall back to an earlier token's successors
                for anchor in [cur] + rng.sample(held, len(held)):
                    options = {t: w for t, w in graph.get(anchor, {}).items() if t not in held}
                    if options:
                        nxt = rng.choices(list(options), weights=[w ** flatten for w in options.values()])[0]
                        break
            if nxt is None:
                break
            held.append(nxt)
            cur = nxt
        month = 2015 * 12 + rng.randint(0, 60)
        events = []
        for tok in held:
            events.append((f"{month // 12:04d}-{month % 12 + 1:02d}", tok))
            month += rng.randint(3, 14)
        people.append(Employee(f"{variant[:3]}-{i:05d}", dept, events))
    return people


# =====================================================================================
# Milestone 2: frequent itemsets and transition patterns
# =====================================================================================
# NOTE: the brief names the `thilimarketbasket` package. It could not be installed or inspected where this
# was built (no network, and its API is unknown to us), so these are self-contained implementations of the
# same techniques: Apriori for co-occurring skills, and gap-bounded sequential patterns for transitions.

def mine_itemsets(seqs: list[Sequence], min_support: float = 0.05, max_len: int = 3,
                  department: str | None = None) -> list[dict]:
    """Apriori on 'baskets' (the set of tokens an employee holds). Returns itemsets with support."""
    baskets = [frozenset(s.tokens) for s in seqs if department in (None, s.department)]
    n = len(baskets)
    if n == 0:
        return []
    need = min_support * n
    counts = Counter(t for b in baskets for t in b)
    level = {frozenset([t]): c for t, c in counts.items() if c >= need}
    found = dict(level)
    for size in range(2, max_len + 1):
        items = sorted({t for s in level for t in s})
        # candidate generation with the Apriori property: every (size-1)-subset must itself be frequent
        cands = [frozenset(c) for c in itertools.combinations(items, size)
                 if all(frozenset(sub) in level for sub in itertools.combinations(c, size - 1))]
        level = {}
        for c in cands:
            cnt = sum(1 for b in baskets if c <= b)
            if cnt >= need:
                level[c] = cnt
        if not level:
            break
        found.update(level)
    return sorted(({"items": sorted(s), "support": c / n, "count": c} for s, c in found.items()),
                  key=lambda r: (-r["support"], r["items"]))


def mine_transitions(seqs: list[Sequence], min_support: float = 0.03, max_gap: int = 3,
                     department: str | None = None) -> list[dict]:
    """Frequent 'A then B' (and 'A, B then C') patterns where each step is at most `max_gap` positions later.
    Each pattern is counted once per employee. For rules A->B we report confidence (= P(B after A | A held))
    and lift (confidence / P(B held))."""
    rows = [s for s in seqs if department in (None, s.department)]
    n = len(rows)
    if n == 0:
        return []
    pair_c, tri_c, item_c = Counter(), Counter(), Counter()
    for s in rows:
        t = s.tokens
        item_c.update(set(t))
        pairs, tris = set(), set()
        for i in range(len(t)):
            for j in range(i + 1, min(i + max_gap, len(t) - 1) + 1):
                pairs.add((t[i], t[j]))
                for k in range(j + 1, min(j + max_gap, len(t) - 1) + 1):
                    tris.add((t[i], t[j], t[k]))
        pair_c.update(pairs)
        tri_c.update(tris)
    need = min_support * n
    out = []
    for (a, b), c in pair_c.items():
        if c >= need:
            conf = c / item_c[a]
            out.append({"pattern": [a, b], "support": c / n, "count": c, "confidence": conf,
                        "lift": conf / (item_c[b] / n)})
    for (a, b, c3), c in tri_c.items():
        if c >= need and pair_c[(a, b)]:
            conf = c / pair_c[(a, b)]
            out.append({"pattern": [a, b, c3], "support": c / n, "count": c, "confidence": conf,
                        "lift": conf / (item_c[c3] / n)})
    return sorted(out, key=lambda r: (-r["support"], r["pattern"]))


# =====================================================================================
# Recommenders: everything exposes scores(department, history) -> array over the vocabulary
# =====================================================================================

class Recommender:
    name = "base"

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def raw_scores(self, department: str, history: list[str]) -> np.ndarray:
        raise NotImplementedError

    def scores(self, department: str, history: list[str]) -> np.ndarray:
        """Scores with held tokens and <pad> removed (-inf), so every model ranks only genuinely new steps."""
        s = np.array(self.raw_scores(department, history), dtype=float)
        s[0] = -np.inf
        for i in self.vocab.ids(history):
            s[i] = -np.inf
        return s

    def top(self, department: str, history: list[str], k: int = 5, allow=None) -> list[tuple[str, float]]:
        s = self.scores(department, history)
        order = [i for i in np.argsort(-s, kind="stable") if np.isfinite(s[i])]
        res = [(self.vocab.tokens[i], float(s[i])) for i in order
               if allow is None or allow(self.vocab.tokens[i])]
        return res[:k]


class PopularityRec(Recommender):
    """Most common next steps in the department. The baseline every model has to beat."""
    name = "popularity"

    def __init__(self, vocab, seqs):
        super().__init__(vocab)
        self.by_dept = defaultdict(lambda: np.zeros(len(vocab)))
        for s in seqs:
            for i in vocab.ids(s.tokens[1:]):
                self.by_dept[s.department][i] += 1

    def raw_scores(self, department, history):
        return self.by_dept[department] if department in self.by_dept else sum(self.by_dept.values())


class BigramRec(Recommender):
    """P(next | previous token, department) with add-alpha smoothing and a department-popularity backoff."""
    name = "bigram"

    def __init__(self, vocab, seqs, alpha=0.1):
        super().__init__(vocab)
        self.alpha, self.pop = alpha, PopularityRec(vocab, seqs)
        self.c = defaultdict(lambda: np.zeros(len(vocab)))
        for s in seqs:
            ids = vocab.ids(s.tokens)
            for a, b in zip(ids, ids[1:]):
                self.c[(s.department, a)][b] += 1

    def raw_scores(self, department, history):
        prior = self.pop.raw_scores(department, history)
        prior = prior / max(prior.sum(), 1)
        last = self.vocab.ids(history[-1:])
        row = self.c.get((department, last[0])) if last else None
        counts = row if row is not None else np.zeros(len(self.vocab))
        return (counts + self.alpha * len(self.vocab) * prior) / (counts.sum() + self.alpha * len(self.vocab))


class RuleRec(Recommender):
    """Recommends from the mined transition patterns (milestone 2): score(c) = sum of the confidences of rules
    h -> c for the last 3 tokens h (recency-decayed), plus a bonus when an (h2, h1) -> c triple rule fires."""
    name = "mined-rules"

    def __init__(self, vocab, seqs, min_support=0.01, max_gap=3, decay=0.6):
        super().__init__(vocab)
        self.decay, self.pairs, self.tris = decay, {}, {}
        for dept in DEPARTMENTS:
            self.pairs[dept], self.tris[dept] = defaultdict(dict), defaultdict(dict)
            for r in mine_transitions(seqs, min_support, max_gap, department=dept):
                if len(r["pattern"]) == 2:
                    self.pairs[dept][r["pattern"][0]][r["pattern"][1]] = r["confidence"]
                else:
                    self.tris[dept][tuple(r["pattern"][:2])][r["pattern"][2]] = r["confidence"]
        self.pop = PopularityRec(vocab, seqs)

    def raw_scores(self, department, history):
        s = np.zeros(len(self.vocab))
        for back, h in enumerate(reversed(history[-3:])):
            for c, conf in self.pairs.get(department, {}).get(h, {}).items():
                s[self.vocab.index[c]] += conf * self.decay ** back
        if len(history) >= 2:
            for c, conf in self.tris.get(department, {}).get(tuple(history[-2:]), {}).items():
                s[self.vocab.index[c]] += conf
        pop = self.pop.raw_scores(department, history)
        return s + 1e-6 * pop / max(pop.max(), 1)          # popularity only breaks ties / covers unseen contexts


class LMRec(Recommender):
    def __init__(self, vocab, model: CareerLM, name="lm"):
        super().__init__(vocab)
        self.model, self.name = model, name

    def raw_scores(self, department, history):
        return self.model.next_probs(department, history)


# =====================================================================================
# Milestone 3 driver: pretrain on the industry corpus, continue on the company corpus
# =====================================================================================

def split_by_employee(seqs: list[Sequence], test_frac=0.2, val_frac=0.1, seed=0):
    rng = random.Random(seed)
    ids = sorted({s.id for s in seqs})
    rng.shuffle(ids)
    n_test, n_val = int(len(ids) * test_frac), int(len(ids) * val_frac)
    test, val = set(ids[:n_test]), set(ids[n_test:n_test + n_val])
    return ([s for s in seqs if s.id not in test and s.id not in val],
            [s for s in seqs if s.id in val], [s for s in seqs if s.id in test])


def train_models(company: list[Sequence], industry: list[Sequence], seed=0, verbose=False) -> dict:
    """Returns the three LM variants compared in the evaluation, plus the split:
       scratch   - trained on the company corpus only
       pretrain  - trained on the industry corpus only (zero-shot on the company)
       continued - `pretrain`, then continued pretraining on the company corpus (lower learning rate)"""
    train, val, test = split_by_employee(company, seed=seed)
    ind_train, ind_val, _ = split_by_employee(industry, test_frac=0.0, seed=seed)
    vocab = Vocab()
    scratch = CareerLM(vocab, seed=seed).fit(train, val, epochs=60, lr=0.01, phase="scratch", seed=seed, verbose=verbose)
    pre = CareerLM(vocab, seed=seed).fit(ind_train, ind_val, epochs=40, lr=0.01, phase="pretrain", seed=seed, verbose=verbose)
    cont = CareerLM(vocab, seed=seed)
    cont.p = {k: v.copy() for k, v in pre.p.items()}
    cont.history = list(pre.history)
    cont.fit(train, val, epochs=40, lr=0.003, phase="continued", seed=seed, verbose=verbose)
    return {"vocab": vocab, "scratch": scratch, "pretrain": pre, "continued": cont,
            "train": train, "val": val, "test": test}


# =====================================================================================
# Milestone 4: evaluation
# =====================================================================================

def _dcg(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / np.log2(rank + 2)


def evaluate(rec: Recommender, test: list[Sequence], k: int = 5, horizon: int = 3, use_oracle: bool = True) -> dict:
    """Replay every prefix of every held-out employee and ask for the top-k next steps.

    Accuracy (against what the employee actually did next)
      hit@1, hit@k, MRR, NDCG@k       - the true next step ranked in the list
      horizon_hit@k                   - any of the next `horizon` real steps is in the list. This is the fairer
                                        'relevance' number: a good suggestion may be taken a little later.
    Relevance (against the career graph, synthetic data only)
      valid@k                         - share of suggested items that are real edges out of the employee's last step
    Breadth
      coverage@k                      - share of the vocabulary that is ever recommended across all prefixes
      diversity@k                     - mean share of distinct skill categories inside one list
    """
    n = 0
    hit1 = hitk = rr = ndcg = hz = valid = div = 0.0
    shown = set()
    for s in test:
        for t in range(1, len(s.tokens)):
            hist, truth = s.tokens[:t], s.tokens[t]
            top = [tok for tok, _ in rec.top(s.department, hist, k)]
            if not top:
                continue
            n += 1
            rank = top.index(truth) if truth in top else None
            hit1 += rank == 0
            hitk += rank is not None
            rr += 1.0 / (rank + 1) if rank is not None else 0.0
            ndcg += _dcg(rank)                                    # one relevant item, so IDCG = 1
            hz += any(x in top for x in s.tokens[t:t + horizon])
            shown.update(top)
            div += len({category(x) for x in top}) / len(top)
            if use_oracle:
                ok = valid_next(s.department, hist[-1])
                valid += sum(x in ok for x in top) / len(top)
    n = max(n, 1)
    res = {"n_prefixes": n, "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n, f"ndcg@{k}": ndcg / n,
           f"horizon_hit@{k}": hz / n, f"coverage@{k}": len(shown) / (len(rec.vocab) - 1),
           f"diversity@{k}": div / n}
    if use_oracle:
        res[f"valid@{k}"] = valid / n
    return res


# =====================================================================================
# Helpers for the CLI: token profiles (for filters) and team gap analysis
# =====================================================================================

def token_profiles(seqs: list[Sequence]) -> dict[str, dict]:
    """For each token: how often it appears per department and at each seniority level (when it was acquired)."""
    prof: dict[str, dict] = defaultdict(lambda: {"dept": Counter(), "level": Counter(), "n": 0})
    for s in seqs:
        for tok, m in zip(s.tokens, s.months):
            p = prof[tok]
            p["dept"][s.department] += 1
            p["level"][seniority_at(m)] += 1
            p["n"] += 1
    return prof


def make_filter(profiles, department=None, seniority=None, categories=None, kind="any", min_share=0.10):
    """Predicate over tokens. Department/seniority: the token must make up at least `min_share` of its
    acquisitions in that department/level. Category and kind are exact matches."""
    def allow(tok: str) -> bool:
        if kind == "skill" and not tok.startswith("skill:"):
            return False
        if kind == "role" and not tok.startswith("role:"):
            return False
        if categories and category(tok) not in categories:
            return False
        p = profiles.get(tok)
        if p is None or p["n"] == 0:
            return False
        if department and p["dept"][department] / p["n"] < min_share:
            return False
        if seniority and p["level"][seniority] / p["n"] < min_share:
            return False
        return True
    return allow


def team_gaps(rec: Recommender, team: list[Sequence], k: int = 5, allow=None) -> list[dict]:
    """Common gaps: for each team member take the top-k next steps; a gap is a skill suggested to many members.
    `share` = fraction of the team that has it in their top-k."""
    votes, mean_score = Counter(), defaultdict(float)
    for s in team:
        for tok, sc in rec.top(s.department, s.tokens, k, allow):
            votes[tok] += 1
            mean_score[tok] += sc
    n = max(len(team), 1)
    return [{"token": t, "members": c, "share": c / n, "mean_score": mean_score[t] / c}
            for t, c in sorted(votes.items(), key=lambda kv: (-kv[1], -mean_score[kv[0]]))]


# =====================================================================================
# Input handling for the CLI
# =====================================================================================

def resolve_token(name: str, vocab: Vocab) -> str:
    """Accept 'SQL', 'sql', 'skill:SQL' or 'Data Analyst' / 'role:Data Analyst' and return the vocabulary token."""
    import difflib
    raw = name.strip()
    lookup = {t.lower(): t for t in vocab.tokens[1:]}
    lookup.update({t.split(":", 1)[1].lower(): t for t in vocab.tokens[1:]})
    if raw.lower() in lookup:
        return lookup[raw.lower()]
    close = difflib.get_close_matches(raw.lower(), [k for k in lookup if ":" not in k], n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(lookup[c].split(':', 1)[1] for c in close)}?" if close else ""
    raise ValueError(f"Unknown skill or role '{raw}'.{hint}")
