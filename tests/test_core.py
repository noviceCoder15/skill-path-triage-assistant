"""One test per milestone. They use small synthetic data so the whole file runs in a few seconds."""
import numpy as np
import pytest

from app.core import (BigramRec, Employee, LMRec, PopularityRec, RuleRec, build_sequences, evaluate,
                      generate_employees, load_events_csv, make_filter, mine_itemsets, mine_transitions,
                      resolve_token, save_events_csv, seniority_at, split_by_employee, team_gaps,
                      token_profiles, train_models)
from app.lm import CareerLM, Vocab


@pytest.fixture(scope="module")
def world():
    company = build_sequences(generate_employees(500, seed=1))
    industry = build_sequences(generate_employees(1500, seed=2, variant="industry"))
    return company, industry


# Milestone 1: skills and roles as sequences over time
def test_milestone_1_sequences_are_time_ordered_and_deduplicated(tmp_path):
    e = Employee("e1", "Data", [("2020-06", "skill:Python"), ("2020-01", "skill:SQL"),
                                ("2021-03", "skill:SQL"), ("2022-01", "role:Data Analyst")])
    (s,) = build_sequences([e])
    assert s.tokens == ["skill:SQL", "skill:Python", "role:Data Analyst"]   # sorted by date, repeat dropped
    assert s.months == [0, 5, 24]
    assert s.seniority == "mid" and seniority_at(3) == "junior" and seniority_at(60) == "senior"
    assert build_sequences([Employee("e2", "Data", [("2020-01", "skill:SQL")])]) == []   # one event is no path

    path = tmp_path / "hr.csv"                                             # CSV round trip
    save_events_csv([e], str(path))
    assert build_sequences(load_events_csv(str(path)))[0].tokens == s.tokens


def test_milestone_1_generator_is_reproducible_and_follows_the_graph():
    a, b = generate_employees(50, seed=3), generate_employees(50, seed=3)
    assert [x.events for x in a] == [x.events for x in b]
    assert {x.department for x in a} <= {"Data", "Engineering", "Marketing", "People Ops"}


# Milestone 2: frequent itemsets and transitions
def test_milestone_2_mining_finds_known_patterns_with_correct_numbers():
    mk = lambda i, toks: build_sequences([Employee(f"e{i}", "Data", [(f"2020-{j + 1:02d}", t) for j, t in enumerate(toks)])])[0]
    seqs = [mk(0, ["skill:A", "skill:B", "skill:C"]), mk(1, ["skill:A", "skill:B"]),
            mk(2, ["skill:A", "skill:C"]), mk(3, ["skill:D", "skill:C"])]
    sets = {tuple(r["items"]): r for r in mine_itemsets(seqs, min_support=0.5, max_len=2)}
    assert sets[("skill:A",)]["support"] == 0.75
    assert sets[("skill:A", "skill:B")]["support"] == 0.5
    assert ("skill:B", "skill:D") not in sets
    tr = {tuple(r["pattern"]): r for r in mine_transitions(seqs, min_support=0.5)}
    ab = tr[("skill:A", "skill:B")]
    assert ab["count"] == 2 and ab["confidence"] == pytest.approx(2 / 3)
    assert ab["lift"] == pytest.approx((2 / 3) / (2 / 4))
    assert ("skill:B", "skill:A") not in tr                                 # order matters


def test_milestone_2_mining_recovers_the_strongest_career_edge(world):
    pats = mine_transitions(world[0], 0.05, department="Data")
    pairs = {tuple(r["pattern"]) for r in pats if len(r["pattern"]) == 2}
    assert ("skill:Statistics", "skill:Machine Learning") in pairs


# Milestone 3: the language model
def test_milestone_3_lm_learns_masks_held_tokens_and_round_trips(world, tmp_path):
    company, _ = world
    train, val, _ = split_by_employee(company)
    m = CareerLM(Vocab(), seed=0)
    before = m.perplexity(val)
    m.fit(train, val, epochs=8, lr=0.01)
    assert m.perplexity(val) < before * 0.6                                 # it learned something
    p = m.next_probs("Data", ["skill:SQL", "skill:Python"])
    assert p.sum() == pytest.approx(1.0) and p[m.vocab.index["skill:SQL"]] == 0.0
    m.save(str(tmp_path / "m.npz"))
    m2 = CareerLM.load(str(tmp_path / "m.npz"))
    assert np.allclose(m2.next_probs("Data", ["skill:SQL"]), m.next_probs("Data", ["skill:SQL"]))


def test_milestone_3_gradients_match_finite_differences():
    m = CareerLM(Vocab(), k=2, dim=4, hidden=5, seed=1)
    seqs = build_sequences(generate_employees(20, seed=0))
    from app.lm import make_examples
    data = make_examples(m.vocab, seqs, m.k)
    g = m._grads(data)
    rng = np.random.default_rng(0)
    for name in ("W1", "W2", "E", "Dep"):
        idx = tuple(rng.integers(0, s) for s in m.p[name].shape)
        old = m.p[name][idx]
        m.p[name][idx] = old + 1e-5; up = m.loss(data)
        m.p[name][idx] = old - 1e-5; dn = m.loss(data)
        m.p[name][idx] = old
        assert g[name][idx] == pytest.approx((up - dn) / 2e-5, abs=1e-6, rel=1e-3)


def test_milestone_3_continued_pretraining_keeps_weights_then_improves_on_company_data(world):
    company, industry = world
    out = train_models(company, industry, seed=0)
    phases = [h["phase"] for h in out["continued"].history]
    assert "pretrain" in phases and phases[-1] == "continued"               # same model, two phases
    ppl = lambda k: out[k].perplexity(out["test"])
    assert ppl("continued") < ppl("pretrain")                               # adapting to the company helps


# Milestone 4: evaluation
def test_milestone_4_metrics_are_bounded_and_ordered_sensibly(world):
    company, industry = world
    train, _, test = split_by_employee(company)
    v = Vocab()
    pop, big = evaluate(PopularityRec(v, train), test), evaluate(BigramRec(v, train), test)
    for res in (pop, big):
        assert all(0.0 <= x <= 1.0 for k, x in res.items() if k != "n_prefixes")
        assert res["hit@1"] <= res["hit@5"] <= res["horizon_hit@5"]
    assert big["mrr"] > pop["mrr"] and big["valid@5"] > pop["valid@5"]      # context beats plain popularity


def test_milestone_4_a_perfect_oracle_scores_one():
    from app.core import Recommender, Sequence
    v = Vocab()
    seq = Sequence("x", "Data", ["skill:Excel", "skill:SQL", "skill:Python"], [0, 5, 10])

    class Oracle(Recommender):
        def raw_scores(self, department, history):
            s = np.zeros(len(v))
            s[v.index[seq.tokens[len(history)]]] = 1.0
            return s

    res = evaluate(Oracle(v), [seq], k=5, use_oracle=False)
    assert res["hit@1"] == 1.0 and res["mrr"] == 1.0 and res["ndcg@5"] == 1.0


# Milestone 5: the CLI
def test_milestone_5_suggest_filters_and_input_resolution(world):
    company, _ = world
    v = Vocab()
    rec = BigramRec(v, company)
    assert resolve_token("sql", v) == "skill:SQL" and resolve_token("data analyst", v) == "role:Data Analyst"
    with pytest.raises(ValueError, match="Did you mean: SQL"):
        resolve_token("SQLL", v)
    allow = make_filter(token_profiles(company), kind="role")
    top = rec.top("Data", ["skill:SQL"], 3, allow)
    assert top and all(t.startswith("role:") for t, _ in top)
    assert all(t != "skill:SQL" for t, _ in rec.top("Data", ["skill:SQL"], 5))   # never re-suggests a held skill
    team = [s for s in company if s.department == "Data"][:10]
    gaps = team_gaps(rec, team, 3)
    assert gaps and gaps[0]["members"] >= gaps[-1]["members"] and gaps[0]["share"] <= 1.0


def test_milestone_5_cli_end_to_end(tmp_path, monkeypatch, capsys):
    from app.cli import main
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))
    main(["generate-data", "--company", "300", "--industry", "600"])
    main(["mine", "--min-support", "0.1"])
    main(["train"])
    main(["evaluate"])
    main(["suggest", "--department", "Data", "--skills", "SQL,Python", "--top", "3"])
    out = capsys.readouterr().out
    assert "Suggested next steps" in out and "lm-continued" in out and "Test perplexity" in out
