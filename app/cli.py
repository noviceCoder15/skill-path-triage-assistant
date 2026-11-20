"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import os
import sys
from pathlib import Path

from .storage import Store


def _dir() -> Path:
    d = Path(os.environ.get("DATA_PATH", "data/app.json")).parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store() -> Store:
    return Store(os.environ.get("DATA_PATH", "data/app.json"))


def _need(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"Missing {path}. {hint}")
    return path


def _company_seqs():
    from .core import build_sequences, load_events_csv
    return build_sequences(load_events_csv(str(_need(_dir() / "company.csv", "Run `generate-data` first (or put your HR export there)."))))


def _fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


# ---------------------------------------------------------------------------- commands
def cmd_generate(a):
    from .core import generate_employees, save_events_csv
    save_events_csv(generate_employees(a.company, a.seed, "company"), str(_dir() / "company.csv"))
    save_events_csv(generate_employees(a.industry, a.seed + 1, "industry"), str(_dir() / "industry.csv"))
    print(f"Wrote {a.company} company and {a.industry} industry employee histories to {_dir()}/ (synthetic).")


def cmd_mine(a):
    from .core import mine_itemsets, mine_transitions
    seqs = _company_seqs()
    sets = mine_itemsets(seqs, a.min_support, a.max_len, a.department)
    trans = mine_transitions(seqs, a.min_support, a.max_gap, a.department)
    _store().put("itemsets", sets)
    _store().put("transitions", trans)
    short = lambda t: t.split(":", 1)[1]
    print(f"{len(sets)} frequent skill sets, {len(trans)} frequent transitions "
          f"(min support {a.min_support:.0%}{', ' + a.department if a.department else ''}).")
    print(f"\nTop {a.top} transitions (A -> B):")
    for r in sorted(trans, key=lambda r: -r["support"]):
        if len(r["pattern"]) == 2:
            print(f"  {short(r['pattern'][0])} -> {short(r['pattern'][1])}   support {r['support']:.1%}  "
                  f"confidence {r['confidence']:.0%}  lift {r['lift']:.2f}")
            a.top -= 1
            if a.top == 0:
                break
    print("\nTop skill sets that go together:")
    for r in [r for r in sets if len(r["items"]) >= 2][:5]:
        print(f"  {{{', '.join(short(i) for i in r['items'])}}}   support {r['support']:.1%}")


def cmd_train(a):
    from .core import build_sequences, load_events_csv, train_models
    company = _company_seqs()
    industry = build_sequences(load_events_csv(str(_need(_dir() / "industry.csv", "Run `generate-data` first."))))
    m = train_models(company, industry, seed=a.seed, verbose=a.verbose)
    for name in ("scratch", "pretrain", "continued"):
        m[name].save(str(_dir() / f"lm_{name}.npz"))
    _store().put("train_meta", {"seed": a.seed, "n_company": len(company), "n_industry": len(industry),
                                "n_train": len(m["train"]), "n_val": len(m["val"]), "n_test": len(m["test"])})
    print(f"Trained on {len(m['train'])} company employees (val {len(m['val'])}, test {len(m['test'])}), "
          f"pretrained on {len(industry)} industry employees.")
    print(f"Saved lm_scratch / lm_pretrain / lm_continued to {_dir()}/. Next: `evaluate`.")


def _recommenders(seed):
    from .core import BigramRec, LMRec, PopularityRec, RuleRec, split_by_employee
    from .lm import CareerLM
    seqs = _company_seqs()
    train, _, test = split_by_employee(seqs, seed=seed)
    lms = {n: CareerLM.load(str(_need(_dir() / f"lm_{n}.npz", "Run `train` first."))) for n in ("scratch", "pretrain", "continued")}
    v = lms["continued"].vocab
    recs = {"popularity": PopularityRec(v, train), "bigram": BigramRec(v, train), "rules": RuleRec(v, train),
            "lm-scratch": LMRec(v, lms["scratch"], "lm-scratch"),
            "lm-pretrain": LMRec(v, lms["pretrain"], "lm-pretrain only"),
            "lm-continued": LMRec(v, lms["continued"], "lm-continued")}
    return recs, train, test, lms


def cmd_evaluate(a):
    from .core import evaluate
    seed = (_store().get("train_meta") or {}).get("seed", 0)
    recs, train, test, lms = _recommenders(seed)
    results = {}
    print(f"Held-out: {len(test)} employees. All models rank only skills the employee does not have yet.\n")
    keys = None
    for key, r in recs.items():
        res = evaluate(r, test, k=a.k)
        results[key] = res
        keys = keys or [k for k in res if k != "n_prefixes"]
        if key == next(iter(recs)):
            print(f"{'model':18s}" + "".join(f"{k:>16s}" for k in keys))
        print(f"{r.name:18s}" + "".join(f"{res[k]:16.3f}" for k in keys))
    ppl = {n: m.perplexity(test) for n, m in lms.items()}
    print("\nTest perplexity (lower is better):", "  ".join(f"{n}={p:.2f}" for n, p in ppl.items()))
    print(f"({results['popularity']['n_prefixes']} prefixes scored. valid@k is capped because many steps have fewer than k real "
          "successors; it is computed against the synthetic career graph, so it only exists for synthetic data.)")
    _store().put("evaluation", {"k": a.k, "metrics": results, "perplexity": ppl})


def cmd_suggest(a):
    from .core import make_filter, resolve_token, token_profiles
    recs, train, _, lms = _recommenders((_store().get("train_meta") or {}).get("seed", 0))
    rec = recs[a.model]
    try:
        history = [resolve_token(x, rec.vocab) for x in a.skills.split(",") if x.strip()]
    except ValueError as e:
        sys.exit(str(e))
    if not history:
        sys.exit("Give at least one skill or role, e.g. --skills 'SQL,Python'.")
    allow = make_filter(token_profiles(train), a.department_filter, a.seniority,
                        a.category.split(",") if a.category else None, a.kind)
    top = rec.top(a.department, history, a.top, allow)
    print(f"{a.department}: {' -> '.join(t.split(':', 1)[1] for t in history)}")
    if not top:
        print("No suggestions match those filters.")
        return
    print(f"Suggested next steps ({rec.name}):")
    for i, (tok, sc) in enumerate(top, 1):
        from .ontology import category
        kind, name = tok.split(":", 1)
        print(f"  {i}. {name:24s} [{'role' if kind == 'role' else category(tok)}]  score {sc:.2f}")


def cmd_gaps(a):
    from .core import build_sequences, load_events_csv, make_filter, resolve_token, team_gaps, token_profiles
    recs, train, _, _ = _recommenders((_store().get("train_meta") or {}).get("seed", 0))
    rec = recs[a.model]
    team = build_sequences(load_events_csv(a.team))
    if not team:
        sys.exit("No employees with at least two dated events found in that file.")
    allow = make_filter(token_profiles(train), a.department_filter, kind=a.kind)
    gaps = team_gaps(rec, team, a.top, allow)[: a.show]
    print(f"{len(team)} team members. Skills most often suggested next (top {a.top} per person):")
    for g in gaps:
        print(f"  {g['token'].split(':', 1)[1]:24s} {g['members']:3d} of {len(team)} members ({g['share']:.0%})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="skill-path-triage-assistant",
                                description="HR managers analyze employee skill acquisition sequences to suggest relevant "
                                            "next courses or identify common skill gaps for team training.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    g = sub.add_parser("generate-data", help="create synthetic employee histories (M1)")
    g.add_argument("--company", type=int, default=1500)
    g.add_argument("--industry", type=int, default=6000)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)

    m = sub.add_parser("mine", help="frequent skill sets and transition patterns (M2)")
    m.add_argument("--min-support", type=float, default=0.05)
    m.add_argument("--max-gap", type=int, default=3)
    m.add_argument("--max-len", type=int, default=3)
    m.add_argument("--department")
    m.add_argument("--top", type=int, default=10)
    m.set_defaults(fn=cmd_mine)

    t = sub.add_parser("train", help="pretrain on industry data, continue on company data (M3)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="compare models on held-out employees (M4)")
    e.add_argument("--k", type=int, default=5)
    e.set_defaults(fn=cmd_evaluate)

    models = ["lm-continued", "lm-pretrain", "lm-scratch", "bigram", "rules", "popularity"]
    s = sub.add_parser("suggest", help="next steps for one employee (M5)")
    s.add_argument("--department", required=True)
    s.add_argument("--skills", required=True, help="comma-separated, oldest first: 'SQL,Python,Data Analyst'")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--model", choices=models, default="lm-continued")
    s.add_argument("--department-filter", help="only items that are typical in this department")
    s.add_argument("--seniority", choices=["junior", "mid", "senior"], help="only items typically acquired at this level")
    s.add_argument("--category", help="comma-separated: technical,analytical,business,leadership,role")
    s.add_argument("--kind", choices=["any", "skill", "role"], default="any")
    s.set_defaults(fn=cmd_suggest)

    gp = sub.add_parser("gaps", help="common skill gaps for a team, from an HR export CSV (M5)")
    gp.add_argument("--team", required=True, help="CSV with employee_id,department,date,item")
    gp.add_argument("--top", type=int, default=5, help="suggestions considered per person")
    gp.add_argument("--show", type=int, default=8)
    gp.add_argument("--model", choices=models, default="lm-continued")
    gp.add_argument("--department-filter")
    gp.add_argument("--kind", choices=["any", "skill", "role"], default="skill")
    gp.set_defaults(fn=cmd_gaps)

    args = p.parse_args(argv)
    if args.command == "hello":
        print("Skill Path Triage Assistant is ready. Open README.md for your milestones.")
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
