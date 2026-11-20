"""A small, hand-written career ontology: departments, skill categories and the career graph.

It has two jobs:
  * it is the *generator* for the synthetic employee histories used when no real HR export is available;
  * it is the *oracle* in evaluation: an employee's next step counts as "valid" if the graph has that edge.
    (With real data there is no oracle; the evaluation then reports only the data-driven metrics.)

Tokens are strings: ``skill:<name>`` or ``role:<name>``.
"""

DEPARTMENTS = ["Data", "Engineering", "Marketing", "People Ops"]


def _parse(spec: str) -> dict[str, dict[str, float]]:
    """'A > B:5, C:3' lines -> {A: {B: 5.0, C: 3.0}}. Bare tokens default to weight 1."""
    graph: dict[str, dict[str, float]] = {}
    for line in spec.strip().splitlines():
        head, tail = line.split(">")
        out = {}
        for part in tail.split(","):
            name, _, w = part.strip().rpartition(":")
            out[name] = float(w)
        graph[head.strip()] = out
    return graph


# Every department's graph: token -> {possible next token: weight}.
GRAPH: dict[str, dict[str, dict[str, float]]] = {
    "Data": _parse("""
        skill:Excel > skill:SQL:5, skill:Data Visualization:3, skill:Statistics:2
        skill:SQL > skill:Python:5, skill:Data Visualization:3, role:Data Analyst:4
        role:Data Analyst > skill:Statistics:4, skill:Python:3, skill:Data Storytelling:2
        skill:Data Visualization > skill:Data Storytelling:4, skill:Python:2, role:Data Analyst:2
        skill:Python > skill:Statistics:3, skill:Machine Learning:4, skill:Data Engineering:2, skill:Data Visualization:1
        skill:Statistics > skill:Machine Learning:5, skill:A/B Testing:3
        skill:Data Storytelling > skill:Stakeholder Management:3, skill:A/B Testing:1
        skill:A/B Testing > skill:Machine Learning:2, skill:Stakeholder Management:2
        skill:Machine Learning > role:Data Scientist:5, skill:MLOps:3, skill:Deep Learning:3
        role:Data Scientist > skill:MLOps:3, skill:Mentoring:3, skill:Deep Learning:2, skill:People Management:2
        skill:Data Engineering > skill:Cloud Platforms:4, role:Data Engineer:4, skill:MLOps:2
        role:Data Engineer > skill:Cloud Platforms:3, skill:MLOps:3, skill:System Design:2
        skill:Cloud Platforms > skill:MLOps:3, skill:System Design:2
        skill:Deep Learning > skill:MLOps:3, skill:Mentoring:1
        skill:MLOps > skill:Mentoring:2, skill:People Management:2, skill:System Design:2
        skill:Stakeholder Management > skill:Mentoring:1, skill:People Management:3
        skill:Mentoring > skill:People Management:4, skill:Leadership:2
        skill:People Management > skill:Leadership:4
        skill:System Design > skill:Mentoring:2
    """),
    "Engineering": _parse("""
        skill:Git > skill:Python:4, skill:JavaScript:4, skill:Testing:2
        skill:Python > skill:Testing:3, skill:APIs:4, role:Software Engineer:3
        skill:JavaScript > skill:APIs:3, skill:Testing:2, role:Software Engineer:3
        skill:Testing > skill:APIs:3, skill:CI/CD:2, role:Software Engineer:3
        skill:APIs > role:Software Engineer:5, skill:Databases:3
        skill:Databases > skill:System Design:2, role:Software Engineer:2
        role:Software Engineer > skill:System Design:4, skill:Cloud Platforms:3, skill:CI/CD:3, skill:Code Review:3
        skill:Code Review > skill:Mentoring:3, skill:System Design:2
        skill:CI/CD > skill:Cloud Platforms:4, skill:Kubernetes:2
        skill:Cloud Platforms > skill:Kubernetes:4, skill:Security:3, skill:System Design:2
        skill:Kubernetes > skill:Security:2, skill:System Design:3
        skill:Security > skill:System Design:2, role:Senior Software Engineer:2
        skill:System Design > role:Senior Software Engineer:5, skill:Mentoring:2
        role:Senior Software Engineer > skill:Mentoring:4, skill:Technical Leadership:4, skill:People Management:2
        skill:Mentoring > skill:Technical Leadership:3, skill:People Management:3
        skill:Technical Leadership > skill:People Management:2, skill:Leadership:3
        skill:People Management > skill:Leadership:4
    """),
    "Marketing": _parse("""
        skill:Copywriting > skill:Social Media:4, skill:SEO:3, skill:Content Strategy:2
        skill:Social Media > skill:Content Strategy:3, skill:Email Marketing:3, role:Marketing Specialist:3
        skill:SEO > skill:Web Analytics:4, skill:Content Strategy:3
        skill:Web Analytics > skill:Paid Advertising:3, skill:Data Visualization:2, role:Marketing Specialist:3
        skill:Content Strategy > role:Marketing Specialist:4, skill:Email Marketing:2
        skill:Email Marketing > skill:Paid Advertising:3, skill:Web Analytics:2
        role:Marketing Specialist > skill:Paid Advertising:3, skill:Brand Management:3, skill:Budgeting:2, skill:Data Visualization:2
        skill:Paid Advertising > skill:Budgeting:3, skill:Brand Management:2
        skill:Data Visualization > skill:Brand Management:1, skill:Budgeting:1
        skill:Brand Management > role:Marketing Manager:4, skill:Stakeholder Management:3
        skill:Budgeting > role:Marketing Manager:3, skill:Stakeholder Management:2
        skill:Stakeholder Management > role:Marketing Manager:3, skill:People Management:2
        role:Marketing Manager > skill:People Management:4, skill:Leadership:3, skill:Mentoring:2
        skill:Mentoring > skill:People Management:3
        skill:People Management > skill:Leadership:4
    """),
    "People Ops": _parse("""
        skill:Recruiting > skill:Interviewing:5, skill:HRIS:2, skill:Onboarding:3
        skill:Interviewing > skill:Onboarding:3, skill:Employment Law:2, role:HR Generalist:3
        skill:Onboarding > skill:HRIS:3, role:HR Generalist:4
        skill:HRIS > skill:Excel:3, skill:Employment Law:2, role:HR Generalist:3
        skill:Excel > skill:Compensation Analysis:4, skill:People Analytics:3
        skill:Employment Law > skill:Conflict Resolution:4, role:HR Generalist:3
        role:HR Generalist > skill:Performance Management:4, skill:Compensation Analysis:3, skill:Conflict Resolution:3
        skill:Performance Management > skill:Coaching:3, role:HR Business Partner:3
        skill:Compensation Analysis > skill:People Analytics:4, role:HR Business Partner:3
        skill:Conflict Resolution > skill:Coaching:3, skill:Change Management:2
        skill:People Analytics > skill:Change Management:2, role:HR Business Partner:3
        skill:Coaching > role:HR Business Partner:3, skill:Change Management:3
        role:HR Business Partner > skill:Change Management:4, skill:Leadership:3, skill:Mentoring:2
        skill:Change Management > role:HR Manager:4, skill:Leadership:3
        skill:Mentoring > role:HR Manager:3, skill:Leadership:2
        skill:Leadership > role:HR Manager:3
        role:HR Manager > skill:Leadership:2, skill:Mentoring:1
    """),
}

# Where people typically begin (entry token -> weight) per department.
ENTRY: dict[str, dict[str, float]] = {
    "Data": {"skill:Excel": 5, "skill:SQL": 3, "skill:Python": 1},
    "Engineering": {"skill:Git": 5, "skill:Python": 2, "skill:JavaScript": 2},
    "Marketing": {"skill:Copywriting": 5, "skill:Social Media": 3, "skill:SEO": 2},
    "People Ops": {"skill:Recruiting": 5, "skill:HRIS": 1, "skill:Interviewing": 2},
}

CATEGORY_OF = {
    # technical
    **{f"skill:{s}": "technical" for s in [
        "SQL", "Python", "JavaScript", "Git", "Testing", "APIs", "Databases", "CI/CD", "Cloud Platforms",
        "Kubernetes", "Security", "System Design", "Machine Learning", "Deep Learning", "MLOps",
        "Data Engineering", "Code Review", "HRIS"]},
    # analytical
    **{f"skill:{s}": "analytical" for s in [
        "Excel", "Statistics", "A/B Testing", "Data Visualization", "Web Analytics",
        "Compensation Analysis", "People Analytics"]},
    # business / domain
    **{f"skill:{s}": "business" for s in [
        "Copywriting", "Social Media", "SEO", "Content Strategy", "Email Marketing", "Paid Advertising",
        "Brand Management", "Budgeting", "Recruiting", "Interviewing", "Onboarding", "Employment Law",
        "Performance Management", "Data Storytelling"]},
    # leadership / people
    **{f"skill:{s}": "leadership" for s in [
        "Mentoring", "People Management", "Leadership", "Technical Leadership", "Stakeholder Management",
        "Coaching", "Change Management", "Conflict Resolution"]},
}


def category(token: str) -> str:
    if token.startswith("role:"):
        return "role"
    return CATEGORY_OF.get(token, "other")


def all_tokens() -> list[str]:
    """Every token in the ontology, sorted: the fixed vocabulary shared by pretraining and fine-tuning."""
    toks = set()
    for g in GRAPH.values():
        for a, outs in g.items():
            toks.add(a)
            toks.update(outs)
    for e in ENTRY.values():
        toks.update(e)
    return sorted(toks)


def valid_next(department: str, token: str) -> set[str]:
    """Oracle: tokens the graph says may follow `token` in `department`."""
    return set(GRAPH.get(department, {}).get(token, {}))
