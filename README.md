# Skill Path Triage Assistant

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/et82vbu3pf/badge.svg)](https://msrit.ask.thili.ai/b/et82vbu3pf)

The Skill Path Triage Assistant helps HR managers understand and guide employee career progression. By analyzing historical sequences of acquired skills and job roles, it identifies common career paths and predicts likely next skills or roles, streamlining training recommendations and addressing skill gaps within the HR & Hiring domain.

## What you are building

**Goal.** You will build a Skill Path Triage Assistant that helps HR managers analyze employee skill acquisition sequences to suggest relevant next courses or identify common skill gaps for team training.

**Done when.** The project is done when you have applied frequent itemset mining and language model continued pretraining to skill sequences, and the model's ability to suggest relevant next steps in career development has been evaluated and is accessible via a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Define employee skill sets or job roles as sequences over time.**  
      Hint: Data Preparation: Format skill sets or job roles into sequential data.
- [x] **2. Mine frequent skill/role transition patterns using `thilimarketbasket`.**  
      Hint: Frequent Itemset Mining: Apply `lfretail-lfmarketbasket` techniques to discover transition patterns.
- [x] **3. Train an LM on these sequences to predict the next skill or role.**  
      Hint: Language Model Training: Use continued pretraining on the skill sequences to predict next items.
- [x] **4. Evaluate the model's ability to suggest relevant next steps in career development.**  
      Hint: Model Evaluation: Design metrics to assess the relevance of the model's career development suggestions.
- [x] **5. Provide a CLI to input current skills/role and get suggestions for next steps.**  
      Hint: Application Development: Create a command-line interface for user interaction and suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli hello
pytest

# the whole pipeline (everything lands in data/, which git ignores)
python -m app.cli generate-data                # M1  synthetic employee histories (company + industry corpora)
python -m app.cli mine --department Data       # M2  frequent skill sets and transitions
python -m app.cli train                        # M3  pretrain on industry, continue pretraining on company
python -m app.cli evaluate                     # M4  compare all models on held-out employees
python -m app.cli suggest --department Data --skills "SQL,Python,Statistics"          # M5
python -m app.cli suggest --department Engineering --skills "Git,Python" --kind skill --seniority junior
python -m app.cli gaps --team my_team.csv      # common gaps for a team (CSV: employee_id,department,date,item)
```

`suggest` filters: `--kind skill|role`, `--category technical,leadership,...`, `--seniority junior|mid|senior`,
`--department-filter`. `--model` picks which recommender answers (default `lm-continued`).

To use real data, put an HR export with columns `employee_id,department,date,item` at `data/company.csv`
(item is `skill:Name` or `role:Name`) and skip `generate-data`. Departments must be one of those in `app/ontology.py`,
and the model's vocabulary is the ontology's token list, so unknown items are ignored.

## How it works

| Milestone | Where | What it does |
|---|---|---|
| 1 | `app/core.py` | Dated events -> one time-ordered, de-duplicated token sequence per employee. Seniority is derived from time since the first event. |
| 2 | `app/core.py` | Apriori for skill sets that co-occur; gap-bounded sequential patterns for transitions, with support, confidence and lift. |
| 3 | `app/lm.py` | A small causal LM in numpy (last 4 tokens + whole-history bag + department -> softmax). Trained on a noisy industry corpus, then training continues on the company corpus at a lower learning rate. |
| 4 | `app/core.py` | Replays every prefix of held-out employees; hit@k, MRR, NDCG, horizon hit, graph validity, coverage, diversity. |
| 5 | `app/cli.py` | `suggest` and `gaps`, with filters. |

## Results (synthetic data, seed 0: 1,500 company and 6,000 industry employees, 300 held out, 1,969 prefixes)

| model | hit@1 | hit@5 | MRR | NDCG@5 | horizon hit@5 | valid@5 |
|---|---|---|---|---|---|---|
| popularity | 0.172 | 0.566 | 0.307 | 0.371 | 0.866 | 0.241 |
| bigram (prev. skill + dept) | 0.492 | 0.947 | 0.688 | 0.754 | 0.975 | 0.459 |
| mined rules | 0.402 | 0.923 | 0.598 | 0.679 | 0.968 | 0.441 |
| LM, company only | 0.474 | 0.948 | 0.680 | 0.748 | 0.969 | 0.459 |
| LM, industry only | 0.468 | 0.947 | 0.670 | 0.741 | 0.975 | 0.459 |
| **LM, continued pretraining** | 0.483 | 0.950 | 0.687 | 0.755 | 0.974 | 0.459 |

Test perplexity: company-only 3.64, industry-only 3.98, **continued 3.27**.

How to read this honestly:
- Continued pretraining clearly helps the language model itself (perplexity), and it is the best LM on every ranking metric.
- It does **not** clearly beat the simple bigram baseline: the differences are within noise for one seed. That is expected,
  because the synthetic careers are close to first-order Markov chains. Real careers have longer-range structure, so
  re-run `evaluate` on real data before trusting any ranking between these two.
- All of these numbers come from synthetic data generated from the same graph used for `valid@k`, so they show the
  pipeline works, not that it predicts real careers. `valid@k` is capped below 1 because many steps have fewer than 5
  real successors.
- Low-ranked suggestions (score near 0.00) are noise; read the top two or three.

## Known limitations

- `thilimarketbasket` is listed in `requirements.txt` but is **not used**: it could not be installed or inspected when
  this was built, so M2 uses self-contained Apriori/sequential-pattern code. If that git URL is not reachable, remove the
  line or `pip install` and CI will fail.
- The LM is a from-scratch numpy model, not a pretrained text LM; "continued pretraining" means pretrain then keep
  training the same weights on the company corpus.
- Evaluation covers relevance, breadth and diversity, but there is no human judgement of whether a suggestion is useful.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Enhance the CLI to allow filtering suggestions by department, seniority level, or specific skill categories.
- Implement additional evaluation metrics beyond just relevance, such as diversity of suggestions or coverage of potential career paths.
- Explore different language model architectures or fine-tuning strategies to potentially improve prediction accuracy or efficiency.
