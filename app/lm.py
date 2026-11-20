"""Milestone 3: a small neural language model over career tokens, in plain numpy.

Why numpy and not a transformer: the "language" here is ~60 tokens, so a big pretrained text model
adds nothing but weight. This model is a causal LM with the same recipe, though:

    p(next token | the last K tokens, everything earlier as a bag, department)

  * **pretraining**            train from random init on a large, noisy *industry* corpus;
  * **continued pretraining**  keep the weights, keep the same next-token objective, and carry on
                               training on the (small) *company* corpus with a lower learning rate.

Tokens already held by the employee are masked out of the softmax, so the model only ranks genuinely new steps.
"""
from __future__ import annotations

import json

import numpy as np

from .ontology import DEPARTMENTS, all_tokens

PAD = 0  # left-padding for short histories


class Vocab:
    def __init__(self, tokens: list[str] | None = None, departments: list[str] | None = None):
        self.tokens = ["<pad>"] + (tokens or all_tokens())
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.departments = departments or list(DEPARTMENTS)
        self.dept_index = {d: i for i, d in enumerate(self.departments)}

    def __len__(self) -> int:
        return len(self.tokens)

    def ids(self, toks: list[str]) -> list[int]:
        return [self.index[t] for t in toks if t in self.index]


def make_examples(vocab: Vocab, sequences, k: int):
    """One example per (prefix -> next token). Returns ctx (N,k), hist multi-hot (N,V), dept (N,), target (N,)."""
    ctx, hist, dept, target = [], [], [], []
    V = len(vocab)
    for s in sequences:
        ids = vocab.ids(s.tokens)
        d = vocab.dept_index.get(s.department, 0)
        for t in range(1, len(ids)):
            window = ids[max(0, t - k):t]
            ctx.append([PAD] * (k - len(window)) + window)
            h = np.zeros(V, dtype=np.float32)
            h[ids[:t]] = 1.0
            hist.append(h)
            dept.append(d)
            target.append(ids[t])
    if not ctx:
        return (np.zeros((0, k), int), np.zeros((0, V), np.float32), np.zeros(0, int), np.zeros(0, int))
    return np.array(ctx), np.stack(hist), np.array(dept), np.array(target)


class CareerLM:
    """Embedding of the last K tokens + mean-embedding of the whole history + department -> tanh layer -> softmax."""

    def __init__(self, vocab: Vocab, k: int = 4, dim: int = 24, hidden: int = 64, seed: int = 0):
        self.vocab, self.k, self.dim, self.hidden = vocab, k, dim, hidden
        V, D = len(vocab), len(vocab.departments)
        rng = np.random.default_rng(seed)
        in_dim = (k + 1) * dim + 8
        self.p = {
            "E": rng.normal(0, 0.1, (V, dim)),
            "Dep": rng.normal(0, 0.1, (D, 8)),
            "W1": rng.normal(0, 1 / np.sqrt(in_dim), (in_dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, V)),
            "b2": np.zeros(V),
        }
        self._adam = None
        self.history: list[dict] = []  # training log: phase, epoch, train/val loss

    # ---- forward / backward -------------------------------------------------------------
    def _forward(self, ctx, hist, dept):
        p = self.p
        B = ctx.shape[0]
        e_ctx = p["E"][ctx].reshape(B, -1)                         # (B, k*dim)
        n = np.maximum(hist.sum(1, keepdims=True), 1.0)
        bag = (hist @ p["E"]) / n                                  # (B, dim)
        x = np.concatenate([e_ctx, bag, p["Dep"][dept]], axis=1)
        h = np.tanh(x @ p["W1"] + p["b1"])
        z = h @ p["W2"] + p["b2"]
        z = np.where(hist > 0, -1e9, z)                            # held tokens can't be next
        z[:, PAD] = -1e9
        z = z - z.max(1, keepdims=True)
        probs = np.exp(z)
        probs /= probs.sum(1, keepdims=True)
        return probs, (ctx, hist, dept, n, x, h)

    def loss(self, data) -> float:
        ctx, hist, dept, target = data
        if len(target) == 0:
            return float("nan")
        probs, _ = self._forward(ctx, hist, dept)
        return float(-np.log(probs[np.arange(len(target)), target] + 1e-12).mean())

    def _grads(self, data):
        ctx, hist, dept, target = data
        probs, (ctx, hist, dept, n, x, h) = self._forward(ctx, hist, dept)
        B = len(target)
        p, g = self.p, {}
        dz = probs.copy()
        dz[np.arange(B), target] -= 1.0
        dz /= B
        g["W2"], g["b2"] = h.T @ dz, dz.sum(0)
        dpre = (dz @ p["W2"].T) * (1 - h ** 2)
        g["W1"], g["b1"] = x.T @ dpre, dpre.sum(0)
        dx = dpre @ p["W1"].T
        d = self.dim
        dE = np.zeros_like(p["E"])
        np.add.at(dE, ctx.reshape(-1), dx[:, : self.k * d].reshape(-1, d))
        dE += hist.T @ (dx[:, self.k * d : (self.k + 1) * d] / n)
        dDep = np.zeros_like(p["Dep"])
        np.add.at(dDep, dept, dx[:, (self.k + 1) * d :])
        g["E"], g["Dep"] = dE, dDep
        return g

    # ---- training -----------------------------------------------------------------------
    def fit(self, train_seqs, val_seqs=None, *, epochs=40, lr=0.01, batch=64, weight_decay=1e-4,
            seed=0, phase="pretrain", patience=6, verbose=False):
        """Next-token training. Calling it again on new data *continues* from the current weights
        (that is the whole of 'continued pretraining'); Adam state is reset and `lr` is usually lower."""
        tr = make_examples(self.vocab, train_seqs, self.k)
        va = make_examples(self.vocab, val_seqs, self.k) if val_seqs else None
        rng = np.random.default_rng(seed)
        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.p.items()}
        step, best, best_p, bad = 0, np.inf, None, 0
        n = len(tr[3])
        for ep in range(1, epochs + 1):
            order = rng.permutation(n)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                g = self._grads(tuple(a[idx] for a in tr))
                step += 1
                for name, grad in g.items():
                    grad = grad + weight_decay * self.p[name]
                    m, v = self._adam[name]
                    m[:] = 0.9 * m + 0.1 * grad
                    v[:] = 0.999 * v + 0.001 * grad ** 2
                    mh, vh = m / (1 - 0.9 ** step), v / (1 - 0.999 ** step)
                    self.p[name] -= lr * mh / (np.sqrt(vh) + 1e-8)
            tl = self.loss(tr)
            vl = self.loss(va) if va else float("nan")
            self.history.append({"phase": phase, "epoch": ep, "train_loss": tl, "val_loss": vl})
            if verbose:
                print(f"[{phase}] epoch {ep:3d}  train {tl:.4f}  val {vl:.4f}")
            if va is not None:                     # early stopping on validation loss, keep the best weights
                if vl < best - 1e-4:
                    best, best_p, bad = vl, {k: v.copy() for k, v in self.p.items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_p is not None:
            self.p = best_p
        return self

    # ---- inference ----------------------------------------------------------------------
    def next_probs(self, department: str, history: list[str]) -> np.ndarray:
        """Probability over the vocabulary for the next token (held tokens get 0)."""
        ids = self.vocab.ids(history)
        window = ids[-self.k:]
        ctx = np.array([[PAD] * (self.k - len(window)) + window])
        h = np.zeros((1, len(self.vocab)), dtype=np.float32)
        h[0, ids] = 1.0
        d = np.array([self.vocab.dept_index.get(department, 0)])
        probs, _ = self._forward(ctx, h, d)
        return probs[0]

    def perplexity(self, seqs) -> float:
        """exp(mean next-token cross-entropy) over every prefix of every sequence."""
        return float(np.exp(self.loss(make_examples(self.vocab, seqs, self.k))))

    # ---- persistence --------------------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, k=self.k, dim=self.dim, hidden=self.hidden,
                 tokens=np.array(self.vocab.tokens), departments=np.array(self.vocab.departments),
                 history=np.array([json.dumps(h) for h in self.history]), **self.p)

    @classmethod
    def load(cls, path: str) -> "CareerLM":
        z = np.load(path, allow_pickle=False)
        vocab = Vocab(list(z["tokens"][1:]), list(z["departments"]))
        m = cls(vocab, int(z["k"]), int(z["dim"]), int(z["hidden"]))
        m.p = {k: z[k] for k in m.p}
        m.history = [json.loads(str(h)) for h in z["history"]]
        return m
