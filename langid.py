"""Character n-gram language identifier with mixed-language segmentation.
Pure Python, no ML libraries: n-gram frequency profiles + Naive Bayes scoring,
then a Viterbi pass over words so code-switched text gets split by language."""
import collections, glob, math, os, re

ORDERS = (1, 2, 3)
TOKEN = re.compile(r"[^\s\d.,;:!?\"()\[\]{}<>|/\\\u0964\u0965\u2026\u2014\u2013\-]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def ngrams(word):
    w = f" {word} "
    for n in ORDERS:
        for i in range(len(w) - n + 1):
            yield n, w[i:i + n]


class LangID:
    def __init__(self, data_dir, alpha=0.05, switch_penalty=9.0):
        self.alpha, self.switch = alpha, switch_penalty
        self.counts, self.totals = {}, {}
        for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
            lang = os.path.splitext(os.path.basename(path))[0]
            c = collections.Counter()
            with open(path, encoding="utf8") as f:
                for w in tokenize(f.read()):
                    c.update(ngrams(w))
            self.counts[lang] = c
            self.totals[lang] = {n: sum(v for (o, _), v in c.items() if o == n) for n in ORDERS}
        self.langs = list(self.counts)
        self.vocab = {n: len({g for c in self.counts.values() for (o, g) in c if o == n}) for n in ORDERS}

    def word_logp(self, word, lang):
        c, t, a = self.counts[lang], self.totals[lang], self.alpha
        lp = 0.0
        for n, g in ngrams(word):
            lp += math.log((c.get((n, g), 0) + a) / (t[n] + a * self.vocab[n]))
        return lp

    @staticmethod
    def softmax(xs):
        m = max(xs)
        e = [math.exp(x - m) for x in xs]
        s = sum(e)
        return [v / s for v in e]

    def detect(self, text):
        toks = tokenize(text)
        if not toks:
            return {"primary": None, "mixed": False, "scores": [], "words": [], "segments": []}
        K = len(self.langs)
        em = [[self.word_logp(t, l) for l in self.langs] for t in toks]
        # Viterbi: staying in a language is free, switching costs `switch` nats
        score, back = em[0][:], []
        for e in em[1:]:
            best = max(range(K), key=score.__getitem__)
            new, bp = [], []
            for k in range(K):
                if score[k] >= score[best] - self.switch:
                    new.append(score[k] + e[k]); bp.append(k)
                else:
                    new.append(score[best] - self.switch + e[k]); bp.append(best)
            score = new; back.append(bp)
        k = max(range(K), key=score.__getitem__)
        path = [k]
        for bp in reversed(back):
            k = bp[k]; path.append(k)
        path.reverse()

        words = []
        for t, e, p in zip(toks, em, path):
            words.append({"w": t, "lang": self.langs[p], "conf": round(self.softmax(e)[p], 3)})
        segments = []
        for w in words:
            if segments and segments[-1]["lang"] == w["lang"]:
                segments[-1]["text"] += " " + w["w"]
            else:
                segments.append({"lang": w["lang"], "text": w["w"]})

        share = collections.Counter()
        for w in words:
            share[w["lang"]] += len(w["w"])
        total = sum(share.values())
        ranked = share.most_common()
        mixed = len(ranked) > 1 and ranked[1][1] / total >= 0.15 and len(words) >= 3
        if mixed:
            scores = [{"lang": l, "pct": round(100 * v / total, 1)} for l, v in ranked]
        else:
            tot = [sum(r[i] for r in em) / len(toks) * 0.4 for i in range(K)]
            pr = self.softmax(tot)
            scores = sorted(({"lang": l, "pct": round(100 * p, 1)} for l, p in zip(self.langs, pr)),
                            key=lambda s: -s["pct"])[:5]
            segments = [{"lang": scores[0]["lang"], "text": " ".join(toks)}]
            for w in words:
                w["lang"] = scores[0]["lang"]
        return {"primary": scores[0]["lang"], "mixed": mixed, "scores": scores,
                "words": words, "segments": segments,
                "ngram_count": sum(1 for t in toks for _ in ngrams(t))}
