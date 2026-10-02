import html as html_lib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_data(data_dir):
    data_dir = Path(data_dir)
    articles = pd.read_feather(data_dir / "articles.f")
    test = pd.read_feather(data_dir / "test.f")
    calib_path = data_dir / "calibration.f"
    calibration = pd.read_feather(calib_path) if calib_path.exists() else None
    return articles, calibration, test


def parse_ground_truth(gt):
    if gt is None or (isinstance(gt, float) and np.isnan(gt)):
        return []
    return [int(x) for x in str(gt).split()]


_WS_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"<[A-Za-zА-Яа-яЁё_]{0,40}>")  # <DATE>, <NAME> и т.п.


def clean_query(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = _PLACEHOLDER_RE.sub(" ", str(raw))
    text = html_lib.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def clean_html(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = str(raw)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def prepare_articles(articles, max_lexical_chars=20000, max_dense_chars=2000, title_boost=2):
    df = articles.copy()
    df["clean_title"] = df["title"].apply(clean_html)
    df["clean_body"] = df["body"].apply(clean_html).str.slice(0, max_lexical_chars)
    # заголовок дублируем, чтобы поднять его вес в BM25
    title_rep = (df["clean_title"] + " ") * title_boost
    df["text_lexical"] = (title_rep + df["clean_body"]).str.strip()
    # для трансформеров тело режем (модель всё равно смотрит ~512 токенов)
    df["text_dense"] = (
        df["clean_title"] + ". " + df["clean_body"].str.slice(0, max_dense_chars)
    ).str.strip()
    return df


STOPWORDS_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
    "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была",
    "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",
    "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "были",
    "куда", "зачем", "всех", "никогда", "можно", "при", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них",
    "какая", "много", "разве", "эту", "моя", "свою", "этой", "перед", "лучше",
    "чуть", "том", "такой", "им", "более", "всю", "между",
    # приветствия/вежливость — в обращениях в поддержку это шум
    "здравствуйте", "здравствуй", "привет", "добрый", "доброе", "доброго",
    "день", "вечер", "утро", "пожалуйста", "подскажите", "подскажете",
    "подскажи", "скажите", "спасибо", "благодарю", "помогите",
}

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _get_stemmer():
    try:
        from nltk.stem.snowball import SnowballStemmer
        return SnowballStemmer("russian")
    except Exception:
        return None


def tokenize_ru(text, stemmer=None, min_len=2):
    tokens = _TOKEN_RE.findall(text.lower())
    out = []
    for t in tokens:
        if len(t) < min_len or t in STOPWORDS_RU:
            continue
        out.append(stemmer.stem(t) if stemmer is not None else t)
    return out


def build_bm25(article_tokens):
    from rank_bm25 import BM25Okapi
    return BM25Okapi(article_tokens)


# e5 требует префиксы query:/passage:
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


def resolve_device(device="auto"):
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_bi_encoder(model_name, device="cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=resolve_device(device))


def load_cross_encoder(model_name, device="cpu", max_length=512):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=resolve_device(device), max_length=max_length)


def encode_passages(model, texts, batch_size=32, verbose=False):
    inputs = [E5_PASSAGE_PREFIX + t for t in texts]
    return model.encode(inputs, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=verbose, convert_to_numpy=True)


def encode_queries(model, texts, batch_size=32, verbose=False):
    inputs = [E5_QUERY_PREFIX + t for t in texts]
    return model.encode(inputs, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=verbose, convert_to_numpy=True)


def _top_indices(scores, m):
    m = min(m, len(scores))
    idx = np.argpartition(-scores, m - 1)[:m]
    return idx[np.argsort(-scores[idx])]


def rrf_fuse(ranked_lists, weights=None, rrf_k=60, top_k=100):
    # Reciprocal Rank Fusion: документ получает 1/(rrf_k+rank) от каждого списка
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_idx in enumerate(lst, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + w / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_idx for doc_idx, _ in ordered[:top_k]]


def average_precision_at_k(predicted, relevant, k=10):
    # AP@k = (1/min(R,k)) * sum precision@i по попаданиям
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc in enumerate(predicted[:k], start=1):
        if doc in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def mean_average_precision_at_k(predictions, relevants, k=10):
    aps = [average_precision_at_k(p, r, k) for p, r in zip(predictions, relevants)]
    return float(np.mean(aps)) if aps else 0.0


@dataclass
class Config:
    bi_encoder: str = "intfloat/multilingual-e5-base"
    cross_encoder: str = "DiTy/cross-encoder-russian-msmarco"
    device: str = "cpu"
    max_lexical_chars: int = 20000
    max_dense_chars: int = 2000
    title_boost: int = 2
    m_per_ranker: int = 200
    rrf_k: int = 60
    rrf_weights: tuple = (2.0, 1.0)  # (BM25, эмбеддинги)
    candidates_for_rerank: int = 50
    cross_encoder_max_length: int = 512
    top_n: int = 10
    batch_size: int = 32
    verbose: bool = True
    cache_dir: str = "cache"


class Retriever:
    def __init__(self, articles, config=None):
        self.cfg = config or Config()
        self.articles = prepare_articles(
            articles,
            max_lexical_chars=self.cfg.max_lexical_chars,
            max_dense_chars=self.cfg.max_dense_chars,
            title_boost=self.cfg.title_boost,
        )
        self.article_ids = self.articles["article_id"].tolist()
        self._stemmer = None
        self._bm25 = None
        self._doc_embs = None
        self._bi = None
        self._ce = None

    def build(self, use_dense=True):
        self._stemmer = _get_stemmer()
        if self.cfg.verbose:
            print("Токенизация статей и построение BM25...")
        tokens = [tokenize_ru(t, self._stemmer) for t in self.articles["text_lexical"]]
        self._bm25 = build_bm25(tokens)

        if use_dense:
            self._bi = load_bi_encoder(self.cfg.bi_encoder, self.cfg.device)
            texts = self.articles["text_dense"].tolist()
            cache_file = self._emb_cache_path(texts)
            if cache_file is not None and cache_file.exists():
                if self.cfg.verbose:
                    print(f"Эмбеддинги статей из кэша: {cache_file.name}")
                self._doc_embs = np.load(cache_file)
            else:
                if self.cfg.verbose:
                    print(f"Кодирование статей ({self.cfg.bi_encoder})...")
                self._doc_embs = encode_passages(
                    self._bi, texts, batch_size=self.cfg.batch_size, verbose=self.cfg.verbose)
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_file, self._doc_embs)
        return self

    def _emb_cache_path(self, texts):
        if not self.cfg.cache_dir:
            return None
        import hashlib
        h = hashlib.sha256(self.cfg.bi_encoder.encode())
        for t in texts:
            h.update(t.encode())
            h.update(b"\x00")
        return Path(self.cfg.cache_dir) / f"doc_embs_{h.hexdigest()[:16]}.npy"

    def bm25_scores(self, query):
        return np.asarray(self._bm25.get_scores(tokenize_ru(query, self._stemmer)))

    def dense_sims(self, query_emb):
        return self._doc_embs @ query_emb

    def _candidates_for_query(self, bm25_s, dense_s):
        bm25_top = _top_indices(bm25_s, self.cfg.m_per_ranker)
        if dense_s is None:
            return list(bm25_top[:self.cfg.candidates_for_rerank])
        dense_top = _top_indices(dense_s, self.cfg.m_per_ranker)
        return rrf_fuse([bm25_top, dense_top], weights=list(self.cfg.rrf_weights),
                        rrf_k=self.cfg.rrf_k, top_k=self.cfg.candidates_for_rerank)

    def _rerank(self, query, cand_idx):
        if self._ce is None:
            if self.cfg.verbose:
                print(f"Загрузка реранкера ({self.cfg.cross_encoder})...")
            self._ce = load_cross_encoder(self.cfg.cross_encoder, self.cfg.device,
                                          max_length=self.cfg.cross_encoder_max_length)
        pairs = [[query, self.articles["text_dense"].iloc[i]] for i in cand_idx]
        scores = self._ce.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [cand_idx[i] for i in order]

    def rank(self, queries, method="hybrid", top_n=None):
        # method: bm25 | dense | hybrid | rerank
        top_n = top_n or self.cfg.top_n
        use_dense = method in ("dense", "hybrid", "rerank")
        queries = [clean_query(q) for q in queries]

        q_embs = None
        if use_dense:
            q_embs = encode_queries(self._bi, queries, batch_size=self.cfg.batch_size,
                                    verbose=self.cfg.verbose)

        results = []
        iterator = range(len(queries))
        if self.cfg.verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"Ранжирование ({method})")
            except Exception:
                pass

        for qi in iterator:
            query = queries[qi]
            bm25_s = self.bm25_scores(query)
            dense_s = self.dense_sims(q_embs[qi]) if use_dense else None

            if method == "bm25":
                idx = list(_top_indices(bm25_s, top_n))
            elif method == "dense":
                idx = list(_top_indices(dense_s, top_n))
            elif method == "hybrid":
                idx = self._candidates_for_query(bm25_s, dense_s)[:top_n]
            elif method == "rerank":
                cand = self._candidates_for_query(bm25_s, dense_s)
                idx = self._rerank(query, cand)[:top_n]
            else:
                raise ValueError(f"Неизвестный method: {method}")

            results.append([self.article_ids[i] for i in idx])
        return results


def format_answers(query_ids, rankings):
    answers = [" ".join(str(a) for a in ids) for ids in rankings]
    return pd.DataFrame({"query_id": query_ids, "answer": answers})


def validate_answers(answer, test, articles, top_n=10):
    errors = []
    valid_ids = set(articles["article_id"].tolist())

    if list(answer.columns) != ["query_id", "answer"]:
        errors.append(f"Колонки должны быть ['query_id','answer'], а не {list(answer.columns)}")

    ans_qids = answer["query_id"].tolist()
    test_qids = test["query_id"].tolist()
    if set(ans_qids) != set(test_qids):
        miss = set(test_qids) - set(ans_qids)
        extra = set(ans_qids) - set(test_qids)
        if miss:
            errors.append(f"Пропущены query_id: {len(miss)} шт.")
        if extra:
            errors.append(f"Лишние query_id: {len(extra)} шт.")
    if len(ans_qids) != len(set(ans_qids)):
        errors.append("Дублирующиеся строки")

    for _, row in answer.iterrows():
        ids = str(row["answer"]).split()
        if len(ids) == 0:
            errors.append(f"query_id={row['query_id']}: пустой ответ")
            continue
        if len(ids) > top_n:
            errors.append(f"query_id={row['query_id']}: больше {top_n} статей")
        if len(ids) != len(set(ids)):
            errors.append(f"query_id={row['query_id']}: повтор article_id")
        for a in ids:
            if not a.lstrip("-").isdigit() or int(a) not in valid_ids:
                errors.append(f"query_id={row['query_id']}: article_id '{a}' нет в articles")
                break

    return errors
