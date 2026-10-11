"""BBC news retrieval: Boolean, TF-IDF cosine and BM25."""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


def preprocess(text):
    return [t for t in re.findall(r'[a-z]{2,}', text.lower())
            if t not in ENGLISH_STOP_WORDS]


def load_documents(path):
    documents, seen = [], set()
    with open(path, encoding='utf-8', newline='') as f:
        for row_number, row in enumerate(csv.DictReader(f), 1):
            text = row['text'].strip()
            if not text or text in seen:
                continue
            seen.add(text)
            documents.append({'id': row_number, 'text': text,
                              'category': row['category']})
    return documents


class SearchEngine:
    def __init__(self, documents, k1=1.5, b=0.75):
        if not documents:
            raise ValueError('The collection is empty.')
        self.docs, self.k1, self.b = documents, k1, b
        self.tokens = [preprocess(d['text']) for d in documents]
        self.postings = defaultdict(dict)
        for i, tokens in enumerate(self.tokens):
            for term, frequency in Counter(tokens).items():
                self.postings[term][i] = frequency
        self.lengths = np.array([len(t) for t in self.tokens])
        self.avg_length = float(self.lengths.mean())
        self.vectorizer = TfidfVectorizer(
            analyzer=preprocess, sublinear_tf=True, norm='l2')
        self.matrix = self.vectorizer.fit_transform(d['text'] for d in documents)

    def boolean(self, query):
        """Explicit operators; precedence NOT > AND > OR; parentheses supported."""
        parts = re.findall(r'\(|\)|[A-Za-z]+', query)
        position = 0
        universe = set(range(len(self.docs)))

        def atom():
            nonlocal position
            if position >= len(parts):
                raise ValueError('Missing Boolean operand.')
            token = parts[position]
            position += 1
            if token == 'NOT':
                return universe - atom()
            if token == '(':
                result = expression()
                if position >= len(parts) or parts[position] != ')':
                    raise ValueError('Missing closing parenthesis.')
                position += 1
                return result
            if token in {'AND', 'OR', ')'}:
                raise ValueError('Unexpected Boolean operator.')
            terms = preprocess(token)
            return set(self.postings.get(terms[0], {})) if terms else set()

        def conjunction():
            nonlocal position
            result = atom()
            while position < len(parts) and parts[position] == 'AND':
                position += 1
                result &= atom()
            return result

        def expression():
            nonlocal position
            result = conjunction()
            while position < len(parts) and parts[position] == 'OR':
                position += 1
                result |= conjunction()
            return result

        if not parts:
            return []
        result = expression()
        if position != len(parts):
            raise ValueError('Use explicit uppercase AND, OR or NOT.')
        return sorted(result, key=lambda i: self.docs[i]['id'])

    def search(self, query, model='bm25', top_k=5):
        if top_k < 1:
            raise ValueError('top_k must be positive.')
        if model == 'tfidf':
            q = self.vectorizer.transform([query])
            scores = (self.matrix @ q.T).toarray().ravel()
        elif model == 'bm25':
            scores = np.zeros(len(self.docs))
            for term in set(preprocess(query)):
                posting = self.postings.get(term, {})
                df = len(posting)
                if not df:
                    continue
                idf = np.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))
                for i, tf in posting.items():
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * self.lengths[i] / self.avg_length)
                    scores[i] += idf * tf * (self.k1 + 1) / denominator
        else:
            raise ValueError('Unknown ranked retrieval model.')
        ranked = sorted(np.flatnonzero(scores > 0),
                        key=lambda i: (-scores[i], self.docs[i]['id']))[:top_k]
        return [{'id': self.docs[i]['id'], 'category': self.docs[i]['category'],
                 'score': float(scores[i]), 'preview': self.docs[i]['text'][:90]}
                for i in ranked]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--model', choices=['bm25', 'tfidf', 'boolean'], default='bm25')
    parser.add_argument('--data', default=str(Path(__file__).parent / 'data/bbc-text.csv'))
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    engine = SearchEngine(load_documents(args.data))
    if args.model == 'boolean':
        print([engine.docs[i]['id'] for i in engine.boolean(args.query)])
    else:
        for rank, result in enumerate(engine.search(args.query, args.model, args.top_k), 1):
            print(f"{rank}. ID {result['id']} | {result['category']} | {result['score']:.4f}")
            print('   ' + result['preview'])
