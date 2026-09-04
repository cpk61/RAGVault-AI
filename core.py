from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str

class RAGVault:
    def __init__(self, chunk_words=80):
        self.chunk_words = chunk_words
        self.chunks = []
        self.vec = None
        self.matrix = None

    def add_document(self, doc_id: str, text: str):
        words = text.split()
        for i in range(0, len(words), self.chunk_words):
            self.chunks.append(Chunk(doc_id, i // self.chunk_words, ' '.join(words[i:i+self.chunk_words])))
        corpus = [c.text for c in self.chunks]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.matrix = self.vec.fit_transform(corpus)

    def query(self, q: str, k=3, threshold=.05):
        if not self.chunks:
            return {'answer': 'No documents indexed.', 'citations': [], 'confidence': 0.0}
        sims = cosine_similarity(self.vec.transform([q]), self.matrix)[0]
        idx = sims.argsort()[::-1][:k]
        hits = [(self.chunks[i], float(sims[i])) for i in idx if sims[i] >= threshold]
        if not hits:
            return {'answer': 'Insufficient evidence in indexed documents.', 'citations': [], 'confidence': 0.0}
        answer = ' '.join(h[0].text for h in hits[:2])
        return {
            'answer': answer,
            'citations': [
                {'doc_id': c.doc_id, 'chunk_id': c.chunk_id, 'score': round(s, 4)}
                for c, s in hits
            ],
            'confidence': round(max(s for _, s in hits), 4),
        }
