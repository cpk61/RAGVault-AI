from core import RAGVault

def test_retrieval_with_citation():
    r = RAGVault(chunk_words=20)
    r.add_document('policy', 'Refunds are available within 30 days with a valid receipt. Shipping fees are non-refundable.')
    out = r.query('refund period')
    assert out['citations'][0]['doc_id'] == 'policy'
    assert out['confidence'] > 0

def test_empty_guard():
    assert RAGVault().query('x')['confidence'] == 0.0
