"""Reproduce all reported results; categories are coarse relevance proxies."""
import csv, hashlib, json, platform, time
from collections import Counter
from pathlib import Path
import numpy as np
import sklearn
from search import SearchEngine, load_documents, preprocess

ROOT=Path(__file__).parent
DATA=ROOT/'data/bbc-text.csv'
QUERIES=[('business','stock market shares'),('business','economic growth inflation'),
 ('sport','football league team'),('sport','tennis tournament champion'),
 ('politics','election government parliament'),('politics','minister political party'),
 ('tech','mobile phone technology'),('tech','computer software internet'),
 ('entertainment','film cinema awards'),('entertainment','music album singer')]

def main():
    start=time.perf_counter()
    e=SearchEngine(load_documents(DATA))
    build=time.perf_counter()-start
    raw=list(csv.DictReader(DATA.open(encoding='utf-8')))
    stats={'raw_rows':len(raw),'indexed_documents':len(e.docs),
      'duplicates_removed':len(raw)-len(e.docs),'categories':dict(Counter(d['category'] for d in e.docs)),
      'vocabulary':len(e.postings),'raw_vocabulary':len(set(t for d in e.docs for t in __import__('re').findall(r'[a-z]{2,}',d['text'].lower()))),
      'average_tokens':e.avg_length,'tfidf_shape':list(e.matrix.shape),'tfidf_nonzeros':e.matrix.nnz,
      'build_seconds':build,'sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
      'python':platform.python_version(),'sklearn':sklearn.__version__}
    output={'statistics':stats,'preprocessing_example':preprocess('The MOBILE phones are changing technology! 2026'),
       'boolean':{},'examples':{},'evaluation':[]}
    for q in ['mobile','mobile AND phone','mobile OR phone','mobile AND NOT phone']:
        ids=e.boolean(q)
        output['boolean'][q]={'count':len(ids),'first_ids':[e.docs[i]['id'] for i in ids[:10]]}
    for model in ['tfidf','bm25']:
        output['examples'][model]=e.search('mobile phone technology',model)
        for category,q in QUERIES:
            results=e.search(q,model)
            rel=[int(r['category']==category) for r in results]
            dcg=sum(r/np.log2(i+2) for i,r in enumerate(rel))
            idcg=sum(1/np.log2(i+2) for i in range(5))
            rank=next((i+1 for i,r in enumerate(rel) if r),None)
            output['evaluation'].append({'model':model,'query':q,'category':category,
                'p_at_5':sum(rel)/5,'ndcg_at_5':float(dcg/idcg),'rr':1/rank if rank else 0})
    output['summary']={m:{k:float(np.mean([r[k] for r in output['evaluation'] if r['model']==m]))
       for k in ['p_at_5','ndcg_at_5','rr']} for m in ['tfidf','bm25']}
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/results.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
