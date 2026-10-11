"""Download the public Kaggle archive without embedding the corpus in GitHub."""
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import hashlib

URL='https://www.kaggle.com/api/v1/datasets/download/yufengdev/bbc-fulltext-and-category'
EXPECTED='fdaee0f7451cd8db2709d00e992886fe1c387ee332b8c7b7c1554ed3d3e3382e'

def main():
    destination=Path(__file__).parent/'data/bbc-text.csv'
    with urlopen(Request(URL,headers={'User-Agent':'Mozilla/5.0'}),timeout=120) as response:
        with ZipFile(BytesIO(response.read())) as archive:
            name=next(n for n in archive.namelist() if n.endswith('bbc-text.csv'))
            data=archive.read(name)
    digest=hashlib.sha256(data).hexdigest()
    if digest != EXPECTED:
        raise ValueError('Dataset version changed; review before reproducing results.')
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(data)
    print('Saved',destination,'SHA256',digest)

if __name__=='__main__': main()
