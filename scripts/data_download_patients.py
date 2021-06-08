
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

# Patient 1
# Patient%201.zip

#Patient 1521
#201521.zip

# Example URL for patient 1521
# http://ictcf.biocuckoo.cn/patient/CT/Patient%201521.zip


for i in range(1124, 1521):
  try:
    download_and_unzip(
        f'http://ictcf.biocuckoo.cn/patient/CT/Patient%20{i}.zip', extract_to='/Volumes/Data EXT SSD/ALL')
  except Exception:
    pass







