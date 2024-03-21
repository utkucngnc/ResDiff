import gdown

url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
output = 'satelite-dset.zip'
gdown.download(url, output, quiet=False)