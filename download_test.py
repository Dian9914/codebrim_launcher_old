import requests

import base64

def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl

url = 'https://uses0-my.sharepoint.com/:u:/g/personal/edusotcas_alum_us_es/EbkdLdANdkNDmLHr5Jeifx8Bw-8-UBoUY0FkDMONxQ5pFQ?download=1'
#url = create_onedrive_directdownload(url)
r = requests.get(url, allow_redirects=True)
print(r.headers.get('content-type'))
open('test.zip', 'wb').write(r.content)