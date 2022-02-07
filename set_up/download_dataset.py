import requests
import argparse

from mmcv import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads zip file from onedrive')
    parser.add_argument('url', help='Onedrive share link')
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    url = args.url
    url = url + '?download=1'
    out_dir = args.out_dir if args.out_dir else './'

    mkdir_or_exist(out_dir)
    print ('Downloading from provided URL')
    r = requests.get(url, allow_redirects=True)
    open(out_dir+'dataset.zip', 'wb').write(r.content)

if __name__ == '__main__':
    main()