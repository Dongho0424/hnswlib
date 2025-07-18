import os.path
import os

# sift-1b
# links = ['ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz',
#          'ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz',
#          'ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz']

# sift-1m
links = [
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
]

# 결과를 저장할 폴더 이름
output_dir = 'sift1m'

os.makedirs('downloads', exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for link in links:
    name = link.rsplit('/', 1)[-1]
    filename = os.path.join('downloads', name)

    if not os.path.isfile(filename):
        print(f'Downloading: {filename}')
        try:
            os.system(f'wget --output-document={filename} {link}')
        except Exception as inst:
            print(inst)
            print('  Encountered unknown error. Continuing.')
    else:
        print(f'Already downloaded: {filename}')

    # .tar.gz 파일이므로 tar로 압축 해제
    if filename.endswith('.tar.gz'):
        # --directory 옵션으로 지정된 폴더에 압축을 풉니다.
        command = f'tar -zxf {filename} --directory {output_dir}'
    else:
        # 이 부분은 실행되지 않습니다.
        unpacked_name = name.replace(".gz", "")
        command = f'cat {filename} | gzip -dc > {os.path.join(output_dir, unpacked_name)}'

    print(f"Unpacking file: {command}")
    os.system(command)

print(f"\n✅ SIFT-1M dataset downloaded and unpacked into the '{output_dir}/sift/' directory.")