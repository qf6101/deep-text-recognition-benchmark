""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np
from pathlib import Path
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent
import shutil


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache( env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def loadData(tag, gtD, checkValid):
    vocab = set()
    cache = {}
    cnt = 0
    imgDir = gtD.parent / (str(gtD.stem)[:-2] + 'img')
    for gtF in gtD.glob('*'):
        imgF = imgDir / (str(gtF.stem) + '.png')
        if gtF.exists() and imgF.exists():
            with open(str(gtF), 'r') as f:
                label = f.readline().split(',')[-2][1:-1]
            with open(str(imgF), 'rb') as f:
                imageBin = f.read()
            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % str(imgF))
                        continue
                except:
                    print('error occured', str(imgF))
                    continue

            imageKey = 'image-{}'.format(gtF.stem).encode()
            labelKey = 'label-{}'.format(gtF.stem).encode()
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()
            cnt += 1
            for i in label:
                vocab.add(i)
    return tag, cnt, cache, vocab


def createDataset_v2(inputPath, outputPath, checkValid=True, max_workers=50):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image

        params: --inputPath /datadisk2/qfeng/ocr_synthesis/number_outp --outputPath /datadisk2/qfeng/ocr_synthesis/number_lmdb
    """
    fonts = ['simsun', 'heiti', 'lishu', 'fangsong']

    if Path(outputPath).exists():
        shutil.rmtree(outputPath)
        print('Remove exist {}'.format(str(outputPath)))
    os.makedirs(outputPath)
    print('Create {}'.format(str(outputPath)))
    env = lmdb.open(outputPath, map_size=1099511627776)
    inputPath = Path(inputPath)
    nSamples = 0
    vocab = set()

    tasks = [(p, checkValid, font) for font in fonts for p in (inputPath / font).glob('*') if p.stem.endswith('gt')]

    with ProcessPoolExecutor(max_workers=max_workers) as ex, ThreadPoolExecutor(max_workers=max_workers) as tx:

        futures = []
        for i, t in enumerate(tasks, 1):
            futures.append(ex.submit(loadData, t[2], t[0], t[1]))
            if i % max_workers == 0:
                cnt = 0
                subCaches = []
                for future in concurrent.futures.as_completed(futures):
                    tag, subCnt, subCache, subVocab = future.result()
                    cnt = cnt + subCnt
                    subCaches.append(subCache)
                    vocab = vocab.union(subVocab)
                    print('task={} reads n_samples={} tag={}'.format(i, subCnt, tag))

                nSamples = nSamples + cnt
                for _ in concurrent.futures.as_completed([tx.submit(writeCache, env, c) for c in subCaches]):
                    print('task={} writes n_samples={}'.format(i, cnt//len(futures)))

        if len(futures) > 0:
            cnt = 0
            subCaches = []
            for future in concurrent.futures.as_completed(futures):
                tag, subCnt, subCache, subVocab = future.result()
                cnt = cnt + subCnt
                subCaches.append(subCache)
                vocab = vocab.union(subVocab)
                print('task={} reads n_samples={} gtD={}'.format('last', subCnt, tag))

            nSamples = nSamples + cnt
            for _ in concurrent.futures.as_completed([tx.submit(writeCache, env, c) for c in subCaches]):
                print('task={} writes n_samples={}'.format('last', cnt//len(futures)))

    cache = {'num-samples'.encode(): str(nSamples).encode()}
    writeCache(env, cache)

    with open(str(inputPath / 'vocabulary.txt'), 'w') as f:
        for v in vocab:
            f.write(v + '\n')

    print('Created dataset with %d samples' % nSamples)


def createDataset_v3(rootInputPath, outputPath, checkValid=True, max_workers=50):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image

        params: --rootInputPath /datadisk2/qfeng/ocr_synthesis --outputPath /datadisk2/qfeng/ocr_synthesis/all_lmdb
    """
    if Path(outputPath).exists():
        shutil.rmtree(outputPath)
        print('Remove exist {}'.format(str(outputPath)))
    os.makedirs(outputPath)
    print('Create {}'.format(str(outputPath)))
    env = lmdb.open(outputPath, map_size=1099511627776)
    nSamples = 0
    vocab = set()

    fonts = ['simsun']
    for inputPath in ['number_outp']:
        inputPath = Path(os.path.join(rootInputPath, inputPath))
        tasks = [(p, checkValid, font) for font in fonts for p in (inputPath / font).glob('*') if p.stem.endswith('gt')]

        with ProcessPoolExecutor(max_workers=max_workers) as ex, ThreadPoolExecutor(max_workers=max_workers) as tx:

            futures = []
            for i, t in enumerate(tasks, 1):
                futures.append(ex.submit(loadData, str(inputPath.stem) + '_' + t[2], t[0], t[1]))
                if i % max_workers == 0:
                    cnt = 0
                    subCaches = []
                    for future in concurrent.futures.as_completed(futures):
                        tag, subCnt, subCache, subVocab = future.result()
                        cnt = cnt + subCnt
                        subCaches.append(subCache)
                        vocab = vocab.union(subVocab)
                        print('task={} reads n_samples={} tag={}'.format(i, subCnt, tag))

                    nSamples = nSamples + cnt
                    for _ in concurrent.futures.as_completed([tx.submit(writeCache, env, c) for c in subCaches]):
                        print('task={} writes n_samples={} font={}'.format(i, cnt//len(futures), t[2]))

            if len(futures) > 0:
                cnt = 0
                subCaches = []
                for future in concurrent.futures.as_completed(futures):
                    tag, subCnt, subCache, subVocab = future.result()
                    cnt = cnt + subCnt
                    subCaches.append(subCache)
                    vocab = vocab.union(subVocab)
                    print('task={} reads n_samples={} tag={}'.format('last', subCnt, tag))

                nSamples = nSamples + cnt
                for _ in concurrent.futures.as_completed([tx.submit(writeCache, env, c) for c in subCaches]):
                    print('task={} writes n_samples={}'.format('last', cnt//len(futures)))

    cache = {'num-samples'.encode(): str(nSamples).encode()}
    writeCache(env, cache)

    with open(str(inputPath / 'vocabulary.txt'), 'w') as f:
        for v in vocab:
            f.write(v + '\n')

    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset_v3)
