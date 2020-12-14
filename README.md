# libtorch tutorial.

## 概要

本プログラムはlibtorchをちょっと試してみたい人向けのチュートリアルプログラムです。

## できること

Pytorch(torchvision)のPretrain済モデルで画像分類ができます！

使えるモデルは下記3つです

* Resnet18
* Resnet50
* MobileNet

## 必要な環境

### 絶対に必要
* CUDA対応のGPU、GPUドライバ

### ホストで動かす場合

* CUDA (10.2推奨)
* Python (3.6)
* Pytorch (v1.6)
* OpenCV (v4.x)

### Dockerで動かす

**Jetsonは非対応です**

* Docker環境 & NVIDIA container toolkit (※ホストで動かす場合は不要)
    * `/etc/docker/daemon.json`を編集し、`default-runtime`を`nvidia`にしてください

## 環境構築

### Dockerコンテナ作成手順 (Dockerを使う人向け)

#### 1. セットアップ

当リポジトリをクローンし、任意のディレクトリに設置してください

#### 2. Dockerのビルド

下記コマンドでDockerイメージを作成してください

```bash
$ docker build -it libtorch-tutorial:latest .
```

#### 3. 下記コマンドで、Dockerコンテナを作成してください

```bash
$ docker run --rm -it -v${PWD}:/app -w/app libtorch-tutorial:latest bash 
```

### モデルのtorchscript変換

`buildmodel.py`を実行し、モデルをtorchscriptに変換してください

**usage**

```bash
$ python3 buildmodel.py -h
usage: buildmodel.py [-h] [--model {resnet18,resnet50,mobilenet_v2}]
                     [--src SRC]
                     save

positional arguments:
  save                  出力ファイルパス

optional arguments:
  -h, --help            show this help message and exit
  --model {resnet18,resnet50,mobilenet_v2}
                        変換するモデル
  --src SRC             テスト用の画像データ
```

### libtorchアプリケーションのビルド

下記の通り、makeを実行してください。
make完了後、`example/build`ディレクトリに`example`という実行ファイルが生成されます
```bash
$ cd example
$ mkdir build
$ cd build
$ cmake .. 
$ make -j$(nproc)
```

### 画像分類実行

下記のコマンドで画像分類結果が出力されます
```bash
$ ./example \
    <ビルドしたtorchscriptファイル> \
    <ラベルファイル(example/labels.txt)> \
    <識別したい画像ファイル>
```

※実行サンプル
```bash
$ ./example ../../models/resnet18.pt ../labels.txt ../../tmp/cat.jpg 
index: 283
label: Persian cat
score: 0.921259
```

## その他

* このコードはQiita アドベントカレンダー(PyTorch Advent Calendar 2020)の12月17日の記事向けに作成したものです
* Githubにスターとか、上記記事にLGTMとかしてくれるとモチベーションが上がりますm(. .)m
