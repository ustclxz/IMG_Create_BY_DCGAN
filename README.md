# IMG_Create_BY_DCGAN
DCGANで人顔画像生成を体験する（WEB　DEMO機能付き）

## Prerequisites
- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset
- Flask==0.12

## Usage
１、学習モデルを保存する（Checkpoint）
２、保存されたモデルをリロードし、推論を行う（DCGANで画像生成する）
３、WEBアプリへの組み込み（DEMO）－Flaskフレームワーク利用

☆まだボタン１回目だけ実行できる状態、２回目以降エラー、今後更新続く

## Author

lxz

#Reference

http://qiita.com/shu223/items/b6d8dc1fccb7c0f68b6b

https://github.com/carpedm20/DCGAN-tensorflow

