#pix2pix

执行顺序：

```bash
traindb2html.py
maskmerge.py
fakemaskfortest.py


# cpu上训练
python pix2pix.py --mode train --output_dir /train_dir/dsb/dsb_train --max_epochs 200 --input_dir /datasets/dsb/stage1_train --which_direction AtoB

# GPU traininng
CUDA_VISIBLE_DEVICES=3 python pix2pix.py \
--mode train \
--output_dir /home/zsy/train_dir/dsb/dsb_train \
--max_epochs 200 \
--input_dir /home/zsy/datasets/dsb/stage1_train \
--which_direction AtoB

# test the model
CUDA_VISIBLE_DEVICES=3  python pix2pix.py \
--mode test \
--output_dir /home/zsy/train_dir/dsb/dsb_test \
--input_dir /home/zsy/datasets/dsb/stage1_test \
--checkpoint /home/zsy/train_dir/dsb/dsb_train


img2rle.py
deletesmallpixarea.py

```





## 问题1：

mask merge执行过程中，部分图片对比度低，未处理

```bash
3%|▎         | 17/670 [00:01<00:46, 14.01it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\07761fa39f60dc37022dbbe8d8694595fd5b77ceb2af2a2724768c8e524d6770\masksmerged\07761fa39f60dc37022dbbe8d8694595fd5b77ceb2af2a2724768c8e524d6770.png is a low contrast image
  warn('%s is a low contrast image' % fname)
  4%|▍         | 28/670 [00:03<01:23,  7.67it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e\masksmerged\0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e.png is a low contrast image
  warn('%s is a low contrast image' % fname)
14%|█▍        | 94/670 [00:14<01:31,  6.29it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\220b37f4ca7cab486d2b71cd87a46ee7411a5aa142799d96ed98015ab5ba538a\masksmerged\220b37f4ca7cab486d2b71cd87a46ee7411a5aa142799d96ed98015ab5ba538a.png is a low contrast image
  warn('%s is a low contrast image' % fname)
25%|██▌       | 169/670 [00:21<01:03,  7.90it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\3d0ca3498d97edebd28dbc7035eced40baa4af199af09cbb7251792accaa69fe\masksmerged\3d0ca3498d97edebd28dbc7035eced40baa4af199af09cbb7251792accaa69fe.png is a low contrast image
  warn('%s is a low contrast image' % fname)
36%|███▌      | 239/670 [00:33<01:01,  7.03it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921\masksmerged\58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921.png is a low contrast image
  warn('%s is a low contrast image' % fname)
49%|████▉     | 330/670 [00:51<00:53,  6.40it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80\masksmerged\7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80.png is a low contrast image
  warn('%s is a low contrast image' % fname)
52%|█████▏    | 348/670 [00:53<00:49,  6.49it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\84eeec681987753029eb83ea5f3ff7e8b5697783cdb2035f2882d40c9a3f1029\masksmerged\84eeec681987753029eb83ea5f3ff7e8b5697783cdb2035f2882d40c9a3f1029.png is a low contrast image
  warn('%s is a low contrast image' % fname)
53%|█████▎    | 352/670 [00:53<00:48,  6.55it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\866a8cba7bfe1ea73e383d6cf492e53752579140c8b833bb56839a55bf79d855\masksmerged\866a8cba7bfe1ea73e383d6cf492e53752579140c8b833bb56839a55bf79d855.png is a low contrast image
  warn('%s is a low contrast image' % fname)
57%|█████▋    | 384/670 [00:58<00:43,  6.60it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\93c5638e7e6433b5c9cc87c152bcbe28873d2f9d6a392cca0642520807542a77\masksmerged\93c5638e7e6433b5c9cc87c152bcbe28873d2f9d6a392cca0642520807542a77.png is a low contrast image
  warn('%s is a low contrast image' % fname)
C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\93cfd412c7de5210bbd262ec3a602cfea65072e9272e9fce9b5339a5b9436eb7\masksmerged\93cfd412c7de5210bbd262ec3a602cfea65072e9272e9fce9b5339a5b9436eb7.png is a low contrast image
  warn('%s is a low contrast image' % fname)
61%|██████    | 406/670 [01:00<00:39,  6.67it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\9cbc0700317361236a9fca2eb1f8f79e3a7da17b1970c179cf453921a6136001\masksmerged\9cbc0700317361236a9fca2eb1f8f79e3a7da17b1970c179cf453921a6136001.png is a low contrast image
  warn('%s is a low contrast image' % fname)
69%|██████▉   | 461/670 [01:09<00:31,  6.64it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\af576e8ec3a8d0b57eb6a311299e9e4fd2047970d3dd9d6f52e54ea6a91109da\masksmerged\af576e8ec3a8d0b57eb6a311299e9e4fd2047970d3dd9d6f52e54ea6a91109da.png is a low contrast image
  warn('%s is a low contrast image' % fname)
84%|████████▍ | 565/670 [01:24<00:15,  6.67it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\d7ec8003735996458b56ccb8ae34d080eb2a6adabef931323239632515b4b220\masksmerged\d7ec8003735996458b56ccb8ae34d080eb2a6adabef931323239632515b4b220.png is a low contrast image
  warn('%s is a low contrast image' % fname)
87%|████████▋ | 586/670 [01:26<00:12,  6.74it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b\masksmerged\e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b.png is a low contrast image
  warn('%s is a low contrast image' % fname)
90%|████████▉ | 601/670 [01:29<00:10,  6.75it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\e5aeb5b3577abbebe8982b5dd7d22c4257250ad3000661a42f38bf9248d291fd\masksmerged\e5aeb5b3577abbebe8982b5dd7d22c4257250ad3000661a42f38bf9248d291fd.png is a low contrast image
  warn('%s is a low contrast image' % fname)
98%|█████████▊| 655/670 [01:34<00:02,  6.92it/s]C:\Users\kaibin\Anaconda3\lib\site-packages\skimage\io\_io.py:132: UserWarning: /datasets/dsb/stage1_train\fa73f24532b3667718ede7ac5c2e24ad7d3cae17b0a42ed17bbb81b15c28f4ae\masksmerged\fa73f24532b3667718ede7ac5c2e24ad7d3cae17b0a42ed17bbb81b15c28f4ae.png is a low contrast image
  warn('%s is a low contrast image' % fname)
100%|██████████| 670/670 [01:36<00:00,  6.93it/s

```





## 问题2：

发现对于image黑白的图像训练效果较好。但是紫色图片中，色差（对比度）较小的图片，训练效果很差 ，有识别不出细胞核形状的趋势；有一些不规则形状的图片训练效果很差，许多颜色和细胞核相似但是形状差异极大（即不是圆形）的地方被认为是细胞核。

尝试降低L1 loss权重，提升GAN权重。



## 问题3：

有大量细胞核的局部区域被预测成一整块相连的白色区域，无法区别细胞核的边界，e.g. 紫色图片；细胞核中颜色较淡的部分无法画出，造成生成的该细胞核残缺；小部分图片（e.g.类似水墨画的图）存在大量很小的细胞核，后续处理可能被过滤；黑白图片有的白色细胞核周围背景是模糊的（背景发白），导致整个背景被预测成细胞核；

