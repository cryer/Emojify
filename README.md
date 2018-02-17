# Emojify
an implementation of emojify by Keras,pyTorch version to [pyTorch emojify](https://github.com/cryer/emojify-pyTorch)


## Inspiration

* Coursera course by Andrew Ng

## word embeding

glove.6B.50d.txt is word embeding we used in the code,and it is already trained in a very large datasets by glove.
It transforms every word index into 50 dimentional embeding vector.You may know glove and word2vec are both 
common way to do word embeding.

I have uploaded this to
[Google Drive](https://drive.google.com/open?id=13VddkMYxcqrkpuaWqXX_YRjFmZ-15PSA) yet,download and put it into data subdirectory.

## Model

![](https://github.com/cryer/Emojify/raw/master/image/emojifier-v2.png)

Pretty simple and common model,which is very useful for emojify and some tasks like this,say many-to-one model.

## Datasets

![](https://github.com/cryer/Emojify/raw/master/image/data_set.png)

datasets like this,check it in data directory.I have put all test data into train data,so you dont need to test.
Because I want more data to get model better generated. 

## Results 

![](https://github.com/cryer/Emojify/raw/master/image/1.png)

Not bad!

## Checkpoints

In fact,you are supposed to train it yourself,only takes a few minites to train about 100 epochs on GPU.
However,I also upload my checkpoints on [Google Drive](https://drive.google.com/open?id=1xEy5nZklygXWjEb4uDA7OzJItmVHVTB1).

Feel free to download,and put into checkpoints subdirectory,then try to run demo.py with my checkpoint,you can change your test
word in demo,but notice that it should not over 10 words.
