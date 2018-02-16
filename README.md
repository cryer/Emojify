# Emojify
an implementation of emojify by Keras,pyTorch version to [pyTorch emojify](https://github.com/cryer/emojify-pyTorch)


## Inspiration

* Coursera course by Andrew Ng

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
However,I also upload my checkpoints on [Google Driver](https://drive.google.com/open?id=1xEy5nZklygXWjEb4uDA7OzJItmVHVTB1).

Feel free to download,and put into checkpoints subdirectory,then try to run demo.py with my checkpoint,you can change your test
word in demo,but notice that it should not over 10 words.
