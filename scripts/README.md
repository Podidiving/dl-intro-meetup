### Running scripts for LMs

based on [this repo](https://github.com/karpathy/ng-video-lecture)

**Note** before running:
- install `torch` and `tqdm`
```
pip install torch tqdm
```

- download data:
```
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---

Each file is a self-sufficient training pipeline.

In the begging of each script there are some hyperparameters you can tweak if you want.

- `bigram.py` - simple bigram model training
- `self-attention.py` - only single head self-attention + linear projection. (~0.6M parameters)
- `gpt-single-block.py` - GPT with only one block (~2M parameters)
- `gpt.py` - final version (~10M parameters)

Results (loss during training and some generated text) can be found in the `experiments` folder.

---

*Note:* It's a bad idea to run `gpt.py` and `gpt-single-block.py` on cpu. It'd take you forever to train.