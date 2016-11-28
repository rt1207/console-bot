
#### コンソールで会話を行うコマンド
`python EncoderDecoderModelForward.py`
![screen shot 0028-11-28 at 6 57 35 pm](https://cloud.githubusercontent.com/assets/2739661/20663970/b0c85a1a-b59c-11e6-9dd9-bc450816b058.png)

#### 単語リスト
```
Wikipediaタイトルデータ：Data/jawiki-latest-random-titles-in-ns0, wordlist_programming
-> モデル書き込み：word2vec/word2vec.py
-> モデル読み込み：encdec/EncoderDecoderModelForward.py
```

#### コーパス
```コーパス：Data/player_1.txt
対応する発話：Data/player_2.txt
-> 分かち書き：Data/player_1_wakati, Data/player_2_wakati
-> モデル書き込み：EncoderDecoderModel.train() -> Data/ChainerDialogue.weights|spec|srcvocab|trgvocab
-> モデル読み込み：EncoderDecoderModel.test()
```

## 参考
- Chainerで学習した対話用のボットをSlackで使用+Twitterから学習データを取得してファインチューニング
http://qiita.com/GushiSnow/items/79ca7deeb976f50126d7
- 簡易版
http://blog.cgfm.jp/garyu/archives/3385

