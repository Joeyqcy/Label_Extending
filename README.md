# Label_Extending
Some works to promote the accuracy of name-entity-recognizing.

# src

`Label_Extending.py`: Extract graph structure from specified corpus. Then use a ssl algorithm named 'MADDL' to extend the labels for leraning set.

`maddl.py` : 'MADDL' algorithm codes.This version does not use spark.

`text_clean.py` : Clean the corpus, filter illegal characters, delete text which is too short.

# datasets

`kol_recommend_raw_rep.csv` : A corpus which contains about 1700 contents.

ps: A large corpus which contains about 300000 contents lies in database.(database:kol_recommend, table:baidu_zhidao)

`labeled_dict.txt` : Some entities with known label.Used for seed set in MADDL

`merge_pos_dict.txt` : ltp postagger user dict.

`merge_words_dict.txt`: ltp segmentor user dict.(old, abandoned now.)

`newwords_dict.txt` : ltp segmentor user dict.(find from wx docs, used now.)


# TODO
1. Use some supervised models on the result of `Label_Extending` to struct a name-entity classifier.(hzdeng is doing now.)
2. Use spark-version maddl to replace existing.
3. Optimize the graph structure.
 