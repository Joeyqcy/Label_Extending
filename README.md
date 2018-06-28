# Label_Extending
Some works to promote the accuracy of name-entity-recognizing.

# src

`Label_Extending.py`: Extract graph structure from specified corpus. Then use a ssl algorithm named 'MADDL' to extend the labels for leraning set.

`maddl.py` : 'MADDL' algorithm codes.This version does not use spark.

`text_clean.py` : Clean the corpus, filter illegal characters, delete text which is too short.

# datasets

`kol_recommend_raw_rep.csv` : A corpus which contains about 1700 contents.

ps: A large corpus which contains about 300000 contents lies in database.(database: kol_recommend, table: baidu_zhidao)

`labeled_dict.txt` : Some entities with known label.Used for seed set in MADDL

`merge_pos_dict.txt` : ltp postagger user dict.

`merge_words_dict.txt`: ltp segmentor user dict.(old, abandoned now.)

`newwords_dict.txt` : ltp segmentor user dict.(find from wx docs, used now.)

# Example

Set the variables on path first, and then type in shell:
`python Label_Extending.py`

You will get several files as output.

`Potential_list.json` : A list of possible entities without known label which shoud be judged.  

`Filter_list.json` : A list of entities to filter the graph structure.(get from that  Potential_list combines labeled_list)

`targetDict.json` : A dictionary about graph structure.

The dictionary is like  {NE1:{RW1:count, RW2:count,……}, NE2:{RW1:count,……},…… } in which 'NE' is the name-entity in corpus, 'RW' is the related word of NE, 'count' is the frequency of appearance of the relation-pair.

`graph.txt` : The graph file for MADDL.

`seed.txt` : The seed file for MADDL.

`output.txt` : The result of MADDL in which all words in potential_list have been labeled. 

# TODO
1. Use some supervised models on the result of `Label_Extending` to struct a name-entity classifier.(hzdeng is doing now.)
2. Use spark-version maddl to replace existing.
3. Optimize the graph structure.
4. The accuracy of MADDL is about 70%.Consider how to tolerate or alleviate this error in supervised model.
 
