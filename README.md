# Label_Extending
Some works to promote the accuracy of name-entity-recognizing.

# src

`Label_Extending.py`: Extract graph structure from specified corpus. Then use a ssl algorithm named 'MADDL' to extend the labels for leraning set.

`maddl.py` : 'MADDL' algorithm codes.This version does not use spark.

`text_clean.py` : Clean the corpus, filter illegal characters, delete text which is too short.

# TODO
1. Use some supervised models on the result of `Label_Extending` to struct a name-entity classifier.(hzdeng is doing now.)
2. Use spark-version maddl to replace existing.
3. Optimize the graph structure.
 