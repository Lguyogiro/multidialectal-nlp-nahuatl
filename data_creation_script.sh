python normalized_conllu.py ../UD_Western_Sierra_Puebla_Nahuatl-ITML/nhi_itml-ud-test.conllu normalized_nhi_treebank.conllu
python make_xval_data.py normalized_nhi_treebank.conllu nhi nhi nhi_10fold

python normalized_conllu.py ../UD_Highland_Puebla_Nahuatl-ITML/azz_itml-ud-test.conllu normalized_azz_treebank.conllu
python make_xval_data.py normalized_azz_treebank.conllu azz azz azz_10fold
python combine_xval_data_from_variants.py
