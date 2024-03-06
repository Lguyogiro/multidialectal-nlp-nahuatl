import pandas as pd
import random
import re
from string import punctuation
punctuation = punctuation + "—¡¿'"
from elotl.nahuatl.orthography import Normalizer

n = Normalizer("ilv")


def tokenize(t):
    return re.sub(fr"([{punctuation}]+)", r" \1 ", t)


def retokenize(t):
    return re.sub(fr"( [{punctuation}]+)", r"", t)

def normalize(t, overrides):
    return retokenize(n.normalize(tokenize(t), overrides=overrides))


def get_axolotl_lang_id_data():
    df = pd.read_csv("../py-elotl/elotl/corpora/axolotl.csv", header=None)
    df = df[~df[1].isnull()]
    has_lang_code = df[~df[4].isnull()]
    df = has_lang_code[has_lang_code[3] != "Una tortillita nomás - Se taxkaltsin saj"]    
    df = df[~df[4].isin(["nci"])]
    texts = df[1].tolist()
    labels = df[4].tolist()
    labels = ["nhw" if label == "nhe" else label for label in labels]

    overrides = {token.lower(): token.lower() for sent in df[0].tolist() for token in sent.split()}
    normed_texts = [normalize(t, overrides) for t in texts]

    texts_and_labels = list(zip(normed_texts, labels))
    expanded_texts_and_labels = []
    for text, label in texts_and_labels:
        subsents = [s for s in re.split("[\.!] ?", text) if s]
        if len(subsents) > 1:
            for subsent in subsents:
                if all(ch in "0123456789" for ch in subsent) or len(subsent.split()) == 1:
                    continue
                expanded_texts_and_labels.append(f"{subsent}.\t{label}")
        else:
            expanded_texts_and_labels.append(f"{text}\t{label}")
    expanded_texts_and_labels = list(set(expanded_texts_and_labels))
    random.shuffle(expanded_texts_and_labels)

    train_size = int(len(expanded_texts_and_labels) * 0.9)
    langid_train = [
        row for row in expanded_texts_and_labels[:train_size]
    ]
    langid_eval = [
        row for row in expanded_texts_and_labels[train_size:]
    ]

    return langid_train, langid_eval




def get_sents_from_tokens(conllu_file, overrides):
    with open(conllu_file) as f:
        sents = [normalize(line.strip("\n").split(" = ")[1], overrides)
                 for line in f if line.startswith("# text =")]
    return sents

def get_treebank_overrides():
    with open("nhi_10fold/nhi_spanish_vocab.txt") as f:
        nhi_spa_vocab = {word.lower().strip(punctuation+"\n") for word in f}
    with open("nhi_10fold/azz_spanish_vocab.txt") as f:
        azz_spa_vocab = {word.lower().strip(punctuation+"\n") for word in f}
    return {token: token for token in nhi_spa_vocab.union(azz_spa_vocab)}


def create_folds(langid_train, langid_eval):
    tb_overrides = get_treebank_overrides()
    res = {}
    for fold in range(1, 11):
        nhi_train = get_sents_from_tokens(f"nhi_10fold/nhi_{fold}_train.conllu", tb_overrides)
        nhi_train = [f"{s}\tnhi" for s in nhi_train]
        
        nhi_eval = get_sents_from_tokens(f"nhi_10fold/nhi_{fold}_val.conllu", tb_overrides)
        nhi_eval = [f"{s}\tnhi" for s in nhi_eval]
        
        azz_train = get_sents_from_tokens(f"azz_10fold/azz_{fold}_train.conllu", tb_overrides)
        azz_train = [f"{s}\tazz" for s in azz_train]
        
        azz_eval = get_sents_from_tokens(f"azz_10fold/azz_{fold}_val.conllu", tb_overrides)
        azz_eval = [f"{s}\tazz" for s in azz_eval]
        
        all_train = langid_train + nhi_train + azz_train
        all_eval = langid_eval + nhi_eval + azz_eval
        
        random.shuffle(all_train)
        random.shuffle(all_eval)
        res[fold] = {"train": all_train, "eval": all_eval}

    return res
        

if __name__ == "__main__":
    lid_train, lid_eval = get_axolotl_lang_id_data()
    folds = create_folds(lid_train, lid_eval)

    for fold in folds:
        with open(f"langid_nahuatl_10fold/train_{fold}.tsv", "w") as fout:
            fout.write("\n".join(folds[fold]["train"]))
        with open(f"langid_nahuatl_10fold/val_{fold}.tsv", "w") as fout:
            fout.write("\n".join(folds[fold]["eval"]))
    
    