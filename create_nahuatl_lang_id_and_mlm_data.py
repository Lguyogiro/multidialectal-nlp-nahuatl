import pandas as pd
import random
from string import punctuation
from elotl.nahuatl.orthography import Normalizer

n = Normalizer("ilv")

import re
from string import punctuation

with open("nhi_10fold/nhi_spanish_vocab.txt") as f:
    nhi_spa_vocab = {word.lower().strip(punctuation) for word in f}
with open("nhi_10fold/azz_spanish_vocab.txt") as f:
    azz_spa_vocab = {word.lower().strip(punctuation) for word in f}
    
overrides_2 = {token: token for token in nhi_spa_vocab.union(azz_spa_vocab)}

def tokenize(t):
    return re.sub(fr"([{punctuation}]+)", r" \1 ", t)

def retokenize(t):
    return re.sub(fr"( [{punctuation}]+)", r"", t)

def get_sents_from_tokens(conllu_file):
    with open(conllu_file) as f:
        sents = [n.normalize(line.strip("\n").split(" = ")[1], overrides=overrides_2) 
                 for line in f if line.startswith("# text =")]
    return sents

def ignore(s):
    stripped = s.strip(punctuation+"0123456789vix").lower()
    return (
        (len(stripped) < 3) or
        (stripped in ("i", "ii", "iii", "iv", "v", "vi", "vii", "vii", "viii", "ix", "x")) or
        (stripped == "mm")
    )

def split_into_subsents(sents, has_labels=False):
    subsent_split = []
    if has_labels:
        for line in sents:
            text, label = line.split("\t")
            subsents = re.split("[.!] ?", text)
            if len(subsents) > 1:
                for subsent in subsents:
                    if not ignore(subsent):
                        subsent_split.append(f"{subsent}.\t{label}")
            else:
                subsent_split.append(f"{text}\t{label}")
    else:
        for text in sents:
            subsents = re.split("[.!] ?", text)
            if len(subsents) > 1:
                for subsent in subsents:
                    if not ignore(subsent):
                        subsent_split.append(f"{subsent}.")
            else:
                subsent_split.append(f"{text}")
    return subsent_split


def get_axolotl_lang_id_data(axolotl_df_no_null_text):
    df = axolotl_df_no_null_text
    df = df[(df[4] != "nci") & (df[3] != 'Una tortillita nomás - Se taxkaltsin saj')]
    overrides = {token.lower(): token.lower() for sent in df[0].tolist() for token in sent.split()}
    
    texts = df[1].tolist()
    labels = df[4].tolist()
    labels = ["nhw" if label == "nhe" else label for label in labels]
    normed_texts = [n.normalize(t, overrides=overrides) for t in texts]

    texts_and_labels = list(zip(normed_texts, labels))
    random.shuffle(texts_and_labels)

    train_size = int(len(texts_and_labels) * 0.9)

    langid_train = [f"{row[0]}\t{row[1]}" for row in texts_and_labels[:train_size]]
    langid_eval = [f"{row[0]}\t{row[1]}" for row in texts_and_labels[train_size:]]

    lid_train = split_into_subsents(langid_train, has_labels=True)
    lid_eval = split_into_subsents(langid_eval, has_labels=True)

    return lid_train, lid_eval


def make_lid_xfold_data(lid_train, lid_eval):
    res = {}
    for fold in range(1, 11):
        nhi_train = get_sents_from_tokens(f"nhi_10fold/nhi_{fold}_train.conllu")
        nhi_train = [f"{s}\tnhi" for s in nhi_train]
        
        nhi_eval = get_sents_from_tokens(f"nhi_10fold/nhi_{fold}_val.conllu")
        nhi_eval = [f"{s}\tnhi" for s in nhi_eval]
        
        azz_train = get_sents_from_tokens(f"azz_10fold/azz_{fold}_train.conllu")
        azz_train = [f"{s}\tazz" for s in azz_train]
        
        azz_eval = get_sents_from_tokens(f"azz_10fold/azz_{fold}_val.conllu")
        azz_eval = [f"{s}\tazz" for s in azz_eval]
        
        all_train = lid_train + nhi_train #+ azz_train
        all_eval = lid_eval + nhi_eval #+ azz_eval
        
        random.shuffle(all_train)
        random.shuffle(all_eval)
        res[fold] = {"train": all_train, "eval": all_eval}

    return res


def write_xval_data(data, destination_dir, ext="tsv"):
    for fold in data:
        with open(f"{destination_dir}/train_{fold}.{ext}", "w") as fout:
            fout.write("\n".join(data[fold]["train"]))
        with open(f"{destination_dir}/val_{fold}.{ext}", "w") as fout:
            fout.write("\n".join(data[fold]["eval"]))


def get_axolotl_data_to_add(axolotl_df_no_null_text):
    df = axolotl_df_no_null_text
    df =df[(df[4] == "nci") | (df[4].isnull())]
    texts = df[1].tolist()
    normalized_text = [n.normalize(t).replace("yn", "in") for t in texts]
    shorter_sents_normalized_text = split_into_subsents(normalized_text, has_labels=False)

    return shorter_sents_normalized_text


def create_mlm_dataset(lid_xval_data, axolotl_data_to_add):
    random.shuffle(axolotl_data_to_add)
    
    train_length = int(len(axolotl_data_to_add) * 0.9)
    add_train = axolotl_data_to_add[:train_length]
    add_val = axolotl_data_to_add[train_length:]

    for fold in lid_xval_data:
        mlm_fold_data_train = [t.split("\t")[0] for t in lid_xval_data[fold]["train"]] + add_train
        random.shuffle(mlm_fold_data_train)
        lid_xval_data[fold]["train"] = mlm_fold_data_train

        mlm_fold_data_val = [t.split("\t")[0] for t in lid_xval_data[fold]["eval"]] + add_val
        random.shuffle(mlm_fold_data_val)
        lid_xval_data[fold]["eval"] = mlm_fold_data_val
    return lid_xval_data


if __name__ == "__main__":
    df = pd.read_csv("../py-elotl/elotl/corpora/axolotl.csv", header=None)
    df = df[~df[1].isnull()]
    print("Making language id dataset...")
    axolotl_train, axolotl_eval = get_axolotl_lang_id_data(df)
    xval_dataset_lid = make_lid_xfold_data(axolotl_train, axolotl_eval)
    #write_xval_data(xval_dataset_lid, "langid_nahuatl_10fold")
    #print("Wrote langid data...")
    print("Getting additional data for mlm dataset...")
    more_axolotl_data_for_mlm = get_axolotl_data_to_add(df)
    xval_dataset_mlm = create_mlm_dataset(xval_dataset_lid, more_axolotl_data_for_mlm)
    write_xval_data(xval_dataset_mlm, "nahuatl_mlm_no_azz_10fold")
    print("Finished writing nahuatl mlm dataset...")


