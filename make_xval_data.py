import argparse
import random
from utils import read_conllu, write_conllu


def is_mwt_or_enhanced(tokenline):
    if tokenline.startswith("#"):
        return False
    fields = tokenline.split("\t")
    return "-" in fields[0] or "." in fields[0]

    
def get_folds(sentences, nfolds=10):
    random.shuffle(sentences)

    folds = {i: [] for i in range(nfolds)}
    for i, sentence in enumerate(sentences):
        sentence = sentence.split("\n")
        sentence = [line for line in sentence if not is_mwt_or_enhanced(line)]
        sentence = "\n".join(sentence)
        folds[i % nfolds].append(sentence)
    
    return folds

def get_configurations(folds):
    for i in folds:
        val = folds[i]
        train = [sent for fold, sent_bloq in folds.items() for sent in sent_bloq if fold != i]
        yield (train, val)

def add_classification_label(sent, label):
    rows = sent.split("\n")
    comment_rows = [row for row in rows if row.startswith("#")]
    noncomment_rows = [row for row in rows if not row.startswith("#")]
    new_comment_rows = []
    updated = False
    for r in comment_rows:
        if r.startswith("# labels"):
            new_comment_rows.append("# labels = " + label)
            updated = True
        else:
            new_comment_rows.append(r)
    if not updated:
        new_comment_rows.append("# labels = " + label)
    return "\n".join(new_comment_rows + noncomment_rows)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("source_file")
    argparser.add_argument("fold_file_prefix")
    argparser.add_argument("lang_code")
    argparser.add_argument("destination_directory")
    args = argparser.parse_args()
    
    conllu = read_conllu(args.source_file)
    conllu = [add_classification_label(sent, args.lang_code) for sent in conllu]
    folds = get_folds(conllu)
    for fold, (train, val) in enumerate(get_configurations(folds)):
        train_fname = f"{args.destination_directory}/{args.fold_file_prefix}_{fold+1}_train.conllu"
        write_conllu(train, train_fname)
        val_fname = f"{args.destination_directory}/{args.fold_file_prefix}_{fold+1}_val.conllu"
        write_conllu(val, val_fname)
