import argparse
import re
from string import punctuation
from utils import read_conllu, write_conllu
from elotl.nahuatl.orthography import Normalizer

n = Normalizer("ilv")

def get_treebank_overrides():
    with open("nhi_10fold/nhi_spanish_vocab.txt") as f:
        nhi_spa_vocab = {word.lower().strip(punctuation+"\n") for word in f}
    with open("nhi_10fold/azz_spanish_vocab.txt") as f:
        azz_spa_vocab = {word.lower().strip(punctuation+"\n") for word in f}
    return {token: token for token in nhi_spa_vocab.union(azz_spa_vocab)}

overrides = get_treebank_overrides()

def normalize_forms_and_lemmas(sent):
    new_sent = []
    for line in sent.split("\n"):
        if line.startswith("#"):
            new_sent.append(line)
            continue
        else:
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                new_sent.append(line)
                continue
            misc = fields[-1]
            norm_form_match = re.search("NormalizedForm=([^\|]+)", misc)
            norm_lemma_match = re.search("NormalizedLemma=([^\|]+)", misc)
            try:
                fields[1] = (
                    n.normalize(norm_form_match.group(1), overrides=overrides) 
                    if norm_form_match is not None 
                    else n.normalize(fields[1], overrides=overrides)
                )
            except Exception:
                import pdb;pdb.set_trace()

            fields[2] = (
                n.normalize(norm_lemma_match.group(1), overrides=overrides) 
                if norm_lemma_match is not None 
                else n.normalize(fields[2], overrides=overrides)
            )

            new_sent.append("\t".join(fields))
    return new_sent


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("source_file")
    argparser.add_argument("outfile")
    args = argparser.parse_args()
    
    conllu = read_conllu(args.source_file)
    conllu = [normalize_forms_and_lemmas(sent) for sent in conllu if sent]
    conllu = ["\n".join(sent) for sent in conllu]
    write_conllu(conllu, args.outfile)
