import argparse
import re
from utils import read_conllu, write_conllu
from elotl.nahuatl.orthography import Normalizer

n = Normalizer("inali")


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
            if "NormalizedForm" not in misc:
                normed_form = n.normalize(fields[1])
                normed_lemma = n.normalize(fields[2])
                misc += f"|NormalizedForm={normed_form}|NormalizedLemma={normed_lemma}"
                misc = misc.strip("|")
                fields[-1] = misc
            new_sent.append("\t".join(fields))
    return new_sent

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("source_file")
    #argparser.add_argument("outfile")
    args = argparser.parse_args()
    
    conllu = read_conllu(args.source_file)
    conllu = [normalize_forms_and_lemmas(sent) for sent in conllu]
    print("\n\n".join(["\n".join(s) for s in conllu]))
