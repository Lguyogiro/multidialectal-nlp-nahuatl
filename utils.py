def read_conllu(filepath):
    with open(filepath) as f:
        return f.read().split("\n\n")
    
def write_conllu(sentences, output_filepath):
    with open(output_filepath, "w") as fout:
        fout.write("\n\n".join(sentences))
        fout.write("\n")
