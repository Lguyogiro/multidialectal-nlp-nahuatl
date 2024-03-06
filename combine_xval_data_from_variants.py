import random
from utils import read_conllu, write_conllu

for fold in range(1, 11):
    for part in ("train", "val"):
        nhi_file = f"nhi_10fold/nhi_{fold}_{part}.conllu"
        azz_file = f"azz_10fold/azz_{fold}_{part}.conllu"

        nhi = read_conllu(nhi_file)
        random.shuffle(nhi)
        half_nhi = nhi[:len(nhi)//2]
        azz = read_conllu(azz_file)
        random.shuffle(azz)
        half_azz = azz[:len(azz)//2]

        combo = nhi + azz
        half_combo = half_nhi + half_azz
        random.shuffle(combo)
        random.shuffle(half_combo)


        write_conllu(combo, f"nhi+azz_10fold/nhi+azz_{fold}_{part}.conllu")
        write_conllu(half_combo, f"half_combined_nhi+azz_10fold/nhi+azz_{fold}_{part}.conllu")
