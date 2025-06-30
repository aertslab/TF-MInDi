import numpy as np
import pandas as pd
from pycistarget.utils import load_motif_annotations

def load_motif(file_name: str) -> dict[str, np.ndarray]:
    motifs: dict[str, np.ndarray] = {}
    with open(file_name) as f:
        # initialize name
        name = f.readline().strip()
        if not name.startswith(">"):
            raise ValueError(f"First line of {file_name} does not start with '>'.")
        name = name.replace(">", "")
        pwm = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # we are at the start of a new motif
                motifs[name] = np.array(pwm)
                # scale values of motif
                motifs[name] = motifs[name].T / motifs[name].sum(1)
                # reset pwm and read new name
                name = line.replace(">", "")
                pwm = []
            else:
                # we are in the middle of reading the pwm values
                pwm.append([float(v) for v in line.split()])
        # add last motif
        motifs[name] = np.array(pwm)
        # scale values of motif
        motifs[name] = motifs[name].T / motifs[name].sum(1)
    return motifs

def load_motif_to_dbd() -> dict[str, str]:
    motif_to_tf = load_motif_annotations("homo_sapiens", version = "v10nr_clust")
    motif_to_tf = motif_to_tf.apply(lambda row: ", ".join(row.dropna()), axis = 1).str.split(", ").explode().reset_index().rename({0: "TF"}, axis = 1)
    human_tf_annot = pd.read_csv("https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv", index_col = 0)[["HGNC symbol", "DBD"]]
    motif_to_tf = motif_to_tf.merge(
        right=human_tf_annot,
        how="left",
        left_on = "TF",
        right_on = "HGNC symbol"
    )
    motif_to_dbd = (
        motif_to_tf.dropna().groupby('MotifID')['DBD']
        .agg(lambda x: x.mode().iat[0])  # take the first mode if there's a tie
        .reset_index()
    )
    motif_to_dbd = motif_to_dbd.set_index("MotifID")["DBD"].to_dict()
    return motif_to_dbd
