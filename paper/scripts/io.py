import numpy as np
import pandas as pd

def load_motif_annotations(specie: str,
                           version: str = 'v9',
                           fname: str | None = None,
                           column_names=('#motif_id', 'gene_name',
                                         'motif_similarity_qvalue', 'orthologous_identity', 'description'),
                           motif_similarity_fdr: float = 0.001,
                           orthologous_identity_threshold: float = 0.0):
    """
    Load motif annotations from a motif2TF snapshot.
    
    Parameters
    ---------
    specie:
        Specie to retrieve annotations for.
    version:
        Motif collection version.
    fname: 
        The snapshot taken from motif2TF.
    column_names: 
        The names of the columns in the snapshot to load.
    motif_similarity_fdr: 
        The maximum False Discovery Rate to find factor annotations for enriched motifs.
    orthologuous_identity_threshold: 
        The minimum orthologuous identity to find factor annotations for enriched motifs.
    
    Return
    ---------
        A dataframe with motif annotations for each motif
    """
    # Create a MultiIndex for the index combining unique gene name and motif ID. This should facilitate
    # later merging.
    if fname is None:
        if specie == 'mus_musculus':
            name='mgi'
        elif specie == 'homo_sapiens':
            name='hgnc'
        elif specie == 'drosophila_melanogaster':
            name='flybase'
        fname = 'https://resources.aertslab.org/cistarget/motif2tf/motifs-'+version+'-nr.'+name+'-m0.001-o0.0.tbl'
    df = pd.read_csv(fname, sep='\t', usecols=column_names)
    df.rename(columns={'#motif_id':"MotifID",
                       'gene_name':"TF",
                       'motif_similarity_qvalue': "MotifSimilarityQvalue",
                       'orthologous_identity': "OrthologousIdentity",
                       'description': "Annotation" }, inplace=True)
    df = df[(df["MotifSimilarityQvalue"] <= motif_similarity_fdr) &
            (df["OrthologousIdentity"] >= orthologous_identity_threshold)]
    
    # Direct annotation
    df_direct_annot = df[df['Annotation'] == 'gene is directly annotated']
    try:
        df_direct_annot = df_direct_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(list(set(x)))).reset_index()
    except:
        pass
    df_direct_annot.index = df_direct_annot['MotifID']
    df_direct_annot = pd.DataFrame(df_direct_annot['TF'])
    df_direct_annot.columns = ['Direct_annot']
    # Indirect annotation - by motif similarity
    motif_similarity_annot = df[df['Annotation'].str.contains('similar') & ~df['Annotation'].str.contains('orthologous')]
    try:
        motif_similarity_annot = motif_similarity_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(list(set(x)))).reset_index()
    except:
        pass
    motif_similarity_annot.index =  motif_similarity_annot['MotifID']
    motif_similarity_annot = pd.DataFrame(motif_similarity_annot['TF'])
    motif_similarity_annot.columns = ['Motif_similarity_annot']
    # Indirect annotation - by orthology
    orthology_annot = df[~df['Annotation'].str.contains('similar') & df['Annotation'].str.contains('orthologous')]
    try:
        orthology_annot = orthology_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(list(set(x)))).reset_index()
    except:
        pass
    orthology_annot.index = orthology_annot['MotifID']
    orthology_annot = pd.DataFrame(orthology_annot['TF'])
    orthology_annot.columns = ['Orthology_annot']
    # Indirect annotation - by orthology
    motif_similarity_and_orthology_annot = df[df['Annotation'].str.contains('similar') & df['Annotation'].str.contains('orthologous')]
    try:
        motif_similarity_and_orthology_annot = motif_similarity_and_orthology_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(list(set(x)))).reset_index()
    except:
        pass
    motif_similarity_and_orthology_annot.index = motif_similarity_and_orthology_annot['MotifID']
    motif_similarity_and_orthology_annot = pd.DataFrame(motif_similarity_and_orthology_annot['TF'])
    motif_similarity_and_orthology_annot.columns = ['Motif_similarity_and_Orthology_annot']
    # Combine
    df = pd.concat([df_direct_annot, motif_similarity_annot, orthology_annot, motif_similarity_and_orthology_annot], axis=1, sort=False)
    return df
    

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
