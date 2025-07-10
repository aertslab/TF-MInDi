"""Dataset functions for TF-MInDi: fetching and loading motif collections and annotations."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pooch

_motif_index = None


def _get_motif_index():
    """
    Set up the pooch motif collection registry from pycistarget.

    Returns
    -------
    pooch.Pooch
        The motif collection registry.
    """
    global _motif_index

    if _motif_index is None:
        _motif_index = pooch.create(
            path=pooch.os_cache("tfmindi"),
            base_url="https://resources.aertslab.org/cistarget/",
            env="TFMINDI_DATA_DIR",
            registry={
                # Motif collections
                "motif_collections/v10nr_clust_public/v10nr_clust_public.zip": "sha256:70dab42794f42471a3c22f5efe78ec2c8af96127656607cb0a929b4adffc2b97",
                # Motif annotations (motifs-v{version}-nr.{species}-m0.001-o0.0.tbl)
                # v8 (only Drosophila)
                "motif2tf/motifs-v8-nr.flybase-m0.001-o0.0.tbl": "sha256:ffc550325334507c8ab2f6f79170fd0dfbcbecb33aee96040bf1f386fb3f982d",
                # v9 annotations
                "motif2tf/motifs-v9-nr.flybase-m0.001-o0.0.tbl": "sha256:db3458fdb38616758ea54543a42c7fc0e0ebb292cbf928d2648aefbf2cd59314",
                "motif2tf/motifs-v9-nr.hgnc-m0.001-o0.0.tbl": "sha256:9e085e8e6ecd6f73a47fe435ed50d583b92bf7814d1bf38ae358e73c6207da2b",
                "motif2tf/motifs-v9-nr.mgi-m0.001-o0.0.tbl": "sha256:cfab1245dbe770b1073f8048b1316285c9ce4b1a55d8ebca78962a2f406a172d",
                # v10nr_clust annotations
                "motif2tf/motifs-v10nr_clust-nr.chicken-m0.001-o0.0.tbl": "sha256:1ede59e4a737d822b7d6713243b83bf8f618f978c4fae1810fab265a30dfe3ba",
                "motif2tf/motifs-v10nr_clust-nr.flybase-m0.001-o0.0.tbl": "sha256:91284e94b0317b764dc2f8d8147d30db707605df757ff7de16fb0953c63fda2a",
                "motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl": "sha256:81eb754118e27e854974301b1400fcf519489f8be5249239671fb288cb501c31",
                "motif2tf/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl": "sha256:5b64aad9df9804d50c50484c92d5192bdd5d2056cb105bdd343c0af2f94cce83",
            },
        )

    return _motif_index


def fetch_motif_collection() -> str:
    """
    Download motif collection (motif names and PWM) from aertslab' pycistarget resources.

    Returns
    -------
    Path to downloaded motif collection folder

    Examples
    --------
    >>> motif_dir = fetch_motif_collection()
    >>> print(motif_dir)
    """
    # Mapping of collection names to registry keys
    name = "v10nr_clust"  # only one collection name
    collection_mapping = {
        "v10nr_clust": "motif_collections/v10nr_clust_public/v10nr_clust_public.zip",
    }

    registry_key = collection_mapping[name]

    def _extract_singletons(fname, action, pooch_obj):
        """Extract only the singletons folder from the zip file."""
        extract_dir = Path(fname).parent / Path(fname).stem / "singletons"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(fname, "r") as zip_file:
            singletons_files = [
                f for f in zip_file.namelist() if f.startswith("v10nr_clust_public/singletons/") and not f.endswith("/")
            ]
            for file_path in singletons_files:
                file_name = Path(file_path).name
                with zip_file.open(file_path) as source:
                    with open(extract_dir / file_name, "wb") as target:
                        target.write(source.read())
        return str(extract_dir)

    motif_dir = _get_motif_index().fetch(registry_key, processor=_extract_singletons, progressbar=True)

    return motif_dir


def fetch_motif_annotations(species: str = "hgnc", version: str = "v10nr_clust") -> str:
    """
    Download motif annotations from aertslab resources.

    Parameters
    ----------
    species
        Species name. Available options:
        - 'hgnc' (human): v9, v10nr_clust
        - 'mgi' (mouse): v9, v10nr_clust
        - 'flybase' (fly): v8, v9, v10nr_clust
        - 'chicken': v10nr_clust only
    version
        Motif collection version. Available options: 'v8', 'v9', 'v10nr_clust'

    Returns
    -------
    Path to downloaded annotations file

    Examples
    --------
    >>> annotations_file = fetch_motif_annotations("hgnc", "v10nr_clust")
    >>> print(annotations_file)
    /path/to/cache/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl
    """
    # Mapping of species and versions to registry keys
    annotation_mapping = {
        ("flybase", "v8"): "motif2tf/motifs-v8-nr.flybase-m0.001-o0.0.tbl",
        ("flybase", "v9"): "motif2tf/motifs-v9-nr.flybase-m0.001-o0.0.tbl",
        ("hgnc", "v9"): "motif2tf/motifs-v9-nr.hgnc-m0.001-o0.0.tbl",
        ("mgi", "v9"): "motif2tf/motifs-v9-nr.mgi-m0.001-o0.0.tbl",
        ("chicken", "v10nr_clust"): "motif2tf/motifs-v10nr_clust-nr.chicken-m0.001-o0.0.tbl",
        ("flybase", "v10nr_clust"): "motif2tf/motifs-v10nr_clust-nr.flybase-m0.001-o0.0.tbl",
        ("hgnc", "v10nr_clust"): "motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
        ("mgi", "v10nr_clust"): "motif2tf/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl",
    }

    key = (species, version)
    assert key in annotation_mapping, (
        f"Species {species} with version {version} is not recognised. "
        f"Available combinations: {list(annotation_mapping.keys())}"
    )

    registry_key = annotation_mapping[key]

    annotations_file = _get_motif_index().fetch(registry_key, progressbar=True)

    return annotations_file


def load_motif(file_name: str) -> dict[str, np.ndarray]:
    """
    Load motif(s) from a single .cb file.

    Parameters
    ----------
    file_name
        Path to the .cb motif file

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping motif names to PWM matrices (4 x length)

    Examples
    --------
    >>> motifs = load_motif("./motif1.cb")
    >>> print(list(motifs.keys()))
    ['motif1', 'motif2']
    >>> print(motifs["motif1"].shape)
    (4, 12)
    """
    motifs: dict[str, np.ndarray] = {}
    with open(file_name) as f:
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


def load_motif_collection(motif_dir: str) -> dict[str, np.ndarray]:
    """
    Load motif collection from directory of .cb files.

    Parameters
    ----------
    motif_dir
        Directory path containing .cb motif files

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping motif names to PWM matrices (4 x length)

    Examples
    --------
    >>> motifs = load_motif_collection("./motif_collection/")
    >>> print(list(motifs.keys())[:3])
    ['motif1', 'motif2', 'motif3']
    >>> print(motifs["motif1"].shape)
    (4, 12)
    """
    motif_dir = Path(motif_dir)

    if not motif_dir.exists():
        raise FileNotFoundError(f"Directory {motif_dir} does not exist")

    motifs = {}

    cb_files = list(motif_dir.glob("*.cb"))

    for cb_file in cb_files:
        try:
            # Load all motifs from this file
            file_motifs = load_motif(str(cb_file))
            motifs.update(file_motifs)
        except (ValueError, IndexError):
            # Skip malformed files
            continue

    return motifs


def load_motif_annotations(
    annotations_file: str,
    motif_similarity_fdr: float = 0.001,
    orthologous_identity_threshold: float = 0.0,
    column_names: tuple[str, ...] = (
        "#motif_id",
        "gene_name",
        "motif_similarity_qvalue",
        "orthologous_identity",
        "description",
    ),
) -> pd.DataFrame:
    """
    Load motif annotations from a motif2TF TSV file with filtering and categorization.

    Parameters
    ----------
    annotations_file
        Path to the annotations TSV file
    motif_similarity_fdr
        Maximum False Discovery Rate for enriched motifs (default: 0.001)
    orthologous_identity_threshold
        Minimum orthologous identity for enriched motifs (default: 0.0)
    column_names
        Column names to load from the TSV file

    Returns
    -------
    DataFrame with motif annotations categorized by annotation type:
    - Direct_annot: Direct gene annotations
    - Motif_similarity_annot: Annotations by motif similarity
    - Orthology_annot: Annotations by orthology
    - Motif_similarity_and_Orthology_annot: Combined annotations

    Examples
    --------
    >>> annotations = load_motif_annotations("./annotations.tbl")
    >>> print(annotations.columns.tolist())
    ['Direct_annot', 'Motif_similarity_annot', 'Orthology_annot', 'Motif_similarity_and_Orthology_annot']
    """
    # Load as pandas DataFrame
    df = pd.read_csv(annotations_file, sep="\t", usecols=column_names)
    df.rename(
        columns={
            "#motif_id": "MotifID",
            "gene_name": "TF",
            "motif_similarity_qvalue": "MotifSimilarityQvalue",
            "orthologous_identity": "OrthologousIdentity",
            "description": "Annotation",
        },
        inplace=True,
    )

    # Filter based on thresholds
    df = df[
        (df["MotifSimilarityQvalue"] <= motif_similarity_fdr)
        & (df["OrthologousIdentity"] >= orthologous_identity_threshold)
    ]

    # Direct annotation
    df_direct_annot = df[df["Annotation"] == "gene is directly annotated"]
    df_direct_annot = df_direct_annot.groupby(["MotifID"])["TF"].apply(lambda x: ", ".join(list(set(x)))).reset_index()
    df_direct_annot.index = df_direct_annot["MotifID"]
    df_direct_annot = pd.DataFrame(df_direct_annot["TF"])
    df_direct_annot.columns = ["Direct_annot"]

    # Indirect annotation - by motif similarity
    motif_similarity_annot = df[
        df["Annotation"].str.contains("similar") & ~df["Annotation"].str.contains("orthologous")
    ]
    motif_similarity_annot = (
        motif_similarity_annot.groupby(["MotifID"])["TF"].apply(lambda x: ", ".join(list(set(x)))).reset_index()
    )
    motif_similarity_annot.index = motif_similarity_annot["MotifID"]
    motif_similarity_annot = pd.DataFrame(motif_similarity_annot["TF"])
    motif_similarity_annot.columns = ["Motif_similarity_annot"]

    # Indirect annotation - by orthology
    orthology_annot = df[~df["Annotation"].str.contains("similar") & df["Annotation"].str.contains("orthologous")]
    orthology_annot = orthology_annot.groupby(["MotifID"])["TF"].apply(lambda x: ", ".join(list(set(x)))).reset_index()
    orthology_annot.index = orthology_annot["MotifID"]
    orthology_annot = pd.DataFrame(orthology_annot["TF"])
    orthology_annot.columns = ["Orthology_annot"]

    # Indirect annotation - by motif similarity and orthology
    motif_similarity_and_orthology_annot = df[
        df["Annotation"].str.contains("similar") & df["Annotation"].str.contains("orthologous")
    ]
    motif_similarity_and_orthology_annot = (
        motif_similarity_and_orthology_annot.groupby(["MotifID"])["TF"]
        .apply(lambda x: ", ".join(list(set(x))))
        .reset_index()
    )
    motif_similarity_and_orthology_annot.index = motif_similarity_and_orthology_annot["MotifID"]
    motif_similarity_and_orthology_annot = pd.DataFrame(motif_similarity_and_orthology_annot["TF"])
    motif_similarity_and_orthology_annot.columns = ["Motif_similarity_and_Orthology_annot"]

    # Combine all annotation types
    result = pd.concat(
        [df_direct_annot, motif_similarity_annot, orthology_annot, motif_similarity_and_orthology_annot],
        axis=1,
        sort=False,
    )

    return result


def load_motif_to_dbd(motif_annotations: pd.DataFrame) -> dict[str, str]:
    """
    Create motif-to-DNA-binding-domain mapping for human TFs.

    Takes motif annotations and maps motifs to their DNA-binding domains
    based on TF annotations and human TF database information.

    Parameters
    ----------
    motif_annotations
        DataFrame with motif annotations as returned by load_motif_annotations()

    Returns
    -------
    dict[str, str]
        Dictionary mapping motif IDs to DNA-binding domain names

    Examples
    --------
    >>> annotations_file = fetch_motif_annotations("hgnc", "v10nr_clust")
    >>> motif_annotations = load_motif_annotations(annotations_file)
    >>> motif_to_dbd = load_motif_to_dbd(motif_annotations)
    >>> print(motif_to_dbd["hocomoco__FOXO1_HUMAN.H11MO.0.A"])
    'Forkhead'
    """
    motif_to_tf = motif_annotations.copy()

    # Flatten all TF annotations into individual TF names
    motif_to_tf = (
        motif_to_tf.apply(lambda row: ", ".join(row.dropna()), axis=1)
        .str.split(", ")
        .explode()
        .reset_index()
        .rename({0: "TF"}, axis=1)
    )

    # Download human TF annotations with DNA-binding domains
    human_tf_annot = pd.read_csv(
        "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv",
        index_col=0,
    )[["HGNC symbol", "DBD"]]

    motif_to_tf = motif_to_tf.merge(right=human_tf_annot, how="left", left_on="TF", right_on="HGNC symbol")

    # For each motif, take the most common (mode) DBD annotation
    motif_to_dbd = (
        motif_to_tf.dropna()
        .groupby("MotifID")["DBD"]
        .agg(lambda x: x.mode().iat[0])  # take the first mode if there's a tie
        .reset_index()
    )

    motif_to_dbd = motif_to_dbd.set_index("MotifID")["DBD"].to_dict()

    return motif_to_dbd
