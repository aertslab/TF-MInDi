
from dataclasses import dataclass
from typing import Self
import h5py
import numpy as np
import pandas as pd


@dataclass
class Seqlet:
    seq_instance: np.ndarray
    start: int
    end: int
    region_one_hot: np.ndarray
    is_revcomp: bool
    contrib_scores: np.ndarray | None = None
    hypothetical_contrib_scores: np.ndarray | None = None
    @classmethod
    def read_from_file(
        cls,
        p: h5py._hl.group.Group,
        seqlet_idx: int,
        ohs: np.ndarray,
    ):
        """
        Read a seqlet from file

        Parameters
        ----------
        p
            Open hdf5 file.
        seqlet_idx
            Index, relative to ohs, of the seqlet.
        ohs
            Original one hot encoded sequences used to generate the patterns.
        """
        contrib_scores = p["contrib_scores"][seqlet_idx]
        hypothetical_contrib_scores = p["hypothetical_contribs"][seqlet_idx]
        seq_instance = p["sequence"][seqlet_idx]
        start = p["start"][seqlet_idx]
        end = p["end"][seqlet_idx]
        is_revcomp = p["is_revcomp"][seqlet_idx]
        region_idx = p["example_idx"][seqlet_idx]
        region_one_hot = ohs[region_idx].T
        if (
            not np.all(seq_instance == region_one_hot[start : end])
            and not is_revcomp
        ) or (
            not np.all(
                seq_instance[::-1, ::-1] == region_one_hot[start : end]
            )
            and is_revcomp
        ):
            raise ValueError(
                "sequence instance does not match onehot\n"
                + f"region_idx\t{region_idx}\n"
                + f"start\t\t{start}\n"
                + f"end\t\t{end}\n"
                + f"is_revcomp\t{is_revcomp}\n"
                + f"seq. instance sequence: {seq_instance.argmax(1)}\n"
                + f"ONEHOT sequence:        {region_one_hot[start: end].argmax(1)}"
            )
        return cls(
            seq_instance=seq_instance,
            start=start,
            end=end,
            region_one_hot=region_one_hot,
            is_revcomp=is_revcomp,
            contrib_scores=contrib_scores,
            hypothetical_contrib_scores=hypothetical_contrib_scores
        )
    def __repr__(self):
        return f"Seqlet {self.start}:{self.end}"

@dataclass
class ModiscoPattern:
    ppm: np.ndarray
    seqlets: list[Seqlet]
    contrib_scores: np.ndarray | None = None
    hypothetical_contrib_scores: np.ndarray | None = None
    is_pos: bool | None = None
    subpatterns: list[Self] | None = None
    @classmethod
    def read_from_file(
        cls,
        p: h5py._hl.group.Group,
        is_pos: bool,
        ohs: np.ndarray,
    ):
        """
        Read a pattern from a hdf5 file

        Parameters
        ----------
        p
            Open hdf5 file.
        is_pos
            Wether the pattern is positive or not.
        ohs
            Original one hot encoded sequences used to generate the patterns.
        """
        contrib_scores = p["contrib_scores"][:]
        hypothetical_contrib_scores = p["hypothetical_contribs"][:]
        ppm = p["sequence"][:]
        is_pos = is_pos
        seqlets = [
            Seqlet.read_from_file(p=p["seqlets"], seqlet_idx=i, ohs=ohs)
            for i in range(p["seqlets"]["n_seqlets"][0])
        ]
        subpatterns = [
            ModiscoPattern(p[sub], is_pos, ohs)
            for sub in p.keys()
            if sub.startswith("subpattern_")
        ]
        return cls(
            contrib_scores=contrib_scores,
            hypothetical_contrib_scores=hypothetical_contrib_scores,
            ppm=ppm,
            is_pos=is_pos,
            seqlets=seqlets,
            subpatterns=subpatterns
        )
    def __repr__(self):
        return f"ModiscoPattern with {len(self.seqlets)} seqlets"
    def ic(self, bg=np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
        return (
            self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)
        ).sum(1)
    def ic_trim(self, min_v: float, **kwargs) -> tuple[int, int]:
        delta = np.where(np.diff((self.ic(**kwargs) > min_v) * 1))[0]
        if len(delta) == 0:
            return 0, 0
        start_index = min(delta)
        end_index = max(delta)
        return start_index, end_index + 1

def create_pattern(
    seqlet_df: pd.DataFrame,
    strands: np.ndarray,
    offsets: np.ndarray,
    ohs: np.ndarray,
    contribs = np.ndarray
):
    max_s = max(seqlet_df["end"] - seqlet_df["start"])
    seqlet_instances = np.zeros( (seqlet_df.shape[0], max_s, 4) )
    seqlet_contribs = np.zeros( (seqlet_df.shape[0], max_s, 4) )
    seqlets: list[Seqlet] = []
    for i, (_, (start, end)) in enumerate(seqlet_df[["start", "end"]].iterrows()):
        st = strands[i]
        of = offsets[i].astype(int)
        of = of * -1 if st else of
        if not st:
            _start = start + of
            _end = start + of + max_s
        else:
            _start = end + of - max_s
            _end = end + of
        if _start < 0 or _end > ohs.shape[2]:
          print("seqlet exceeds one hot")
          continue
        inst = ohs[i, :, _start: _end].T
        cont = contribs[i, :, _start: _end].T
        if st:
            inst = inst[::-1, ::-1]
            cont = cont[::-1, ::-1]
        seqlet_instances[i] = inst
        seqlet_contribs[i ] = cont
        seqlets.append(
            Seqlet(
                seq_instance=inst,
                start=_start,
                end=_end,
                region_one_hot=ohs[i].T,
                is_revcomp=bool(st),
                contrib_scores=inst * cont,
                hypothetical_contrib_scores=cont
            )
        )
    pattern = ModiscoPattern(
        ppm=seqlet_instances.mean(0),
        seqlets=seqlets,
        contrib_scores=(seqlet_instances*seqlet_contribs).mean(0),
        hypothetical_contrib_scores=seqlet_contribs.mean(0),
    )
    return pattern
