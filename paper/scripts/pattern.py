
import numpy as np
import pandas as pd
from crested.tl.modisco._tfmodisco import Seqlet, ModiscoPattern

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
