from pynei.config import DEF_POP_NAME


def _calc_pops_idxs(pops: dict[list[str]] | None, samples):
    if pops is None:
        pops_idxs = {DEF_POP_NAME: slice(None, None)}
    else:
        if samples is None:
            raise ValueError("Variants should have samples defined if pops is not None")
        samples_idx = {sample: idx for idx, sample in enumerate(samples)}
        pops_idxs = {}
        for pop_id, pop_samples in pops.items():
            pops_idxs[pop_id] = [samples_idx[sample] for sample in pop_samples]
    return pops_idxs
