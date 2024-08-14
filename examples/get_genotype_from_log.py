from __future__ import annotations

from collections import namedtuple

import torch
from torch import tensor  # noqa: F401
import torch.nn.functional as F  # noqa: N812


def extract_darts_alpha_at_epoch(
    log_file_path: str,
    epoch: int,
    end_epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    epoch_str = f"{epoch:03d}"
    with open(log_file_path) as f:
        lines = [line.rstrip() for line in f]
    # print(f"[{epoch_str}-{end_epoch:03d}]")
    for line_num, line in enumerate(lines):  # noqa: B007
        # if f"[{epoch_str}-100]" in line:
        if f"[{epoch_str}-{end_epoch:03d}]" in line:

            break

    alpha_0 = ""
    alpha_1 = ""

    store_alpha_0 = False
    store_alpha_1 = False
    num_alpha_stored = 0
    for line in lines[line_num + 1 :]:
        if num_alpha_stored == 2:
            break

        if "alpha 0 is Parameter containing:" in line:
            store_alpha_0 = True
            store_alpha_1 = False

        if "alpha 1 is Parameter containing:" in line:
            store_alpha_1 = True
            store_alpha_0 = False

        if "requires_grad=True" in line:
            num_alpha_stored += 1

        if store_alpha_0:
            alpha_0 += line
        elif store_alpha_1:
            alpha_1 += line
        else:
            continue

    assert alpha_0 != "", "alpha not present at epoch " + epoch_str
    assert alpha_1 != ""
    # remove till we find tensor
    alpha_0 = alpha_0[alpha_0.find("tensor") : alpha_0.find("device")] + ") "
    alpha_1 = alpha_1[alpha_1.find("tensor") : alpha_1.find("device")] + ") "

    alpha_0 = eval(alpha_0)  # noqa: S307
    alpha_1 = eval(alpha_1)  # noqa: S307

    return alpha_0, alpha_1


Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


class DARTSGenotype(Genotype):
    def __str__(self) -> str:
        return (
            f"DARTSGenotype(normal={self.normal}, "
            f"normal_concat={self.normal_concat}, "
            f"reduce={self.reduce}, "
            f"reduce_concat={self.reduce_concat})"
        )

    def tostr(self) -> str:
        return str(self)


def get_genotype_from_alpha(
    alphas_normal: torch.Tensor, alphas_reduce: torch.Tensor
) -> DARTSGenotype:
    """Get the genotype of the model, representing the architecture.

    Returns:
        Structure: An object representing the genotype of the model, which describes
        the architectural choices in terms of operations and connections between
        nodes.
    """

    def _parse(weights: list[torch.Tensor]) -> list[tuple[str, int]]:
        gene = []
        n = 2
        start = 0
        for i in range(4):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(
                range(i + 2),
                key=lambda x: -max(
                    W[x][k]
                    for k in range(len(W[x]))  # type: ignore
                    if k != PRIMITIVES.index("none")
                ),
            )[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index("none") and (
                        k_best is None or W[j][k] > W[j][k_best]
                    ):
                        k_best = k
                gene.append((PRIMITIVES[k_best], j))  # type: ignore
            start = end
            n += 1
        return gene

    gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2, 6)
    genotype = DARTSGenotype(
        normal=gene_normal,
        normal_concat=concat,
        reduce=gene_reduce,
        reduce_concat=concat,
    )
    return genotype


if __name__ == "__main__":
    log_file_path = "your_log_file_here"
    # get alpha at 100th epoch
    alpha_normal, alpha_reduce = extract_darts_alpha_at_epoch(
        log_file_path, epoch=100, end_epoch=100
    )
    # get genotype
    genotype = get_genotype_from_alpha(alpha_normal, alpha_reduce)
    print(genotype)
