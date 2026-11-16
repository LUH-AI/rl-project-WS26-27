"""Run the VLA: python -m vla [--model edge|full] --prompt "..."."""

from __future__ import annotations

import os
import sys

import torch

from vla.node import OMNIVLA_DIR, get_parser, run_node

CHECKPOINTS = {"edge": "omnivla-edge", "full": "omnivla-original"}


def main() -> None:
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        sys.exit("OmniVLA needs a CUDA GPU.")
    if not os.path.isdir(os.path.join(OMNIVLA_DIR, CHECKPOINTS[args.model])):
        hint = " --full" if args.model == "full" else ""
        sys.exit(f"No {args.model} checkpoint found, run ./scripts/install_vla.sh{hint}")

    # Import only the chosen model, the full one pulls in much more
    if args.model == "full":
        from vla.omnivla_full import OmniVLAFull

        vla = OmniVLAFull(args.prompt)
    else:
        from vla.omnivla_edge import OmniVLAEdge

        vla = OmniVLAEdge(args.prompt, torch.device("cuda:0"))
    print(f"OmniVLA {args.model}, prompt={args.prompt!r}", flush=True)
    run_node(vla, args)


if __name__ == "__main__":
    main()
