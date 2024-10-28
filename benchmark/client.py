import time

import httpx
import msgspec
import pandas as pd
from fire import Fire

from oat.oracles.remote.client import RemoteRMOracle


def main(batch_size: int = 8, max_workers: int = 4, server_addr: str = "0.0.0.0:8000"):
    # A quick validation.
    req = {
        "batch_prompt": [
            "What is the range of the numeric output of a sigmoid node in a neural network?"
        ],
        "batch_candidates": [
            [
                "The output of a tanh node is bounded between -1 and 1.",
                "The output of a sigmoid node is bounded between 0 and 1.",
            ]
        ],
    }
    resp = httpx.post(
        f"http://{server_addr}/compare", content=msgspec.msgpack.encode(req)
    )
    print(resp.status_code, msgspec.msgpack.decode(resp.content))

    # Speed test.
    n = 2000
    remote_oracle = RemoteRMOracle(
        remote_rm_url=f"http://{server_addr}", max_workers=max_workers
    )
    data = pd.read_json(
        "/home/aiops/liuzc/ellm/output/neworacle_dpo_offline_0911T19:18/eval_results/380.json",
        orient="records",
        lines=True,
    )
    prompts = data["prompt"][:n].to_list()
    candidate_1 = data["response"][:n].to_list()
    candidate_2 = data["reference"][:n].to_list()

    st = time.time()
    _ = remote_oracle.compare(
        prompts, candidate_1, candidate_2, batch_size=batch_size, return_probs=True
    )
    print(time.time() - st)


if __name__ == "__main__":
    Fire(main)
