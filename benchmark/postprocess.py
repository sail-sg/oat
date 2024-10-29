import tqdm
import wandb
from fire import Fire


def main(run_name: str, wandb_proj: str = "lkevinzc/oat-benchmark"):
    features_of_interest = [
        "actor/oracle_time",
        "actor/generate_time",
        "misc/gradient_update_elapse",
        "train/learn_batch_time",
    ]

    api = wandb.Api()
    runs = api.runs(wandb_proj)
    for run in tqdm.tqdm(runs):
        if run.name == run_name:
            print(run.name)
            data = run.history(keys=features_of_interest)
            break

    print(data)
    print(data.iloc[range(5, 15), :].mean())


if __name__ == "__main__":
    Fire(main)
