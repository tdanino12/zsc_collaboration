import os
import socket
import sys

import numpy as np
import wandb
from loguru import logger

wandb_name = "tdanino12"
POLICY_POOL_PATH = "/home/user/Desktop/zseval/ZSC-Eval/zsceval/scripts/policy_pool"


def extract_sp_S1_models(layout, exp, env="Overcooked"):
    os.environ["WANDB_API_KEY"] = "495b87eba3dbc88f719508680483181c811852ba"
    os.environ["WANDB_MODE"] = "online"
    wandb.login(key = os.environ["WANDB_API_KEY"])    # Default/Base scheme
    
    api = wandb.Api()
    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"
    runs = api.runs(
        f"{wandb_name}/{env}",
        filters={
            "$and": [
                {"config.experiment_name": exp},
                {layout_config: layout},
                {"state": "finished"},
                {"tags": {"$nin": ["hidden", "unused"]}},
            ]
        },
        order="+config.seed",
    )
    runs = list(runs)
    run_ids = [r.id for r in runs]
    logger.info(f"num of runs: {len(runs)}")
    seeds = set()
    num_agents = None
    print("########################################################")
    print(run_ids)
    print("########################################################")
    path = "/home/user/ZSC/results/Overcooked/"+layout+"/mappo/hsp-S1/wandb/" # "/home/user/ZSC/results/Overcooked/random0_medium/mappo/hsp-S1/wandb/"
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    print(folders)
    #my runs = ['qfc76int', 'dpsj957n', 'dyvgp7k9', '54o3pah9', 'kvxb13na', 'd1houooy', 'iel1lviw', '1tzeu757', '6olr0pm7', 'ven9byw3', '3drto6y8', 'loahg7ti', 'd9cn391r', 'tdj01ne6', '63yy9304', 'wqtnjhtc', 'ym4246uk', 'gqq9rdi6', 'h08j4i63', 'yv2t07pn', 'prqv2jvp', '2l31q6pu', 'pvmqe4cv', 'yvvy9xkg', 'rfkqqyuq', '2ivrthvj', 'q8dck2da', 'k6t9jma3', 'c91wykf4', '9s1dedil']
    for r_i, run_id in enumerate(run_ids):
        run = runs[r_i]
        if run.state == "finished":
           
            run2 = wandb.init(project="Overcooked", name=run_id + "_" + str(r_i))  # this creates a new run
            this_run_id = run2.id
            
            
            folder = next((item for item in folders if run_id in item), None)
            if(folder == None):
                continue
            folder = "/home/user/ZSC/results/Overcooked/"+layout+"/mappo/hsp-S1/wandb/"+folder+"/files/" # "/home/user/ZSC/results/Overcooked/random0_medium/mappo/hsp-S1/wandb/"+folder+"/files/"
            print("###")
            print(folder, run_id)
            print("###")
            wandb.save(os.path.join(folder, "**"))  # Double-asterisk for recursion
            wandb.finish()
            run2 = api.run(f"tdanino12/Overcooked/{this_run_id}")
            files = run2.files()
            print("################")
            print("those are the files:", files)
            print("$$$$$$$$$$$$$$$$$$")
            if num_agents is None:
                num_agents = run.config["num_agents"]
            if run.config["seed"] in seeds:
                continue
            i = run.config["seed"]
            history = run.history()
            history = history[["_step", "ep_sparse_r"]]
            steps = history["_step"].to_numpy().astype(int)
            ep_sparse_r = history["ep_sparse_r"].to_numpy()
            final_ep_sparse_r = np.mean(ep_sparse_r[-5:])
            logger.info(f"hsp{i} Run: {run_id} Seed: {run.config['seed']} Return {final_ep_sparse_r}")
            seeds.add(run.config["seed"])
            #files = run.files()
            actor_pts = [f for f in files if f.name.startswith("files/actor")]
            actor_versions = [eval(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts]
            actor_pts = {v: p for v, p in zip(actor_versions, actor_pts)}
            actor_versions = sorted(actor_versions)
            max_actor_versions = max(actor_versions) + 1
            max_steps = max(steps)

            new_steps = [steps[0]]
            new_ep_sparse_r = [ep_sparse_r[0]]
            for s, er in zip(steps[1:], ep_sparse_r[1:]):
                l_s = new_steps[-1]
                l_er = new_ep_sparse_r[-1]
                for w in range(l_s + 1, s, 100):
                    new_steps.append(w)
                    new_ep_sparse_r.append(l_er + (er - l_er) * (w - l_s) / (s - l_s))
            steps = new_steps
            ep_sparse_r = new_ep_sparse_r

            # select checkpoints
            selected_pts = dict(mid=-1, final=max_steps)
            mid_ep_sparse_r = final_ep_sparse_r / 2
            min_delta = 1e9
            for s, score in zip(steps, ep_sparse_r):
                if min_delta > abs(mid_ep_sparse_r - score):
                    min_delta = abs(mid_ep_sparse_r - score)
                    selected_pts["mid"] = s

            selected_pts = {k: int(v / max_steps * max_actor_versions) for k, v in selected_pts.items()}
            sparse_r_dict = dict(init=0, mid=mid_ep_sparse_r, final=final_ep_sparse_r)
            for tag, exp_version in selected_pts.items():
                version = actor_versions[0]
                for actor_version in actor_versions:
                    if abs(exp_version - version) > abs(exp_version - actor_version):
                        version = actor_version
                logger.info(f"hsp{i}: {tag} Expected: {exp_version} {sparse_r_dict[tag]} Found: {version}")
                actor_pts = []
                for a_i in range(run.config["num_agents"]):
                    actor_pts.append(run2.file(f"files/actor_agent{a_i}_periodic_{version}.pt"))

                tmp_dir = f"/home/user/Desktop/zseval/tmp/{layout}/{exp}"
                for pt in actor_pts:
                    print("here!!!!!!!!!!!!!!!")
                    pt.download(tmp_dir, replace=True)

                hsp_s1_dir = f"{POLICY_POOL_PATH}/{layout}/hsp/s1/{exp.replace('-S1', '')}"
                os.makedirs(hsp_s1_dir, exist_ok=True)
                for a_i in range(run.config["num_agents"]):
                    pt_path = f"{hsp_s1_dir}/hsp{i}_{tag}_w{a_i}_actor.pt"
                    logger.info(f"pt {a_i} store in {pt_path}")
                    os.system(f"mv {tmp_dir}/files/actor_agent{a_i}_periodic_{version}.pt {pt_path}")

    logger.success(f"Extracted {len(seeds)} models for {layout}")


if __name__ == "__main__":
    layout = sys.argv[1]
    env = sys.argv[2]
    assert layout in [
        "random0",
        "random0_medium",
        "random1",
        "random3",
        "small_corridor",
        "unident_s",
        "random0_m",
        "random1_m",
        "random3_m",
        "academy_3_vs_1_with_keeper",
        "all",
    ], layout
    if layout == "all":
        layout = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
            "academy_3_vs_1_with_keeper",
        ]
    else:
        layout = [layout]

    hostname = socket.gethostname()
    exp_names = {
        "random0": "hsp-S1",
        "random0_medium": "hsp-S1",
        "random3_m": "hsp-S1",
        "small_corridor": "hsp-S1",
    }

    # logger.add(f"./extract_log/extract_{layout}_hsp_S1_models.log")
    # logger.info(f"hostname: {hostname}")
    for l in layout:
        exp = exp_names[l]
        logger.info(f"Extracting {exp} for {l}")
        extract_sp_S1_models(l, exp, env)
