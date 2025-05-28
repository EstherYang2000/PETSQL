# ppo_expert_selector.py
import os, json, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from rl_expert_selector import ExpertSelectionEnv
from evaluation import build_foreign_key_map_from_json, evaluate_cc
from utils.file_utils import load_prompts, append_json, write_txt

# ---------- Network ----------
class PolicyNet(nn.Module):
    def __init__(self, dim_in, n_actions, hid=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, hid), nn.ReLU(),
            nn.Linear(hid, hid),    nn.ReLU(),
            nn.Linear(hid, n_actions)
        )

    def forward(self, x):                # x: (B, dim)
        return self.fc(x)                # logits


class ValueNet(nn.Module):
    def __init__(self, dim_in, hid=128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(dim_in, hid), nn.ReLU(),
            nn.Linear(hid, hid),    nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x):                # (B, dim)
        return self.v(x).squeeze(-1)     # (B,)


# ---------- Reward (batched) ----------
@torch.no_grad()
def batched_reward(gold_sqls, pred_sqls, db_path, kmaps):
    etype = "all"
    res = []
    for g, p in zip(gold_sqls, pred_sqls):
        print(g, p)
        ok = evaluate_cc(g, p, db_dir=db_path, etype=etype, kmaps=kmaps)
        
        res.append(int(ok))
    return torch.tensor(res, dtype=torch.float32)


# ---------- PPO ----------
def ppo_train(env, policy, value, device,
              epochs=96, steps=4096, batch=256,
              gamma=0.99, lam=0.95,
              clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):

    opt = Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    global_step = 0
    pbar = tqdm(total=steps, desc="Collecting")

    obs = env.reset().to(device)
    stor = []  # (obs, act, rew, logp, val, done)
    all_rewards = []
    all_entropies = []
    all_actions = []
    n_experts = policy.fc[-1].out_features if hasattr(policy, 'fc') else None

    while global_step < steps:
        logits = policy(obs.unsqueeze(0))
        prob   = torch.softmax(logits, -1)
        dist   = torch.distributions.Categorical(prob)
        act    = dist.sample()
        logp   = dist.log_prob(act)
        val    = value(obs.unsqueeze(0)).squeeze()

        nxt_obs, rew, done, _ = env.step(act.item())
        rew = torch.tensor(rew, dtype=torch.float32, device=device)

        # Store only values (not graph-tensors) for logp and val
        stor.append((obs.cpu(), act.item(), rew.cpu(),
                     logp.item(), val.item(), done))

        # logging for debug
        all_rewards.append(rew.item())
        all_entropies.append(dist.entropy().item())
        all_actions.append(act.item())
        if global_step % 100 == 0 and global_step > 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_entropy = np.mean(all_entropies[-100:])
            if n_experts is None:
                n_experts = logits.size(-1)
            act_hist = np.bincount(all_actions[-100:], minlength=n_experts)
            act_hist = act_hist / np.sum(act_hist)
            print(f"Step {global_step}: avg_reward={avg_reward:.3f}, avg_entropy={avg_entropy:.3f}, action_dist={act_hist}", flush=True)
            pbar.set_description(f"Collecting | avg_r={avg_reward:.2f} | ent={avg_entropy:.2f}")

        obs = env.reset().to(device) if done else nxt_obs.to(device)
        global_step += 1
        pbar.update(1)

    pbar.close()


    # -------- GAE --------
    obs_t, act_t, rew_t, logp_t, val_t, done_t = zip(*stor)
    obs_t   = torch.stack(obs_t).to(device)
    act_t   = torch.tensor(act_t,   dtype=torch.long,   device=device)
    rew_t   = torch.tensor(rew_t,   dtype=torch.float32, device=device)
    logp_t  = torch.tensor(logp_t,  dtype=torch.float32, device=device)
    val_t   = torch.tensor(val_t,   dtype=torch.float32, device=device)
    done_t  = torch.tensor(done_t,  dtype=torch.float32, device=device)

    T = len(rew_t)
    adv = torch.zeros(T, device=device)
    gae = 0
    next_val = 0
    for t in reversed(range(T)):
        delta = rew_t[t] + gamma * next_val * (1 - done_t[t]) - val_t[t]
        gae   = delta + gamma * lam * (1 - done_t[t]) * gae
        adv[t] = gae
        next_val = val_t[t]

    ret = adv + val_t
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # -------- Update --------
    ds = torch.utils.data.TensorDataset(obs_t, act_t, logp_t, adv, ret)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    for _ in range(epochs):
        for ob, ac, old_lp, ad, rt in loader:
            logits  = policy(ob)
            prob    = torch.softmax(logits, -1)
            dist    = torch.distributions.Categorical(prob)
            lp      = dist.log_prob(ac)
            ratio   = torch.exp(lp - old_lp)

            pol_loss = -torch.min(ratio * ad,
                                  torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * ad).mean()
            ent      = dist.entropy().mean()
            val_pred = value(ob)
            vf_loss  = F.mse_loss(val_pred, rt)

            loss = pol_loss + vf_coef * vf_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            opt.step()



# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_generate", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--start_num_prompts", type=int, default=0)
    parser.add_argument("--end_num_prompts",   type=int, default=None)
    parser.add_argument("--dataset_type", choices=["dev","test"], default="dev")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ---- logging ----
    os.makedirs(args.path_generate, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.path_generate, "ppo.log"),
                        level=logging.INFO, format="%(asctime)s %(message)s")

    # ---- data & experts ----
    questions_path = os.path.join(args.path_generate, "questions.json")
    prompts = load_prompts(args.path_generate, args.start_num_prompts, args.end_num_prompts)
    with open(args.gold) as f:
        gold_sqls = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    with open(questions_path) as f:
        questions = json.load(f)
    # <-- 這裡請依照你的路徑載入 expert_list 並填 raw_sql_outputs -->
    expert_list = [
        {"name": "llamaapi_3.3", "model": "llamaapi", "version": "3.3","path":"data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_llama/llamaapi_3.3_output.txt"},
        {"name": "gpt-4o", "model": "gptapi", "version": "gpt-4o","path":"data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_4o/gptapi_gpt-4o_output.txt"},
        {"name": "o3-mini", "model": "gptapi", "version": "o3-mini","path":"data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_o3_mini/gptapi_o3-mini_output.txt"},
        {"name": "qwen_api_32b-instruct-fp16", "model": "qwen_api", "version": "32b-instruct-fp16","path":"data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_qwen_32b/qwen_api_32b-instruct-fp16_output.txt"},
        {"name": "qwen_api_2_5_72b", "model": "qwen_api", "version": "2_5_72b","path":"data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_qwen_72b/qwen_api_2_5_72b_output.txt"},
        {"name": "gemini", "model": "googlegeminiapi", "version": "gemini-2.5-pro-exp-03-25","path":"data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_gemini/googlegeminiapi_gemini-2.5-pro-exp-03-25_output.txt"},

    ]
    for expert in expert_list:
        path = expert['path']
        with open(path) as f:
            raw_sql_outputs = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
        expert['raw_sql_outputs'] = raw_sql_outputs

    # ---- embedding ----
    embedder = SentenceTransformer("all-mpnet-base-v2")
    prompt_embs = torch.tensor(embedder.encode(prompts), dtype=torch.float32)

    # ---- reward helper ----
    table_path = "./data/spider/tables.json" if args.dataset_type=="dev" \
                 else "./data/spider/test_tables.json"
    kmaps = build_foreign_key_map_from_json(table_path)

    # ---- build env ----
    db_path = "./data/spider/database" if args.dataset_type=="dev" \
              else "./data/spider/test_database"
    env = ExpertSelectionEnv(
        prompts, expert_list, gold_sqls,
        prompt_embs,
        batched_evaluate_fn=batched_reward,
        kmaps=kmaps,
        db_path=db_path,
        dataset_type=args.dataset_type
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pol = PolicyNet(prompt_embs.size(1), len(expert_list), args.hidden_dim).to(device)
    val = ValueNet(prompt_embs.size(1), args.hidden_dim).to(device)

    # ---- train PPO ----
    ppo_train(env, pol, val, device)

    # ---- inference & save ----
    results, finals = [], []
    with torch.no_grad():
        for idx, q in enumerate(questions):
            obs = env.get_obs(idx).unsqueeze(0).to(device)
            logits = pol(obs)
            probs  = torch.softmax(logits, -1).cpu().numpy().flatten()
            top_i  = int(np.argmax(probs))
            chosen = expert_list[top_i]["name"]
            final_sql = expert_list[top_i]["raw_sql_outputs"][idx][0]

            results.append({
                "index": idx + args.start_num_prompts,
                "question": q["question"],
                "chosen_expert": chosen,
                "final_sql": final_sql,
                "probs": probs.tolist()
            })
            finals.append(final_sql)

    # write files
    append_json(os.path.join(args.path_generate, "results_rl.json"), results)
    write_txt(os.path.join(args.path_generate, "final_sql_rl.txt"), finals)
    print("✅ RL 推論完成")


if __name__ == "__main__":
    main()


"""
python src/sources/rl_selection/cc_gpt_rl.py \
  --path_generate data/vote/rl/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_rl \
  --gold ./data/spider/dev_gold.sql \
  --start_num_prompts 0 --end_num_prompts 1034 \
  --dataset_type dev --device cuda

"""