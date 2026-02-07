import torch
import logging
import time
from retro_star.common import (
    prepare_starting_molecules,
    prepare_mlp,
    prepare_molstar_planner,
    smiles_to_fp,
)
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger

import os

dirpath = os.path.dirname(os.path.abspath(__file__))


class RSPlanner:
    def __init__(
        self,
        gpu=-1,
        expansion_topk=50,
        iterations=500,
        use_value_fn=False,
        starting_molecules=dirpath + "/dataset/origin_dict.csv",
        mlp_templates=dirpath + "/one_step_model/template_rules_1.dat",
        mlp_model_dump=dirpath + "/one_step_model/saved_rollout_state_1_2048.ckpt",
        save_folder=dirpath + "/saved_models",
        value_model="best_epoch_final_4.pt",
        fp_dim=2048,
        viz=False,
        viz_dir="viz",
        max_routes=1,
    ):
        setup_logger()
        device = torch.device("cuda:%d" % gpu if gpu >= 0 else "cpu")
        starting_mols = prepare_starting_molecules(starting_molecules)

        one_step = prepare_mlp(mlp_templates, mlp_model_dump)

        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=fp_dim,
                latent_dim=128,
                dropout_rate=0.1,
                device=device,
            ).to(device)
            model_f = "%s/%s" % (save_folder, value_model)
            logging.info("Loading value nn from %s" % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x: 0.0

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir,
            max_routes=max_routes,
        )

    def plan(self, target_mol):
        t0 = time.time()
        succ, msg = self.plan_handle(target_mol)
        routes_list, num_iterations = msg

        if succ and routes_list:
            best = routes_list[0]
            result = {
                "succ": succ,
                "time": time.time() - t0,
                "iter": num_iterations,
                # Backward-compatible fields (best route)
                "routes": best.serialize(),
                "route_cost": best.total_cost,
                "route_len": best.length,
                # All found routes
                "all_routes": [
                    {
                        "route": r.serialize(),
                        "route_cost": r.total_cost,
                        "route_len": r.length,
                    }
                    for r in routes_list
                ],
                "num_routes": len(routes_list),
            }
            return result

        else:
            logging.info(
                "Synthesis path for %s not found. Please try increasing "
                "the number of iterations." % target_mol
            )
            return None


if __name__ == "__main__":
    planner = RSPlanner(gpu=0, use_value_fn=True, iterations=100, expansion_topk=50)

    result = planner.plan("CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC")
    print(result)

    result = planner.plan(
        "CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C"
    )
    print(result)

    result = planner.plan("CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1")
    print(result)
