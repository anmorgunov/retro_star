from retro_star.api import RSPlanner

planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=100,
    expansion_topk=50,
    starting_molecules='data/dataset/origin_dict.csv',
    mlp_templates='data/one_step_model/template_rules_1.dat',
    mlp_model_dump='data/one_step_model/saved_rollout_state_1_2048.ckpt',
    save_folder='data/saved_models',

)

result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
print(result)

result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
print(result)

result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
print(result)