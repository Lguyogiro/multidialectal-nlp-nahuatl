import glob
import json
import numpy as np

exp_dir_names = {
    "azz": "azz monolingual",
    "nhi": "nhi monolingual",
    "azz-nhi": "xlingual train=azz eval=nhi",
    "nhi-azz": "xlingual train=nhi eval=azz",
}

path_to_logs = "/home/pughrob/machamp/logs"

for dirname, exp in exp_dir_names.items():
    print(exp)
    metrics_files = [glob.glob(f"{path_to_logs}/straight_exps_no_pretraining/{dirname}_fold={i}/*/metrics.json") for i in range(1, 11)]
    metrics_files = [l[0] for l in metrics_files if len(l) != 0]
    tasks = {
         'best_dev_feats_multi_acc': [],
         'best_dev_lemma_accuracy': [],
         'best_dev_upos_accuracy': [],
         'best_dev_dependency_uas': [],
         'best_dev_dependency_las': []


    }
    task2name = {
         'best_dev_feats_multi_acc': "Morphological Analysis",
         'best_dev_lemma_accuracy': "Lemmatization",
         'best_dev_upos_accuracy': "Part-of-speech",
         'best_dev_dependency_uas': "UAS",
         'best_dev_dependency_las': "LAS"
    }

    for mf in metrics_files:
        with open(mf) as f:
            metrics_dict = json.load(f)
            for taskname in tasks:
                task_result = metrics_dict[taskname]
                tasks[taskname].append(task_result)

    for task, results in tasks.items():
        print(f"\t{task2name[task][:3]}\t{np.mean(results)}\t{np.std(results)}")
    


joint_exp_dir_names = {
    "joint_full_data_azz": "joint training eval=azz",
    "joint_full_data_nhi": "join training  eval=nhi",
    "joint_half_data_azz": "joint training (half-data) eval=azz",
    "joint_half_data_nhi": "joint training (half-data) eval=nhi",
    "joint_langid_full_data_nhi": "joint training with langid eval=nhi",
    "joint_langid_full_data_azz": "joint training with langid eval=azz",
    "mlm_nahuatl_pretraining_nhi": "pretrained on Nahuatl MLM eval=nhi",
    "mlm_nahuatl_pretraining_azz": "pretrained on Nahuatl MLM eval=azz",
    "es_ud_pretraining_azz": "pretrained on Spanish UD, eval=azz",
    "es_ud_pretraining_nhi": "pretrained on Spanish UD, eval=nhi"
}
pred_metrics_path = '/home/pughrob/machamp/joint-training-predictions'
for dirname, exp in joint_exp_dir_names.items():
    print(exp)
    metrics_files = [glob.glob(f"{pred_metrics_path}/{dirname}-{i}.out.eval")[0] for i in range(1, 11)]

    tasks = {
         'best_dev_feats_multi_acc': [],
         'best_dev_lemma_accuracy': [],
         'best_dev_upos_accuracy': [],
         'best_dev_dependency_uas': [],
         'best_dev_dependency_las': []
    }
    for mf in metrics_files:
        with open(mf) as f:
            res = json.load(f)
            tasks['best_dev_lemma_accuracy'].append(res["lemma"]["accuracy"]["accuracy"])
            tasks['best_dev_upos_accuracy'].append(res["upos"]["accuracy"]["accuracy"])
            tasks['best_dev_feats_multi_acc'].append(res["feats"]["multi_acc"]["multi_acc"])
            tasks['best_dev_dependency_uas'].append(res["dependency"]["uas"]["uas"])
            tasks['best_dev_dependency_las'].append(res["dependency"]["las"]["las"])
    for task, results in tasks.items():
        print(f"\t{task2name[task][:3]}\t{np.mean(results)}\t{np.std(results)}")
