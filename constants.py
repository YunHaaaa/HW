task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}

acc_tasks = ["mnli","qnli", "rte", "sst2"]
f1_tasks = ["mrpc", "qqp"]