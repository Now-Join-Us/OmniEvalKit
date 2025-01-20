import os
import numpy as np
from collections import Counter, defaultdict
# from dataloaders.calculators.utils import CODE_WARNING
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataloaders.calculators.code_metric import check_correctness, estimate_pass_at_k

def code_eval(filtered_r, is_filtered, test_case, **kwargs):
    candidates = filtered_r
    # references = gold['references']
    task_id = kwargs.get('id')
    num_workers = kwargs.get('num_workers',4)
    timeout = kwargs.get('timeout', 3.0)  # seconds
    k = kwargs.get('k', [1, 10, 100])

    if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
        raise ValueError(CODE_WARNING)

    if os.name == "nt":
        raise NotImplementedError("This metric is currently not supported on Windows.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for candidate in candidates:
            
            test_program = candidate + "\n" + test_case
            args = (test_program, timeout, task_id, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    if not isinstance(ks, (list, tuple)):
        ks = [ks]
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

    return {
        'pass_at_k': pass_at_k,
        'results': results
    }

CODE_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""