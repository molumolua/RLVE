import re
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
def compute_score(solution_str, ground_truth):
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    acc = 0
    pred = ""
    format_verify = 0.0
    try:
        answer_str = solution_str
        # format_verify,answer_str = format_verify_and_extract(solution_str)
        acc,_=verify_func([ground_truth], [answer_str])
    except Exception as e:
        print(e)
    except TimeoutException:
        print("TimeoutException in math-verify.")

    reward = 1.0 if acc else -1.0

    return dict(
        reward = reward,
        accuracy = acc,
        format_score = format_verify,
    )
