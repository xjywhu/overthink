from runner.main import Runner
import json
import numpy as np


def run():
    server = Runner()

    # server.start("results/codeexecution/deepseek-v3/demo.json", max_workers=10)

    # server.start("results/codegeneration/qwq-32b/demo.json", max_workers=10)
    # server.start("results/codegeneration/deepseek-r1/lcb_generation.json", max_workers=10)
    server.start("results/codegeneration/deepseek-v3/lcb_generation.json", max_workers=10)
    # server.start("results/codegeneration/qwen2.5-72b-Ins/lcb_generation.json", max_workers=10)

def analyze_score(res_file="results/codegeneration/qwq-32b/lcb_generation.json"):
    model_name = res_file.split("/")[-2]
    scores = []
    with open(res_file, "r") as fs:
        questions = json.loads(fs.read())
        for question in questions:
            # try:
            scores.append(question["gpt_4o_metrc"]["score"])
            # except:
            #     print(question["question_id"])
    unique_scores = set(scores)
    labels = [f"Score {score}" for score in unique_scores]
    frequencies = [scores.count(score) for score in unique_scores]
    # Calculate proportions
    total = sum(frequencies)
    proportions = [round(freq / total, 4) for freq in frequencies]

    # Create dictionary with score:proportion pairs
    result = {score: prop for score, prop in zip(unique_scores, proportions)}
    # print(result)
    print(model_name, np.mean(scores))
    # print(labels)
    # print(frequencies)
    return scores


# scores = analyze_score()


if __name__ == "__main__":
    models = ["deepseek-r1", "deepseek-v3", "qwq-32b", "qwen2.5-72b-Ins"]
    for model in models:
        analyze_score(res_file=f"results/codegeneration/{model}/lcb_generation.json")

    # run()
    pass