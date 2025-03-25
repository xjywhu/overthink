import json
import re

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")


def count_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


# def extract_think_answer(output):
#     think_content = re.findall(r'<think>(.*?)</think>', output)
#     think_content = think_content.group(1) if think_content else None
#
#     # 提取[ANSWER][/ANSWER]标签之间的内容
#     answer_content = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', output)
#     answer_content = answer_content.group(1) if answer_content else None
#
#     return think_content, answer_content

# def extract_think_answer(text):
#     # 提取 <think> 和 </think> 之间的内容
#     think_pattern = r'<think>\n(.*?)\n</think>'
#     think_content = re.findall(think_pattern, text)
#
#     # 提取 [ANSWER] 和 [/ANSWER] 之间的内容
#     answer_pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
#     answer_content = re.findall(answer_pattern, text)
#
#     return think_content, answer_content

def extract_think_answer(text):
    # 提取 <think> 和 </think> 之间的内容
    think_start = text.find('<think>')  # 查找 <think> 的位置
    think_end = text.find('</think>')  # 查找 </think> 的位置

    think_content = ""
    if think_start != -1 and think_end != -1:
        think_content = text[think_start + len('<think>'):think_end]  # 提取 <think> 和 </think> 之间的内容

    # 提取 [ANSWER] 和 [/ANSWER] 之间的内容
    answer_start = text.find('[ANSWER]')  # 查找 [ANSWER] 的位置
    answer_end = text.find('[/ANSWER]')  # 查找 [/ANSWER] 的位置

    answer_content = ""
    if answer_start != -1 and answer_end != -1:
        answer_content = text[answer_start + len('[ANSWER]'):answer_end]  # 提取 [ANSWER] 和 [/ANSWER] 之间的内容

    return think_content, answer_content


def get_think_process(task_name, result_file):
    with open(f"../results/{task_name}/{result_file}", "r") as fs:
        res = json.loads(fs.read())
        for question in res:
            output_list = question["output_list"]
            # print(output_list[0])
            think, answer = extract_think_answer(output_list[0])
            print([think, answer])
            print(count_tokens(think), count_tokens(answer))

    pass


if __name__ == "__main__":
    get_think_process(task_name="codeexecution", result_file="lcb_output_VAN_direct.json")
