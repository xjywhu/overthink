import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import re

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")


def count_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


# def plot_distribution(data_list, name):
#     # 转换为numpy数组方便处理
#     data = np.array(data_list)
#
#     # 创建画布
#     plt.figure(figsize=(10, 6))
#
#     # 绘制直方图（Histogram）
#     sns.histplot(data, kde=False, bins='auto', color='skyblue', alpha=0.7, label='Histogram')
#
#     # 绘制核密度估计（KDE）
#     sns.kdeplot(data, color='red', linewidth=2, label='KDE')
#
#     # 添加均值和标准差标注
#     mean = np.mean(data)
#     std = np.std(data)
#     plt.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
#     plt.axvline(mean + std, color='orange', linestyle=':', linewidth=2, label=f'±1 Std: {std:.2f}')
#     plt.axvline(mean - std, color='orange', linestyle=':', linewidth=2)
#
#     # 添加图例和标题
#     plt.legend()
#     plt.title(f'Thinking Token Distribution ({name})', fontsize=14)
#     plt.xlabel('Number of Tokens')
#     plt.ylabel('Frequency/Density')
#     plt.grid(axis='y', alpha=0.3)
#
#     # 显示图形
#     plt.show()


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
    tokens = []
    with open(f"../results/{task_name}/{result_file}", "r") as fs:
        res = json.loads(fs.read())
        for question in res:
            output_list = question["output_list"]
            # print(output_list[0])
            think, answer = extract_think_answer(output_list[0])
            print([think, answer])
            print(count_tokens(think), count_tokens(answer))
            tokens.append(count_tokens(think))
    # plot_distribution(tokens, result_file)
    return tokens


if __name__ == "__main__":
    get_think_process(task_name="codeexecution", result_file="lcb_output_VAN_direct.json")
    get_think_process(task_name="codeexecution", result_file="crux_output_VAN_direct_record.json")

    # plot_distribution([1,2,3], "test")
