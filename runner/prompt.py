METRIC_PROMPT = """
You are an AI judge focused on detecting if models are overthinking when solving a coding task.

<Task>
{task}
</Task>

<Think>
{think}
</Think>

<Solution>
{solution}
</Solution>

Analyze the content in <Task>, <Think> and <Solution>, given <Task> and <Solution>, please determine if the reasoning process in <Think> is overthink or not. 

<SCORING SYSTEM (0-10)>
 0-3 points (efficient thinking):
- The thinking process is concise and clear, cutting directly to the core of the problem
- Necessary boundary conditions and error handling are considered
- No unnecessary analysis or assumptions
- The optimal solution is quickly identified
- Accurately estimate the complexity of the problem

4-7 points (moderate thinking):
- Multiple solutions are considered, but the analysis is reasonable
- Some unlikely boundary cases are moderately analyzed
- Some overthinking in some parts, but the overall structure is still there
- Some unnecessary implementation details are analyzed
- There is a certain degree of repetitive analysis

8-10 points (overthinking):
- A large number of unrelated solutions are analyzed
- Over-complexification of obviously simple problems
- Repeatedly reiterate the same point of view or analysis
- Spend too much space on non-critical points
- Lengthy analysis of edge cases or almost impossible scenarios
- The solution exceeds the complexity required by the problem

</SCORING SYSTEM>

Give your reason and a score from 0 to 10 in JSON format:
{{
    "reason": "<reason>",
    "score": <from 0 to 10>
}}
"""


# FILTER_PROMPT = """
# You are an AI judge focused on extracting ONLY the useful reasoning steps that directly contribute to answering the question.
#
# <Task>
# {task}
# </Task>
#
# <thinking_process>
# {reasoning_content}
# </thinking_process>
#
# Your job is to:
# - Copy only the reasoning that logically leads to the final answer.
# - Skip irrelevant thoughts, re-checks, confirmations, meta-thinking, or commentary.
# - Do NOT paraphrase. Do NOT summarize. Just copy the useful reasoning steps.
#
# Output your result within <useful_content> and </useful_content> tags.
# """

FILTER_PROMPT = """
You are an AI assistant tasked with identifying and extracting ONLY the critical reasoning steps that directly lead to the final code solution.

<Task>
{task}
</Task>

<thinking_process>
{reasoning_content}
</thinking_process>

Your job is to:
- Copy only the reasoning that logically leads to the final code.
- Skip irrelevant thoughts, re-checks, confirmations, meta-thinking, or commentary.
- Do NOT paraphrase. Do NOT summarize. Just copy the useful reasoning steps.

Output your result within <useful_content> and </useful_content> tags.
"""