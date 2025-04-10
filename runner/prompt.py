METRIC_PROMPT = """
Given the code snippet:
{code_snippet}

Given the thinking content:
{reasoning_content}

Please determine if the reaosning is overthink or not. Give your reason and a score from 0 to 10 in JSON format:
{{
    "reason": "<reason>",
    "score": <from 0 to 10>
}}
"""