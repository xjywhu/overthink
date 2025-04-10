import os
import re
import json
import traceback
from typing import Optional
from copy import deepcopy
from tqdm.auto import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm import OpenKeyChat, Message
from .prompt import METRIC_PROMPT


def extract_think(text: str) -> str:
    return text.split("<think>")[-1].split("</think>")[0].strip()

def extract_json_format(content: str) -> dict:
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}")
            return None
    return None

def multiprocess_run(
    data_list,
    process_function,
    max_workers=5,
    progress_bar_desc="Processing",
    additional_args=None
):
    additional_args = additional_args or {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_function, idx, item, **additional_args)
            for idx, item in enumerate(data_list)
        ]

        for future in tqdm(as_completed(futures), total=len(data_list), desc=progress_bar_desc):
            try:
                future.result()
            except Exception as e:
                print(f"Error during processing: {e}")
                traceback.print_exc()


class Runner:
    def __init__(self) -> None:
        self.lock = Lock()
        self.model = OpenKeyChat("gpt-4o", temperature=0, top_p=0.95, max_tokens=2000, delay=4)
        self.system_prompt = "You are a helpful assistant."
        self.filepath = None
        self.results = []

    def load(self) -> list:
        """
        Load the dataset from a JSON file.
        """
        with open(self.filepath, "r") as f:
            self.results = json.load(f)
        return self.results

    def save(self) -> None:
        """
        Save the results to a JSON file.
        """
        with open(self.filepath, "w") as f:
            json.dump(self.results, f, indent=4)

    def run_demo(self, question: dict) -> str:
        think = extract_think(question["output_list"][0])
        inputs = {
            "code_snippet": question["code"],
            "reasoning_content": think,
        }
        prompt = METRIC_PROMPT.format(**inputs).strip()
        input_prompt = [Message(role="system", content=self.system_prompt), Message(role="user", content=prompt)]
        response = self.model.chat(input_prompt)
        try:
            json_response = extract_json_format(response[0])
            return json_response
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {response[0]}")
            return None

    def start(self, filepath: str, max_workers: int = 30, **kwargs) -> None:
        self.filepath = filepath
        self.load()

        def process_function(idx, _):
            try:
                question = self.results[idx]
                if "gpt_4o_metrc" in question:
                    return

                response = self.run_demo(question)
                if response:
                    question["gpt_4o_metrc"] = response
                    with self.lock:
                        self.save()
                else:
                    print(f"Failed to process question {question.get('id', idx)}: No response")
            except Exception as e:
                print(f"Error in process_function for question {question.get('id', idx)}: {e}")
                traceback.print_exc()

        multiprocess_run(
            data_list=list(enumerate(self.results)),
            process_function=process_function,
            max_workers=max_workers,
            progress_bar_desc=f"{self.model.model_name}: Processing Questions",
            additional_args=kwargs,
        )