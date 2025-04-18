import os
import json

question_path = "bird/process/BIRD-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-10000/questions.json"
prompt_path = "bird/process/BIRD-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-10000/prompts.txt"
questions_json = json.load(open(os.path.join(question_path), "r"))
prompt = ""

questions = [_["prompt"] for _ in questions_json["questions"]]

with open(prompt_path, "w", encoding='utf-8') as f_out:
    for i, question in enumerate(questions_json["questions"]):
        prompt += question["prompt"]
        # prompt += "### Here are some external knowledge about the question you can refer to.\n"
        # prompt += f"{question['response']}\n"
        f_out.write(prompt + "\n\n\n\n")
