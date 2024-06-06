
from transformers import pipeline

# create pieline for QA
qa = pipeline('question-answering')

ctx = "My name is Ganesh and I am studying Data Science"
que = "What is Ganesh studying?"
qa(context=ctx, question=que)

# prompt = f"""
# Your task is to select
#
# Summarize the review below, delimited by triple
# backticks, in at most 30 words.
#
# Review: ```{prod_review}```
# """