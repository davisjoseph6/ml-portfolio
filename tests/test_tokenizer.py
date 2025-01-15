#!/usr/bin/env python3

from transformers import pipeline

pipe = pipeline("summarization", model="../my_summarization_model", tokenizer="../my_summarization_model")
print(pipe("Test text"))  
