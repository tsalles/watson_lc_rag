## TODO List

- [ ] Conversation with memory
- [ ] Conversation with memory and RAG
- [ ] Other sources, like WebLoader
- [ ] Include OpenAI

When using without langchain:

```
    Answer the following question using only information from the article. If there is no good answer in
    the article, say "I don't know".

    Article: 
    ###
    {context}
    ###
    
    Question: What is the answer to life and the universe and everything?
    Answer: 42.
    
    Question: {question}
    Answer:
```