# What is a Transformer?

**One line:** Transformer is a deep learning architecture that processes sequences (text, audio, images) using **attention** instead of recurrence.

---

**The problem before Transformers:**

RNN and LSTM read data **word by word** — like reading a book one letter at a time and trying to remember everything. The further back a word was, the more it got "forgotten."

Also — it was **slow** because everything was sequential. Word 2 had to wait for Word 1 to finish.

---

**What Transformer does differently:**

It looks at the **entire sentence at once** — all words in parallel — and figures out which words are related to which.

> Example: *"The trophy didn't fit in the bag because **it** was too big."*
> 
> What does "it" refer to — trophy or bag? **Transformer figures this out** by looking at all words together and their relationships.

---

**Why it changed everything:**

- Faster to train (parallel processing)
- Handles long-range dependencies better
- Works on text, images, audio, code — anything sequential
- Foundation of: **BERT, GPT, Claude, ChatGPT, Gemini** — literally all modern AI

---

**Interview Q&A:**

**Q: What problem did Transformers solve that RNNs couldn't?**
> RNNs process sequences step-by-step and struggle with long-range dependencies due to vanishing gradients. Transformers use attention to process all tokens in parallel, capturing long-range relationships efficiently.

**Q: In one line, what is a Transformer?**
> A deep learning architecture that uses self-attention to process sequences in parallel, without recurrence.

---
