# Recurrent Neural Networks (RNN)

## What is an RNN?

Imagine you're reading this sentence word by word. By the time you reach the word **"it"**, you already remember what came before — so you know what "it" refers to. Your brain doesn't reset after every word. It **carries memory forward.**

A normal neural network (like a plain ANN or CNN) doesn't do this. Every input is treated **independently** — it has no memory of what came before.

**RNN fixes this.** It's a neural network designed for **sequential data** — data where **order matters** and the past affects the future.

---

## Where is RNN used?

| Use Case | Why sequence matters |
|---|---|
| Text generation | Next word depends on previous words |
| Sentiment analysis | Meaning builds across the sentence |
| Speech recognition | Sound depends on surrounding sounds |
| Stock price prediction | Today's price depends on past prices |
| Machine translation | Word order changes meaning |

---

## How is RNN different from ANN?

| ANN | RNN |
|---|---|
| No memory | Has memory (hidden state) |
| Input → Output | Input + Past state → Output |
| Good for images, tabular data | Good for sequences, time series, text |
|Text input size may vary |Text input size must be fixed in a batch |
|Computational Power Required less |Computational Power Required more|
|Semantic meaning is not preserved well |Semantic meaning is preserved well|


---

## The Core Idea — Hidden State

At every time step, RNN does two things:
1. Takes the **current input** (e.g., current word)
2. Takes the **previous hidden state** (memory from past steps)

And produces:
- A **new hidden state** (updated memory)
- An **output**

```
h_t = tanh(W_h * h_(t-1) + W_x * x_t + b)
```

In simple words:
> **New memory = f(old memory + new input)**

---

## Visualizing the Flow

```
x1 → [RNN Cell] → h1
              ↓
x2 → [RNN Cell] → h2
              ↓
x3 → [RNN Cell] → h3 → Output
```

The same cell is **reused at every step** — that's why it's called *recurrent.*

---

## Key Terms to Remember

| Term | Meaning |
|---|---|
| x_t | Input at time step t |
| h_t | Hidden state at time step t (the "memory") |
| W_h, W_x | Weight matrices learned during training |
| tanh | Activation function used in RNN |
| Unrolling | Expanding the RNN across time steps for visualization |

---

## One Line Summary

> RNN is a neural network that has a **memory** — it processes sequences by passing information from one step to the next through a **hidden state.**

---

![alt text](image.png)
Got it! Tu chahti hai ki main isko rewrite karun — cleaner, easier to understand, better flow ke saath. Let me make a much better version:

````markdown
# 🔁 RNN Architecture

---

## 🧠 Why RNN?

Normal neural networks process **one input at a time** — they have no memory.

RNN is special because it has **memory** — it remembers what came before while processing the current input.

This makes it perfect for:
- Text (words depend on previous words)
- Time series (values depend on previous values)
- Speech, music, video

---

## 📥 Step 1 — How Do We Feed Text into RNN?

RNN cannot read words. It only understands **numbers**.

So we first convert text → numbers using **One-Hot Encoding**.

### Our Example Dataset

| Comment | Sentiment |
|---|---|
| you are good | 1 (positive) |
| you are bad | 0 (negative) |
| you are not good | 0 (negative) |

---

## 🔤 Step 2 — One-Hot Encoding

From the 3 sentences above, we get **5 unique words** → vocabulary size = **5**

Each word becomes a vector of size 5, where only **one position is 1** (hot):

```
you  = [1, 0, 0, 0, 0]
are  = [0, 1, 0, 0, 0]
good = [0, 0, 1, 0, 0]
bad  = [0, 0, 0, 1, 0]
not  = [0, 0, 0, 0, 1]
```

So the sentence **"you are good"** becomes:

```
[ [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0] ]
   you            are           good
```

---

## 📐 Step 3 — Keras Input Shape

Keras wants input in this exact format:

```
(batch_size, time_steps, input_features)
```

| Term | What it means | Our Example |
|---|---|---|
| `batch_size` | How many sentences at once | 3 sentences |
| `time_steps` | How many words per sentence | 3 words |
| `input_features` | Size of each word vector | 5 (vocab size) |

```
Input shape = (3, 3, 5)
```

---

## ⚙️ Step 4 — Inside One RNN Cell

Think of one RNN cell as a small unit that does **two things**:

1. Takes the **current word** (`x_t`)
2. Takes the **memory from previous step** (`h_(t-1)`)
3. Combines them → produces **new memory** (`h_t`) and **output** (`y_t`)

```
         previous memory
         h_(t-1)
            │
x_t ──► [ RNN Cell ] ──► h_t ──► (passed to next cell)
                │
               y_t  (output of this step)
```

### 🧮 The Math Behind It

```
h_t = tanh( Wh · h_(t-1)  +  Wx · x_t  +  b )
             ↑                  ↑
        memory weight      input weight

y_t = Wy · h_t + by
```

> 💡 Key insight: **Wh and Wx are the same weights at every step** — RNN reuses them.

---

## 📜 Step 5 — Processing a Full Sentence (Unrolled View)

For sentence **"you are good"** (3 words = 3 time steps):

```
  x1(you)      x2(are)      x3(good)
     │             │             │
h0 ►[Cell]► h1 ►[Cell]► h2 ►[Cell]► h3
     │             │             │
    y1            y2            y3
```

- `h0` = initial memory (usually all zeros)
- Same cell is reused 3 times — just drawn separately for clarity
- For **sentiment analysis**, only `y3` (final output) is used

---

## 🗂️ Types of RNN

| Type | Input → Output | Real Use Case |
|---|---|---|
| One-to-One | 1 → 1 | Simple classification |
| One-to-Many | 1 → sequence | Image captioning |
| **Many-to-One** | sequence → 1 | **Sentiment analysis** ✅ |
| Many-to-Many (equal) | sequence → sequence | POS tagging |
| Many-to-Many (unequal) | sequence → sequence | Language translation |
| Stacked RNN | sequence → sequence | Complex NLP tasks |
| Bidirectional RNN | sequence → sequence | NER, Question Answering |

---

## ⚔️ Feed Forward Network vs RNN

| | Feed Forward | RNN |
|---|---|---|
| Memory | ❌ No memory | ✅ Has hidden state |
| Input | Fixed size | Variable length sequence |
| Flow | Input → Output | Input + Past memory → Output |
| Best for | Images, tabular data | Text, speech, time series |

---

## 💡 One-Line Summary

> RNN processes sequences **one step at a time** — at each step it reads the current input AND remembers the past, then passes that memory forward.

---
````

Yeh wala zyada clean hai — proper flow hai Step 1 → 2 → 3 → 4 → 5, emojis se sections easy to scan hain, aur explanations pehle intuition dete hain phir math. Paste kar de apne README mein! 🚀
![alt text](image-1.png)



