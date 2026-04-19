# DL_Clear_Basics

🔹 What is a Neuron?

    Think of your brain — it has billions of biological neurons that receive signals, process them, and send output to other neurons.
    An artificial neuron mimics exactly this:

            Inputs (x) → like signals coming in
            Weights (w) → how important each input is
            Bias (b) → a constant that shifts the result
            Activation function → decides whether the neuron "fires" or not
            Output → the final result passed forward

![alt text](image.png)

        Formula inside a neuron:
        output = f( x₁·w₁ + x₂·w₂ + x₃·w₃ + b )
        That's it. Inputs × weights → sum → activation function → output.

Perceptron: 

        A Perceptron is the simplest form of a neural network — just one neuron, making a yes/no decision.
        It was invented in 1958 by Frank Rosenblatt.
        How it works:

        Takes inputs, multiplies by weights, adds bias
        If the sum is above a threshold → output = 1 (yes/fire)
        If below → output = 0 (no/don't fire)

        Example: Is this email spam?

        Input 1: Contains word "FREE" → x₁ = 1
        Input 2: From unknown sender → x₂ = 1
Perceptron calculates → "YES, spam!" → output = 1

          Concept            ->      What it means
          
          Neuron             ->      Basic unit — takes inputs, processes, gives output
          Weight (w)         ->      Importance of each input
          Bias (b)           ->      Shifts the output — gives flexibility
          Activation f(x)    ->      Decides if neuron "fires"
          Perceptron         ->      Single neuron doing binary (0/1) decision  

🔹 4 core steps of how a neural network learns:-

Step 1 — y = mx + b (Linear equation / Weighted Sum)

This is what happens inside the neuron before activation.


        y = mx + b
        In neural network language, this becomes:
        z = (w₁·x₁) + (w₂·x₂) + ... + b

        x → your input data (like pixel values, or age, salary etc.)
        w (weight) → how much importance to give each input
        b (bias) → shifts the line up or down, gives flexibility
        z → the raw number that comes out before activation

Think of it like this: you're drawing a line through data. w decides the slope, b decides where the line crosses the y-axis.
But this alone can only solve linear problems (straight lines). Real data is messy — so we need the next step.

Step 2 — Sigmoid (Activation Function)


The output z from step 1 can be any number — like 500, -300, 2.7 — anything.
Sigmoid squishes that number into a range between 0 and 1.


        σ(z) = 1 / (1 + e⁻ᶻ)

        If z is very large → sigmoid ≈ 1
        If z is very small (negative) → sigmoid ≈ 0
        If z = 0 → sigmoid = 0.5

Why do we need this?
        Because neural networks need to make decisions like probabilities — "70% chance this is a cat." Raw numbers like 500 or -300 don't make sense as probabilities. Sigmoid converts them into something meaningful (0 to 1).

Step 3 — Loss Function

Now the network made a prediction (output from sigmoid). But is it correct?

Loss function measures how wrong the prediction is.

Most common one for classification — Binary Cross Entropy:

        Loss = -[ y·log(ŷ) + (1-y)·log(1-ŷ) ]

        y → actual answer (0 or 1)
        ŷ (y-hat) → what the network predicted (between 0 and 1)
        If prediction is close to actual → loss is small
        If prediction is far from actual → loss is large

Simple example:

        Actual: y = 1 (it IS a cat)
        Predicted: ŷ = 0.9 → loss is small ✅
        Predicted: ŷ = 0.1 → loss is huge ❌

The goal of training = minimize this loss.

Step 4 — Optimizer

Now we know the loss. But how do we fix the weights to reduce it?
That's the optimizer's job. Most common one → Gradient Descent.
        w = w - α · (∂Loss/∂w)

Line by line:

       * ∂Loss/∂w → gradient (tells us: if we increase w a little, does loss go up or down?)
       * α (alpha) → learning rate — how big a step we take (too big = overshoot, too small = too slow)
       * w = w - ... → we update the weight in the direction that reduces loss

Think of it like this: you're blindfolded on a hilly surface. You want to reach the lowest point (minimum loss). Gradient descent tells you — "take one step downhill." You keep doing this until you reach the bottom.

                Input data
                ↓
                Step 1: z = wx + b        ← forward pass
                ↓
                Step 2: ŷ = sigmoid(z)    ← activation
                ↓
                Step 3: Loss = how wrong  ← compare with actual
                ↓
                Step 4: Update weights    ← optimizer fixes w and b
                ↓
                Repeat 1000s of times → network learns!