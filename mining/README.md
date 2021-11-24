# Stage I: Mining
> In the first
stage, called mining, we find out possible FN samples from the N/A set by
heuristically leveraging the memory mechanism of deep neural networks.
According to (Arpit et al., 2017), deep neural networks tend to learn and
memorize patterns from clean instances within a noisy dataset.
We design a Transformer-based (Vaswani et al., 2017) deep filter to mine
FN samples from the N/A set.

**TRAINING:**
~~~bash
python run.py
~~~

**INFERRING:**
~~~bash
python infer.py
~~~




