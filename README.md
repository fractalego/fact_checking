## Fact checking

This generative model - trained on FEVER - aims to predict whether a claim is consistent with the provided evidence.


### Installation and simple usage
One quick way to install it is to type

```bash
pip install fact_checking
```

and then use the following code:

```python
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from fact_checking import FactChecker

_evidence = """
Justine Tanya Bateman (born February 19, 1966) is an American writer, producer, and actress . She is best known for her regular role as Mallory Keaton on the sitcom Family Ties (1982 -- 1989). Until recently, Bateman ran a production and consulting company, SECTION 5 . In the fall of 2012, she started studying computer science at UCLA.
"""

_claim = 'Justine Bateman is a poet.'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
fact_checking_model = GPT2LMHeadModel.from_pretrained('fractalego/fact-checking')
fact_checker = FactChecker(fact_checking_model, tokenizer)
is_claim_true = fact_checker.validate(_evidence, _claim)

print(is_claim_true)
```

which gives the output

```bash
False
```

### Probabilistic output with replicas
The output can include a probabilistic component, obtained by iterating a number of times the output generation.
The system generates an ensemble of answers and groups them by Yes or No.

For example, one can ask
```python
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from fact_checking import FactChecker

_evidence = """
Jane writes code for Huggingface.
"""

_claim = 'Jane is an engineer.'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
fact_checking_model = GPT2LMHeadModel.from_pretrained('fractalego/fact-checking')
fact_checker = FactChecker(fact_checking_model, tokenizer)
is_claim_true = fact_checker.validate_with_replicas(_evidence, _claim)

print(is_claim_true)

```

with output
```bash
{'Y': 0.95, 'N': 0.05}
```


### Score on FEVER

The predictions are evaluated on a subset of the FEVER dev dataset, 
restricted to the SUPPORTING and REFUTING options:

| precision | recall | F1|
| --- | --- | --- |
|0.94|0.98|0.96|

These results should be taken with many grains of salt. This is still a work in progress, 
and there might be leakage coming from the underlining GPT2 model unnaturally raising the scores.

 