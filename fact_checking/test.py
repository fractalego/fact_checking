from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from fact_checking import FactChecker


_evidence = """
The clock time is 12,30.
"""

_claim = 'The clock says it after 12'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
fact_checking_model = GPT2LMHeadModel.from_pretrained('fractalego/fact-checking')
fact_checker = FactChecker(fact_checking_model, tokenizer)
is_claim_true = fact_checker.validate(_evidence, _claim)

print(is_claim_true)
