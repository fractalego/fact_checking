from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from fact_checking import FactChecker

_evidence = 'There is only one User. The User\'s house name is Anastasia.'

_claim = "The User is called George."


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
fact_checking_model = GPT2LMHeadModel.from_pretrained('fractalego/fact-checking')
fact_checker = FactChecker(fact_checking_model, tokenizer)
is_claim_true = fact_checker.validate_with_replicas(_evidence, _claim)

print(is_claim_true)
