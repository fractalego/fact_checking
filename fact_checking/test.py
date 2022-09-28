from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from fact_checking import FactChecker


_evidence = "The system has only one User. The User's dog is called George. The User is called Anastasia."
#_evidence = 'User: my dog\'s name is Anastasia.'

_claim = "The User is called Anastasia."

print(_evidence)
print(_claim)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
fact_checking_model = GPT2LMHeadModel.from_pretrained('fractalego/fact-checking')
fact_checker = FactChecker(fact_checking_model, tokenizer)
is_claim_true = fact_checker.validate(_evidence, _claim)

print(is_claim_true)
