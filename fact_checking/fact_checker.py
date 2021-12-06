from fact_checking.utils import (
    create_prompt_from_evidence_and_claim,
    generate_y_n_answer,
    generate_multiple_y_n_answers)


class FactChecker:

    def __init__(self, model, tokenizer):
        '''
        :param model: One of the models in this repository
        :param tokenizer: The appropriate tokenizer
        '''

        self._model = model
        self._tokenizer = tokenizer

    def validate(self, evidence, claim):
        """
        :param evidence: A short text containing the evidence for the claim
        :param claim: A claim to check against the evidence
        :return: True or False boolean values
        """
        prompt = create_prompt_from_evidence_and_claim(evidence.strip(), claim.strip())
        answer = generate_y_n_answer(self._model, self._tokenizer, prompt)
        return True if answer == 'Y' else False

    def validate_with_replicas(self, evidence, claim, num_replicas=20):
        """
        :param evidence: A short text containing the evidence for the claim
        :param claim: A claim to check against the evidence
        :param num_replicas: the number of generated predictions
        :return: A dict like {'Y': 0.9, 'N': 0.1}, showing the model's belief
                 whether the claim is consistent with the evidence or not
        """
        prompt = create_prompt_from_evidence_and_claim(evidence.strip(), claim.strip())
        return generate_multiple_y_n_answers(self._model, self._tokenizer, prompt, num_replicas)
