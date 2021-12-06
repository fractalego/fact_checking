import torch

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_answer_from_text(text):
    _claim_yn = 'The evidence supports the claim:\n'
    pos = text.find(_claim_yn) + len(_claim_yn)
    return text[pos]


def generate_y_n_answer(model, tokenizer, prompt):
    model.eval()
    model.to(_device)
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt')
        _length = 1
        tokens_length = tokens.shape[1]
        if tokens_length + _length > 1024:
            return ''

        output = model.generate(
            tokens.to(_device),
            max_length=tokens_length + _length,
            pad_token_id=50256
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = get_answer_from_text(text)

    return answer


def create_prompt_from_evidence_and_claim(evidence, claim):
    if not evidence:
        return 'N'
    if evidence[-1] != '.':
        evidence += '.'
    if claim[-1] != '.':
        claim += '.'
    text = 'Evidence:\n'
    text += evidence.capitalize().replace('\n\n', '\n') + '\n\n'
    text += 'Claim:\n'
    text += claim.capitalize() + '\n\n'
    text += 'The evidence supports the claim:\n'
    return text


def generate_multiple_y_n_answers(model, tokenizer, prompt, num_replicas):
    model.train()
    model.to(_device)
    outputs_count = {}
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt')
        tokens = tokens.repeat(num_replicas, 1)
        _length = 50
        tokens_length = tokens.shape[1]
        if tokens_length + _length > 1024:
            return ''

        output = model.generate(
            tokens.to(_device),
            max_length=tokens_length + _length,
            pad_token_id=50256
        )
        for index in range(num_replicas):
            text = tokenizer.decode(output[index, :], skip_special_tokens=True)
            answer = get_answer_from_text(text)
            outputs_count.setdefault(answer, 0)
            outputs_count[answer] += 1

    total = sum(v for v in outputs_count.values())
    return {k: v / total for k, v in outputs_count.items()}
