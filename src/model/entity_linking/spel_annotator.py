"""
This file is a connector which loads SpEL and performs annotation using its pretrained checkpoints.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../../spel/src'))
from timeout_decorator.timeout_decorator import TimeoutError
from spel.src.spel.model import SpELAnnotator, dl_sa, tokenizer
from spel.src.spel.span_annotation import WordAnnotation, PhraseAnnotation
from spel.src.spel.utils import get_subword_to_word_mapping

class SpELAnnotate:
    def __init__(self, finetuned_after_step = 4, device = 'cpu'):
        self.spel = SpELAnnotator()
        self.spel.init_model_from_scratch(device=device)
        if finetuned_after_step == 3:
            self.spel.shrink_classification_head_to_aida(device)
        self.spel.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=finetuned_after_step)

    def annotate(self, sentence):
        inputs = tokenizer(sentence, return_tensors="pt")
        token_offsets = list(zip(inputs.encodings[0].tokens,inputs.encodings[0].offsets))
        subword_annotations = self.spel.annotate_subword_ids(
            inputs.input_ids, k_for_top_k_to_keep=10, token_offsets=token_offsets)
        tokens_offsets = token_offsets[1:-1]
        subword_annotations = subword_annotations[1:]
        try:
            word_annotations = [WordAnnotation(subword_annotations[m[0]:m[1]], tokens_offsets[m[0]:m[1]])
                                for m in get_subword_to_word_mapping(inputs.tokens(), sentence)]
        except TimeoutError:
            return []
        phrase_annotations = []
        for w in word_annotations:
            if not w.annotations:
                continue
            if phrase_annotations and phrase_annotations[-1].resolved_annotation == w.resolved_annotation:
                phrase_annotations[-1].add(w)
            else:
                phrase_annotations.append(PhraseAnnotation(w))
        final_result = [{
            "annotation": dl_sa.mentions_itos[phrase_annotation.resolved_annotation],
            "begin_character": phrase_annotation.begin_character,
            "end_character": phrase_annotation.end_character
        } for phrase_annotation in phrase_annotations if phrase_annotation.resolved_annotation != 0]
        return final_result

if __name__ == '__main__':
    m = SpELAnnotate()
    annotations = m.annotate("Grace Kelly by Mika reached the top of the UK Singles Chart in 2007.")
    print(annotations)
