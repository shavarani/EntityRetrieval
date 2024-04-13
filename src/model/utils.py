
class LLMModel:
    """
    Used to unify the different LLM class implementations.
    """
    def annotate(self, record, summarize=False, verbose=False):
        raise NotImplementedError