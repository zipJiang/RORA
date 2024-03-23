from .collate_fn import CollateFn
from .strategyqa_collate_fn import (
    StrategyQACollateFn,
    StrategyQANGramClassificationCollateFn,
    StrategyQAEmbeddingClassificationCollateFn,
    StrategyQAGenerationCollateFn,
    StrategyQAInfillingCollateFn,
    StrategyQAIRMCollateFn,
    StrategyQAIRMEmbeddingClassificationCollateFn
)
from .ecqa_collate_fn import (
    ECQACollateFn,
    ECQALstmClassificationCollateFn,
    ECQAGenerationCollateFn,
    ECQAInfillingCollateFn,
    ECQAIRMCollateFn,
)