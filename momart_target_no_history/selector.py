"""One frozen five-task selector shared by all ten policy data sources."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
RESEARCH=Path(__file__).resolve().parents[1]/'experiments/momart_target_research_20261101'
sys.path.insert(0,str(RESEARCH))
from runtime import SingleBoxSelector
from .train_unified import FAMILIES,family


class UnifiedTargetSelector:
    def __init__(self,bundle):
        self.runtime=SingleBoxSelector(Path(bundle)/'selected.json',bundle)
        if self.runtime.tasks != FAMILIES:
            raise ValueError('Expected frozen five-task vocabulary')

    def select(self,frame,source,episode_id,sensor_step):
        task=family(source)
        context=np.r_[frame.context[:276],np.eye(5,dtype=np.float32)[FAMILIES.index(task)]]
        return self.runtime.select(replace(frame,context=context,task=task),episode_id,sensor_step)
