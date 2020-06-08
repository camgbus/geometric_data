import torch
import torch.nn.functional as F

from ptt.agents.agent import Agent
from ptt.eval.metrics import dice

class SegmentationAgent(Agent): 
    def __init__(self, config, base_criterion=dice, verbose=True):
        super().__init__(config=config, base_criterion=base_criterion, verbose=verbose)
        # Extend dictionary
        self.metrics['dice'] = dice
        self.metrics['bce'] = F.binary_cross_entropy_with_logits

    def criterion(self, outputs, targets):
        bce_weight=0.5
        bce = F.binary_cross_entropy_with_logits(outputs, targets)
        dice_loss = dice(outputs, targets)
        loss = bce * bce_weight + dice_loss * (1 - bce_weight)
        return loss