import torch


def compute_class_loss(rpn_true, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: (batch, anchors, 1). Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_class_logits: (batch, anchors, 2). RPN classifier logits for BG/FG.
    """

