"""Read-only diagnostic probabilities. These are not action sampling weights."""
import numpy as np

def distribution(logits, mask=None):
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite diagnostic logits')
    legal = np.ones(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if legal.shape != values.shape:
        raise ValueError('diagnostic mask shape mismatch')
    result = np.zeros(values.shape, dtype=np.float64)
    if legal.any():
        result[legal] = np.exp(values[legal] - values[legal].max())
        result /= result.sum()
    return result.tolist()

def probability_snapshot(card_logits, position_logits, cards, positions, hand, deck, *, tick, side, timing):
    if np.shape(card_logits) != (4,) or np.shape(position_logits) != (4, 576):
        raise ValueError('unexpected diagnostic output shape')
    return dict(tick=tick, side=side, play_probability=timing, threshold=.5,
                semantics='greedy; card=P(slot|ordinary play); cell=P(cell|slot); not sampling',
                card_ids=[int(deck[i]['card_id']) if 0 <= i < len(deck) else None for i in hand],
                card_forms=[int(deck[i].get('form_flags', 0)) if 0 <= i < len(deck) else 0 for i in hand],
                card_raw=distribution(card_logits), card_legal=distribution(card_logits, cards),
                legal_cards=np.asarray(cards, dtype=bool).tolist(),
                position_raw=[distribution(row) for row in position_logits],
                position_legal=[distribution(row, mask) for row, mask in zip(position_logits, positions)],
                legal_positions=np.asarray(positions, dtype=bool).tolist())

def screen_cell(column, row, side):
    """Same camera mapping as ScreenLayout.deployment_point, top row first."""
    return (31-row)*18 + (17-column if side == 1 else column)
