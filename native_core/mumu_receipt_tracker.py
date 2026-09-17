"""Evidence-based receipt reconciliation; an unresolved click is never retried."""
from .mumu_live_actions import own_player,card_receipt

class ReceiptTracker:
    SOFT_SECONDS=1.8
    REVIEW_SECONDS=4.

    def __init__(self,before,*,side,slot,card_id):
        self.before=before;self.side=side;self.slot=slot;self.card_id=card_id
        player=own_player(before,side)
        self.hand=list(player['hand_deck_indices']);self.next_index=player.get('next_deck_index',-1)
        self.last_tick=before['game_tick'];self.first_rotation_tick=None;self.rotation_count=0
        self.max_elixir_drop=0;self.last_elixir=player['elixir_raw'];self.drop_seen=False
        self.sent_at=None;self.samples=0;self.soft_reported=False;self.review_reported=False
        self.ambiguous=False;self.last_evidence={}

    def arm(self,when):
        if self.sent_at is None:self.sent_at=when

    def observe(self,frame,now):
        receipt=card_receipt(self.before,frame,side=self.side,slot=self.slot,card_id=self.card_id)
        elapsed=max(0.,now-self.sent_at) if self.sent_at is not None else 0.
        result=dict(receipt,accepted=False,phase='waiting',elapsed_ms=round(elapsed*1000),samples=self.samples)
        if receipt['reason']=='battle_changed':
            self.ambiguous=True;result.update(phase='ambiguous',reason='battle_changed');return result
        player=own_player(frame,self.side);tick=frame.get('game_tick',-1)
        if not frame.get('coherent') or player is None or tick<=self.last_tick:
            result.update(phase='waiting',reason='awaiting_new_coherent_frame');return result
        self.last_tick=tick;self.samples+=1
        hand=list(player.get('hand_deck_indices',[]))
        if len(hand)!=4 or any(i not in range(8) for i in hand):
            result.update(reason='transient_hand_unavailable');return result
        elixir=player['elixir_raw']
        # Preserve a real downward transition; later regeneration cannot erase it.
        self.max_elixir_drop=max(self.max_elixir_drop,self.last_elixir-elixir)
        self.drop_seen=self.drop_seen or receipt.get('elixir_decrease_raw',0)>0 or self.max_elixir_drop>0
        self.last_elixir=elixir
        other_slots_same=all(hand[i]==self.hand[i] for i in range(4) if i!=self.slot)
        changed=hand[self.slot]!=self.hand[self.slot]
        expected_rotation=changed and hand[self.slot]==self.next_index and self.next_index in range(8)
        if not other_slots_same or (changed and not expected_rotation):self.ambiguous=True
        if changed and other_slots_same and expected_rotation:
            if self.first_rotation_tick is None:self.first_rotation_tick=tick
            self.rotation_count+=1
        else:
            if self.first_rotation_tick is not None:self.ambiguous=True
            self.rotation_count=0
        # Exact selected-slot -> known-next-card transition, stable across two
        # unique ticks, is primary evidence. Net mana decline is corroboration.
        confirmed=(not self.ambiguous and expected_rotation and other_slots_same and self.rotation_count>=2)
        result.update(samples=self.samples,expected_next_index=self.next_index,
                      expected_rotation=expected_rotation,other_slots_same=other_slots_same,
                      rotation_samples=self.rotation_count,first_rotation_tick=self.first_rotation_tick,
                      max_observed_elixir_drop_raw=self.max_elixir_drop,elixir_drop_seen=self.drop_seen,
                      evidence_ambiguous=self.ambiguous)
        if confirmed:
            result.update(accepted=True,phase='confirmed',reason='stable_expected_hand_rotation')
        elif self.ambiguous:
            result.update(accepted=False,phase='ambiguous',reason='unattributed_hand_change')
        elif elapsed>=self.REVIEW_SECONDS:
            result.update(accepted=False,phase='review',reason='execution_still_unknown_no_retry')
        elif elapsed>=self.SOFT_SECONDS:
            result.update(accepted=False,phase='grace',reason='extended_receipt_observation')
        else:result['accepted']=False
        self.last_evidence=result
        return result
