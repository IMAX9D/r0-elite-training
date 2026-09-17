"""Non-blocking click accounting. Never authorizes, delays, retries or stops input."""
from .mumu_live_actions import own_player

class PassiveReceipts:
    def __init__(self):
        self.reset()

    def reset(self):
        self.records={};self.transitions=[];self.hand=None;self.generations=[0]*4;self.last_tick=-1

    def add(self,request):
        item=dict(request);item['generation']=self.generations[item['slot']];item['sent']=False
        self.records[item['id']]=item

    def sent(self,request_id,when):
        if request_id in self.records:self.records[request_id].update(sent=True,completed_at=when)

    def cancel(self,request_id):
        self.records.pop(request_id,None)

    def observe(self,frame,side,now):
        events=[];player=own_player(frame,side);tick=frame['game_tick']
        if not frame.get('coherent') or player is None or tick<=self.last_tick:return events
        hand=list(player['hand_deck_indices'])
        if len(hand)!=4 or any(i not in range(8) for i in hand):return events
        # Finalize transitions only after a second, distinct tick. Multiple clicks
        # for one card can produce one rotation; never count all of them as plays.
        for transition in self.transitions:
            slot=transition['slot'];candidates=transition['candidates']
            same_action=bool(candidates) and all((r['card'],r['position'])==(candidates[0]['card'],candidates[0]['position']) for r in candidates)
            certain=(same_action and all(r['sent'] for r in candidates) and transition['single_slot'] and
                     hand[slot]==transition['new_index'] and
                     all(r['expected_next']==transition['new_index'] for r in candidates))
            events.append(dict(event='play_observed',tick=transition['tick'],slot=slot,
                attribution=('unique_click' if len(candidates)==1 else 'consistent_repeated_clicks') if certain else 'unknown',
                candidate_clicks=len(candidates),request=candidates[0] if certain and len(candidates)==1 else None,
                history_action=dict(card=candidates[0]['card'],position=candidates[0]['position']) if certain else None))
        self.transitions=[]
        if self.hand is not None:
            changed=[s for s in range(4) if hand[s]!=self.hand[s]]
            for slot in changed:
                candidates=[r for r in self.records.values() if r['slot']==slot and
                            r['deck_index']==self.hand[slot] and r['generation']==self.generations[slot]]
                self.transitions.append(dict(tick=tick,slot=slot,new_index=hand[slot],
                                             candidates=candidates,single_slot=len(changed)==1))
                for r in candidates:self.records.pop(r['id'],None)
                self.generations[slot]+=1
        self.hand=hand;self.last_tick=tick
        for request_id,r in list(self.records.items()):
            if now-r.get('completed_at',r['submitted_at'])>=4:
                events.append(dict(event='touch_not_confirmed',tick=r['tick'],slot=r['slot'],
                                   request_id=request_id,reason='passive_timeout_no_block'))
                self.records.pop(request_id,None)
        return events
