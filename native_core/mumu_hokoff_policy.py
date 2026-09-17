"""Fixed4 release adapter for the verified read-only MuMu observation schema.

Cross-build deployment is explicit. Public history requires past, consumed commands
with observed spending; it is never inferred from spawned units.
"""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from .mumu_live_controller import _native_state,_position_masks
from .gui import CARD_COSTS
from .card_catalog import metadata
from expert_selfplay_v1.native_observation import NativeActorFrame
from expert_v1.tick_store_v1.schema import normalize_native_state
from .mumu_probabilities import probability_snapshot
from .mumu_timing_gate import TimingGate

MODEL_ROOT=Path(__file__).resolve().parents[1]/'models/hokoff-restart319051'

def verify_source(root:Path):
    manifest=json.loads((root/'LOCAL_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,item in manifest.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('invalid model source manifest path')
        path=root/rel
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('model source changed: '+name)

class LivePolicy:
    warnings=[
        '训练环境150535029 → 实时160402002/16.402.10，属于跨版本适配，未宣称数值等价',
        '公开历史仅接收已消费且扣费核验通过的原生命令；缺失/歧义保持未知，不从单位生成猜出牌',
        '仅普通下牌；在线英雄技能控制未认证；部署掩码为保守实时几何近似',
        '原生历史使用命令执行Tick；旧读取器回退使用观察确认Tick，不混合两种历史来源',
    ]
    def __init__(self,checkpoint,contract,source,device,*,timing_config=None,calibration=None):
        source=Path(source).resolve();root=source.parent;verify_source(root)
        existing=sys.modules.get('hokoff_model')
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(source):
            raise RuntimeError('another hokoff_model source is already loaded')
        sys.path.insert(0,str(source))
        from hokoff_model.match_agent import load_release
        from hokoff_model.online_history import OnlinePlayHistory,PublicPlay
        self.event_type=PublicPlay
        self.model,self.encoder,self.identity=load_release(checkpoint,contract,device=str(device))
        self.device=device;self.history=OnlinePlayHistory(4);self.side=0
        self.timing_gate=TimingGate(timing_config,calibration=calibration,model_digest=self.identity['model_digest'])
        self.stats=dict(decisions=0,gap_resets=0,confirmed_history=0,unknown_manual_history=0)
        self.reset(0)
    def reset(self,side):
        self.side=side;self.history.reset();self.hidden=self.model.initial_hidden(1,device=self.device)
        self.last_tick=None;self.warm_decisions=0
        self.visualization={};self.next_visualization_at=0.
        self.native_history_supported=False;self.native_history_seen=set();self.revealed_enemy={}
        self.native_history_counts=[0,0];self.native_history_source_valid=False
        if hasattr(self,'timing_gate'):self.timing_gate.reset()
    def token(self,card):
        cid=int(card['card_id']);flags=int(card.get('form_flags',0))
        if flags not in (0,1,2):raise ValueError('unknown card form')
        if flags:cid=int(metadata(cid)['evolution_form_id' if flags==1 else 'hero_form_id'])
        if cid not in self.encoder.card_id_to_token:raise ValueError(f'卡牌/形态 {cid} 不在模型词表中')
        return self.encoder.card_id_to_token[cid]
    def record_confirmation(self,tick,card,position):
        if self.native_history_counts[self.side]:return False
        absolute=position if self.side==0 else 575-position
        self.history.record(self.event_type(tick,self.side,self.token(card),absolute,True))
        self.stats['confirmed_history']+=1
        return True
    def record_manual_unknown(self,tick):
        if self.native_history_counts[self.side]:return False
        self.history.record(self.event_type(tick,self.side,0,0,False))
        self.stats['unknown_manual_history']+=1
    def record_unattributed(self,tick):
        if self.native_history_counts[self.side]:return False
        self.history.record(self.event_type(tick,self.side,0,0,False))
        self.stats['unknown_unattributed_history']=self.stats.get('unknown_unattributed_history',0)+1
    def ingest_public_history(self,frame,decks,side):
        if frame.get('public_history_schema')!='native_spend_confirmed_v1':return
        self.native_history_source_valid=bool(frame.get('public_history_source_valid'))
        if not self.native_history_source_valid:return
        for event in sorted(frame.get('public_plays',[]),key=lambda e:(e['tick'],e['side'])):
            key=(event['id'],event['tick'],event['side'])
            if key in self.native_history_seen:continue
            if event['side'] not in (0,1) or not 0<=event['tick']<frame['game_tick']:
                raise ValueError('public play is not strictly past')
            if not 0<=event['x']<18000 or not 0<=event['y']<32000:
                raise ValueError('public play position invalid')
            card=dict(card_id=event['card_id'],form_flags=event['form_flags'])
            if event['side']==side:
                own=next((c for c in decks[side] if c['card_id']==event['card_id']),None)
                if own is not None:card=own
            known=bool(event['known']);token=0
            if known:
                try:token=self.token(card)
                except (ValueError,KeyError,TypeError):known=False
            cell=(event['y']//1000)*18+event['x']//1000
            if not self.native_history_counts[event['side']] and hasattr(self.history,'events'):
                s=event['side']
                self.history.events[s]=[];self.history.last_tick[s]=-1;self.history.last_event[s]=None;self.history.accepted_plays[s]=0
            self.history.record(self.event_type(event['tick'],event['side'],token if known else 0,cell if known else 0,known))
            self.native_history_supported=True
            self.native_history_seen.add(key);self.native_history_counts[event['side']]+=1
            self.stats['native_history_events']=self.stats.get('native_history_events',0)+1
            if known and event['side']!=side:self.revealed_enemy[event['card_id']]=token
    def evaluate(self,frame,decks,side,*,blocked_slots=()):
        tick=int(frame['game_tick'])
        if tick%4 or (self.last_tick is not None and tick<=self.last_tick):return None,{'reason':'not_new_fixed4_tick'}
        if self.side!=side:self.reset(side)
        delta=0 if self.last_tick is None else tick-self.last_tick
        if delta not in (0,4):
            self.hidden=self.model.initial_hidden(1,device=self.device);delta=0;self.warm_decisions=0
            self.stats['gap_resets']+=1
        raw=_native_state(frame,frame)
        if len(raw['entities'])!=frame.get('decoded_entity_count'):raise ValueError('native entity count mismatch')
        state=normalize_native_state(raw)
        encoded=self.encoder.encode_batch([NativeActorFrame(state,side,decks[side],delta_ticks=delta,
                    revealed_enemy_tokens=tuple(self.revealed_enemy.values()))])
        batch=dict(encoded);batch['prev_elapsed_ticks']=batch.pop('delta_ticks')
        batch['frame_ticks']=torch.tensor([[tick]],dtype=torch.long);batch['frame_mask']=torch.ones(1,1,dtype=torch.bool)
        batch.update(self.history.query(side,tick))
        batch={k:v.to(self.device) for k,v in batch.items()}
        with torch.inference_mode():out,h=self.model.forward_stream(batch,self.hidden)
        if not all(bool(torch.isfinite(v).all()) for v in [*out.values(),*h]):raise FloatingPointError('nonfinite live prediction')
        self.hidden=tuple(v.detach().clone() for v in h);self.last_tick=tick;self.warm_decisions+=1;self.stats['decisions']+=1
        logit=float(out['timing'][0,0])
        timing=self.timing_gate.evaluate(logit,tick=tick,elixir_raw=state.players[side].elixir_raw)
        probability=timing['play_probability'];threshold=timing['threshold']
        audit=dict(tick=tick,**timing,encoded_entities=encoded.encoded_entity_counts[0],
                   history_entries=int(batch['history_mask'].sum()),warm_decisions=self.warm_decisions,
                   enemy_history_available=bool(self.native_history_counts[1-side]),
                   history_source='native_spend_confirmed' if self.native_history_supported else 'local_click_receipts',
                   history_counts_by_side=list(self.native_history_counts),history_source_valid=self.native_history_source_valid)
        audit['history_slots_by_relation']=[int(batch['history_mask'][0,0,r].sum()) for r in (0,1)]
        audit['history_known_by_relation']=[int(batch['history_known'][0,0,r].sum()) for r in (0,1)]
        audit['observed_units_by_side']=[sum(e.get('side')==s and e.get('card_id',-1)>=0 and e.get('hp',0)>0 for e in raw['entities']) for s in (0,1)]
        capture=time.monotonic()>=self.next_visualization_at
        if probability<threshold and not capture:return None,{**audit,'reason':'below_timing_threshold'}
        hand=list(state.players[side].hand)
        cards,positions=_position_masks(decks,side,hand,state.players[side].elixir_raw,raw)
        # Do not guess unknown or dynamic card costs; these cards remain unavailable.
        for slot,di in enumerate(hand):
            if di<0:continue
            cid=int(decks[side][di]['card_id']);cost=CARD_COSTS.get(cid)
            if cid==28000006 or cost is None or cost<0:cards[slot]=False;positions[slot]=False
        for slot in blocked_slots:cards[slot]=False;positions[slot]=False
        if capture:
            # One small CPU copy, no extra inference. UI publication is capped at 2 Hz.
            logits=torch.cat((out['card'][0,0].flatten(),out['position'][0,0].flatten())).detach().float().cpu().numpy()
            self.visualization=probability_snapshot(logits[:4],logits[4:].reshape(4,576),cards,positions,
                hand,decks[side],tick=tick,side=side,timing=probability)
            self.visualization.update(timing)
            self.visualization['observed_units_by_side']=audit['observed_units_by_side']
            self.visualization['encoded_entities']=audit['encoded_entities']
            self.visualization['history_counts_by_side']=audit['history_counts_by_side']
            self.visualization['history_source']=audit['history_source']
            self.visualization['history_slots_by_relation']=audit['history_slots_by_relation']
            self.next_visualization_at=time.monotonic()+.5
        if probability<threshold:return None,{**audit,'reason':'below_timing_threshold'}
        if not cards.any():return None,{**audit,'reason':'no_supported_legal_card'}
        slot=int(out['card'][0,0].masked_fill(~torch.as_tensor(cards,device=self.device),-torch.inf).argmax())
        position=int(out['position'][0,0,slot].masked_fill(~torch.as_tensor(positions[slot],device=self.device),-torch.inf).argmax())
        di=hand[slot]
        return dict(slot=slot,deck_index=di,card_id=int(decks[side][di]['card_id']),position=position),audit
