"""Automatic fixed4 expert takeover on an EXISTING MuMu device; never starts matches."""
from __future__ import annotations
import argparse,collections,json,os,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from . import mumu_live_controller as legacy
from .mumu_live_protocol import start_reader,stop_owned_reader
from .mumu_live_actions import UI_READY_MIN_TICK,send_card_taps
from .mumu_hokoff_policy import LivePolicy,MODEL_ROOT
from .mumu_model_catalog import MODELS,model_paths
from .mumu_timing_gate import TimingConfig
from .mumu_receipt_tracker import ReceiptTracker
from .mumu_live_actions import own_player
from .gui import CARD_COSTS
from .mumu_passive_receipts import PassiveReceipts
from .mumu_battle_view import ScenePublisher

STOP_PATH=legacy.LOG_ROOT/'stop.request'

class PreflightCancelled(RuntimeError):
    """Raised only before any Android touch command has been issued."""

class BufferedStream(legacy.JsonLineStream):
    def __init__(self,process,event):
        self.frames=collections.deque(maxlen=256);self.counter=0
        super().__init__(process,event)
    def _read(self):
        for line in self.process.stdout:
            try:value=json.loads(line)
            except ValueError:continue
            if value.get('event')!=self.event:continue
            with self._lock:
                self.latest=value;self.updated=time.monotonic();self.counter+=1
                self.frames.append((self.counter,value,self.updated))
    def drain(self,last):
        with self._lock:
            lost=bool(self.frames and last and last<self.frames[0][0]-1)
            return [row for row in self.frames if row[0]>last],lost

class Controller(legacy.MuMuExpertController):
    def _load_controller_model(self,args,device):
        self.adapter=LivePolicy(args.checkpoint,args.encoder_contract,args.model_source,device,
            timing_config=TimingConfig(mode=args.timing_mode,base=args.timing_base,floor=args.timing_floor),
            calibration=args.timing_calibration)
        meta={**self.adapter.identity,'card_id_to_token':self.adapter.encoder.card_id_to_token,
              'ability_id_to_token':self.adapter.encoder.ability_id_to_token}
        return self.adapter.model,meta
    def __init__(self,args):
        super().__init__(args)
        self.touch_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='mumu-touch')
        self.touch_future=None;self.last_hand=None;self.last_model_audit={};self.stop_requested=False
        self.touch_request=None
        self.read_sequence=0;self.last_live_frame=None;self.read_frames=0
        self.published_visualization=None
        self.scene_publisher=None
        self.slot_ready_ticks=[0]*4;self.next_touch_tick=0;self.last_receipt={};self.receipt_owns_block=False
        self.passive_receipts=PassiveReceipts();self.click_sequence=0
        self.event_counts.update(play_observed=0,receipt_ambiguous=0)
    def _status_payload(self):
        value=super()._status_payload()
        value.update(policy_backend='hokoff-fixed4',decision_period_ticks=4,
                     policy_stats=dict(self.adapter.stats),last_model_audit=self.last_model_audit,
                     compatibility_warnings=self.adapter.warnings,control_scope='manual_start_live_match_only',
                     auto_matchmaking=False,read_frames=self.read_frames,stop_request_path=str(STOP_PATH))
        if value.get('tick') is None and self.clock_guard.tick>=0:value['tick']=self.clock_guard.tick
        value.update(timing_config=vars(self.adapter.timing_gate.config),last_receipt=self.last_receipt,
                     receipt_pending=self.pending is not None,receipt_mode=self.args.receipt_mode,
                     passive_pending_clicks=len(self.passive_receipts.records))
        return value
    def _start_streams(self,pid):
        if self.runtime_evidence.get('resource_version')!='16.402.10' or self.runtime_evidence.get('resource_fingerprint')!='da8b002364d0392fd150a9cfc607c1637201a441':
            raise RuntimeError('资源指纹未核对；本入口限定已检查的16.402.10快照')
        self.entity_stream=BufferedStream(start_reader(self.adb,self.serial,pid,interval_ms=20),'mumu_live_frame')
        self.private_stream=self.entity_stream
    def _stable_active(self,entity,private):
        # Warm the recurrent policy before UI-ready, but never touch before Tick150.
        self.observation_health=self.clock_guard.observe(entity)
        return bool(entity and self.observation_health.get('can_control') and self._visible_local_side(entity) is not None
                    and 0<=entity.get('game_tick',-1)<legacy.TERMINAL_GAME_TICK)
    def _reset_policy(self):
        super()._reset_policy();self.adapter.reset(self.local_side);self.last_hand=None;self.last_model_audit={}
        self.slot_ready_ticks=[0]*4;self.next_touch_tick=0;self.last_receipt={};self.receipt_owns_block=False
        if hasattr(self,'passive_receipts'):self.passive_receipts.reset()
    def _bind_visible_deck(self,player):
        good=super()._bind_visible_deck(player)
        if good:
            for card in self.decks[self.local_side]:self.adapter.token(card)
        return good
    def _send_async(self,pending):
        frame,_=self.entity_stream.snapshot()
        identity=lambda f:(f.get('pid'),(f.get('chain') or {}).get('battle'))
        if self.stop_requested or not self.in_battle or self.battle_epoch!=pending['epoch'] or not frame or identity(frame)!=identity(pending['before_frame']):
            raise PreflightCancelled('touch cancelled: battle or controller changed')
        if frame['game_tick']-pending['tick']>4:raise PreflightCancelled('touch cancelled: stale decision')
        player=own_player(frame,pending['side']);before=own_player(pending['before_frame'],pending['side'])
        cost=CARD_COSTS.get(pending['card_id'])
        if (not frame.get('coherent') or player is None or before is None or
            player.get('hand_deck_indices')!=before.get('hand_deck_indices') or
            cost is None or player['elixir_raw']<cost*10000):
            raise PreflightCancelled('touch cancelled: hand/elixir changed before input')
        result=send_card_taps(self.adb,self.serial,self.layout,pending['slot'],pending['position'],side=pending['side'])
        return {**result,'completed_monotonic':time.monotonic()}

    def _receipt_diagnostic(self,frame,evidence):
        pending=self.pending
        path=legacy.LOG_ROOT/f'receipt-b{self.battle_number}-t{pending["tick"]}-{evidence["phase"]}.json'
        payload=dict(model_step=self.model_meta.get('training_step'),request={k:pending[k] for k in ('tick','slot','deck_index','card_id','position','side')},
                     before_frame=pending['before_frame'],observed_frame=frame,evidence=evidence)
        try:path.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8');return str(path)
        except OSError:return 'diagnostic_write_failed'

    def _reconcile_pending(self,frame):
        pending=self.pending
        if pending is None:return
        tracker=pending['tracker'];evidence=tracker.observe(frame,time.monotonic());self.last_receipt=evidence
        if evidence['phase']=='confirmed':
            self.log('touch_accepted',battle=self.battle_number,tick=pending['tick'],slot=pending['slot'],
                     latency_ms=evidence['elapsed_ms'],receipt=evidence)
            self.adapter.record_confirmation(evidence['first_rotation_tick'],self.decks[self.local_side][pending['deck_index']],pending['position'])
            self.slot_ready_ticks[pending['slot']]=frame['game_tick']+8
            self.next_touch_tick=frame['game_tick']+4
            self.pending=None
            if self.receipt_owns_block:
                self.actions_blocked=False;self.receipt_owns_block=False
                self.log('receipt_recovered',message='迟到回执已核实，恢复新的模型决策；没有补发旧点击')
        elif evidence['phase']=='grace' and not tracker.soft_reported:
            tracker.soft_reported=True
            self.log('receipt_grace',battle=self.battle_number,tick=pending['tick'],receipt=evidence,
                     diagnostic=self._receipt_diagnostic(frame,evidence))
        elif evidence['phase'] in ('ambiguous','review') and not tracker.review_reported:
            tracker.review_reported=True
            self.receipt_owns_block=not self.actions_blocked
            self.actions_blocked=True
            self.log('touch_not_confirmed',battle=self.battle_number,tick=pending['tick'],slot=pending['slot'],
                     receipt=evidence,diagnostic=self._receipt_diagnostic(frame,evidence))
    def _step_frame(self,frame,arrived):
        if frame.get('public_history_schema')=='native_spend_confirmed_v1':
            if not self._bind_visible_deck(own_player(frame,self.local_side)):raise ValueError('历史对应卡组未解析')
            self.adapter.ingest_public_history(frame,self.decks,self.local_side)
        if self.args.receipt_mode=='observe':return self._step_without_receipt_gate(frame,arrived)
        tick=int(frame['game_tick']);side=self.local_side
        player=next(p for p in frame['players'] if p['side']==side)
        if not self._bind_visible_deck(player):raise ValueError('当前完整卡组尚未可靠识别')
        hand=[int(x) for x in player['hand_deck_indices']]
        self.live_player=dict(elixir_raw=player['elixir_raw'],elixir=round(player['elixir_raw']/10000.,2),
                              hand_deck_indices=hand,next_deck_index=player['next_deck_index'])
        if self.touch_future is not None and self.touch_future.done():
            future=self.touch_future;error=future.exception();self.touch_future=None
            if isinstance(error,PreflightCancelled):
                self.pending=None;self.next_touch_tick=tick+4
                self.log('touch_cancelled',message=str(error))
            elif error:
                if self.pending:self.pending['tracker'].arm(time.monotonic())
                self.actions_blocked=True;self.receipt_owns_block=False
                self.log('inference_error',error=str(error))
            elif self.touch_request:
                sent=self.touch_request
                if self.pending:self.pending['tracker'].arm(future.result()['completed_monotonic'])
                if self.last_action:self.last_action['sent']=True
                self.log('touch_sent',battle=sent['battle'],**{k:sent[k] for k in ('tick','slot','deck_index','card_id','position')})
            self.touch_request=None
        was_pending=self.pending is not None
        if self.pending:self._reconcile_pending(frame)
        if self.last_hand is not None and hand!=self.last_hand and not was_pending:
            departed=set(self.last_hand)-set(hand)
            if len(departed)==1:self.adapter.record_manual_unknown(tick)
            elif departed:raise ValueError('手牌发生多次未观察变化，停止本局触屏')
        if self.last_hand is not None:
            for slot in range(4):
                if hand[slot]!=self.last_hand[slot]:self.slot_ready_ticks[slot]=tick+8
        self.last_hand=hand;self.last_tick=tick
        self.lifecycle='blocked' if self.actions_blocked else 'warming' if tick<UI_READY_MIN_TICK or self.adapter.warm_decisions<4 else 'controlling'
        action,audit=self.adapter.evaluate(frame,self.decks,side,blocked_slots=[s for s in range(4) if tick<self.slot_ready_ticks[s]])
        if audit.get('reason')!='not_new_fixed4_tick':self.last_model_audit=audit
        if action is None:
            if audit.get('reason')!='not_new_fixed4_tick':
                self.last_model_audit['control_reason']='actions_blocked' if self.actions_blocked else 'waiting_receipt' if self.pending else audit.get('reason')
            return
        latest,_=self.entity_stream.snapshot()
        fresh=latest and tick>=latest['game_tick']-1 and time.monotonic()-arrived<.15
        gate=('actions_blocked' if self.actions_blocked else 'waiting_receipt' if self.pending is not None or self.touch_future is not None
              else 'stale_decision' if not fresh else 'card_ui_settling' if tick<self.next_touch_tick
              else 'ui_warming' if tick<UI_READY_MIN_TICK or self.adapter.warm_decisions<4 else 'eligible')
        self.last_model_audit['control_reason']=gate
        if gate!='eligible':return
        if self.args.dry_run:
            self.last_action={**action,'tick':tick,'play_probability':audit['play_probability'],'sent':False}
            # Only publish the latest candidate; no pretend acknowledgement/history.
            return
        self.pending={**action,'tick':tick,'side':side,'epoch':self.battle_epoch,'before_frame':frame,
                      'sent_at':time.monotonic(),'retried':False,
                      'tracker':ReceiptTracker(frame,side=side,slot=action['slot'],card_id=action['card_id'])}
        self.touch_request={**action,'tick':tick,'battle':self.battle_number}
        self.touch_future=self.touch_pool.submit(self._send_async,dict(self.pending))
        self.last_action={**action,'tick':tick,'play_probability':audit['play_probability'],'sent':False}
        self.log('touch_scheduled',battle=self.battle_number,tick=tick,**action,play_probability=audit['play_probability'],
                 threshold=audit['threshold'],timing_mode=audit['timing_mode'],raw_play_probability=audit['raw_play_probability'])

    def _step_without_receipt_gate(self,frame,arrived):
        tick=int(frame['game_tick']);side=self.local_side
        player=own_player(frame,side)
        if not self._bind_visible_deck(player):raise ValueError('当前完整卡组尚未可靠识别')
        self.live_player=dict(elixir_raw=player['elixir_raw'],elixir=round(player['elixir_raw']/10000.,2),
                              hand_deck_indices=player['hand_deck_indices'],next_deck_index=player['next_deck_index'])
        if self.touch_future is not None and self.touch_future.done():
            future=self.touch_future;request=self.touch_request;error=future.exception()
            self.touch_future=None;self.touch_request=None
            if isinstance(error,PreflightCancelled):
                self.passive_receipts.cancel(request['id']);self.log('touch_cancelled',message=str(error))
            elif error:
                # Transport errors are not confirmation timeouts: input ordering
                # and even whether the second tap ran are unknown.
                self.actions_blocked=True;self.log('inference_error',error=str(error))
            else:
                self.passive_receipts.sent(request['id'],future.result()['completed_monotonic'])
                if self.last_action:self.last_action['sent']=True
                self.log('touch_sent',battle=request['battle'],request_id=request['id'],
                         **{k:request[k] for k in ('tick','slot','deck_index','card_id','position')})
        for receipt in self.passive_receipts.observe(frame,side,time.monotonic()):
            self.last_receipt={k:v for k,v in receipt.items() if k not in ('request','history_action')}
            self.log(receipt['event'],battle=self.battle_number,**{k:v for k,v in self.last_receipt.items() if k!='event'})
            if receipt['event']=='play_observed':
                request=receipt['request']
                history_action=receipt.get('history_action')
                if history_action:
                    if self.adapter.record_confirmation(receipt['tick'],history_action['card'],history_action['position']):
                        self.log('history_recorded',battle=self.battle_number,tick=receipt['tick'],
                                 source=receipt['attribution'],**history_action)
                if request:
                    self.log('touch_accepted',battle=self.battle_number,tick=request['tick'],slot=request['slot'],
                             request_id=request['id'],receipt=dict(reason='passive_unique_rotation',tick_after=receipt['tick']))
                elif not history_action:
                    self.adapter.record_unattributed(receipt['tick'])
                    self.log('receipt_ambiguous',battle=self.battle_number,tick=receipt['tick'],message='观察到轮转但不能归因到唯一点击；位置历史标为未知，不停手')
        self.last_tick=tick;self.last_hand=list(player['hand_deck_indices'])
        self.lifecycle='blocked' if self.actions_blocked else 'warming' if tick<UI_READY_MIN_TICK or self.adapter.warm_decisions<4 else 'controlling'
        # No pending-receipt gate and no post-confirmation or slot cooldown masks.
        action,audit=self.adapter.evaluate(frame,self.decks,side)
        if audit.get('reason')=='not_new_fixed4_tick':return
        self.last_model_audit=audit
        latest,_=self.entity_stream.snapshot()
        fresh=latest and tick>=latest['game_tick']-1 and time.monotonic()-arrived<.15
        gate=('actions_blocked' if self.actions_blocked else 'touch_command_in_flight' if self.touch_future is not None
              else 'stale_decision' if not fresh else 'ui_warming' if tick<UI_READY_MIN_TICK or self.adapter.warm_decisions<4
              else audit.get('reason','no_action') if action is None else 'eligible')
        audit['control_reason']=gate
        if action is None or gate!='eligible':return
        if self.args.dry_run:
            self.last_action={**action,'tick':tick,'play_probability':audit['play_probability'],'sent':False};return
        self.click_sequence+=1
        request={**action,'id':self.click_sequence,'tick':tick,'side':side,'epoch':self.battle_epoch,
                 'battle':self.battle_number,'before_frame':frame,'submitted_at':time.monotonic(),
                 'expected_next':player['next_deck_index'],'card':dict(self.decks[side][action['deck_index']])}
        self.passive_receipts.add(request);self.touch_request=request
        self.touch_future=self.touch_pool.submit(self._send_async,request)
        self.last_action={**action,'tick':tick,'play_probability':audit['play_probability'],'sent':False}
        self.log('touch_scheduled',battle=self.battle_number,request_id=request['id'],tick=tick,**action,
                 play_probability=audit['play_probability'],threshold=audit['threshold'],receipt_mode='observe')
    def _publish_probabilities(self):
        snapshot=self.adapter.visualization
        key=(self.battle_number,snapshot.get('tick'),self.in_battle)
        if key==self.published_visualization:return
        payload=dict(snapshot=snapshot if self.in_battle else {},battle=self.battle_number,
                     checkpoint=str(self.args.checkpoint),model_step=self.model_meta.get('training_step'),
                     published_unix=time.time(),in_battle=self.in_battle)
        path=legacy.LOG_ROOT/'controller-probabilities.json';temp=path.with_suffix('.tmp')
        try:
            temp.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
            os.replace(temp,path);self.published_visualization=key
        except OSError:
            pass  # A viewer must never interrupt game control; retry next iteration.
    def run(self):
        self.prepare();self.log('compatibility_notice',message='；'.join(self.adapter.warnings))
        self.scene_publisher=ScenePublisher(self.entity_stream,lambda:self.local_side)
        started=time.monotonic()
        try:
            while not STOP_PATH.exists() and (not self.args.seconds or time.monotonic()-started<self.args.seconds):
                if self.entity_stream.process.poll() is not None:raise RuntimeError('只读采样器退出；不会自动启动另一个设备')
                rows,lost=self.entity_stream.drain(self.read_sequence)
                if lost:
                    self.adapter.reset(self.local_side);self.log('memory_reset',message='采样缓存溢出，重置策略记忆并重新热身')
                for seq,frame,arrived in rows:
                    self.read_sequence=seq;self.read_frames+=1
                    active=self._stable_active(frame,frame)
                    if self.in_battle and self.observation_health.get('epoch')!=self.battle_epoch:self._end_battle()
                    if self.in_battle and self.observation_health.get('local_side')!=self.local_side:self._end_battle()
                    if not active:
                        if self.in_battle and self.observation_health.get('status') in ('unresolved','terminal_or_unverified_towers','both_hands_visible_readonly','paused_or_stalled'):self._end_battle()
                        continue
                    if not self.in_battle:self._begin_battle(frame['game_tick'],frame)
                    self.last_live_frame=frame
                    try:self._step_frame(frame,arrived)
                    except Exception as e:
                        if not self.actions_blocked:self.log('inference_error',error=f'{type(e).__name__}: {e}')
                        self.actions_blocked=True;self.lifecycle='blocked'
                latest,stamp=self.entity_stream.snapshot()
                if time.monotonic()-stamp>1.:
                    self.observation_health={'can_control':False,'status':'stale','epoch':self.battle_epoch}
                    if self.in_battle:self._end_battle()
                self._publish_status();self._publish_probabilities();time.sleep(.005)
        except BaseException as e:
            self.lifecycle='error';self.last_message=str(e);self._publish_status(force=True);raise
        finally:
            self.stop_requested=True;self.in_battle=False
            if self.scene_publisher:self.scene_publisher.close()
            if self.touch_future:self.touch_future.cancel()
            self.touch_pool.shutdown(wait=True,cancel_futures=True)
            latest,_=self.entity_stream.snapshot()
            if latest:stop_owned_reader(self.adb,self.serial,latest.get('reader_pid'),self.game_pid)
            self.entity_stream.stop();self.lifecycle='stopped';self.last_message='接管已停止，MuMu与游戏保留';self._publish_status(force=True)

def parser():
    p=legacy.build_parser();p.description=__doc__
    p.set_defaults(checkpoint=MODEL_ROOT/'inference.pt',play_rate_scale=1.,choice_mode='greedy-placement')
    p.add_argument('--encoder-contract',type=Path,default=MODEL_ROOT/'encoder-contract.json')
    p.add_argument('--model-source',type=Path,default=MODEL_ROOT/'runtime')
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--model-id',choices=tuple(MODELS),help='选择已经校验的模型包')
    p.add_argument('--timing-mode',choices=('fixed','dynamic'),default='dynamic')
    p.add_argument('--timing-base',type=float,default=.5)
    p.add_argument('--timing-floor',type=float,default=.25)
    p.add_argument('--timing-calibration',type=Path,default=None)
    p.add_argument('--receipt-mode',choices=('observe','guarded'),default='observe',help='observe仅记录回执，不等待确认或施加换牌冷却')
    return p
def main():
    args=parser().parse_args()
    if args.model_id:
        args.checkpoint,args.encoder_contract,args.model_source=model_paths(args.model_id)
    if args.play_rate_scale!=1:raise SystemExit('新fixed4模型不沿用旧行动率倍率；请使用1.0')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Refuse a second known controller targeting the same existing Android device.
    import psutil
    own_tree={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(['name']):
        try:
            if process.pid in own_tree or not (process.info['name'] or '').lower().startswith('python'):continue
            cmd=process.cmdline()
            if not any(m in cmd for m in ('native_core.mumu_live_controller','native_core.mumu_hokoff_controller')):continue
            serial=cmd[cmd.index('--serial')+1] if '--serial' in cmd else legacy.DEFAULT_SERIAL
            if serial==args.serial:raise RuntimeError('该设备已有接管器运行，请先在监控中停止旧入口')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    legacy.LOG_ROOT.mkdir(parents=True,exist_ok=True)
    lock=(legacy.LOG_ROOT/'controller.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            if lock.tell()==0:lock.write(b'0');lock.flush()
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if STOP_PATH.exists():STOP_PATH.unlink()
        Controller(args).run()
    finally:lock.close()
if __name__=='__main__':main()
