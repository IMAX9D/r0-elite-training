"""The familiar MuMu monitor, with a fixed4 launcher and cooperative stop."""
from pathlib import Path
import os,sys,subprocess,time,json
from tkinter import messagebox,ttk,StringVar
from . import mumu_live_monitor as base
from .mumu_live_protocol import DEFAULT_LOG_ROOT
from .mumu_model_catalog import available_models,identify_checkpoint,DEFAULT_MODEL
from .mumu_probability_view import ProbabilityView
from .mumu_battle_view import BattleView

class Monitor(base.Monitor):
    COLORS={**base.Monitor.COLORS,'warming':('观察热身 · 暂不触屏','#eab308'),
            'blocked':('已停手 · 检查输入/回执','#ef4444')}
    def __init__(self):
        super().__init__();self.title('HOKOFF · 模型选择与实时接管');self.geometry('1050x750')
        self.protocol('WM_DELETE_WINDOW',self._close)
        self.after(500,self._write_ready)
        self.after(700,self._show_probabilities)
        self.after(850,self._show_battle)
        self.after(1000,self._recording_layout)
    def _build(self):
        super()._build()
        self.choices=available_models();self.probability_window=None;self.battle_window=None;self.starting_until=0.
        current=base._read_json(base.STATUS_PATH) or {}
        saved=base._read_json(DEFAULT_LOG_ROOT/'model-selection.json') or {}
        selected=identify_checkpoint(current.get('checkpoint')) or saved.get('model_id') or DEFAULT_MODEL
        self.model_choice=StringVar(value=self.choices.get(selected,''))
        root=self.detail.master
        toolbar=ttk.Frame(root,style='Root.TFrame')
        toolbar.pack(fill='x',pady=(0,12),before=root.winfo_children()[2])
        ttk.Label(toolbar,text='下次启动使用：',style='Sub.TLabel').pack(side='left')
        self.model_selector=ttk.Combobox(toolbar,textvariable=self.model_choice,values=list(self.choices.values()),width=31,state='readonly')
        self.model_selector.pack(side='left',padx=6)
        ttk.Button(toolbar,text='实时概率 / 落点热图',command=self._show_probabilities).pack(side='right')
        ttk.Button(toolbar,text='实时战场',command=self._show_battle).pack(side='right',padx=4)
        ttk.Label(toolbar,text='换模型：停止接管 → 选择 → 启动',style='Sub.TLabel').pack(side='left',padx=8)
        timingbar=ttk.Frame(root,style='Root.TFrame');timingbar.pack(fill='x',pady=(0,8),after=toolbar)
        self.timing_mode=StringVar(value='动态 0.25–0.5' if current.get('timing_config',{}).get('mode','dynamic')=='dynamic' else '固定 0.5')
        ttk.Label(timingbar,text='下次启动的时机方案：',style='Sub.TLabel').pack(side='left')
        self.timing_selector=ttk.Combobox(timingbar,textvariable=self.timing_mode,values=['动态 0.25–0.5','固定 0.5'],state='readonly',width=20)
        self.timing_selector.pack(side='left',padx=6)
        self.timing_label=ttk.Label(timingbar,text='动态门槛为试验配置，尚未做概率校准。',style='Sub.TLabel')
        self.timing_label.pack(side='left',padx=6)
        ttk.Button(timingbar,text='录制布局',command=self._recording_layout).pack(side='right')
    def _recording_layout(self):
        self._show_probabilities();self._show_battle()
        width,height=self.winfo_screenwidth(),self.winfo_screenheight()
        # Reserve the rightmost 29% for the user's existing portrait MuMu.
        heat_width=round(width*.415);battle_width=round(width*.295)
        content_height=max(560,height-82)
        for window,w,x in ((self.probability_window,heat_width,0),
                           (self.battle_window,battle_width,heat_width)):
            window.state('normal')
            window.geometry(f'{w-12}x{content_height}+{x}+0')
            window.lift()
    def _show_battle(self):
        if self.battle_window is not None and self.battle_window.winfo_exists():
            self.battle_window.lift();return
        self.battle_window=BattleView(self)
    def _show_probabilities(self):
        if self.probability_window is not None and self.probability_window.winfo_exists():
            self.probability_window.lift();return
        self.probability_window=ProbabilityView(self)
        self.after(150,self._write_ready)
    def _write_ready(self):
        import json
        DEFAULT_LOG_ROOT.mkdir(parents=True,exist_ok=True)
        (DEFAULT_LOG_ROOT/'monitor-status.json').write_text(json.dumps(dict(pid=os.getpid(),title=self.title(),
            mapped=bool(self.winfo_ismapped()),viewable=bool(self.winfo_viewable()),geometry=self.geometry(),
            probability_maps=len(self.probability_window.canvases) if self.probability_window and self.probability_window.winfo_exists() else 0,
            probability_viewable=bool(self.probability_window and self.probability_window.winfo_exists() and self.probability_window.winfo_viewable()))),encoding='utf-8')
    def _start_controller(self):
        current=base._read_json(base.STATUS_PATH) or {}
        if base._pid_alive(int(current.get('controller_pid') or 0)) or time.monotonic()<self.starting_until:
            messagebox.showinfo('先停止接管','请先停止当前接管器，等待退出后再选择模型。');return
        model_id=next((k for k,v in self.choices.items() if v==self.model_choice.get()),None)
        if model_id is None:messagebox.showerror('模型不可用','请选择已经校验的模型包。');return
        folder=DEFAULT_LOG_ROOT/'launcher';folder.mkdir(parents=True,exist_ok=True)
        stamp=str(int(time.time()))
        with (folder/f'hokoff-{stamp}.out.log').open('w',encoding='utf-8') as out,(folder/f'hokoff-{stamp}.err.log').open('w',encoding='utf-8') as err:
            mode='dynamic' if self.timing_mode.get().startswith('动态') else 'fixed'
            subprocess.Popen([sys.executable,'-m','native_core.mumu_hokoff_controller','--model-id',model_id,'--timing-mode',mode],cwd=base.PROJECT_ROOT,
                             stdout=out,stderr=err,creationflags=base.NO_WINDOW,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        (DEFAULT_LOG_ROOT/'model-selection.json').write_text(json.dumps(dict(model_id=model_id)),encoding='utf-8')
        self.starting_until=time.monotonic()+10
        self.detail.configure(text='正在加载所选模型并重置记忆；不自动匹配对局。')
    def _stop_controller(self):
        DEFAULT_LOG_ROOT.mkdir(parents=True,exist_ok=True)
        (DEFAULT_LOG_ROOT/'stop.request').write_text('user requested stop\n',encoding='utf-8')
        self.detail.configure(text='正在停止接管；不关闭游戏，不再提交新动作。')
    def _close(self):
        if self.status and base._pid_alive(int(self.status.get('controller_pid') or 0)):
            if not messagebox.askyesno('关闭监控','同时停止 AI 接管并关闭监控？游戏和模拟器保持打开。'):return
            self._stop_controller()
        self.destroy()
    def _refresh(self):
        super()._refresh()
        alive=bool(self.status and base._pid_alive(int(self.status.get('controller_pid') or 0)))
        busy=alive or time.monotonic()<self.starting_until
        self.model_selector.configure(state='disabled' if busy else 'readonly')
        self.timing_selector.configure(state='disabled' if busy else 'readonly')
        if busy:self.start_button.configure(state='disabled')
        if self.status and self.status.get('policy_backend')=='hokoff-fixed4':
            health=self.status.get('observation_health',{}).get('status','--')
            stats=self.status.get('policy_stats',{})
            audit=self.status.get('last_model_audit') or {};threshold=audit.get('threshold')
            self.timing_label.configure(text=f"当前门槛 {threshold:.1%} · 满费持续 {audit.get('cap_seconds',0):.1f}s · {'已校准' if audit.get('calibrated') else '未校准分数'}" if threshold is not None else '等待本局时机数据；动态方案为待验证试验配置。')
            self.detail.configure(text=f"HOKOFF step {self.status.get('model_step')} · 每4Tick判断 · 推理 {stats.get('decisions',0)} 次 · 观察状态 {health}\n仅普通下牌；敌方精确出牌历史未知。请手动进入允许测试的对局，不会自动匹配。")
            if self.status.get('receipt_mode')=='observe':
                counts=self.status.get('event_counts',{})
                self.metric_values['actions'].master.winfo_children()[0].configure(text='点击 / 观察到轮转')
                self.metric_values['actions'].configure(text=f"{counts.get('touch_sent',0)} / {counts.get('play_observed',0)}")
                history=audit.get('history_counts_by_side',[0,0]);side=int(self.status.get('local_side') or 0)
                slots=audit.get('history_slots_by_relation',[0,0]);known=audit.get('history_known_by_relation',[0,0])
                self.detail.configure(text=f"HOKOFF step {self.status.get('model_step')} · 每4Tick判断 · 回执只记录，不阻塞下牌\n模型历史槽：己 {known[0]}/{slots[0]}、敌 {known[1]}/{slots[1]}（已知/存在） · 原生事件 {history[side]}/{history[1-side]} · {health}")
            if self.status.get('last_message') and self.status.get('actions_blocked'):
                msg=self.status['last_message'];self.action_label.configure(text=base.EVENT_ZH.get(msg,msg),wraplength=850)

def main():
    base.EVENT_ZH.update(touch_scheduled='准备下牌',compatibility_notice='适配范围说明',memory_reset='重新热身',
                         receipt_grace='回执延长核验',receipt_recovered='迟到回执已核实',touch_cancelled='点击前取消（未触屏）',
                         play_observed='观察到手牌轮转',receipt_ambiguous='点击归因不明（不停手）')
    Monitor().mainloop()
if __name__=='__main__':main()
