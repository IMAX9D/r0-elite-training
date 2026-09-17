"""Four simultaneous hand-card probability maps; telemetry only, never input."""
import math,time
import tkinter as tk
from tkinter import ttk
from .mumu_live_monitor import _read_json,STATUS_PATH,CARD_ZH
from .mumu_live_protocol import DEFAULT_LOG_ROOT
from .mumu_probabilities import screen_cell

REASONS={
    'below_timing_threshold':'行动概率低于阈值', 'no_supported_legal_card':'没有可用的合法手牌',
    'waiting_receipt':'等待上次下牌回执', 'actions_blocked':'保护停手', 'stale_decision':'观察/决策过期',
    'ui_warming':'入场/记忆热身', 'eligible':'通过控制检查','card_ui_settling':'换牌/点击保护间隔',
    'touch_command_in_flight':'上一组双击命令尚未完成（不等待游戏回执）',
}

class ProbabilityView(tk.Toplevel):
    CELL_W=21
    CELL_H=17

    def __init__(self,parent):
        super().__init__(parent)
        self.title('四张手牌 · 实时落点概率');self.geometry('980x650');self.minsize(480,560)
        self.configure(bg='#0f172a');self.snapshot={};self.signature=None
        self.mode=tk.StringVar(value='合法筛选后')
        self.heading=ttk.Label(self,text='等待实时概率……',style='Sub.TLabel')
        controls=ttk.Frame(self,style='Root.TFrame');controls.pack(fill='x',padx=14)
        selector=ttk.Combobox(controls,textvariable=self.mode,values=['合法筛选后','原始分布'],state='readonly',width=14)
        selector.pack(side='right');selector.bind('<<ComboboxSelected>>',lambda e:self.draw())
        body=ttk.Frame(self,style='Root.TFrame');body.pack(fill='both',expand=True,padx=14,pady=(10,0))
        maps=body
        self.panels=[];self.cell_sizes=[(21,17)]*4;self._layout_job=None
        self.canvases=[];self.cells=[];self.card_labels=[];self.summaries=[]
        for slot in range(4):
            panel=ttk.Frame(maps,style='Root.TFrame');panel.grid(row=0,column=slot,padx=4,sticky='nsew')
            panel.grid_propagate(False);panel.pack_propagate(False);self.panels.append(panel)
            label=ttk.Label(panel,text=f'槽 {slot+1} · 等待手牌',style='Hand.TLabel')
            label.pack(fill='x',pady=(0,6));self.card_labels.append(label)
            canvas=tk.Canvas(panel,width=1,height=1,bg='#0f172a',highlightthickness=0)
            self.canvases.append(canvas);cells=[]
            for row in range(32):
                for col in range(18):
                    x,y=col*self.CELL_W,row*self.CELL_H
                    box=canvas.create_rectangle(x,y,x+self.CELL_W,y+self.CELL_H,fill='#172033',outline='#334155')
                    number=canvas.create_text(x+self.CELL_W/2,y+self.CELL_H/2,text='',fill='white',font=('Consolas',6))
                    cells.append((box,number))
            self.cells.append(cells)
            canvas.bind('<Motion>',lambda e,s=slot:self._hover(e,s))
            canvas.bind('<Leave>',lambda e:self.hover.place_forget())
            info=ttk.Label(panel,text='',style='Sub.TLabel',wraplength=378,justify='left')
            info.pack(side='bottom',fill='x',pady=4);self.summaries.append(info)
            canvas.pack(fill='both',expand=True)
            canvas.bind('<Configure>',lambda e,s=slot:self._resize_map(e,s))
        self.hover=ttk.Label(self,text='鼠标移到任意一张图的格子上查看精确概率。',style='Sub.TLabel')
        self.note=ttk.Label(self,style='Sub.TLabel',justify='left',text=(
            '落点为选定该卡后的条件概率；四图共享色阶。小窗隐藏格内数字，悬停可查。'
            '灰格不可下；行动分数未经校准。'))
        maps.bind('<Configure>',lambda e:self._schedule_layout(maps,e.width))
        self.after(100,self.refresh)

    def _schedule_layout(self,maps,width):
        if self._layout_job is not None:self.after_cancel(self._layout_job)
        self._layout_job=self.after(60,lambda:self._layout(maps,width))

    def _layout(self,maps,width):
        self._layout_job=None
        columns=4 if width>=840 else 2
        rows=4//columns
        for i in range(4):
            maps.columnconfigure(i,weight=int(i<columns),uniform='cards' if i<columns else '')
            maps.rowconfigure(i,weight=int(i<rows),uniform='rows' if i<rows else '')
        for slot,panel in enumerate(self.panels):
            panel.grid(row=slot//columns,column=slot%columns,padx=4,pady=3,sticky='nsew')
            self.card_labels[slot].configure(wraplength=max(80,width//columns-12))
            self.summaries[slot].configure(wraplength=max(80,width//columns-12))
        for label in (self.heading,self.hover,self.note):label.configure(wraplength=max(100,self.winfo_width()-32))

    def _resize_map(self,event,slot):
        size=max(.1,min(event.width/18,event.height/32))
        self.cell_sizes[slot]=(size,size)
        canvas=self.canvases[slot]
        for n,(box,number) in enumerate(self.cells[slot]):
            row,col=divmod(n,18);x,y=col*size,row*size
            canvas.coords(box,x,y,x+size,y+size)
            canvas.coords(number,x+size/2,y+size/2)
            canvas.itemconfigure(number,state='normal' if size>=18 else 'hidden')

    def _hover(self,event,slot):
        cw,ch=self.cell_sizes[slot]
        if not self.snapshot or not 0<=event.x<18*cw or not 0<=event.y<32*ch:return
        index=screen_cell(int(event.x//cw),int(event.y//ch),self.snapshot['side'])
        raw=self.snapshot['position_raw'][slot][index];legal=self.snapshot['position_legal'][slot][index]
        self.hover.configure(text=f'槽 {slot+1} · 模型格 {index}（x={index%18}, y={index//18}） · 原始 {raw:.8%} · 合法筛选后 {legal:.8%}')
        # On-demand tooltip replaces the persistent footer text.
        self.hover.place(x=12,y=max(0,self.winfo_height()-32),width=max(1,self.winfo_width()-24))
        self.hover.lift()

    def refresh(self):
        status=_read_json(STATUS_PATH) or {};payload=_read_json(DEFAULT_LOG_ROOT/'controller-probabilities.json') or {}
        snap=payload.get('snapshot') or {}
        valid=(bool(snap) and status.get('in_battle') and payload.get('in_battle') and
               status.get('checkpoint')==payload.get('checkpoint') and status.get('battle_number')==payload.get('battle') and
               time.time()-payload.get('published_unix',0)<2 and time.time()-status.get('heartbeat_unix',0)<3)
        if not valid:
            if self.snapshot:self.snapshot={};self.signature=None;self.draw()
            self.heading.configure(text='等待实时对局 / 概率数据尚未更新（不展示旧局概率）')
        else:
            audit=status.get('last_model_audit') or {};reason=audit.get('control_reason',audit.get('reason','--'))
            signature=(payload.get('checkpoint'),payload.get('battle'),snap.get('tick'))
            counts=snap.get('observed_units_by_side',[0,0]);side=snap.get('side',0)
            history=snap.get('history_counts_by_side',[0,0])
            self.heading.configure(text=f"step {payload.get('model_step')} · 采样 Tick {snap.get('tick')} · 行动分数 {snap.get('play_probability',0):.2%} / 门槛 {snap.get('threshold',.5):.1%}（{snap.get('timing_mode','fixed')}） · 满费 {snap.get('cap_seconds',0):.1f}s\n单位：己 {counts[side]} / 敌 {counts[1-side]} · 编码 {snap.get('encoded_entities','--')} · 历史核验：己 {history[side]} / 敌 {history[1-side]} · Tick {audit.get('tick','--')}：{REASONS.get(reason,reason)}")
            if signature!=self.signature:self.signature=signature;self.snapshot=snap;self.draw()
        self.after(250,self.refresh)

    def draw(self):
        snap=self.snapshot
        if not snap:
            self.hover.place_forget()
            for slot,canvas in enumerate(self.canvases):
                for box,number in self.cells[slot]:canvas.itemconfigure(box,fill='#172033');canvas.itemconfigure(number,text='')
                self.card_labels[slot].configure(text=f'槽 {slot+1} · 等待手牌');self.summaries[slot].configure(text='')
            self.hover.configure(text='等待本局新数据。');return
        mode='position_legal' if self.mode.get()=='合法筛选后' else 'position_raw'
        peak=max(max(values) for values in snap[mode])
        for slot,canvas in enumerate(self.canvases):
            cid=snap['card_ids'][slot];prefix={1:'觉醒 ',2:'英雄 '}.get(snap['card_forms'][slot],'')
            name=prefix+CARD_ZH.get(cid,str(cid) if cid else '空')
            self.card_labels[slot].configure(text=f"槽 {slot+1} · {name}\n候选：原始 {snap['card_raw'][slot]:.2%} / 合法后 {snap['card_legal'][slot]:.2%}")
            values=snap[mode][slot];best=max(range(576),key=values.__getitem__);local_peak=values[best]
            self.summaries[slot].configure(text=f'{self.mode.get()} · 最大 {local_peak:.4%} · 格 {best}\n概率和 {sum(values):.6f} · 共享色阶上限 {peak:.4%}')
            for n,(box,number) in enumerate(self.cells[slot]):
                row,col=divmod(n,18);idx=screen_cell(col,row,snap['side']);v=values[idx]
                allowed=snap['legal_positions'][slot][idx];q=math.sqrt(v/peak) if peak else 0
                color=f'#{int(20+210*q):02x}{int(30+100*q):02x}{int(50+40*q):02x}'
                txt=f'{v*100:.1f}' if v>=.001 else f'{v*100:.2f}'
                if not allowed and mode=='position_legal':color='#30343d';txt='×'
                canvas.itemconfigure(box,fill=color,outline='#f8fafc' if idx==best and local_peak else '#334155')
                canvas.itemconfigure(number,text=txt)
