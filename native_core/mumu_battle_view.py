"""Responsive registry battle canvas, fed by the existing MuMu reader only.

Adapts the unit/name/HP display of cr_re/src/cr_live/gui.py to the current
MuMu camera and telemetry contract; never starts a reader or sends input.
"""
import os,time,threading,json,math
import tkinter as tk
from tkinter import ttk
from .mumu_live_protocol import DEFAULT_LOG_ROOT,visible_sides
from .mumu_live_monitor import _read_json,STATUS_PATH,CARD_ZH
from .card_catalog import catalog,observed_card

SCENE_PATH=DEFAULT_LOG_ROOT/'controller-scene.json'

def scene_payload(frame,*,controller_pid,published_unix,local_side=None):
    fields=('category','card_id','side','x','y','hp','max_hp','level')
    sides=visible_sides(frame)
    return dict(schema='mumu-scene-v1',controller_pid=controller_pid,published_unix=published_unix,
        battle=str((frame.get('chain') or {}).get('battle','')),game_pid=frame.get('pid'),
        tick=frame.get('game_tick'),coherent=bool(frame.get('coherent')),
        active=bool(frame.get('battle_active')),side=sides[0] if len(sides)==1 else local_side,
        entities=[{key:e.get(key) for key in fields} for e in frame.get('entities',[])])

class ScenePublisher:
    """Bounded latest-frame telemetry on a separate thread, at most 20 Hz."""
    def __init__(self,stream,side_getter,path=SCENE_PATH):
        self.stream=stream;self.side_getter=side_getter;self.path=path
        self.stopping=threading.Event()
        self.thread=threading.Thread(target=self._run,name='mumu-scene-view',daemon=True)
        self.thread.start()
    def _run(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_name(self.path.name+f'.{os.getpid()}.tmp')
        while not self.stopping.wait(.05):
            frame,arrived=self.stream.snapshot()
            if not frame or time.monotonic()-arrived>1:continue
            try:
                payload=scene_payload(frame,controller_pid=os.getpid(),published_unix=time.time(),local_side=self.side_getter())
                temp.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
                os.replace(temp,self.path)
            except (OSError,ValueError,TypeError):
                continue  # A viewer cannot stop the controller.
    def close(self):
        self.stopping.set();self.thread.join(timeout=1)

def camera_xy(x,y,side):
    # Current MuMu keeps world X; own side 0 is at low world Y.
    return x/18000,(1-y/32000) if side==0 else y/32000

def valid_scene(payload,status,now):
    return bool(payload and status and payload.get('schema')=='mumu-scene-v1'
        and payload.get('controller_pid')==status.get('controller_pid')
        and 0<=now-payload.get('published_unix',0)<1
        and 0<=now-status.get('heartbeat_unix',0)<3
        and status.get('lifecycle') not in ('stopped','error')
        and payload.get('active') and payload.get('coherent'))

class BattleView(tk.Toplevel):
    def __init__(self,parent):
        super().__init__(parent);self.title('MuMu · 实时战场重建');self.geometry('460x760');self.minsize(300,420)
        self.configure(bg='#0f172a');self.scene=None;self.marker=None;self.changed=time.monotonic();self.side=0
        self.status=ttk.Label(self,text='等待实时场景',style='Sub.TLabel');self.status.pack(fill='x',padx=10,pady=6)
        self.canvas=tk.Canvas(self,bg='#101a29',highlightthickness=0);self.canvas.pack(fill='both',expand=True,padx=8,pady=(0,8))
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.job=self.after(50,self.refresh);self.protocol('WM_DELETE_WINDOW',self.close)
    def close(self):
        self.after_cancel(self.job);self.destroy()
    def refresh(self):
        payload=_read_json(SCENE_PATH);status=_read_json(STATUS_PATH)
        if valid_scene(payload,status,time.time()):
            marker=(payload.get('game_pid'),payload.get('battle'),payload.get('tick'))
            if marker!=self.marker:self.changed=time.monotonic()
            self.marker=marker;self.scene=payload
            if payload.get('side') in (0,1):self.side=payload['side']
            self.draw()
        elif self.scene is not None or self.marker is None:
            self.scene=None;self.marker=None;self.draw()
        self.job=self.after(50,self.refresh)
    def draw(self):
        c=self.canvas;c.delete('all')
        scale=max(1,min(c.winfo_width()/18,c.winfo_height()/32))
        w,h=18*scale,32*scale;left=(c.winfo_width()-w)/2;top=(c.winfo_height()-h)/2
        def xy(x,y):
            a,b=camera_xy(x,y,self.side);return left+(1-a)*w,top+b*h
        c.create_rectangle(left,top,left+w,top+h,fill='#172331',outline='#334155')
        for col in range(19):c.create_line(left+col*scale,top,left+col*scale,top+h,fill='#243447')
        for row in range(33):c.create_line(left,top+row*scale,left+w,top+row*scale,fill='#243447')
        c.create_rectangle(left,top+15*scale,left+w,top+17*scale,fill='#173e59',outline='')
        if not self.scene:
            self.status.configure(text='等待实时场景');return
        units=towers=0
        for e in self.scene.get('entities',[]):
            x,y=e.get('x'),e.get('y');side=e.get('side');hp=e.get('hp')
            if side not in (0,1) or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y)):continue
            if not 0<=x<=18000 or not 0<=y<=32000 or (isinstance(hp,(int,float)) and hp==0):continue
            cid=e.get('card_id');tower=cid==-1 and (e.get('max_hp') or 0)>0
            if tower:
                name='国王塔' if 6500<=x<=11500 else '公主塔';towers+=1
            else:
                info=observed_card(cid) if isinstance(cid,int) and cid>0 else {}
                base=info.get('base_card_id',cid)
                name=CARD_ZH.get(base) or catalog().get(base,{}).get('display_name') or f'实体 {cid}'
                units+=1
            px,py=xy(x,y);radius=max(4,scale*(.65 if tower else .3));color='#60a5fa' if side==self.side else '#fb7185'
            shape=c.create_rectangle if tower else c.create_oval
            shape(px-radius,py-radius,px+radius,py+radius,fill=color,outline='#e2e8f0')
            c.create_text(px,max(top+6,py-radius-7),text=name,fill='#f1f5f9',font=('Microsoft YaHei UI',8))
            maximum=e.get('max_hp')
            if isinstance(hp,(int,float)) and isinstance(maximum,(int,float)) and maximum>0 and hp>=0:
                ratio=min(1,hp/maximum);half=max(6,radius)
                c.create_line(px-half,py+radius+4,px+half,py+radius+4,fill='#475569',width=3)
                c.create_line(px-half,py+radius+4,px-half+2*half*ratio,py+radius+4,fill='#4ade80',width=3)
        paused=' · 时钟暂停' if time.monotonic()-self.changed>1 else ''
        self.status.configure(text=f'Tick {self.scene.get("tick")} · {units} 单位 / {towers} 塔{paused}')
