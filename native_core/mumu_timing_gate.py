"""Bounded experimental timing gate; never changes the fixed4 observation cadence."""
from dataclasses import dataclass
import json,math
from pathlib import Path

@dataclass(frozen=True)
class TimingConfig:
    mode: str = 'dynamic'
    base: float = .5
    floor: float = .25
    high_elixir: float = 9.
    cap_ramp_seconds: float = 3.

    def __post_init__(self):
        if self.mode not in ('fixed','dynamic'):
            raise ValueError('unknown timing mode')
        if not all(math.isfinite(x) for x in (self.base,self.floor,self.high_elixir,self.cap_ramp_seconds)):
            raise ValueError('nonfinite timing configuration')
        if not 0 < self.floor <= self.base < 1 or not 0 <= self.high_elixir < 10 or self.cap_ramp_seconds <= 0:
            raise ValueError('invalid timing bounds')

class TimingGate:
    def __init__(self,config=None,*,calibration=None,model_digest=None):
        self.config=config or TimingConfig();self.a=1.;self.b=0.;self.calibrated=False
        if calibration:
            saved=json.loads(Path(calibration).read_text(encoding='utf-8'))
            if saved.get('schema')!='mumu_timing_calibration_v1' or saved.get('model_digest')!=model_digest or saved.get('decision_ticks')!=4:
                raise ValueError('calibration/model contract mismatch')
            self.a=float(saved['a']);self.b=float(saved['b'])
            if not math.isfinite(self.a) or self.a<=0 or not math.isfinite(self.b) or not saved.get('heldout_dataset_digest'):
                raise ValueError('invalid calibration parameters/provenance')
            self.calibrated=True
        self.reset()

    def reset(self):
        self.last_tick=None;self.cap_ticks=0;self.was_capped=False

    def evaluate(self,logit,*,tick,elixir_raw):
        if not math.isfinite(logit) or not 0<=elixir_raw<=100000:
            raise ValueError('invalid timing observation')
        delta=0 if self.last_tick is None else tick-self.last_tick
        # Discontinuities cannot manufacture long cap duration.
        if delta not in (0,4):self.cap_ticks=0;self.was_capped=False
        if elixir_raw>=99900:
            if delta==4 and self.was_capped:self.cap_ticks+=4
        else:self.cap_ticks=0
        self.was_capped=elixir_raw>=99900
        self.last_tick=tick
        cfg=self.config;elixir=elixir_raw/10000.;cap_seconds=self.cap_ticks*.05
        pressure=max(0.,min(1.,(elixir-cfg.high_elixir)/(10-cfg.high_elixir)))
        cap_pressure=min(1.,cap_seconds/cfg.cap_ramp_seconds)
        # 60% of the bounded adjustment follows high elixir; 40% requires sustained cap.
        adjustment=(cfg.base-cfg.floor)*(.6*pressure+.4*cap_pressure) if cfg.mode=='dynamic' else 0.
        threshold=max(cfg.floor,min(cfg.base,cfg.base-adjustment))
        sigmoid=lambda z: 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
        raw=sigmoid(logit);score=sigmoid(self.a*logit+self.b)
        return dict(raw_play_probability=raw,play_probability=score,threshold=threshold,
                    timing_mode=cfg.mode,calibrated=self.calibrated,cap_seconds=cap_seconds,
                    high_elixir_pressure=pressure,timing_experimental=cfg.mode=='dynamic',
                    timing_eligible=score>=threshold)
