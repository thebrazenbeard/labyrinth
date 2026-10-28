"""Experimental numeric neural multiplication; no product rule in model.forward.

Shared continuous scalar output avoids a learned per-class output table.
This does not establish arithmetic discovery; supplied Fourier input prior.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, math, random, time
from typing import Any
import torch
from torch import nn

MOD=97
@dataclass(frozen=True)
class NumericConfig:
    seed:int=20261028
    seeds:tuple[int,int]=(20261028,20261029)
    width:int=24
    depth:int=2
    epochs:int=18
    batch_size:int=512
    learning_rate:float=.003
    max_optimizer_seconds:float=35.0
    holdout_fraction:float=.10
    def validate(self):
        if (type(self.seed) is not int or not 0<=self.seed<2**31 or
            type(self.seeds) is not tuple or len(self.seeds)!=2 or
            len(set(self.seeds))!=2 or
            any(type(k) is not int or not 0<=k<2**31 for k in self.seeds)
            or type(self.width) is not int or not 8<=self.width<=48
            or type(self.depth) is not int or not 1<=self.depth<=3
            or type(self.epochs) is not int or not 1<=self.epochs<=160
            or type(self.batch_size) is not int or not 16<=self.batch_size<=2048
            or type(self.learning_rate) is not float or not 0<self.learning_rate<=.02
            or type(self.max_optimizer_seconds) is not float or not 1<=self.max_optimizer_seconds<=90
            or type(self.holdout_fraction) is not float or not .02<=self.holdout_fraction<=.3):
            raise ValueError("invalid bounded numeric configuration")
        return self

@dataclass(frozen=True)
class Batch:
    x:Any
    y:Any
    target:Any
    def __len__(self): return int(self.x.numel())

@dataclass(frozen=True)
class Split:
    train:Batch
    independent_holdout:Batch
    target_53:Batch
    input_53:Batch
    modulus:int=MOD

def multiplication_targets(x,y,modulus=MOD):
    """Offline teacher ONLY; never called inside neural forward."""
    return torch.remainder(x*y,modulus)

def _batch(pairs,modulus=MOD):
    pairs=list(pairs)
    x=torch.tensor([a for a,b in pairs],dtype=torch.long)
    y=torch.tensor([b for a,b in pairs],dtype=torch.long)
    return Batch(x,y,multiplication_targets(x,y,modulus))

def build_scientific_split(*,modulus=MOD,seed=20261028,holdout_fraction=.1):
    if modulus!=MOD or type(seed) is not int or seed<0 or not .02<=holdout_fraction<=.3:
        raise ValueError("invalid split")
    groups=[(a,b) for a in range(MOD) for b in range(a,MOD)]
    rnd=random.Random(seed)
    rnd.shuffle(groups)
    eligible=[]; blocked_output=[]; blocked_input=[]
    for a,b in groups:
        label=(a*b)%MOD
        if a==53 or b==53:
            blocked_input.append((a,b))
        elif label==53:
            blocked_output.append((a,b))
        else:
            eligible.append((a,b))
    n=max(1,round(len(eligible)*holdout_fraction))
    def directions(groups):
        return [(a,b) for a,b in groups for a,b in
                (((a,b),(b,a)) if a!=b else ((a,b),))]
    return Split(_batch(directions(eligible[n:])),
                 _batch(directions(eligible[:n])),
                 _batch(directions(blocked_output)),
                 _batch(directions(blocked_input)))

class NumericNetwork(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        w=config.width
        self.encoder=nn.Sequential(nn.Linear(9,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh())
        layers=[nn.Linear(2*w,w),nn.GELU()]
        for _ in range(config.depth-1):
            layers += [nn.Linear(w,w),nn.GELU()]
        layers += [nn.Linear(w,1),nn.Sigmoid()]
        self.decoder=nn.Sequential(*layers)
    def _features(self,z,p):
        t=z.float()/p.float()
        f=[t]
        for k in (1,2,4,8):
            f.extend([torch.sin(t*(2*math.pi*k)),torch.cos(t*(2*math.pi*k))])
        return torch.stack(f,dim=-1)
    def forward(self,x,y,modulus):
        if (not all(isinstance(v,torch.Tensor) for v in (x,y,modulus))
            or x.dtype!=torch.long or y.dtype!=torch.long
            or modulus.dtype!=torch.long or x.ndim!=1 or
            x.shape!=y.shape or x.shape!=modulus.shape
            or bool(((modulus<3)|(modulus>101)).any())
            or bool(((x<0)|(y<0)|(x>=modulus)|(y>=modulus)).any())):
            raise ValueError("invalid bounded residues")
        a=self.encoder(self._features(x,modulus))
        b=self.encoder(self._features(y,modulus))
        merged=torch.cat((a+b,a*b),dim=-1)
        return self.decoder(merged).squeeze(-1)*modulus.float()

def build_numeric_model(config=NumericConfig()):
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return NumericNetwork(config)

def _digest(model):
    h=hashlib.sha256()
    for value in model.state_dict().values():
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def _score(model,batch,modulus):
    model.eval()
    with torch.no_grad():
        out=[]
        for i in range(0,len(batch),1024):
            p=torch.full_like(batch.x[i:i+1024],modulus)
            out.extend(model(batch.x[i:i+1024],batch.y[i:i+1024],p).round().long().clamp(0,modulus-1).tolist())
    targets=batch.target.tolist()
    hits=sum(x==y for x,y in zip(out,targets))
    return {"correct":hits,"count":len(targets),
            "accuracy":hits/len(targets) if targets else 0.0}

def run_study(config=NumericConfig()):
    config.validate()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    split=build_scientific_split(modulus=MOD,seed=config.seed,
                                 holdout_fraction=config.holdout_fraction)
    external=_batch(((a,b) for a in range(101) for b in range(101)),101)
    reports=[]
    for seed in config.seeds:
        cfg=NumericConfig(seed=seed,seeds=config.seeds,width=config.width,depth=config.depth,
                          epochs=config.epochs,batch_size=config.batch_size,
                          learning_rate=config.learning_rate,max_optimizer_seconds=config.max_optimizer_seconds,
                          holdout_fraction=config.holdout_fraction)
        model=build_numeric_model(cfg)
        initial=_digest(model)
        names={"independent_holdout":split.independent_holdout,
               "target_53":split.target_53,"input_53":split.input_53,
               "modulus_101":external}
        frozen={name:_score(model,examples,101 if name=="modulus_101" else MOD)
                for name,examples in names.items()}
        optim=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate)
        g=torch.Generator().manual_seed(seed+1)
        started=time.perf_counter();updates=0;timed_out=False
        model.train()
        for _ in range(cfg.epochs):
            order=torch.randperm(len(split.train),generator=g)
            for j in range(0,len(order),cfg.batch_size):
                if time.perf_counter()-started>=cfg.max_optimizer_seconds:
                    timed_out=True;break
                pick=order[j:j+cfg.batch_size]
                modulus=torch.full_like(split.train.x[pick],MOD)
                prediction=model(split.train.x[pick],split.train.y[pick],modulus)
                loss=torch.nn.functional.smooth_l1_loss(prediction,split.train.target[pick].float())
                if not torch.isfinite(loss): raise RuntimeError("nonfinite neural loss")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step();updates+=1
            if timed_out: break
        elapsed=time.perf_counter()-started
        evaluated={name:_score(model,examples,101 if name=="modulus_101" else MOD)
                   for name,examples in names.items()}
        const=int(torch.bincount(split.train.target,minlength=MOD).argmax())
        baselines={name:{"accuracy":(ex.target==const).float().mean().item() if len(ex) else 0.0}
                   for name,ex in names.items()}
        improvement=evaluated["target_53"]["accuracy"] > max(
            frozen["target_53"]["accuracy"],baselines["target_53"]["accuracy"])
        reports.append({"seed":seed,"split":{"train_count":len(split.train),
            "independent_holdout_count":len(split.independent_holdout),
            "target_53_train_rows":int((split.train.target==53).sum()),
            "input_53_train_rows":int(((split.train.x==53)|(split.train.y==53)).sum())},
            "controls":{"constant":baselines,"frozen_random":frozen},
            "evaluation":evaluated,
            "training":{"optimizer_steps":updates,"optimizer_seconds":elapsed,
            "stopped_at_budget":timed_out,"initial_sha256":initial,"trained_sha256":_digest(model)},
            "conclusion":"target_53_improves_over_controls" if improvement else
                        "no_target_53_improvement_over_controls"})
    return {"schema":"labyrinth.minotaur.v7-numeric.v1",
            "model":{"architecture":"learned_symmetric_scalar_output_fourier_bias",
                     "parameters":sum(p.numel() for p in model.parameters()),
                     "supplied_prior":"periodic Fourier features and symmetric weight sharing",
                     "forward_product_or_modulo":False},
            "config":asdict(config),"runs":reports,
            "promotion_recommendation":"DO_NOT_PROMOTE","qualification":"NOT_ESTABLISHED",
            "limits":"Public synthetic mod97 and evaluation-only mod101, not blind or general reasoning."}

