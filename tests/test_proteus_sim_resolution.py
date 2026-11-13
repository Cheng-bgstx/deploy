"""Catch the actual high-frequency joint oscillation seen with the 5-ms Python PD loop."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from sim2real.config.robots import get_robot_cfg

ROOT=Path(__file__).resolve().parents[1]
ROBOTS=[n for n in ['q1','t800'] if (ROOT/f'sim2real/config/robots/{n}.py').is_file()]

@pytest.mark.parametrize('robot',ROBOTS)
def test_default_simulation_tracks_without_pd_oscillation(robot,tmp_path,monkeypatch):
    from scripts.proteus import evaluate as evaluation
    contract={'q1':'primebot_q1','t800':'engineai_t800'}[robot]
    policy=ROOT/f'checkpoints/proteus/proteus_wamseq_62/{contract}/policy-sim.yaml'
    source=ROOT/f'checkpoints/{contract}/motions'
    cfg=get_robot_cfg(robot)
    if not policy.is_file() or not cfg.mjcf_path.is_file() or not (source/'manifest.json').is_file():
        pytest.skip('Requires locally prepared model, policy and reference motion')
    dataset=tmp_path/'motion';(dataset/'motions').mkdir(parents=True)
    shutil.copy2(source/'manifest.json',dataset/'manifest.json')
    (dataset/'motions/walk1_subject1.npz').symlink_to(source/'motions/walk1_subject1.npz')
    velocities=[];seen=set()
    real=evaluation.IntegratedSim2Sim
    class Capture(real):
        def _append_trajectory_frame(self):
            super()._append_trajectory_frame()
            if not self.playback_started:return
            if round(self.sim.mj_data.time/self.args.sim_dt)%self.args.decimation:return
            frame=int(self.state_processor.motion_t[0])
            if frame in seen:return
            seen.add(frame)
            m=self.sim.mj_model
            ids=[m.joint(n).dofadr[0] for n in cfg.joint_names]
            velocities.append(self.sim.mj_data.qvel[ids].copy())
    monkeypatch.setattr(evaluation,'IntegratedSim2Sim',Capture)
    output=tmp_path/'evaluation'
    evaluation.evaluate(robot,policy,dataset,output)
    v=np.stack(velocities)
    if robot=='q1':
        acceleration_rms=float(np.sqrt(np.mean((np.diff(v,axis=0)/.02)**2)))
        assert acceleration_rms<60.,f'Q1 joint oscillation: acceleration RMS={acceleration_rms:.2f} rad/s²'
    else:
        report=json.loads((output/'summary.json').read_text())['motions'][0]
        assert report['mean_world_body_error_m']<.08, f'T800 body error={report["mean_world_body_error_m"]:.3f} m'
        # The initial static portion must not develop fast head-joint oscillation.
        assert float(np.sqrt(np.mean(v[:100]**2)))<1.,'T800 static-pose joint velocity is unstable'
