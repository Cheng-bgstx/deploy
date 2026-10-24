# sim2real

Deploys humanoid policies trained in other codebases onto real robots and into sim2sim. This fork adds the Scalecar vehicle-entry deployment on the Unitree G1.

## Language

### Scalecar vehicle entry

**Tracker** (跟踪策略):
The Scalecar policy that drives all joints so the robot follows a Reference.
_Avoid_: policy (on its own), actor, controller

**Reference** (参考):
The target whole-body motion the Tracker is asked to follow at each control step.
_Avoid_: target, command, motion (for the concept)

**Reference Token** (参考 token):
One encoded future frame of the Reference, expressed relative to the current Reference pelvis position and heading; the Tracker sees a fixed set of them per step.
_Avoid_: task token

**Motion** (动作文件):
A recorded 50 Hz whole-body trajectory stored as a file. File Playback steps through it, and with either Reference Source its first frame defines the start pose.
_Avoid_: clip, take

**Reference Source** (参考源):
Where the Reference comes from; it is either File Playback or the Planner.

**File Playback** (文件回放):
The Reference Source that steps through a Motion frame by frame.
_Avoid_: replay, offline reference

**Planner** (规划器):
The Reference Source that generates the Reference online from depth images and the robot's recent history.
_Avoid_: DP (for the runtime role), diffusion policy

**Chunk** (规划块):
One Planner output: a 2.56 s, 50 Hz whole-body plan anchored at the moment its inputs were observed; a newer Chunk replaces the rest of the older one.
_Avoid_: plan segment, trajectory

### Operator stages

**Shaping** (塑形):
The stage that moves the robot from its powered-on posture to the Motion's first-frame pose, kept within soft joint limits, before the Tracker takes over.
_Avoid_: init

**Hold** (冻结保持):
The stage after the Tracker takes over and before Go, in which the Reference stays frozen at the Motion's first frame; with the Planner it also lasts until the first Chunk arrives.
_Avoid_: pause

**Go**:
The operator action that starts the Reference moving from the Motion's first frame; with the Planner it starts planning.
_Avoid_: play, start

**Final Hold** (末帧保持):
The stage after the Motion has ended in which the Tracker keeps following the last Reference frame.
_Avoid_: end hold

**E-stop** (急停):
The stage, reachable from any other stage, in which every joint target follows that joint's own measured position so the robot goes compliant.
_Avoid_: damping, zero

**Controller** (操作输入设备):
A device the operator uses to switch stages and give commands: the Unitree joystick, the keyboard, or a Pico headset.
_Avoid_: joystick (for the general concept), remote

### Vehicle sim2sim

**Vehicle Scene** (车辆场景):
The simulated ground and static car, with the driver door fixed open at 70°, in which sim2sim episodes run.
_Avoid_: car world, environment

**Task Envelope** (任务包络):
The part of the car the robot can reach during vehicle entry; the Vehicle Scene keeps collision geometry only inside it, and an episode in which the robot nears its boundary is invalid.
_Avoid_: crop box

**Entry** (进车):
The outcome in which both feet and the pelvis stay inside the car's cabin region continuously for 0.75 s.

**Fall** (倒地):
The outcome in which the pelvis or torso drops too low, or the body tilts too far, continuously for 0.2 s.

**Task Success** (任务成功):
An episode that reaches Entry, never Falls, and runs to completion.

### Machines and processes

**Host** (外置主机):
The external x86 machine with an NVIDIA GPU that runs the Tracker and the Planner and talks to the robot over Ethernet.
_Avoid_: PC, laptop, x86 machine

**Onboard Computer** (机载计算机):
The G1's built-in Orin NX; in this deployment it only runs depth capture.
_Avoid_: Orin, NX, robot PC

**Bridge** (桥接进程):
A separate process that relays robot state and commands between the robot and a policy process.
_Avoid_: controller
