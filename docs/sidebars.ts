import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        {
          type: 'doc',
          id: 'getting-started/overview',
          label: 'Overview',
        },
        {
          type: 'doc',
          id: 'getting-started/network-configuration',
          label: 'Network Configuration',
        },
        {
          type: 'doc',
          id: 'getting-started/root-project',
          label: 'Root Project',
        },
        {
          type: 'doc',
          id: 'getting-started/teleop-x86-64',
          label: 'Teleop Project (x86_64 PC)',
        },
        {
          type: 'doc',
          id: 'getting-started/teleop-onboard-orin',
          label: 'Teleop Project (Onboard Orin)',
        },
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/offline-motion-tracking',
        'tutorials/pico-teleoperation',
        'tutorials/motion-recording',
        'tutorials/run-external-policies',
        'tutorials/scalecar-vehicle-entry',
        {type: 'doc', id: 'reference/proteus-g1-pc', label: 'Proteus G1 (PC)'},
        {type: 'doc', id: 'reference/proteus-g1-onboard', label: 'Proteus G1 (Onboard Orin)'},
        {type: 'doc', id: 'reference/proteus-et1-onboard', label: 'Proteus ET1 (Onboard)'},
        'tutorials/proteus-oli',
        'tutorials/proteus-x2',
        'tutorials/proteus-x2-onboard',
        'tutorials/proteus-a3',
        'tutorials/proteus-t800',
        'tutorials/proteus-pm01',
        'tutorials/proteus-q1',
      ],
    },
    'faq',
    {
      type: 'category',
      label: 'Reference',
      items: [
        {
          type: 'doc',
          id: 'reference/artifacts',
          label: 'Download Artifacts',
        },
        {
          type: 'doc',
          id: 'robot_io',
          label: 'Robot I/O',
        },
        {
          type: 'doc',
          id: 'reference/tracking-framework',
          label: 'Tracking Framework',
        },
        {
          type: 'doc',
          id: 'reference/pico-to-g1',
          label: 'Pico To G1',
        },
        {
          type: 'doc',
          id: 'reference/teleop-implementation',
          label: 'Teleop Implementation',
        },
        {
          type: 'doc',
          id: 'reference/h2',
          label: 'Unitree H2 Model',
        },
        {
          type: 'doc',
          id: 'reference/policy-yaw-alignment',
          label: 'Yaw Alignment',
        },
        {
          type: 'doc',
          id: 'reference/motion-stream-disconnects',
          label: 'Motion Stream Disconnects',
        },
        {
          type: 'doc',
          id: 'reference/npz-motion-publisher',
          label: 'NPZ Motion Publisher',
        },
        {
          type: 'doc',
          id: 'reference/sonic-smpl-input',
          label: 'SONIC SMPL Input',
        },
        {
          type: 'doc',
          id: 'reference/xrobot-grpc-jetpack5',
          label: 'XRobot gRPC JetPack 5',
        },
        {
          type: 'doc',
          id: 'reference/onboard-jetpack5-inference-backends',
          label: 'Onboard JetPack 5 Inference Backends',
        },
        {
          type: 'doc',
          id: 'reference/xrobotoolkit-pc-service-arm64-build',
          label: 'XRoboToolkit PC Service ARM64 Build',
        },
      ],
    },
  ],
};

export default sidebars;
