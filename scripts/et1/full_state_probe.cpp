// Read-only ET1 state probe through HDEP: every joint motor, IMU, gamepad and robot state.
//
// Calls only Bridge::load, initialize (subscribe and wait for the first state) and getters.
// Never calls enter_debug_mode, set_joint_motor_mode, set_joint_motor_commands, set_rgb_light,
// speak or any audio setter. The first output line is metadata; each further line is one sample.
//
//   full_state_probe ROBOT_INTERFACE_PREFIX BRIDGE_LIBRARY [SAMPLES] [INTERVAL_MS]
#include <hdep/hdep.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

namespace {

[[noreturn]] void finish(int code) {
  // HDEP has no listener-detach API; destroying the Bridge can race a CycloneDDS
  // callback. A one-shot read-only process leaves teardown to process exit.
  std::cout.flush();
  std::cerr.flush();
  if (!std::cout && code == 0) code = 1;
  std::_Exit(code);
}

std::uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void number(double value) {
  if (std::isfinite(value)) std::cout << value; else std::cout << "null";
}

template <class T> void optional_value(const std::optional<T>& value) {
  if (value) number(static_cast<double>(*value)); else std::cout << "null";
}

template <class T, std::size_t N> void array(const std::array<T, N>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) std::cout << ',';
    number(static_cast<double>(values[i]));
  }
  std::cout << ']';
}

template <class F> void guarded(const char* key, F&& body) {
  // Optional getters throw std::logic_error when the bridge lacks them.
  std::cout << ",\"" << key << "\":";
  try {
    body();
  } catch (const std::logic_error&) {
    std::cout << "null";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: full_state_probe ROBOT_INTERFACE_PREFIX BRIDGE_LIBRARY [SAMPLES] [INTERVAL_MS]\n";
    return 2;
  }
  const int samples = argc > 3 ? std::atoi(argv[3]) : 50;
  const int interval_ms = argc > 4 ? std::atoi(argv[4]) : 100;
  if (samples <= 0 || samples > 10000 || interval_ms < 0 || interval_ms > 10000) {
    std::cerr << "SAMPLES must be in [1, 10000] and INTERVAL_MS in [0, 10000]\n";
    return 2;
  }
  std::shared_ptr<hdep::Bridge> bridge;  // kept alive until _Exit
  try {
    bridge = hdep::Bridge::load(argv[2]);
    hdep::InitializeOptions options;
    options.network_interface = argv[1];
    options.startup_timeout = std::chrono::milliseconds(5000);
    options.joint_motor_mode = hdep::JointMotorMode::PR;
    bridge->initialize(options);

    const auto names = bridge->get_joint_motor_names();
    const auto valid = bridge->get_joint_motor_valid(hdep::JointMotorMode::PR);
    const auto count = bridge->get_joint_motor_count();
    if (count > names.size()) throw std::runtime_error("invalid joint motor count");
    const auto info = bridge->get_robot_info();
    const auto version = bridge->get_version();

    std::cout << std::setprecision(10);
    std::cout << "{\"meta\":true,\"vendor\":" << std::quoted(info.vendor) << ",\"model\":" << std::quoted(info.model)
              << ",\"bridge_version\":" << std::quoted(version.bridge_string())
              << ",\"capabilities\":" << bridge->get_capabilities()
              << ",\"gamepad_supported_buttons\":" << bridge->get_gamepad_supported_buttons()
              << ",\"joint_motor_count\":" << count << ",\"joints\":[";
    for (unsigned i = 0; i < count; ++i) {
      if (i) std::cout << ',';
      std::cout << "{\"slot\":" << i << ",\"name\":" << std::quoted(names[i])
                << ",\"pr_valid\":" << (valid[i] ? "true" : "false") << '}';
    }
    std::cout << ']';
    guarded("diagnostics", [&] {
      const auto states = bridge->get_diagnostic_states();
      std::cout << '[';
      for (std::size_t i = 0; i < states.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"level\":" << static_cast<int>(states[i].level) << ",\"name\":" << std::quoted(states[i].name)
                  << ",\"code\":" << states[i].code << ",\"message\":" << std::quoted(states[i].message) << '}';
      }
      std::cout << ']';
    });
    std::cout << '}' << std::endl;

    for (int n = 0; n < samples; ++n) {
      const auto joints = bridge->get_joint_motor_state();
      const auto imu = bridge->get_imu_state();
      const auto pad = bridge->get_gamepad_state();
      std::cout << "{\"sample\":" << n << ",\"read_ns\":" << now_ns()
                << ",\"joint_timestamp_ns\":" << joints.timestamp_ns
                << ",\"joint_mode\":" << static_cast<unsigned>(joints.mode) << ",\"joints\":[";
      bool first = true;
      for (unsigned i = 0; i < count; ++i) {
        if (!valid[i]) continue;
        const auto& s = joints.samples[i];
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"slot\":" << i << ",\"q\":"; number(s.position);
        std::cout << ",\"dq\":"; number(s.velocity);
        std::cout << ",\"tau\":"; number(s.torque);
        std::cout << ",\"raw_mode\":"; optional_value(s.raw_mode);
        std::cout << ",\"raw_state\":"; optional_value(s.raw_state);
        std::cout << ",\"rotor_temp\":"; optional_value(s.rotor_temperature);
        std::cout << ",\"driver_temp\":"; optional_value(s.driver_temperature);
        std::cout << ",\"voltage\":"; optional_value(s.voltage);
        std::cout << ",\"fault\":"; optional_value(s.fault_code);
        std::cout << '}';
      }
      std::cout << "],\"imu\":{\"timestamp_ns\":" << imu.timestamp_ns << ",\"quat_wxyz\":";
      array(imu.quaternion);
      std::cout << ",\"gyro\":"; array(imu.gyroscope);
      std::cout << ",\"accel\":"; array(imu.accelerometer);
      std::cout << ",\"rpy\":";
      if (imu.rpy) array(*imu.rpy); else std::cout << "null";
      std::cout << ",\"temperature\":"; optional_value(imu.temperature);
      std::cout << "},\"gamepad\":{\"timestamp_ns\":" << pad.timestamp_ns << ",\"buttons\":" << pad.buttons
                << ",\"lx\":" << pad.left_x << ",\"ly\":" << pad.left_y << ",\"rx\":" << pad.right_x
                << ",\"ry\":" << pad.right_y << '}';
      guarded("robot", [&] {
        const auto robot = bridge->get_robot_state();
        std::cout << "{\"timestamp_ns\":" << robot.timestamp_ns << ",\"machine_mode\":" << robot.machine_mode
                  << ",\"control_tick\":" << robot.control_tick << ",\"firmware\":";
        array(robot.firmware_version);
        std::cout << '}';
      });
      std::cout << '}' << std::endl;
      if (!std::cout) throw std::runtime_error("cannot write probe output");
      if (n + 1 < samples) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    std::cerr << samples << " read-only samples collected; exiting without SDK teardown.\n";
    finish(0);
  } catch (const std::exception& error) {
    std::cerr << "Read-only probe failed: " << error.what() << '\n';
    finish(1);
  }
}
