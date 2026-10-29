// Minimal C ABI over HDEP for ctypes (scripts/et1/hdep_bridge.py).
//
// hdep_py ships only for x86_64/CPython 3.11; the ET1 board runs aarch64/CPython 3.10.
// Slots are HDEP joint-motor indices (0..63). Functions return 0 on success and
// write a message to `error` otherwise. There is no close: HDEP has no listener
// detach API and destroying a Bridge can race a CycloneDDS callback, so the
// owning process exits without teardown.
#include <hdep/hdep.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct Handle {
  std::shared_ptr<hdep::Bridge> bridge;
};

void set_error(char* error, int size, const char* message) {
  if (error && size > 0) {
    std::strncpy(error, message, static_cast<std::size_t>(size) - 1);
    error[size - 1] = '\0';
  }
}

template <class F> int guarded(char* error, int size, F&& body) {
  try {
    body();
    return 0;
  } catch (const std::exception& e) {
    set_error(error, size, e.what());
  } catch (...) {
    set_error(error, size, "unknown HDEP error");
  }
  return -1;
}

}  // namespace

extern "C" {

void* et1_open(const char* bridge_library, const char* interface_prefix, int startup_timeout_ms, char* error,
               int size) {
  auto* handle = new Handle();
  const int code = guarded(error, size, [&] {
    handle->bridge = hdep::Bridge::load(bridge_library);
    hdep::InitializeOptions options;
    options.network_interface = interface_prefix;
    options.startup_timeout = std::chrono::milliseconds(startup_timeout_ms);
    options.joint_motor_mode = hdep::JointMotorMode::PR;
    handle->bridge->initialize(options);
  });
  if (code != 0) return nullptr;  // the half-initialized bridge is leaked on purpose
  return handle;
}

int et1_joint_layout(void* h, char* names, int names_size, std::uint8_t* pr_valid, std::uint32_t* count, char* error,
                     int size) {
  return guarded(error, size, [&] {
    auto& bridge = *static_cast<Handle*>(h)->bridge;
    const auto& motor_names = bridge.get_joint_motor_names();
    const auto& valid = bridge.get_joint_motor_valid(hdep::JointMotorMode::PR);
    *count = bridge.get_joint_motor_count();
    std::string joined;
    for (std::uint32_t i = 0; i < *count; ++i) {
      joined += motor_names[i];
      joined += '\n';
      pr_valid[i] = valid[i];
    }
    if (static_cast<int>(joined.size()) >= names_size) throw std::runtime_error("joint name buffer too small");
    std::memcpy(names, joined.c_str(), joined.size() + 1);
  });
}

// Arrays: q, dq, tau [64]; imu quat_wxyz[4], gyro[3], accel[3];
// stamps[3] = joint, imu, gamepad host-steady-clock ns; sticks[4] = lx, ly, rx, ry.
int et1_read(void* h, double* q, double* dq, double* tau, double* quat, double* gyro, double* accel,
             std::uint64_t* stamps, std::uint32_t* buttons, float* sticks, char* error, int size) {
  return guarded(error, size, [&] {
    auto& bridge = *static_cast<Handle*>(h)->bridge;
    const auto joints = bridge.get_joint_motor_state();
    for (std::size_t i = 0; i < hdep::max_joint_motor_count; ++i) {
      q[i] = joints.samples[i].position;
      dq[i] = joints.samples[i].velocity;
      tau[i] = joints.samples[i].torque;
    }
    const auto imu = bridge.get_imu_state();
    std::copy(imu.quaternion.begin(), imu.quaternion.end(), quat);
    std::copy(imu.gyroscope.begin(), imu.gyroscope.end(), gyro);
    std::copy(imu.accelerometer.begin(), imu.accelerometer.end(), accel);
    const auto pad = bridge.get_gamepad_state();
    stamps[0] = joints.timestamp_ns;
    stamps[1] = imu.timestamp_ns;
    stamps[2] = pad.timestamp_ns;
    *buttons = pad.buttons;
    sticks[0] = pad.left_x;
    sticks[1] = pad.left_y;
    sticks[2] = pad.right_x;
    sticks[3] = pad.right_y;
  });
}

int et1_enter_debug_mode(void* h, char* error, int size) {
  return guarded(error, size, [&] { static_cast<Handle*>(h)->bridge->enter_debug_mode(); });
}

// Arrays [64] by HDEP slot; slots without a command must carry kp = kd = 0.
int et1_write(void* h, const double* q, const double* dq, const double* tau, const double* kp, const double* kd,
              char* error, int size) {
  return guarded(error, size, [&] {
    std::array<hdep::JointMotorCommand, hdep::max_joint_motor_count> commands{};
    for (std::size_t i = 0; i < commands.size(); ++i) {
      commands[i].position = q[i];
      commands[i].velocity = dq[i];
      commands[i].torque = tau[i];
      commands[i].kp = kp[i];
      commands[i].kd = kd[i];
    }
    static_cast<Handle*>(h)->bridge->set_joint_motor_commands(commands);
  });
}

}  // extern "C"
