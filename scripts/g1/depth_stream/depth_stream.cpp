// Timestamped depth-only TCP source. No Unitree SDK or motor publisher.
#include <librealsense2/rs.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <signal.h>
#include <zlib.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cerrno>
#include "depth_clock.hpp"

static volatile sig_atomic_t running = 1;
static void stop(int) { running = 0; }
static double monotonic_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double wall_s() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
static bool transfer(int fd, void* data, size_t size, bool sending) {
    auto* ptr = static_cast<char*>(data);
    while (size && running) {
        ssize_t count = sending ? send(fd, ptr, size, MSG_NOSIGNAL) : recv(fd, ptr, size, 0);
        if(count<0 && (errno==EINTR || errno==EAGAIN || errno==EWOULDBLOCK)) continue;
        if (count <= 0) return false;
        size -= count; ptr += count;
    }
    return size == 0;
}
// Wire v2 is little endian; each depth frame is followed by a clock exchange.
#pragma pack(push, 1)
struct Header {
    char magic[8] = {'S','C','D','E','P','T','H','2'};
    uint32_t version = 2, bytes = sizeof(Header);
    uint64_t session = 0, frame = 0;
    double device_ms = 0, acquired_s = 0, mapping_bracket_s = 0;
    uint32_t width = 640, height = 480, domain = 0, payload_bytes = 0, crc = 0;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 76, "Depth header wire layout changed");

int main(int argc, char** argv) {
    int listener = -1, client = -1;
    try {
        uint16_t endian = 1;
        if (*reinterpret_cast<char*>(&endian) != 1) throw std::runtime_error("Little endian required");
        int port = 18741;
        for (int i=1; i<argc; ++i) {
            if (std::string(argv[i]) == "--port" && i+1<argc) port = std::stoi(argv[++i]);
            else throw std::runtime_error("Usage: depth_stream --port PORT");
        }
        if (port<1024 || port>65535) throw std::runtime_error("Invalid port");
        signal(SIGINT, stop); signal(SIGTERM, stop); signal(SIGHUP, stop); signal(SIGPIPE, stop);
        listener = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1; setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{}; address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
        if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || listen(listener, 1))
            throw std::runtime_error("Cannot bind depth listener");
        std::cerr << "Depth listener ready on 127.0.0.1:" << port << std::endl;
        pollfd connection{listener, POLLIN, 0};
        if (poll(&connection, 1, 15000)<=0 || !running) throw std::runtime_error("Depth client did not connect within 15 s");
        client = accept(listener, nullptr, nullptr);
        if (client<0) throw std::runtime_error("Depth accept failed");
        timeval timeout{3,0}; setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        char request = 0;
        while (transfer(client, &request, 1, false) && request == 'C') {
            double times[2] = {monotonic_s(), 0}; times[1] = monotonic_s();
            if (!transfer(client, times, sizeof(times), true)) throw std::runtime_error("Clock probe failed");
        }
        if (request != 'S') throw std::runtime_error("Expected stream request");
        rs2::pipeline pipe; rs2::config config;
        config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
        // Preserve the sensor hardware clock; SDK global mapping can jump by 10–23 ms.
        rs2::context context; auto devices = context.query_devices();
        if (devices.size()!=1) throw std::runtime_error("Exactly one RealSense camera required");
        auto sensor = devices[0].first<rs2::depth_sensor>();
        if (!sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
            throw std::runtime_error("RealSense hardware time configuration unavailable");
        sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0);
        config.enable_device(devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        auto profile = pipe.start(config);
        auto active_sensor = profile.get_device().first<rs2::depth_sensor>();
        active_sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0);
        if (std::abs(profile.get_device().first<rs2::depth_sensor>().get_depth_scale()-.001)>1e-7)
            throw std::runtime_error("Camera depth scale must be 1 mm");
        const auto intrinsics = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
        if (std::abs(intrinsics.fx-390.409454)>0.01 || std::abs(intrinsics.fy-390.409454)>0.01 ||
            std::abs(intrinsics.ppx-324.806091)>0.01 || std::abs(intrinsics.ppy-237.971497)>0.01)
            throw std::runtime_error("Camera intrinsics differ from frozen DP contract");
        for (float c : intrinsics.coeffs) if (std::abs(c)>1e-7)
            throw std::runtime_error("Camera distortion differs from frozen DP contract");
        uint64_t session = static_cast<uint64_t>(monotonic_s()*1e9);
        double stream_started = monotonic_s();
        double last_acquired = -1;
        scalecar::DepthClockMapper clock_mapper;
        uint64_t clock_rejected = 0, delivered = 0;
        while (running) {
            rs2::frameset frames;
            if (!pipe.try_wait_for_frames(&frames,1000)) continue;
            auto depth = frames.get_depth_frame();
            auto domain = depth.get_frame_timestamp_domain();
            if (domain != RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK) {
                if(monotonic_s()-stream_started<12) continue;
                throw std::runtime_error("Expected hardware camera clock after warmup");
            }
            double before = monotonic_s(), after = monotonic_s();
            Header header; header.session = session; header.frame = depth.get_frame_number();
            header.device_ms = depth.get_timestamp();
            header.acquired_s = clock_mapper.map(header.device_ms/1000, after);
            header.mapping_bracket_s = after-before; header.domain = static_cast<uint32_t>(domain);
            if (after-stream_started<12) continue;
            if (!scalecar::usable_depth_clock(header.acquired_s, after, header.mapping_bracket_s, last_acquired)) {
                ++clock_rejected;
                if(clock_rejected==1 || clock_rejected%30==0)
                    std::cerr << "Depth frame skipped: clock age=" << after-header.acquired_s
                              << " domain=" << header.domain << " skipped=" << clock_rejected << std::endl;
                continue;
            }
            uLongf size = compressBound(depth.get_data_size()); std::vector<unsigned char> packed(size);
            if (compress2(packed.data(), &size, static_cast<const Bytef*>(depth.get_data()), depth.get_data_size(),1)!=Z_OK)
                throw std::runtime_error("Depth compression failed");
            header.payload_bytes = size;
            header.crc = crc32(0, static_cast<const Bytef*>(depth.get_data()), depth.get_data_size());
            if (!transfer(client,&header,sizeof(header),true) || !transfer(client,packed.data(),size,true)) break;
            last_acquired=header.acquired_s;
            ++delivered;
            if (!transfer(client,&request,1,false)) break;
            if (request!='C') throw std::runtime_error("Expected per-frame clock probe");
            double times[2] = {monotonic_s(), 0}; times[1] = monotonic_s();
            if (!transfer(client,times,sizeof(times),true)) break;
        }
        std::cerr << "Depth stream summary: delivered=" << delivered << " clock_skipped=" << clock_rejected << std::endl;
        pipe.stop(); close(client); close(listener); return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        if (client>=0) close(client); if (listener>=0) close(listener); return 1;
    }
}
