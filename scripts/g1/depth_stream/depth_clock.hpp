#pragma once

#include <cmath>
#include <deque>
#include <algorithm>
#include <limits>

namespace scalecar {
// Map the device clock using the minimum observed arrival offset over 60 s.
// Arrival latency is an upper bound on acquisition, not an exact USB latency
// measurement. SDK global-wall-clock adjustments are deliberately excluded.
class DepthClockMapper {
public:
    double map(double device_s, double observed) {
        if(!std::isfinite(device_s) || !std::isfinite(observed))
            return std::numeric_limits<double>::quiet_NaN();
        if(device_s<=last_device) return std::numeric_limits<double>::quiet_NaN();
        last_device=device_s;
        offsets.emplace_back(observed,observed-device_s);
        while(offsets.size()>1800 || observed-offsets.front().first>60.) offsets.pop_front();
        double offset=offsets.front().second;
        for(const auto& sample:offsets) offset=std::min(offset,sample.second);
        return device_s+offset;
    }
private:
    std::deque<std::pair<double,double>> offsets;
    double last_device=-1.;
};
// Reject only this frame. Clock mapping jitter must not close the stream.
inline bool usable_depth_clock(double acquired, double observed,
                               double bracket, double previous) {
    return std::isfinite(acquired) && std::isfinite(observed) &&
           std::isfinite(bracket) && bracket >= 0 && bracket <= .01 &&
           acquired <= observed + .01 && observed - acquired <= .2 &&
           acquired > previous;
}
}
