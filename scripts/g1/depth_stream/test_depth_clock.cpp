#include "depth_clock.hpp"
#include <cassert>
#include <limits>

int main() {
    assert(scalecar::usable_depth_clock(10., 10.03, .0001, 9.97));
    assert(!scalecar::usable_depth_clock(10.0101742, 10., .0001, 9.97));
    assert(!scalecar::usable_depth_clock(9.79, 10., .0001, 9.7));
    assert(!scalecar::usable_depth_clock(9.96, 10., .0001, 9.97));
    assert(!scalecar::usable_depth_clock(std::numeric_limits<double>::quiet_NaN(), 10., .0001, 9.97));
    // The next valid frame recovers immediately without restarting a session.
    assert(scalecar::usable_depth_clock(10.02, 10.05, .0001, 9.97));
    scalecar::DepthClockMapper mapper;
    double prev=-1;
    for(int frame=0; frame<3600; ++frame) {
        double device=frame/30.;
        double observed=100.+device+.004+(frame%17==0 ? .025 : 0.);
        double mapped=mapper.map(device,observed);
        assert(mapped<=observed+1e-9 && mapped>prev);
        assert(observed-mapped<.03);
        prev=mapped;
    }
    assert(std::isnan(mapper.map(1., 221.)));

}
