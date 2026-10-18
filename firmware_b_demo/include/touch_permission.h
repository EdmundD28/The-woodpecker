#pragma once
#include <cstdint>

// Physical equivalent of firmware A + collector HoldGate's space-key session.
// Observe release first, hold one second, consume one attempt, release to rearm.
class TouchPermission {
public:
    static constexpr uint32_t ARM_DELAY_MS=1000;

    void update(bool pressed,uint32_t now) {
        if(!pressed) {
            releasedOnce=true; holding=false; used=false;
        } else if(releasedOnce && !holding) {
            holding=true; pressedAt=now;
        }
    }
    bool held() const { return holding; }
    bool consumed() const { return used; }
    bool canStart(uint32_t now) const {
        return holding && !used && uint32_t(now-pressedAt)>=ARM_DELAY_MS;
    }
    void consume() { used=true; }

private:
    bool releasedOnce=false,holding=false,used=false;
    uint32_t pressedAt=0;
};
