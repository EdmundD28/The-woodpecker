#pragma once
#include <stdint.h>

// G开启按住许可，g恢复旧自动触发；A续租许可，a立即锁定。
// 心跳停止250ms后自动锁定，防止电脑退出或断线后一直允许触发。
class CapturePermission {
public:
    static constexpr uint32_t LEASE_MS = 250;

    void enable(bool value) {
        enabled = value;
        lock();
    }

    void lock() {
        armed = false;
        consumed = false;
    }

    void renew(uint32_t now) {
        if (!enabled) return;
        armed = true;
        renewedAt = now;
    }

    bool allowed(uint32_t now) const {
        return !enabled || (armed && uint32_t(now - renewedAt) < LEASE_MS);
    }

    bool canStart(uint32_t now) const {
        return allowed(now) && (!enabled || !consumed);
    }

    void consume() { if (enabled) consumed = true; }

private:
    bool enabled = false;
    bool armed = false;
    bool consumed = false;
    uint32_t renewedAt = 0;
};
