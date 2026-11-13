#!/usr/bin/env python3
"""Query X2 system state, or explicitly migrate to Develop_MC / Ready over ROS 2.

Official contract: https://x2-aimdk.agibot.com/zh-cn/dev/Interface/system_state/
No arguments is read-only. A successful migration response is only acceptance;
completion requires the target mode and IN_READY, as verified by polling.
"""
from __future__ import annotations

import argparse
import json
import math
import time


def migration_complete(target, state, status):
    # Ready automatically proceeds to Business on the robot.
    return status == 1 and (state == target or target == 'Ready' and state == 'Business')


def response_error(header):
    if header.header.code != 0 or header.status.value != header.status.SUCCESS:
        return f'code={header.header.code}, status={header.status.value}: {header.message}'
    return None


class ServiceResponseError(RuntimeError):
    def __init__(self, header):
        super().__init__(response_error(header))
        self.retryable = False
        if header.message.startswith('State ') and 'not configured' in header.message:
            self.error_code = 'system_mode_not_configured'
            self.hint = ('The robot does not configure this system mode. Installing a host SDK '
                         'does not add robot-side modes. Use supported robot firmware/configuration '
                         'or the authenticated legacy MC-stop path; do not guess other mode names.')
        else:
            self.error_code = 'service_request_rejected'
            self.hint = 'The robot rejected the request; inspect its state and response before retrying.'


def check_response(header):
    if response_error(header) is not None:
        raise ServiceResponseError(header)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['Develop_MC', 'Ready'],
                        help='Changes robot mode; keep support in place. Stop Tracking before restoring Ready.')
    parser.add_argument('--require-develop-mc', action='store_true',
                        help='Read-only check: fail unless Develop_MC migration is complete.')
    parser.add_argument('--timeout', type=float, default=60., help='Migration completion timeout in seconds.')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    if args.target and args.require_develop_mc:
        parser.error('--require-develop-mc is read-only and cannot be combined with --target')

    import rclpy
    from aimdk_msgs.srv import GetSystemState, MigrateSystemState

    rclpy.init()
    node = rclpy.create_node('sim2real_x2_system_state')

    def call(client, request, *, retries=1, timeout=1.):
        if not client.wait_for_service(timeout_sec=3.):
            raise RuntimeError(f'Service unavailable: {client.srv_name}')
        for _ in range(retries):
            request.header.header.stamp = node.get_clock().now().to_msg()
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            if future.done():
                result = future.result()
                check_response(result.header)
                return result
            client.remove_pending_request(future)
        raise TimeoutError(f'No response from {client.srv_name}')

    def query():
        result = call(getter, GetSystemState.Request(), retries=3)
        return result.cur_state, int(result.curr_status.value)

    def show(state, status, **extra):
        print(json.dumps({'cur_state': state, 'curr_status': status,
                          'in_ready': status == 1, **extra}), flush=True)

    try:
        getter = node.create_client(GetSystemState, '/aimdk_5Fmsgs/srv/GetSystemState')
        state, status = query()
        show(state, status)
        if args.target is None:
            if args.require_develop_mc and not migration_complete('Develop_MC', state, status):
                raise RuntimeError('Develop_MC is not ready; release native control before starting Tracking')
            return 0
        if migration_complete(args.target, state, status):
            show(state, status, migration_complete=True, request_sent=False)
            return 0
        if status != 1 or state in {'Startup', 'EStop', 'OTA', 'Poweroff', 'Reboot'}:
            raise RuntimeError(f'Refusing migration from {state}, status={status}; resolve system state first')

        setter = node.create_client(MigrateSystemState, '/aimdk_5Fmsgs/srv/MigrateSystemState')
        request = MigrateSystemState.Request()
        request.state = args.target
        # Never automatically retry a mutating request with an uncertain outcome.
        try:
            result = call(setter, request, timeout=3.)
            print(json.dumps({'request_accepted': True, 'message': result.header.message}), flush=True)
        except TimeoutError as exc:
            print(json.dumps({'request_outcome_unknown': str(exc), 'polling_without_resending': True}), flush=True)
        deadline = time.monotonic() + args.timeout
        previous = None
        while time.monotonic() < deadline:
            state, status = query()
            if (state, status) != previous:
                show(state, status)
                previous = state, status
            if migration_complete(args.target, state, status):
                show(state, status, migration_complete=True)
                return 0
            if status in {3, 4, 5} or state in {'EStop', 'OTA', 'Poweroff', 'Reboot'}:
                raise RuntimeError(f'Migration interrupted: {state}, status={status}')
            time.sleep(.25)
        raise TimeoutError('Migration completion not confirmed; query state before any further action')
    except ServiceResponseError as exc:
        print(json.dumps({'error': str(exc), 'error_code': exc.error_code,
                          'retryable': exc.retryable, 'hint': exc.hint}), flush=True)
        return 1
    except (RuntimeError, TimeoutError) as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
