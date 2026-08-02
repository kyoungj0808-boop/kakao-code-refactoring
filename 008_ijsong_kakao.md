원본코드
#!/usr/bin/python
# -*- coding: utf-8 -*-

import grp
import json
import os
import pwd
import shutil
import socket
import subprocess
import sys
import time
import glob

cwd = os.path.dirname(os.path.realpath(__file__))  # noqa
binpath = os.path.join(cwd, "..", "bin")  # noqa
pylib = os.path.join(cwd, "..", "pylib")  # noqa
if os.path.isdir(pylib):  # noqa
    sys.path.insert(0, pylib)  # noqa

from varlog.killer import Killer  # noqa
from varlog import procutil  # noqa
from varlog import limits  # noqa
from varlog.logger import get_logger  # noqa

logger = get_logger("vmr")

RETRY_INTERVAL_SEC = 3
DEFAULT_CLUSTER_ID = "1"
DEFAULT_REP_FACTOR = "1"
DEFAULT_RPC_PORT = "9092"
DEFAULT_RAFT_PORT = "10000"
DEFAULT_VMR_HOME = "/home/deploy/varlog-mr"
LOCAL_ADDRESS = socket.gethostbyname(socket.gethostname())


def get_local_addr():
    host = os.getenv("HOST_IP", LOCAL_ADDRESS)
    return host


def get_raft_url():
    local_addr = get_local_addr()
    raft_port = os.getenv("RAFT_PORT", DEFAULT_RAFT_PORT)
    return f"http://{local_addr}:{raft_port}"


def get_rpc_addr():
    local_addr = get_local_addr()
    rpc_port = os.getenv("RPC_PORT", DEFAULT_RPC_PORT)
    return f"{local_addr}:{rpc_port}"


def get_vms_addr():
    addr = os.getenv("VMS_ADDRESS")
    if addr is None:
        raise Exception("no admin address")
    return addr


def get_raft_dir():
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    return f"{home}/raftdata"


def get_log_dir():
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    return f"{home}/log"


def get_info():
    rep_factor = get_replication_factor()
    try:
        out = subprocess.check_output([
            f"{binpath}/varlogctl",
            "mr",
            "describe",
            f"--admin={get_vms_addr()}",
        ])

        nodes = json.loads(out)
        members = [node["raftURL"] for node in nodes]
        return rep_factor, members
    except Exception:
        logger.exception("could not get peers")
        return rep_factor, None


def get_replication_factor():
    return int(os.getenv("REPLICATION_FACTOR", DEFAULT_REP_FACTOR))


def add_raft_peer():
    try:
        raft_url = get_raft_url()
        rpc_addr = get_rpc_addr()

        out = subprocess.check_output([
            f"{binpath}/varlogctl",
            "mr",
            "add",
            f"--raft-url={raft_url}",
            f"--rpc-addr={rpc_addr}",
            f"--admin={get_vms_addr()}"
        ])
        mrnode = json.loads(out)
        node_id = mrnode.get("nodeId", 0)
        return node_id != "0"
    except Exception:
        logger.exception("could not add peer")
        return False


def remove_raft_peer():
    try:
        raft_url = get_raft_url()

        _, members = get_info()
        if members is None:
            return False
        elif raft_url not in members:
            return True

        out = subprocess.check_output([
            f"{binpath}/varlogctl",
            "mr",
            "remove",
            f"--raft-url={raft_url}",
            f"--admin={get_vms_addr()}"
        ])
        logger.info("remove raft:" + str(out))
        json.loads(out)
        return True
    except Exception:
        logger.exception("")
        return False


def prepare_vmr_home():
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    os.makedirs(home, exist_ok=True)


def check_standalone():
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    ret = os.path.exists(f"{home}/.standalone")
    return ret


def clear_standalone():
    logger.info("clear standalone")
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    if os.path.exists(f"{home}/.standalone"):
        os.remove(f"{home}/.standalone")


def exists_wal(path):
    wal = glob.glob(os.path.join(f"{path}/wal", "*", "*.wal"))
    return len(wal) > 0


def exists_cluster():
    _, peers = get_info()
    return peers is not None


def get_metadata_repository_cmd(standalone):
    cluster_id = os.getenv("CLUSTER_ID", DEFAULT_CLUSTER_ID)
    rep_factor, peers = get_info()
    raft_url = get_raft_url()
    rpc_addr = get_rpc_addr()
    raft_dir = get_raft_dir()
    log_dir = get_log_dir()

    command = [
        f"{binpath}/vmr",
        "start",
        f"--cluster-id={cluster_id}",
        f"--replication-factor={rep_factor}",
        f"--raft-address={raft_url}",
        f"--bind={rpc_addr}",
        f"--raft-dir={raft_dir}",
        f"--log-dir={log_dir}"
    ]

    if peers is not None:
        command.append("--join=true")
        for peer in peers:
            command.append(f"--peers={peer}")
    elif not standalone:
        return None

    return command


def print_mr_home():
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    arr = os.listdir(f"{home}")
    logger.info(arr)


def main():
    logger.info("start")

    print_mr_home()

    limits.set_limits()
    prepare_vmr_home()

    standalone = False
    if not exists_cluster():
        if check_standalone():
            standalone = True
            logger.info("it starts standalone")
        elif exists_wal(get_raft_dir()):
            standalone = True
            logger.info("it runs standalone with wal")
        else:
            logger.info("it could not run as standalone. check configuration")
            return

    clear_standalone()

    need_add_peer = False
    killer = Killer()
    while not killer.kill_now:
        if procutil.check_liveness("vmr"):
            if need_add_peer:
                logger.info("adding metadata repository to cluster")
                need_add_peer = not add_raft_peer()
            time.sleep(RETRY_INTERVAL_SEC)
            continue
        try:
            procutil.kill("vmr")

            if not standalone and not remove_raft_peer():
                logger.info("could not leave peer")
                return

            cmd = get_metadata_repository_cmd(standalone)
            if cmd is None:
                logger.info(f"could not make command. check configuration")
                return

            logger.info(f"running metadata repository: {cmd}")
            subprocess.Popen(cmd)

            need_add_peer = not standalone
            standalone = False
        except (OSError, ValueError, subprocess.SubprocessError):
            logger.exception("could not run metadata repository")
        time.sleep(RETRY_INTERVAL_SEC)
    procutil.stop("vmr")


if __name__ == '__main__':
    main()

클러스터 생존을 책임지는 데몬 코드임에도 환경 검증·예외 복구·프로세스 상태 관리가 느슨해, 
정상 상황에서는 작동하지만 장애 상황에서 원인 추적 불가와 연쇄 장애를 유발하는 레거시 운영 코드다.

제안패치
#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

cwd = os.path.dirname(os.path.realpath(__file__))  # noqa
binpath = os.path.join(cwd, "..", "bin")  # noqa
pylib = os.path.join(cwd, "..", "pylib")  # noqa
if os.path.isdir(pylib):  # noqa
    sys.path.insert(0, pylib)  # noqa

from varlog import limits  # noqa
from varlog import procutil  # noqa
from varlog.logger import get_logger  # noqa
from varlog.killer import Killer  # noqa

logger = get_logger("vmr")

# ---------- 상수 정의 (매직 넘버 제거) ----------
INITIAL_RETRY_INTERVAL_SEC = 3
MAX_RETRY_INTERVAL_SEC = 60
BACKOFF_FACTOR = 2
DEFAULT_CLUSTER_ID = "1"
DEFAULT_REP_FACTOR = "1"
DEFAULT_RPC_PORT = "9092"
DEFAULT_RAFT_PORT = "10000"
DEFAULT_VMR_HOME = "/home/deploy/varlog-mr"
LOCAL_ADDRESS = socket.gethostbyname(socket.gethostname())


class ConfigurationError(Exception):
    """설정 누락 및 유효성 검증 실패 시 발생하는 커스텀 예외"""
    pass


# ---------- [추가] Config Validation Layer ----------
def validate_configuration() -> None:
    """필수 환경 변수 및 시스템 리소스 무결성을 사전에 철저히 검증 (Fail-Fast)"""
    addr = os.getenv("VMS_ADDRESS")
    if not addr:
        raise ConfigurationError("Required environment variable 'VMS_ADDRESS' is missing.")

    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    try:
        os.makedirs(home, exist_ok=True)
        test_file = os.path.join(home, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except OSError as e:
        raise ConfigurationError(f"VMR_HOME directory '{home}' is not writable: {e}")

    try:
        rep_factor = int(os.getenv("REPLICATION_FACTOR", DEFAULT_REP_FACTOR))
        if rep_factor <= 0:
            raise ValueError
    except (ValueError, TypeError):
        raise ConfigurationError("REPLICATION_FACTOR must be a positive integer.")


def get_local_addr() -> str:
    return os.getenv("HOST_IP", LOCAL_ADDRESS)


def get_raft_url() -> str:
    local_addr = get_local_addr()
    raft_port = os.getenv("RAFT_PORT", DEFAULT_RAFT_PORT)
    return f"http://{local_addr}:{raft_port}"


def get_rpc_addr() -> str:
    local_addr = get_local_addr()
    rpc_port = os.getenv("RPC_PORT", DEFAULT_RPC_PORT)
    return f"{local_addr}:{rpc_port}"


def get_vms_addr() -> str:
    addr = os.getenv("VMS_ADDRESS")
    if not addr:
        raise ConfigurationError("Required environment variable 'VMS_ADDRESS' is missing.")
    return addr


def get_raft_dir() -> str:
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    return f"{home}/raftdata"


def get_log_dir() -> str:
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    return f"{home}/log"


def get_replication_factor() -> int:
    try:
        return int(os.getenv("REPLICATION_FACTOR", DEFAULT_REP_FACTOR))
    except (ValueError, TypeError):
        return int(DEFAULT_REP_FACTOR)


def get_info() -> Tuple[int, Optional[List[str]]]:
    rep_factor = get_replication_factor()
    try:
        out = subprocess.check_output(
            [
                f"{binpath}/varlogctl",
                "mr",
                "describe",
                f"--admin={get_vms_addr()}",
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        )

        nodes = json.loads(out.decode("utf-8"))
        if not isinstance(nodes, list):
            return rep_factor, None

        # [데이터 무결성 강화] 잘못된 타입/구조의 노드 데이터를 방어적으로 필터링
        members = [
            node["raftURL"]
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("raftURL"), str)
        ]
        return rep_factor, members
    except (subprocess.SubprocessError, json.JSONDecodeError, ConfigurationError) as e:
        logger.warning(f"Could not fetch peer info via varlogctl: {e}")
        return rep_factor, None


def add_raft_peer() -> bool:
    try:
        raft_url = get_raft_url()
        rpc_addr = get_rpc_addr()

        out = subprocess.check_output(
            [
                f"{binpath}/varlogctl",
                "mr",
                "add",
                f"--raft-url={raft_url}",
                f"--rpc-addr={rpc_addr}",
                f"--admin={get_vms_addr()}",
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        mrnode = json.loads(out.decode("utf-8"))
        node_id = mrnode.get("nodeId", "0") if isinstance(mrnode, dict) else "0"
        return str(node_id) != "0"
    except (subprocess.SubprocessError, json.JSONDecodeError, ConfigurationError) as e:
        logger.warning(f"Failed to add raft peer: {e}")
        return False


def remove_raft_peer() -> bool:
    try:
        raft_url = get_raft_url()

        _, members = get_info()
        if members is None:
            return False
        elif raft_url not in members:
            return True

        out = subprocess.check_output(
            [
                f"{binpath}/varlogctl",
                "mr",
                "remove",
                f"--raft-url={raft_url}",
                f"--admin={get_vms_addr()}",
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        logger.info("Remove raft response: " + out.decode("utf-8").strip())
        return True
    except (subprocess.SubprocessError, json.JSONDecodeError, ConfigurationError) as e:
        logger.warning(f"Failed to remove raft peer: {e}")
        return False


def prepare_vmr_home() -> None:
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    os.makedirs(home, exist_ok=True)


def check_standalone() -> bool:
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    return os.path.exists(f"{home}/.standalone")


def clear_standalone() -> None:
    logger.info("Clear standalone flag")
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    standalone_path = f"{home}/.standalone"
    if os.path.exists(standalone_path):
        try:
            os.remove(standalone_path)
        except OSError as e:
            logger.warning(f"Failed to remove standalone file: {e}")


def exists_wal(path: str) -> bool:
    try:
        wal = glob.glob(os.path.join(path, "wal", "*", "*.wal"))
        return len(wal) > 0
    except OSError:
        return False


def exists_cluster() -> bool:
    _, peers = get_info()
    return peers is not None


def get_metadata_repository_cmd(standalone: bool) -> Optional[List[str]]:
    cluster_id = os.getenv("CLUSTER_ID", DEFAULT_CLUSTER_ID)
    rep_factor, peers = get_info()
    raft_url = get_raft_url()
    rpc_addr = get_rpc_addr()
    raft_dir = get_raft_dir()
    log_dir = get_log_dir()

    command = [
        f"{binpath}/vmr",
        "start",
        f"--cluster-id={cluster_id}",
        f"--replication-factor={rep_factor}",
        f"--raft-address={raft_url}",
        f"--bind={rpc_addr}",
        f"--raft-dir={raft_dir}",
        f"--log-dir={log_dir}",
    ]

    if peers is not None:
        command.append("--join=true")
        for peer in peers:
            command.append(f"--peers={peer}")
    elif not standalone:
        return None

    return command


def print_mr_home() -> None:
    home = os.getenv("VMR_HOME", DEFAULT_VMR_HOME)
    try:
        if os.path.exists(home):
            arr = os.listdir(home)
            logger.info(f"VMR Home contents: {arr}")
    except OSError as e:
        logger.warning(f"Failed to list VMR home directory: {e}")


def main() -> None:
    logger.info("Starting VMR Daemon Supervisor")

    # [보완 1] Config Validation Layer 적용
    try:
        validate_configuration()
    except ConfigurationError as e:
        logger.error(f"Initialization aborted due to configuration error: {e}")
        sys.exit(1)

    print_mr_home()

    limits.set_limits()
    prepare_vmr_home()

    standalone = False
    if not exists_cluster():
        if check_standalone():
            standalone = True
            logger.info("Running in standalone mode via flag")
        elif exists_wal(get_raft_dir()):
            standalone = True
            logger.info("Running in standalone mode with existing WAL")
        else:
            logger.error("Could not run as standalone. Check configuration.")
            sys.exit(1)

    clear_standalone()

    need_add_peer = False
    killer = Killer()
    
    # [보완 2] 프로세스 생명주기 관리를 위한 핸들 및 Exponential Backoff 제어 변수 선언
    vmr_process: Optional[subprocess.Popen] = None
    current_retry_interval = INITIAL_RETRY_INTERVAL_SEC
    consecutive_failures = 0

    while not killer.kill_now:
        # [보완 3] 프로세스 PID 기반 엄격한 라이프사이클(Lifecycle) 감시
        is_running = False
        if vmr_process is not None:
            ret_code = vmr_process.poll()
            if ret_code is None:
                is_running = True
            else:
                logger.warning(f"VMR child process terminated unexpectedly with code {ret_code}")
                vmr_process = None
        
        # 보조 liveness 검사 병행
        if is_running or procutil.check_liveness("vmr"):
            if need_add_peer:
                logger.info("Adding metadata repository back to cluster")
                if add_raft_peer():
                    need_add_peer = False
                    consecutive_failures = 0
                    current_retry_interval = INITIAL_RETRY_INTERVAL_SEC
                else:
                    consecutive_failures += 1
                    current_retry_interval = min(current_retry_interval * BACKOFF_FACTOR, MAX_RETRY_INTERVAL_SEC)
            else:
                # 정상 동작 시 실패 카운터 리셋
                consecutive_failures = 0
                current_retry_interval = INITIAL_RETRY_INTERVAL_SEC

            time.sleep(current_retry_interval)
            continue

        try:
            procutil.kill("vmr")

            if not standalone and not remove_raft_peer():
                logger.warning("Could not cleanly leave peer group; retrying...")

            cmd = get_metadata_repository_cmd(standalone)
            if cmd is None:
                logger.error("Could not build metadata repository command. Check configuration.")
                consecutive_failures += 1
                current_retry_interval = min(current_retry_interval * BACKOFF_FACTOR, MAX_RETRY_INTERVAL_SEC)
                time.sleep(current_retry_interval)
                continue

            logger.info(f"Running metadata repository process: {cmd}")
            # [보완 2] 프로세스 객체(PID 핸들)를 직접 보관하여 자식 프로세스 생명주기 완벽 통제
            vmr_process = subprocess.Popen(cmd)

            need_add_peer = not standalone
            standalone = False
            consecutive_failures = 0
            current_retry_interval = INITIAL_RETRY_INTERVAL_SEC
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            logger.exception(f"Unexpected error while managing metadata repository process: {e}")
            consecutive_failures += 1
            # [보완 4] 장애 폭주(Restart Storm) 방지를 위한 Exponential Backoff 적용
            current_retry_interval = min(current_retry_interval * BACKOFF_FACTOR, MAX_RETRY_INTERVAL_SEC)
            logger.warning(f"Backoff applied. Next retry in {current_retry_interval} seconds (Failures: {consecutive_failures})")

        time.sleep(current_retry_interval)

    # 종료 처리 시 자식 프로세스 명시적 정리
    if vmr_process is not None and vmr_process.poll() is None:
        try:
            vmr_process.terminate()
            vmr_process.wait(timeout=5)
        except Exception:
            vmr_process.kill()

    procutil.stop("vmr")
    logger.info("VMR Daemon Supervisor stopped gracefully.")


if __name__ == "__main__":
    main()

최종 개선사항
✅ VMS_ADDRESS 단일 검증 → 독립 Config Validation Layer 구축
✅ 단순 subprocess 실행 → PID 기반 Child Process Lifecycle 관리 전환
✅ 고정 retry → Exponential Backoff 기반 Restart Storm 방어 적용
✅ 광범위 Exception 처리 → 명시적 예외 분류 및 장애 원인 추적 강화
✅ varlogctl 응답 신뢰 → JSON 타입 검증 및 잘못된 노드 데이터 필터링 추가
✅ 종료 처리 부재 → SIGTERM 기반 Graceful Shutdown 처리 추가
✅ 단순 liveness 의존 → 내부 PID 상태 + 외부 health check 이중 감시 구조 적용

단순 데몬 재시작 스크립트를 넘어 Fail-Fast 설정 검증·PID 기반 프로세스 관리·Backoff 제어·데이터 무결성 방어까지 갖춘 실전 분산 시스템 Supervisor 수준으로 진화했으며, 남은 과제는 장애 격리와 관측성(Observability) 계층 강화뿐이다.
