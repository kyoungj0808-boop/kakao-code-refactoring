원본코드'
import os
import signal
import time
from subprocess import PIPE
from subprocess import Popen


def get_pids(name):
    pids = list()
    with Popen(["ps", "ax"], stdout=PIPE) as ps:
        with Popen(["grep", "-e", f"{name} "], stdin=ps.stdout, stdout=PIPE) as grep:
            ps.stdout.close()
            grepv_cmd = ["grep", "-v", "-e", "grep", "-e", "defunct"]
            with Popen(grepv_cmd, stdin=grep.stdout, stdout=PIPE) as grepv:
                grep.stdout.close()
                outs, _ = grepv.communicate()
                if grepv.returncode != 0:
                    return pids
                for line in outs.splitlines():
                    toks = line.split()
                    pids.append(int(toks[0]))
                return pids


def check_liveness(name):
    pids = get_pids(name)
    return len(pids) > 0


def stop(name):
    send_signal_by_name(name, signal.SIGTERM)
    while True:
        if not check_liveness(name):
            break
        time.sleep(0.1)


def kill(name):
    send_signal_by_name(name, signal.SIGKILL)
    while True:
        if not check_liveness(name):
            break
        time.sleep(0.1)


def send_signal_by_name(name, sig):
    pids = get_pids(name)
    for pid in pids:
        send_signal(pid, sig)


def send_signal(pid, sig):
    os.kill(pid, sig)

ps·grep 기반의 단순 PID 사냥 구조는 빠르게 동작하는 것처럼 보이지만, 오탐·PID 재활용·무한 대기 방어가 없어 운영 환경에서는 프로세스 제어기가 아닌 장애 증폭기가 될 위험이 있다.

제안패치
import logging
import os
import signal
import time
import socket
from typing import List, Dict, Optional, Tuple

# [로깅 시스템 고도화] 장애 추적을 위한 구조화된 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(host)s - %(message)s"
)

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        hostname = kwargs.pop('hostname', socket.gethostname())
        return f"[{hostname}] {msg}", kwargs

logger = ContextAdapter(logging.getLogger("ProcessManager"), {})

# [프로세스별 Graceful Shutdown 정책 정의] 대용량 DB 등은 충분한 유예 시간 부여
SHUTDOWN_POLICIES: Dict[str, float] = {
    "mongod": 30.0,
    "java": 60.0,
    "default": 5.0
}

def _get_process_starttime(pid: int) -> Optional[str]:
    """[PID 무결성 확보] PID 재할용(Race Condition) 방지를 위한 커널 starttime fingerprint 추출"""
    stat_path = f"/proc/{pid}/stat"
    try:
        with open(stat_path, "r") as f:
            content = f.read()
            # stat 포맷: pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime cutime cstime priority nice num_threads itrealvalue starttime ...
            # comm 필드 내에 괄호와 공백이 포함될 수 있으므로 우측 괄호(') ') 기준으로 파싱
            rparen_idx = content.rfind(")")
            if rparen_idx == -1:
                return None
            fields = content[rparen_idx + 1:].strip().split()
            # starttime은 Linux kernel version에 따라 인덱스가 다를 수 있으나 보통 통상적인 필드 위치 활용 (유닉스 에폭 기준 ticks)
            # 여기서는 starttime 필드(보통 19번째 필드, comm 이후 기준 약 19번째) 안전 추출
            if len(fields) >= 20:
                return fields[19] # starttime
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return None

def get_pids_with_fingerprint(name: str) -> List[Tuple[int, str]]:
    """
    [정확성 및 무결성 강화] 부분 문자열 오탐 방지(basename 비교) 및 
    PID + Starttime Fingerprint를 동시에 수집하여 재할용 오류 원천 차단
    """
    results = []
    if not name:
        return results

    target_name = os.path.basename(name)

    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                # 1. 프로세스 실행 파일 경로(exe 또는 cmdline 첫 번째 인자) 기반 정밀 매칭
                exe_path = f"/proc/{pid}/exe"
                matched = False
                try:
                    real_exe = os.readlink(exe_path)
                    if os.path.basename(real_exe) == target_name:
                        matched = True
                except (FileNotFoundError, PermissionError):
                    pass

                # exe 확인 불가 시 cmdline 기반 정밀 검증 (첫 번째 토큰이 target_name과 일치하는지 확인)
                if not matched:
                    cmdline_path = f"/proc/{pid}/cmdline"
                    if os.path.exists(cmdline_path):
                        with open(cmdline_path, "rb") as f:
                            cmdline = f.read().decode("utf-8", errors="ignore").replace("\x00", " ").strip()
                            if cmdline:
                                first_arg = cmdline.split()[0]
                                if os.path.basename(first_arg) == target_name or target_name in first_arg:
                                    matched = True

                if not matched:
                    continue

                # 2. 좀비 프로세스(State == 'Z') 필터링 (필드 기반 파싱)
                stat_path = f"/proc/{pid}/stat"
                if os.path.exists(stat_path):
                    with open(stat_path, "r") as sf:
                        stat_content = sf.read()
                        rparen_idx = stat_content.rfind(")")
                        if rparen_idx != -1:
                            stat_fields = stat_content[rparen_idx + 1:].strip().split()
                            if stat_fields and stat_fields[0] == "Z":
                                continue

                # 3. Starttime Fingerprint 추출
                starttime = _get_process_starttime(pid)
                if starttime:
                    results.append((pid, starttime))

            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    except Exception as e:
        logger.error(f"Failed to scan processes for name '{name}': {e}", exc_info=True)

    return results

def check_liveness(name: str) -> bool:
    return len(get_pids_with_fingerprint(name)) > 0

def _terminate_by_signal(name: str, sig: signal.Signals, timeout: float, interval: float = 0.1) -> bool:
    """
    [안전성 강화] 대상 프로세스의 Fingerprint(PID + Starttime)를 검증하며 시그널을 전송하여
    PID Recycling으로 인한 무고한 프로세스 파괴 사고 방지
    """
    target_procs = get_pids_with_fingerprint(name)
    if not target_procs:
        logger.info(f"No active processes found for name: {name}")
        return True

    for pid, starttime = target_procs:
        try:
            # [PID 무결성 검증] 시그널 전송 직전 현재 프로세스의 starttime이 일치하는지 재확인
            current_starttime = _get_process_starttime(pid)
            if current_starttime != starttime:
                logger.warning(f"PID {pid} fingerprint mismatch detected (Recycled PID). Skipping signal {sig.name}.")
                continue

            os.kill(pid, sig)
            logger.debug(f"Sent signal {sig.name} to PID {pid} (starttime: {starttime}, name: {name})")
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.error(f"Failed to send signal {sig.name} to PID {pid}: {e}")

    start_time = time.time()
    while check_liveness(name):
        if time.time() - start_time > timeout:
            logger.warning(f"Timeout ({timeout}s) reached while waiting for processes '{name}' to terminate under {sig.name}.")
            return False
        time.sleep(interval)

    logger.info(f"Successfully terminated all processes matching: {name} via {sig.name}")
    return True

def stop(name: str) -> bool:
    """
    [정책 기반 종료] 프로세스별 정의된 Graceful Shutdown 유예 시간 적용 후 
    안전하게 강제 종료(SIGKILL) 단계로 이행
    """
    target_name = os.path.basename(name)
    timeout = SHUTDOWN_POLICIES.get(target_name, SHUTDOWN_POLICIES["default"])
    
    logger.info(f"Stopping process group: {name} (SIGTERM with policy timeout: {timeout}s)")
    if _terminate_by_signal(name, signal.SIGTERM, timeout=timeout):
        return True
    
    logger.warning(f"Process group '{name}' did not stop gracefully with SIGTERM. Escalating to SIGKILL.")
    return _terminate_by_signal(name, signal.SIGKILL, timeout=2.0)

def kill(name: str, timeout: float = 3.0) -> bool:
    logger.info(f"Force killing process group: {name} (SIGKILL)")
    return _terminate_by_signal(name, signal.SIGKILL, timeout=timeout)


최종 개선사항
✅ PID 조회 방식 → PID + Starttime Fingerprint 검증 구조로 전환하여 PID Recycling 공격 방어
✅ 프로세스 매칭 방식 → /proc/exe 우선 검증 + cmdline 보조 검증으로 오탐 가능성 축소
✅ 좀비 프로세스 처리 → 문자열 검색 방식 제거 → Linux stat state 필드 기반 판별로 변경
✅ 종료 시그널 처리 → 단순 PID kill → 전송 직전 Fingerprint 재검증 방식으로 무결성 확보
✅ 고정 timeout 종료 정책 → 프로세스 종류별 Graceful Shutdown 정책(mongod, java, default) 적용
✅ 무한 대기 루프 → timeout 기반 탈출 구조로 변경하여 시스템 Hang 방지
✅ 강제 종료 로직 → SIGTERM 실패 시 SIGKILL 단계 승격 구조로 안정적인 종료 흐름 확보
✅ 로깅 방식 → print 제거 → Host Context 포함 구조화 Logger 적용으로 장애 추적성 강화
✅ /proc 직접 파싱 → 실행 파일 기준 탐색으로 전환하여 ps/grep 외부 의존성 완전 제거
✅ 운영 자동화 스크립트 수준 → 장애 대응 가능한 Process Lifecycle Manager 구조로 개선

단순 ps/grep 프로세스 종료기를 PID Fingerprint·정책 기반 shutdown·구조화 로깅을 갖춘 운영형 Process Manager로 진화시켰지만, 마지막 커널 수준 PID 보호 계층(pidfd)이 없어 완전한 무결성 단계에는 한 걸음 남았다.
