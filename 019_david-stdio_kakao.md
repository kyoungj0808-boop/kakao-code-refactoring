원본코드
import paramiko

DEBUG_MODE = False

class ServerCommand:
    def __init__(self, host):
        self.host = host
        self.user = "root"
        self.debug = DEBUG_MODE

    def _executor(self, command):
        try:
            if self.debug:
                print(self.host, command)
            ssh=paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname = self.host, username = self.user)
            try:
                stdin, stdout, stderr = ssh.exec_command(command)
            except Exception as e:
                print(f"connection is failed {e}")

            rtnValue = stdout.channel.recv_exit_status()
            retOut = stdout.read().decode('utf-8')
            retErr = stderr.read().decode('utf-8')

            if self.debug:
                print(rtnValue, retOut, retErr)

            if self.debug and rtnValue == 0:
                print(f"command is successfully executed, {command} ")
        except Exception as e:
            print(f"exception occur {e}")
        finally:
            ssh.close()

    def dependency(self):
        self._executor("yum install snappy-devel libzstd-devel zlib-devel -y")

    def rsync(self, hostname, mongo_path, db, mypath, mydb):
        self._executor(f"rsync --bwlimit=150M -av --no-perms -e \"ssh -o StrictHostKeyChecking=no\" root@{hostname}:{mongo_path}/{db} {mypath}/{mydb}")

    def salvage(self, dbpath, colpath):
        self._executor(f"/tmp/wt -v -h {dbpath} -R salvage {colpath}")

    def start(self, command: str): #TODO
        if command:
            self._executor(command)
        else:
            self._executor("systemctl restart mongod")

SSH 원격 제어라는 핵심 인프라 코드를 예외·보안·자원 생명주기 검증 없이 작성해, 작은 네트워크 장애가 세션 누수와 운영 장애로 전파되는 취약한 관리 스크립트다.

제안패차
import logging
import os
import shlex
from typing import Optional, Tuple
import paramiko

# [로깅 시스템 전환] 표준 print 대신 운영 환경에 적합한 logging 모듈 구성
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ServerCommand")

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "t")

class ServerCommand:
    def __init__(self, host: str, user: Optional[str] = None, port: int = 22, timeout: float = 10.0):
        self.host = host
        self.user = user or os.getenv("SSH_USER", "root")
        self.port = port
        self.timeout = timeout
        self.debug = DEBUG_MODE
        if self.debug:
            logger.setLevel(logging.DEBUG)

    def _get_ssh_client(self) -> paramiko.SSHClient:
        ssh = paramiko.SSHClient()
        # [보안 강화] MITM 방지를 위한 RejectPolicy 적용
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        
        try:
            ssh.load_system_host_keys()
        except Exception:
            pass

        key_filename = os.getenv("SSH_KEY_PATH", None)
        try:
            ssh.connect(
                hostname=self.host,
                username=self.user,
                port=self.port,
                key_filename=key_filename,
                timeout=self.timeout,
                allow_agent=True,
                look_for_keys=True
            )
        except Exception as e:
            logger.error(f"Failed to establish SSH connection to {self.host}:{self.port} - {e}")
            raise ConnectionError(f"Failed to establish SSH connection to {self.host}:{self.port} - {e}") from e
        
        return ssh

    def _executor(self, command: str) -> Tuple[int, str, str]:
        logger.debug(f"Host: {self.host} | Command: {command}")

        ssh = None
        try:
            ssh = self._get_ssh_client()
            stdin, stdout, stderr = ssh.exec_command(command, timeout=self.timeout)
            
            rtn_value = stdout.channel.recv_exit_status()
            ret_out = stdout.read().decode('utf-8', errors='replace')
            ret_err = stderr.read().decode('utf-8', errors='replace')

            logger.debug(f"Exit Code: {rtn_value}\nSTDOUT:\n{ret_out}\nSTDERR:\n{ret_err}")

            if rtn_value != 0:
                logger.warning(f"Command failed on {self.host} with exit code {rtn_value}: {command} | STDERR: {ret_err}")

            return rtn_value, ret_out, ret_err

        except Exception as e:
            logger.error(f"Exception occurred during execution on {self.host}: {e}", exc_info=True)
            raise
        finally:
            if ssh:
                ssh.close()

    def dependency(self):
        self._executor("yum install snappy-devel libzstd-devel zlib-devel -y")

    def rsync(self, hostname: str, mongo_path: str, db: str, mypath: str, mydb: str, bwlimit: int = 150):
        # [보안 강화] 문자열 포맷팅 기반 셸 주입 원천 차단 및 shlex.quote 적용
        # [보안 강화] StrictHostKeyChecking=no 제거 후 known_hosts 검증 수행 구조 반영
        known_hosts_path = os.getenv("SSH_KNOWN_HOSTS", "~/.ssh/known_hosts")
        ssh_opt = f"ssh -o StrictHostKeyChecking=yes -o UserKnownHostsFile={known_hosts_path}"

        safe_hostname = shlex.quote(hostname)
        safe_mongo_path = shlex.quote(mongo_path)
        safe_db = shlex.quote(db)
        safe_mypath = shlex.quote(mypath)
        safe_mydb = shlex.quote(mydb)
        safe_ssh_opt = shlex.quote(ssh_opt)
        safe_user = shlex.quote(self.user)

        cmd = (
            f"rsync --bwlimit={int(bwlimit)}M -av --no-perms "
            f"-e {safe_ssh_opt} "
            f"{safe_user}@{safe_hostname}:{safe_mongo_path}/{safe_db} {safe_mypath}/{safe_mydb}"
        )
        self._executor(cmd)

    def salvage(self, dbpath: str, colpath: str):
        safe_dbpath = shlex.quote(dbpath)
        safe_colpath = shlex.quote(colpath)
        self._executor(f"/tmp/wt -v -h {safe_dbpath} -R salvage {safe_colpath}")

    def start(self, command: Optional[str] = None):
        if command:
            self._executor(command)
        else:
            self._executor("systemctl restart mongod")

최종개선사항
✅ print 기반 출력 제거 → logging 기반 운영 추적 체계 전환
✅ AutoAddPolicy 제거 → RejectPolicy + known_hosts 검증 구조 적용
✅ SSH 연결 예외 은닉 제거 → ConnectionError 원인 추적 가능 구조 전환
✅ exec_command 미할당 오류 제거 → 안전한 SSH 세션 생명주기 관리 적용
✅ rsync StrictHostKeyChecking=no 제거 → MITM 방어 가능한 SSH 검증 구조 전환
✅ f-string 직접 명령 조합 제거 → shlex.quote 기반 Shell Injection 방어 적용
✅ SSH 사용자 하드코딩 제거 → 환경 변수 및 외부 주입 방식 지원
✅ 반환값 없는 실행 구조 개선 → exit code/stdout/stderr 반환 기반 상태 검증 가능
✅ 경로 정규화 의존 제거 → shell escape 기반 입력 무결성 강화

SSH 연결·인증·쉘 주입·로그 추적의 핵심 방어선을 모두 구축했지만, 마지막 임의 command 실행 인터페이스가 남아 있어 운영 자동화 시스템의 최종 방화벽이 완전히 닫히지는 않았다.
