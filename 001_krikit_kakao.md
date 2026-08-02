원본코드
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
command line part-of-speech tagger demo
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import sys

from khaiii.train.tagger import PosTagger


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    tgr = PosTagger(args.model_dir, args.gpu_num)
    for line_num, line in enumerate(sys.stdin, start=1):
        if line_num % 100000 == 0:
            logging.info('%d00k-th line..', (line_num // 100000))
        line = line.rstrip('\r\n')
        if not line:
            print()
            continue
        pos_sent = tgr.tag_raw(line)
        for pos_word in pos_sent.pos_tagged_words:
            print(pos_word.raw, end='\t')
            print(' + '.join([str(m) for m in pos_word.pos_tagged_morphs]))
        print()


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='command line part-of-speech tagger demo')
    parser.add_argument('-m', '--model-dir', help='model dir', metavar='DIR', required=True)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--gpu-num', help='GPU number to use <default: -1 for CPU>', metavar='INT',
                        type=int, default=-1)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

CLI 구조와 스트리밍 처리 설계는 깔끔하지만, sys.stdin/stdout 전역 변조, 리소스 해제 누락, 예외 격리 부재로 인해 프로덕션 환경에서는 안정성과 확장성이 크게 떨어지는 전형적인 데모 수준의 코드다.

제안패치
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line part-of-speech tagger (Production Ready - v2)
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""

###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
import logging
import sys
from typing import TextIO

from khaiii.train.tagger import PosTagger


#############
# functions #
#############
def run(args: Namespace) -> None:
    """
    Run function incorporating advanced production standards:
    - with statements for safe resource management
    - Granular exception handling (UnicodeError, RuntimeError, etc.)
    - logging.exception for automatic traceback capture
    - Generator expression and removed unnecessary per-line flush()
    """
    # with 문과 nullcontext를 활용하여 표준 입출력 및 파일 스트림을 안전하게 통합 관리
    input_cm = open(args.input, 'r', encoding='UTF-8') if args.input else nullcontext(sys.stdin)
    output_cm = open(args.output, 'w', encoding='UTF-8') if args.output else nullcontext(sys.stdout)

    try:
        with input_cm as input_stream, output_cm as output_stream:
            tgr = PosTagger(args.model_dir, args.gpu_num)
            
            for line_num, line in enumerate(input_stream, start=1):
                if line_num % 100000 == 0:
                    logging.info('%d00k-th line processed..', (line_num // 100000))
                
                line = line.rstrip('\r\n')
                if not line:
                    output_stream.write('\n')
                    continue
                
                # 구체적인 예외 유형 분리를 통한 방어적 격리 및 logging.exception 적용
                try:
                    pos_sent = tgr.tag_raw(line)
                    for pos_word in pos_sent.pos_tagged_words:
                        output_stream.write(f"{pos_word.raw}\t")
                        # 불필요한 리스트 생성을 없앤 제너레이터 표현식 적용
                        output_stream.write(' + '.join(str(m) for m in pos_word.pos_tagged_morphs) + '\n')
                    output_stream.write('\n')
                
                except (UnicodeDecodeError, UnicodeEncodeError) as ue:
                    logging.exception("Unicode encoding/decoding error at line %d ('%s'): %s", line_num, line, ue)
                    continue
                except (RuntimeError, ValueError) as re:
                    logging.exception("Runtime or Value error during tagging at line %d ('%s'): %s", line_num, line, re)
                    continue
                except Exception as e:
                    logging.exception("Unexpected error processing line %d ('%s'): %s", line_num, line, e)
                    continue
            
            # 대용량 배치의 I/O 성능 극대화를 위해 루프 내 매 줄 flush 제거 후 최종 종료 시점 반영
            output_stream.flush()

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting gracefully...")
    except Exception as e:
        logging.critical("Critical error during initialization or execution: %s", e)
        sys.exit(1)


########
# main #
########
def main() -> None:
    """
    Main function processes only argument parsing and global logging configuration.
    """
    parser = ArgumentParser(description='Command-line part-of-speech tagger demo (Production Ready v2)')
    parser.add_argument('-m', '--model-dir', help='model dir', metavar='DIR', required=True)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--gpu-num', help='GPU number to use <default: -1 for CPU>', metavar='INT',
                        type=int, default=-1)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(message)s')

    run(args)


if __name__ == '__main__':
    main()

최종 개선사항
✅ sys.stdin/stdout 전역 변조 제거 → nullcontext와 with 문 기반 스트림 관리로 리소스 안전성 향상
✅ finally 기반 수동 close() 제거 → 컨텍스트 매니저를 통한 자동 자원 해제로 코드 단순화
✅ 라인 단위 예외 격리 강화 → 개별 입력 오류가 전체 형태소 분석 파이프라인을 중단하지 않도록 개선
✅ Exception 단일 처리 개선 → UnicodeError, RuntimeError, ValueError 등 예외 유형별 분리 처리 적용
✅ logging.error → logging.exception 전환 → Traceback 자동 기록으로 운영 환경 디버깅 효율 향상
✅ 매 라인 flush() 제거 → 종료 시점 일괄 Flush 방식으로 불필요한 I/O 오버헤드 감소
✅ 리스트 컴프리헨션 제거 → 제너레이터 표현식 사용으로 메모리 할당 최소화
✅ logging 설정 단순화 → 로그 레벨 선택을 단일 변수(log_level)로 통합하여 가독성 향상
✅ 입출력 스트림과 형태소 분석 로직 분리 → CLI 데모 수준에서 운영 가능한 구조로 유지보수성 향상
✅ 배치 처리 안정성 강화 → 장시간·대용량 입력 환경에서도 장애 전파를 최소화하는 방어적 실행 구조 적용

데모 수준의 CLI 형태소 분석기를 운영 환경을 고려한 배치 처리 구조로 개선하여, 리소스 관리·예외 격리·로그 추적성을 모두 강화한 프로덕션 지향 리팩토링이다.
    
