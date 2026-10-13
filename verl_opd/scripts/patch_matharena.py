import re, io
G='/home/sya225/opd/matharena/src/matharena/grader.py'
s=open(G).read()
assert 'def _extract_and_grade_impl' not in s
s=s.replace('def extract_and_grade(messages, output_tokens, gold_answer, competition_config, problem=None, debug_info=""):',
'''import signal as _signal, threading as _threading

_MAX_ANS_CHARS = 3000      # a \\boxed{...} tail longer than this is never a real answer; sympy/regex parsing of it can eat 100s of GB
_GRADE_TIMEOUT_S = 120     # hard cap on one extract+grade call (main thread only)


class _GradeTimeout(Exception):
    pass


def extract_and_grade(messages, output_tokens, gold_answer, competition_config, problem=None, debug_info=""):
    """Guarded wrapper (added 2026-09-17): length cap + hard timeout + never let a grading blow-up kill the run."""
    try:
        last = (messages[-1].get("content") or "") if messages else ""
        if isinstance(last, str):
            i = max(last.rfind("\\\\boxed"), last.rfind("\\\\fbox"))
            if i >= 0 and len(last) - i > _MAX_ANS_CHARS:
                logger.warning(f"[{debug_info}] boxed tail is {len(last) - i} chars (> {_MAX_ANS_CHARS}); treated as no answer")
                return None, False, WarningType.MAJOR.value
    except Exception as e:  # noqa E722
        logger.warning(f"[{debug_info}] length guard failed: {e}")
    use_alarm = _threading.current_thread() is _threading.main_thread()
    old = None
    if use_alarm:
        def _h(signum, frame):
            raise _GradeTimeout()
        old = _signal.signal(_signal.SIGALRM, _h); _signal.alarm(_GRADE_TIMEOUT_S)
    try:
        return _extract_and_grade_impl(messages, output_tokens, gold_answer, competition_config, problem=problem, debug_info=debug_info)
    except (_GradeTimeout, MemoryError, RecursionError) as e:
        logger.error(f"[{debug_info}] grading aborted ({type(e).__name__}); run scored as incorrect")
        return None, False, WarningType.MAJOR.value
    finally:
        if use_alarm:
            _signal.alarm(0); _signal.signal(_signal.SIGALRM, old)


def _extract_and_grade_impl(messages, output_tokens, gold_answer, competition_config, problem=None, debug_info=""):''',1)
open(G,'w').write(s)
R='/home/sya225/opd/matharena/scripts/run.py'
r=open(R).read()
assert 'RLIMIT_AS' not in r
r=r.replace('import argparse\n','import argparse\nimport os, resource\n_cap = int(os.environ.get("MATHARENA_MEM_GB", "64")) << 30   # 2026-09-17: cap the grader process; a runaway parse must not take the node down\ntry:\n    resource.setrlimit(resource.RLIMIT_AS, (_cap, _cap))\nexcept Exception as _e:\n    print("RLIMIT_AS not set:", _e)\n',1)
open(R,'w').write(r)
print("patched")
