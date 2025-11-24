#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Any
import re
import multiprocessing
from multiprocessing.connection import Connection

ExecuteResult = Tuple[bool, str, Tuple[bool]]


def extract_python_code(text_string: str) -> List[str]:

    code_blocks = re.findall(r"```python(.*?)```", text_string, re.DOTALL)
    if not code_blocks:
        code_blocks = [text_string]

    results = []
    for block in code_blocks:
        funcs = re.findall(r"(def\s+\w+\(.*?:[\s\S]*?)(?=^def\s|\Z)", block.strip(), re.MULTILINE)
        for func in funcs:
            results.append(func.strip())

    return results

def _exec_code_and_capture(code: str, conn: Connection):

    try:
        local_ns = {}
        exec(code, local_ns)

        for name, func in local_ns.items():
            if callable(func) and name.startswith("test_"):
                func()
        conn.send(True)  
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()

class PyExecutor:
    def _run_with_timeout(self, code: str, timeout: int) -> Any:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_exec_code_and_capture, args=(code, child_conn))
        
        p.start()
        p.join(timeout)  
        
        if p.is_alive():
            p.kill()
            p.join()  
            raise TimeoutError("Test execution timed out")

        if parent_conn.poll():
            result = parent_conn.recv()
            if isinstance(result, Exception):
                raise result
            return result
        else:
            raise RuntimeError("Child process terminated unexpectedly without sending a result.")

    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        success_tests = []
        failed_tests = []
        is_passing = True

        for test_code in tests:
            cleaned_test = re.sub(r"^\s*from\s+solution\s+import\s+\w+\s*", "", test_code, flags=re.MULTILINE)
            code_to_run = func + "\n" + cleaned_test
            try:
                self._run_with_timeout(code_to_run, timeout)
                success_tests.append(test_code)
            except Exception as e:
                failed_tests.append(f"{test_code}  # output: {e}")
                is_passing = False

        state = tuple(test in success_tests for test in tests)
        feedback = (
            "Tests passed:\n" + "\n".join(success_tests)
            + "\n\nTests failed:\n" + "\n".join(failed_tests)
        )
        return is_passing, feedback, state

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        cleaned_test = re.sub(r"^\s*from\s+solution\s+import\s+\w+\s*", "", test, flags=re.MULTILINE)
        code_to_run = func + "\n" + cleaned_test
        try:
            self._run_with_timeout(code_to_run, timeout)
            return True
        except Exception:
            return False