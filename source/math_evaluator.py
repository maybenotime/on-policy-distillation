import re
import json
from typing import List, Dict, Any, Tuple, Optional
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig

class MathSolutionEvaluator:
    def __init__(self):
        #对于模型输出使用两种配置
        self.prediction_configs = [
            LatexExtractionConfig(),
            ExprExtractionConfig()
        ]

        # 对于标准答案，根据格式选择配置
        self.gold_latex_config = [LatexExtractionConfig()]
        self.gold_expr_config = [ExprExtractionConfig()]

    def extract_boxed_answers(self, text: str) -> List[str]:
        """
        提取所有 \\boxed{} 中的答案
        """
        boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(boxed_pattern, text)
        return matches

    def extract_final_numerical_answers(self, text: str) -> List[str]:
        """
        提取文本末尾的数值答案
        """
        patterns = [
            r'(\d+(?:\.\d+)?)\s*km/h',  # 带单位的数值
            r'(\d+(?:\.\d+)?)\s*(?:km|m|cm|kg|g)(?:\s|$)',  # 其他单位
            r'(\d+(?:\.\d+)?)\s*$',  # 纯数值
            r'=\s*(\d+(?:\.\d+)?)',  # 等号后的数值
        ]

        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            results.extend(matches)

        return results

    def evaluate_single_solution(self, model_output: str, gold_answer: str,
                                problem_type: str = "auto") -> Dict[str, Any]:
        """
        评估单个解题过程

        Args:
            model_output: 模型的完整输出（包含解题步骤）
            gold_answer: 标准答案
            problem_type: 问题类型 ("latex", "number", "auto")
        """
        result = {
            "model_output": model_output,
            "gold_answer": gold_answer,
            "extracted_answers": [],
            "verification_results": [],
            "final_correct": False,
            "parsing_errors": []
        }

        try:
            # 1. 从模型输出中提取答案
            extracted_answers = []

            # 优先提取 \boxed{} 答案
            boxed_answers = self.extract_boxed_answers(model_output)
            extracted_answers.extend(boxed_answers)

            # 如果没有 boxed 答案，尝试其他方法
            if not boxed_answers:
                numerical_answers = self.extract_final_numerical_answers(model_output)
                extracted_answers.extend(numerical_answers)

            result["extracted_answers"] = extracted_answers

            # 2. 解析标准答案
            if problem_type == "auto":
                # 自动判断答案类型
                if "$" in gold_answer or "\\" in gold_answer:
                    gold_configs = self.gold_latex_config
                else:
                    gold_configs = self.gold_expr_config
            elif problem_type == "latex":
                gold_configs = self.gold_latex_config
            else:
                gold_configs = self.gold_expr_config

            try:
                gold_parsed = parse(gold_answer, extraction_config=gold_configs)
            except Exception as e:
                result["parsing_errors"].append(f"Gold parsing error: {e}")
                return result

            # 3. 验证每个提取的答案
            for answer in extracted_answers:
                try:
                    # 使用 math_verify 解析模型答案
                    answer_parsed = parse(str(answer), extraction_config=self.prediction_configs)
                    is_correct = verify(gold_parsed, answer_parsed)

                    result["verification_results"].append({
                        "extracted_answer": answer,
                        "correct": is_correct
                    })

                    # 如果有任何一个答案正确，就认为整体正确
                    if is_correct:
                        result["final_correct"] = True

                except Exception as e:
                    result["parsing_errors"].append(f"Answer '{answer}' parsing error: {e}")
                    result["verification_results"].append({
                        "extracted_answer": answer,
                        "correct": False,
                        "error": str(e)
                    })

        except Exception as e:
            result["parsing_errors"].append(f"General error: {e}")

        return result
