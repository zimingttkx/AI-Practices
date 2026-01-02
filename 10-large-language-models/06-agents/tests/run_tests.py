"""
06-agents 模块测试运行器

运行所有单元测试并生成报告。

使用方法：
    python run_tests.py
    python run_tests.py -v  # 详细输出
    python run_tests.py --cov  # 覆盖率报告

作者: AI-Practices
许可证: MIT
"""

import subprocess
import sys
import os


def main():
    """运行测试主函数。"""
    # 获取当前目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(test_dir)
    
    # 构建pytest命令
    cmd = [sys.executable, "-m", "pytest"]
    
    # 添加测试目录
    cmd.append(test_dir)
    
    # 添加命令行参数
    cmd.extend(sys.argv[1:])
    
    # 如果没有指定详细程度，默认使用-v
    if "-v" not in sys.argv and "--verbose" not in sys.argv:
        cmd.append("-v")
    
    # 运行测试
    print(f"运行测试: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=project_dir)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
