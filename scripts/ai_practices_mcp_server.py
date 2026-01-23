#!/usr/bin/env python3
"""
AI-Practices 自定义 MCP 服务器

这是一个为 AI-Practices 项目定制的 MCP 服务器，提供项目特定的工具和功能。
"""

from fastmcp import FastMCP
from pathlib import Path
import json
import subprocess
import sys

# 创建 MCP 服务器实例
mcp = FastMCP("AI-Practices Custom Server")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@mcp.tool()
def analyze_project_structure() -> dict:
    """
    分析 AI-Practices 项目结构
    
    Returns:
        dict: 项目结构统计信息
    """
    stats = {
        "modules": [],
        "total_files": 0,
        "python_files": 0,
        "notebook_files": 0,
        "test_files": 0
    }
    
    # 扫描模块目录
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and item.name.startswith(('0', '1')):
            module_info = {
                "name": item.name,
                "submodules": []
            }
            
            # 扫描子模块
            for subitem in item.iterdir():
                if subitem.is_dir():
                    module_info["submodules"].append(subitem.name)
            
            stats["modules"].append(module_info)
    
    # 统计文件
    for py_file in PROJECT_ROOT.rglob("*.py"):
        stats["total_files"] += 1
        stats["python_files"] += 1
        if "test" in py_file.name.lower():
            stats["test_files"] += 1
    
    for nb_file in PROJECT_ROOT.rglob("*.ipynb"):
        stats["total_files"] += 1
        stats["notebook_files"] += 1
    
    return stats


@mcp.tool()
def run_module_tests(module_name: str) -> dict:
    """
    运行指定模块的测试
    
    Args:
        module_name: 模块名称，例如 "07-reinforcement-learning"
    
    Returns:
        dict: 测试结果
    """
    module_path = PROJECT_ROOT / module_name
    
    if not module_path.exists():
        return {
            "success": False,
            "error": f"Module {module_name} not found"
        }
    
    try:
        # 运行 pytest
        result = subprocess.run(
            ["pytest", str(module_path), "-v"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Test execution timeout (5 minutes)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def get_module_readme(module_name: str) -> str:
    """
    获取模块的 README 内容
    
    Args:
        module_name: 模块名称
    
    Returns:
        str: README 内容
    """
    readme_path = PROJECT_ROOT / module_name / "README.md"
    
    if not readme_path.exists():
        return f"README not found for module {module_name}"
    
    return readme_path.read_text(encoding="utf-8")


@mcp.tool()
def list_notebooks(module_name: str = None) -> list:
    """
    列出项目中的 Jupyter notebooks
    
    Args:
        module_name: 可选，指定模块名称
    
    Returns:
        list: notebook 文件列表
    """
    if module_name:
        search_path = PROJECT_ROOT / module_name
    else:
        search_path = PROJECT_ROOT
    
    notebooks = []
    for nb_file in search_path.rglob("*.ipynb"):
        if ".ipynb_checkpoints" not in str(nb_file):
            notebooks.append({
                "path": str(nb_file.relative_to(PROJECT_ROOT)),
                "name": nb_file.name,
                "size": nb_file.stat().st_size
            })
    
    return notebooks


@mcp.tool()
def check_dependencies() -> dict:
    """
    检查项目依赖状态
    
    Returns:
        dict: 依赖检查结果
    """
    try:
        # 检查 pip 依赖
        result = subprocess.run(
            ["pip", "check"],
            capture_output=True,
            text=True
        )
        
        pip_status = {
            "success": result.returncode == 0,
            "output": result.stdout if result.returncode == 0 else result.stderr
        }
        
        # 检查已安装的包
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        
        installed_packages = json.loads(result.stdout) if result.returncode == 0 else []
        
        return {
            "pip_check": pip_status,
            "installed_packages_count": len(installed_packages),
            "installed_packages": installed_packages[:20]  # 只返回前 20 个
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def get_project_stats() -> dict:
    """
    获取项目统计信息
    
    Returns:
        dict: 项目统计
    """
    stats = {
        "total_modules": 0,
        "total_python_files": 0,
        "total_notebooks": 0,
        "total_tests": 0,
        "total_lines_of_code": 0
    }
    
    # 统计模块
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and item.name.startswith(('0', '1')):
            stats["total_modules"] += 1
    
    # 统计文件和代码行数
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if ".venv" not in str(py_file) and "node_modules" not in str(py_file):
            stats["total_python_files"] += 1
            try:
                lines = py_file.read_text(encoding="utf-8").count('\n')
                stats["total_lines_of_code"] += lines
            except:
                pass
            
            if "test" in py_file.name.lower():
                stats["total_tests"] += 1
    
    # 统计 notebooks
    for nb_file in PROJECT_ROOT.rglob("*.ipynb"):
        if ".ipynb_checkpoints" not in str(nb_file):
            stats["total_notebooks"] += 1
    
    return stats


@mcp.tool()
def search_code(query: str, file_extension: str = "py") -> list:
    """
    在项目代码中搜索
    
    Args:
        query: 搜索关键词
        file_extension: 文件扩展名，默认 "py"
    
    Returns:
        list: 搜索结果
    """
    results = []
    pattern = f"*.{file_extension}"
    
    for file_path in PROJECT_ROOT.rglob(pattern):
        if ".venv" in str(file_path) or "node_modules" in str(file_path):
            continue
        
        try:
            content = file_path.read_text(encoding="utf-8")
            if query.lower() in content.lower():
                # 找到匹配的行
                lines = content.split('\n')
                matching_lines = [
                    (i + 1, line) for i, line in enumerate(lines)
                    if query.lower() in line.lower()
                ]
                
                results.append({
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "matches": len(matching_lines),
                    "lines": matching_lines[:5]  # 只返回前 5 个匹配
                })
        except:
            pass
    
    return results[:20]  # 只返回前 20 个文件


@mcp.tool()
def get_module_dependencies(module_name: str) -> dict:
    """
    分析模块的依赖关系
    
    Args:
        module_name: 模块名称
    
    Returns:
        dict: 依赖分析结果
    """
    module_path = PROJECT_ROOT / module_name
    
    if not module_path.exists():
        return {
            "success": False,
            "error": f"Module {module_name} not found"
        }
    
    imports = set()
    
    # 扫描所有 Python 文件
    for py_file in module_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    # 提取导入的包名
                    if line.startswith('import '):
                        pkg = line.replace('import ', '').split()[0].split('.')[0]
                    else:
                        pkg = line.replace('from ', '').split()[0].split('.')[0]
                    
                    imports.add(pkg)
        except:
            pass
    
    return {
        "module": module_name,
        "dependencies": sorted(list(imports)),
        "count": len(imports)
    }


@mcp.resource("project://structure")
def get_project_structure() -> str:
    """
    获取项目结构的文本表示
    """
    structure = ["AI-Practices/"]
    
    for item in sorted(PROJECT_ROOT.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            structure.append(f"├── {item.name}/")
            
            # 列出子目录
            try:
                for subitem in sorted(item.iterdir())[:5]:  # 只显示前 5 个
                    if subitem.is_dir():
                        structure.append(f"│   ├── {subitem.name}/")
            except:
                pass
    
    return "\n".join(structure)


@mcp.resource("project://readme")
def get_main_readme() -> str:
    """
    获取项目主 README
    """
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "README.md not found"


if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run()
