"""
代码助手智能体单元测试。
"""

import numpy as np
import pytest

from src.code_retriever import (
    CodeBlockType,
    CodeChunker,
    CodeDocument,
    CodeEmbedding,
    CodeLanguage,
    CodeRetriever,
    SearchResult,
)
from src.code_agent import (
    CodeAction,
    CodeActionType,
    CodeAgent,
    CodeResult,
)
from src.review_agent import (
    IssueCategory,
    IssueSeverity,
    ReviewAgent,
    ReviewIssue,
    ReviewResult,
)


# =============================================================================
# CodeDocument Tests
# =============================================================================

class TestCodeDocument:
    """CodeDocument测试。"""
    
    def test_create_document(self):
        doc = CodeDocument(content="def foo(): pass")
        assert doc.content == "def foo(): pass"
        assert doc.language == CodeLanguage.UNKNOWN
    
    def test_document_with_metadata(self):
        doc = CodeDocument(
            content="class Bar: pass",
            language=CodeLanguage.PYTHON,
            block_type=CodeBlockType.CLASS,
            name="Bar",
            file_path="test.py",
        )
        assert doc.name == "Bar"
        assert doc.language == CodeLanguage.PYTHON
    
    def test_num_lines(self):
        doc = CodeDocument(content="line1\nline2\nline3")
        assert doc.num_lines == 3
    
    def test_content_hash(self):
        doc1 = CodeDocument(content="same")
        doc2 = CodeDocument(content="same")
        assert doc1.content_hash() == doc2.content_hash()


# =============================================================================
# CodeChunker Tests
# =============================================================================

class TestCodeChunker:
    """CodeChunker测试。"""
    
    @pytest.fixture
    def chunker(self):
        return CodeChunker(max_chunk_lines=50, min_chunk_lines=2)
    
    def test_detect_python(self, chunker):
        assert chunker.detect_language("main.py") == CodeLanguage.PYTHON
    
    def test_detect_javascript(self, chunker):
        assert chunker.detect_language("app.js") == CodeLanguage.JAVASCRIPT
    
    def test_detect_typescript(self, chunker):
        assert chunker.detect_language("index.ts") == CodeLanguage.TYPESCRIPT
    
    def test_detect_unknown(self, chunker):
        assert chunker.detect_language("file.xyz") == CodeLanguage.UNKNOWN
    
    def test_chunk_python_file(self, chunker):
        code = '''def foo():
    return 1

def bar():
    return 2

class MyClass:
    def method(self):
        pass
'''
        chunks = chunker.chunk_file(code, "test.py")
        assert len(chunks) >= 1
    
    def test_chunk_by_lines(self, chunker):
        code = "\n".join([f"line {i}" for i in range(100)])
        chunks = chunker.chunk_file(code, "unknown.txt")
        assert len(chunks) >= 2


# =============================================================================
# CodeEmbedding Tests
# =============================================================================

class TestCodeEmbedding:
    """CodeEmbedding测试。"""
    
    @pytest.fixture
    def embedding(self):
        return CodeEmbedding(dim=128)
    
    def test_encode_single(self, embedding):
        emb = embedding.encode("def hello(): pass")
        assert emb.shape == (1, 128)
    
    def test_encode_multiple(self, embedding):
        codes = ["def a(): pass", "def b(): pass"]
        emb = embedding.encode(codes)
        assert emb.shape == (2, 128)
    
    def test_encode_document(self, embedding):
        doc = CodeDocument(content="class Foo: pass", name="Foo")
        emb = embedding.encode_document(doc)
        assert emb.shape == (128,)
    
    def test_normalized(self, embedding):
        emb = embedding.encode("test code")[0]
        norm = np.linalg.norm(emb)
        assert np.isclose(norm, 1.0, atol=1e-5)


# =============================================================================
# CodeRetriever Tests
# =============================================================================

class TestCodeRetriever:
    """CodeRetriever测试。"""
    
    @pytest.fixture
    def retriever(self):
        return CodeRetriever()
    
    def test_add_document(self, retriever):
        doc = CodeDocument(content="def test(): pass")
        doc_id = retriever.add_document(doc)
        assert retriever.num_documents == 1
    
    def test_index_file(self, retriever):
        code = """def foo():
    x = 1
    y = 2
    z = 3
    return x + y + z

def bar():
    a = 10
    b = 20
    return a * b
"""
        doc_ids = retriever.index_file(code, "test.py")
        assert len(doc_ids) >= 1
    
    def test_search(self, retriever):
        retriever.add_document(CodeDocument(
            content="def quicksort(arr): pass",
            name="quicksort",
        ))
        retriever.add_document(CodeDocument(
            content="def binary_search(arr): pass",
            name="binary_search",
        ))
        
        results = retriever.search("排序算法", top_k=2)
        assert len(results) <= 2
    
    def test_search_with_filter(self, retriever):
        retriever.add_document(CodeDocument(
            content="def py_func(): pass",
            language=CodeLanguage.PYTHON,
        ))
        retriever.add_document(CodeDocument(
            content="function jsFunc() {}",
            language=CodeLanguage.JAVASCRIPT,
        ))
        
        results = retriever.search("func", language_filter=CodeLanguage.PYTHON)
        for r in results:
            assert r.document.language == CodeLanguage.PYTHON
    
    def test_search_by_name(self, retriever):
        retriever.add_document(CodeDocument(content="...", name="myFunction"))
        results = retriever.search_by_name("myFunc")
        assert len(results) == 1
    
    def test_get_document(self, retriever):
        doc = CodeDocument(content="test")
        retriever.add_document(doc)
        found = retriever.get_document(doc.doc_id)
        assert found is not None
    
    def test_clear(self, retriever):
        retriever.add_document(CodeDocument(content="test"))
        retriever.clear()
        assert retriever.num_documents == 0


# =============================================================================
# CodeAgent Tests
# =============================================================================

class TestCodeAction:
    """CodeAction测试。"""
    
    def test_create_action(self):
        action = CodeAction(
            action_type=CodeActionType.GENERATE,
            params={"task": "sort"},
        )
        assert action.action_type == CodeActionType.GENERATE


class TestCodeResult:
    """CodeResult测试。"""
    
    def test_result_properties(self):
        result = CodeResult(code="line1\nline2\nline3")
        assert result.num_lines == 3
        assert result.success


class TestCodeAgent:
    """CodeAgent测试。"""
    
    @pytest.fixture
    def agent(self):
        return CodeAgent()
    
    def test_generate(self, agent):
        result = agent.generate("快速排序", language=CodeLanguage.PYTHON)
        assert isinstance(result, CodeResult)
        assert result.code
        assert result.language == CodeLanguage.PYTHON
    
    def test_complete(self, agent):
        result = agent.complete("def hello():", language=CodeLanguage.PYTHON)
        assert isinstance(result, CodeResult)
    
    def test_explain(self, agent):
        result = agent.explain("def foo(): return 1")
        assert result.explanation
    
    def test_refactor(self, agent):
        code = "def foo(): return 1"
        result = agent.refactor(code, "添加类型注解")
        assert isinstance(result, CodeResult)
    
    def test_fix(self, agent):
        code = "def foo() return 1"
        result = agent.fix(code, "SyntaxError: invalid syntax")
        assert isinstance(result, CodeResult)
    
    def test_history(self, agent):
        agent.generate("test")
        assert len(agent.get_history()) == 1
        agent.clear_history()
        assert len(agent.get_history()) == 0


# =============================================================================
# ReviewAgent Tests
# =============================================================================

class TestReviewIssue:
    """ReviewIssue测试。"""
    
    def test_create_issue(self):
        issue = ReviewIssue(
            message="测试问题",
            severity=IssueSeverity.WARNING,
            category=IssueCategory.STYLE,
            line=10,
        )
        assert issue.severity == IssueSeverity.WARNING
        assert issue.line == 10


class TestReviewResult:
    """ReviewResult测试。"""
    
    def test_empty_result(self):
        result = ReviewResult()
        assert result.num_errors == 0
        assert result.num_warnings == 0
        assert result.passed
    
    def test_result_with_issues(self):
        issues = [
            ReviewIssue("err", IssueSeverity.ERROR, IssueCategory.BUG),
            ReviewIssue("warn", IssueSeverity.WARNING, IssueCategory.STYLE),
        ]
        result = ReviewResult(issues=issues, score=80.0)
        assert result.num_errors == 1
        assert result.num_warnings == 1


class TestReviewAgent:
    """ReviewAgent测试。"""
    
    @pytest.fixture
    def agent(self):
        return ReviewAgent()
    
    def test_review_clean_code(self, agent):
        code = '''def add(a, b):
    """Add two numbers."""
    return a + b
'''
        result = agent.review(code, CodeLanguage.PYTHON)
        assert isinstance(result, ReviewResult)
        assert result.score > 0
    
    def test_detect_bare_except(self, agent):
        code = '''try:
    risky()
except:
    pass
'''
        result = agent.review(code, CodeLanguage.PYTHON)
        assert any("except" in i.message for i in result.issues)
    
    def test_detect_eval(self, agent):
        code = "result = eval(user_input)"
        result = agent.review(code, CodeLanguage.PYTHON)
        assert any(i.severity == IssueSeverity.ERROR for i in result.issues)
    
    def test_detect_hardcoded_password(self, agent):
        code = 'password = "secret123"'
        result = agent.review(code, CodeLanguage.PYTHON)
        assert any("密码" in i.message for i in result.issues)
    
    def test_detect_long_line(self, agent):
        code = "x = " + "a" * 150
        result = agent.review(code, CodeLanguage.PYTHON)
        assert any("行过长" in i.message for i in result.issues)
    
    def test_review_javascript(self, agent):
        code = "var x = 1; if (x == 1) { console.log(x); }"
        result = agent.review(code, CodeLanguage.JAVASCRIPT)
        assert len(result.issues) > 0
    
    def test_skip_style_check(self, agent):
        code = "from module import *"
        result_with = agent.review(code, CodeLanguage.PYTHON, check_style=True)
        result_without = agent.review(code, CodeLanguage.PYTHON, check_style=False)
        assert len(result_with.issues) >= len(result_without.issues)
    
    def test_suggest_fixes(self, agent):
        code = "try:\n    x()\nexcept:\n    pass"
        result = agent.review(code, CodeLanguage.PYTHON)
        fixes = agent.suggest_fixes(code, result.issues)
        assert isinstance(fixes, dict)
    
    def test_review_document(self, agent):
        doc = CodeDocument(
            content="eval(input())",
            language=CodeLanguage.PYTHON,
        )
        result = agent.review_document(doc)
        assert not result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
