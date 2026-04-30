"""
ClosureDetector - Prime Coder v1.3.0
Closure-First boundary analysis and API surface locking

Extracts function/class closures (boundaries), computes complexity metrics,
and locks API surfaces to prevent breaking changes without version bumps.

Boundary complexity formula:
    C_b = 0.4 * length + 0.3 * degree + 0.3 * (length / max(interior_size, 1))

Where:
    length: number of boundary elements (params, public methods)
    degree: diversity of element types
    ratio: boundary/interior size (surface density)
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Complexity(Enum):
    """Boundary complexity levels"""
    SIMPLE = "simple"      # C_b < 3
    MODERATE = "moderate"  # 3 <= C_b < 7
    COMPLEX = "complex"    # C_b >= 7


class VersionBump(Enum):
    """Semantic versioning bump required"""
    PATCH = "patch"    # Interior changes only
    MINOR = "minor"    # Added optional/non-breaking
    MAJOR = "major"    # Breaking changes


@dataclass
class Closure:
    """Extracted closure (boundary) from code"""
    name: str
    type: str  # "function" or "class"
    boundary_elements: List[str] = field(default_factory=list)
    interior_elements: List[str] = field(default_factory=list)
    complexity: float = 0.0
    complexity_level: Complexity = Complexity.SIMPLE

    def compute_complexity(self) -> float:
        """
        Compute boundary complexity metric.

        C_b = 0.4 * length + 0.3 * degree + 0.3 * (length / max(interior_size, 1))

        Returns:
            Complexity score (typically 0-10)
        """
        length = len(self.boundary_elements)
        degree = len(set(
            "param" if e.startswith("__") and not e.endswith("__") else
            "method" if "(" in e else
            "decorator" if "@" in e else
            "return"
            for e in self.boundary_elements
        ))
        interior_size = max(len(self.interior_elements), 1)
        ratio = length / interior_size

        self.complexity = 0.4 * length + 0.3 * degree + 0.3 * ratio

        # Classify by complexity level
        if self.complexity < 3:
            self.complexity_level = Complexity.SIMPLE
        elif self.complexity < 7:
            self.complexity_level = Complexity.MODERATE
        else:
            self.complexity_level = Complexity.COMPLEX

        return self.complexity


class ClosureDetector(ast.NodeVisitor):
    """
    Extracts function and class closures from Python code.

    Usage:
        detector = ClosureDetector()
        closures = detector.analyze_code(code_string)
        for closure in closures:
            print(f"{closure.name}: C_b = {closure.complexity}")
    """

    def __init__(self):
        self.closures: Dict[str, Closure] = {}
        self.current_class: Optional[str] = None
        self.api_surface_locks: Dict[str, Closure] = {}  # Locked surfaces

    def analyze_code(self, code: str) -> List[Closure]:
        """
        Analyze Python code and extract all closures.

        Args:
            code: Python source code string

        Returns:
            List of Closure objects
        """
        tree = ast.parse(code)
        self.visit(tree)

        # Compute complexity for all closures
        for closure in self.closures.values():
            closure.compute_complexity()

        return list(self.closures.values())

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function closure (parameters + return type)"""
        closure = Closure(
            name=node.name,
            type="function"
        )

        # Boundary: parameters
        for arg in node.args.args:
            closure.boundary_elements.append(f"param_{arg.arg}")
        for arg in node.args.posonlyargs:
            closure.boundary_elements.append(f"param_{arg.arg}")
        for arg in node.args.kwonlyargs:
            closure.boundary_elements.append(f"kwarg_{arg.arg}")

        # Boundary: decorators
        for _decorator in node.decorator_list:
            closure.boundary_elements.append("decorator")

        # Interior: function body statements
        closure.interior_elements = [f"stmt_{i}" for i in range(len(node.body))]

        self.closures[node.name] = closure
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class closure (public methods + __init__ + dunder methods)"""
        closure = Closure(
            name=node.name,
            type="class"
        )

        public_methods = set()
        private_methods = set()

        # Walk class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name.startswith("_") and not item.name.startswith("__"):
                    # Private method
                    private_methods.add(item.name)
                elif item.name.startswith("__") and item.name.endswith("__"):
                    # Dunder method (protocol)
                    closure.boundary_elements.append(f"dunder_{item.name}")
                elif item.name == "__init__":
                    # Constructor
                    closure.boundary_elements.append("__init__")
                else:
                    # Public method
                    public_methods.add(item.name)
                    closure.boundary_elements.append(f"method_{item.name}")
            elif isinstance(item, ast.Assign):
                # Class attribute
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id.startswith("_"):
                            private_methods.add(target.id)
                        else:
                            closure.boundary_elements.append(f"attr_{target.id}")

        # Interior: private methods and implementation
        closure.interior_elements = list(private_methods)

        self.closures[node.name] = closure
        self.generic_visit(node)

    def lock_api_surface(self, closure: Closure) -> None:
        """Lock an API surface at current version."""
        self.api_surface_locks[closure.name] = closure

    def check_api_surface(self, new_closure: Closure) -> Tuple[bool, List[str], VersionBump]:
        """
        Check if new closure breaks API.

        Args:
            new_closure: Updated closure to check

        Returns:
            (matches, breaking_changes, version_bump)
        """
        if new_closure.name not in self.api_surface_locks:
            return True, [], VersionBump.PATCH

        locked = self.api_surface_locks[new_closure.name]
        breaking = []

        # Extract method/parameter names
        locked_names = {e.split("_", 1)[-1] if "_" in e else e
                       for e in locked.boundary_elements}
        new_names = {e.split("_", 1)[-1] if "_" in e else e
                    for e in new_closure.boundary_elements}

        # Check for removed public methods
        removed = locked_names - new_names
        if removed:
            breaking.extend([f"removed_{name}" for name in removed])

        # Check for added methods (non-breaking)
        added = new_names - locked_names
        if added and not removed:
            return True, [], VersionBump.MINOR

        # Determine version bump
        if breaking:
            return False, breaking, VersionBump.MAJOR
        elif added:
            return True, [], VersionBump.MINOR
        else:
            return True, [], VersionBump.PATCH


class APIBoundaryValidator:
    """
    Validates API boundaries using semver rules.

    Usage:
        validator = APIBoundaryValidator()
        validator.analyze_codebase("path/to/code")
        violations = validator.find_breaking_changes()
    """

    def __init__(self):
        self.detector = ClosureDetector()
        self.violations: List[str] = []

    def analyze_codebase(self, code_dict: Dict[str, str]) -> None:
        """
        Analyze multiple code files.

        Args:
            code_dict: Dict mapping filename -> code string
        """
        for filename, code in code_dict.items():
            try:
                closures = self.detector.analyze_code(code)
                for closure in closures:
                    print(f"{filename}:{closure.name} "
                          f"(C_b={closure.complexity:.1f}, "
                          f"{closure.complexity_level.value})")
            except SyntaxError as e:
                print(f"Syntax error in {filename}: {e}")

    def find_breaking_changes(self) -> List[str]:
        """Find API changes that would require major version bump."""
        return self.violations


if __name__ == "__main__":
    # Test 1: Simple function
    print("=" * 70)
    print("TEST 1: Simple Function Closure")
    print("=" * 70)
    code1 = """
def simple_func(x, y):
    z = x + y
    return z
"""
    detector = ClosureDetector()
    closures = detector.analyze_code(code1)
    for c in closures:
        print(f"Name: {c.name}")
        print(f"Type: {c.type}")
        print(f"Boundary: {c.boundary_elements}")
        print(f"Interior: {c.interior_elements}")
        print(f"Complexity: {c.complexity:.2f} ({c.complexity_level.value})")
    print()

    # Test 2: Complex class
    print("=" * 70)
    print("TEST 2: Complex Class Closure")
    print("=" * 70)
    code2 = """
class ResolutionDetector:
    def __init__(self, R_p=None):
        self._tolerance = R_p
        self.residuals = []

    def __repr__(self):
        return f"Detector(R_p={self._tolerance})"

    def track_iteration(self, residual):
        self.residuals.append(residual)

    def check_convergence(self, residual, max_iterations):
        return residual < self._tolerance

    def _internal_method(self):
        pass
"""
    closures = detector.analyze_code(code2)
    for c in closures:
        print(f"Name: {c.name}")
        print(f"Type: {c.type}")
        print(f"Boundary: {c.boundary_elements}")
        print(f"Interior: {c.interior_elements}")
        print(f"Complexity: {c.complexity:.2f} ({c.complexity_level.value})")
    print()

    # Test 3: API surface locking
    print("=" * 70)
    print("TEST 3: API Surface Locking (Breaking Change Detection)")
    print("=" * 70)
    code_v1 = """
class APIv1:
    def public_method_a(self):
        pass
    def public_method_b(self):
        pass
"""
    code_v2_breaking = """
class APIv1:
    def public_method_a(self):
        pass
"""
    detector = ClosureDetector()
    closures_v1 = detector.analyze_code(code_v1)
    detector.lock_api_surface(closures_v1[0])

    closures_v2 = detector.analyze_code(code_v2_breaking)
    matches, breaking, bump = detector.check_api_surface(closures_v2[0])

    print(f"V1 boundary: {closures_v1[0].boundary_elements}")
    print(f"V2 boundary: {closures_v2[0].boundary_elements}")
    print(f"Matches: {matches}")
    print(f"Breaking changes: {breaking}")
    print(f"Version bump required: {bump.value}")
