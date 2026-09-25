"""Semantic signatures, similarity, diversity, spectral ranking - Tez Section 3.14-3.20"""

import numpy as np
from typing import Set, Dict, List

# ========== SEMANTIC SIGNATURE (Section 3.15) ==========
def get_semantic_signature(test_case: Dict) -> Set[str]:
    """Σ(t) = {τ_method, τ_path, τ_auth, τ_polarity, τ_status, τ_shape, τ_roles}"""
    tokens = set()
    tokens.add(f"method_{test_case.get('http_method', 'UNKNOWN')}")
    tokens.add(f"path_{test_case.get('path', '')}")
    tokens.add(f"auth_{test_case.get('auth_required', False)}")
    
    expected_status = test_case.get('expected_status', 200)
    polarity = 'positive' if (isinstance(expected_status, int) and expected_status < 400) else 'negative'
    tokens.add(f"polarity_{polarity}")
    
    status_class = (expected_status // 100) if isinstance(expected_status, int) else 2
    tokens.add(f"status_{status_class}xx")
    
    request_body = test_case.get('request_body')
    if request_body is None:
        body_shape = "empty"
    elif isinstance(request_body, dict):
        body_shape = f"object_{len(request_body)}"
    elif isinstance(request_body, list):
        body_shape = f"array_{len(request_body)}"
    else:
        body_shape = "scalar"
    tokens.add(f"shape_{body_shape}")
    
    for role in test_case.get('parameter_roles', []):
        tokens.add(f"role_{role}")
    
    return tokens

# ========== JACCARD SIMILARITY (Section 3.15.1) ==========
def jaccard_similarity(sig_i: Set[str], sig_j: Set[str]) -> float:
    """J(t_i, t_j) = |∩| / |∪|"""
    intersection = len(sig_i & sig_j)
    union = len(sig_i | sig_j)
    return intersection / union if union > 0 else 0.0

# ========== INTRA-GENERATOR SIMILARITY (Section 3.16) ==========
def intra_generator_similarity(generator_id: str, test_cases: List[Dict]) -> float:
    """IGS(g) = Average Jaccard within generator"""
    gen_tests = [tc for tc in test_cases if tc.get('generator') == generator_id]
    if len(gen_tests) < 2:
        return 0.0
    
    signatures = [get_semantic_signature(tc) for tc in gen_tests]
    total_similarity = 0.0
    count = 0
    
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            total_similarity += jaccard_similarity(signatures[i], signatures[j])
            count += 1
    
    return total_similarity / count if count > 0 else 0.0

# ========== PAIRWISE GENERATOR SIMILARITY (Section 3.17) ==========
def pairwise_generator_similarity(gen_a_id: str, gen_b_id: str, test_cases: List[Dict]) -> float:
    """PGS(g_a, g_b) = Average Jaccard between two generators"""
    tests_a = [tc for tc in test_cases if tc.get('generator') == gen_a_id]
    tests_b = [tc for tc in test_cases if tc.get('generator') == gen_b_id]
    
    if not tests_a or not tests_b:
        return 0.0
    
    sigs_a = [get_semantic_signature(tc) for tc in tests_a]
    sigs_b = [get_semantic_signature(tc) for tc in tests_b]
    
    total_similarity = 0.0
    for sig_a in sigs_a:
        for sig_b in sigs_b:
            total_similarity += jaccard_similarity(sig_a, sig_b)
    
    return total_similarity / (len(sigs_a) * len(sigs_b))

# ========== DIVERSITY SCORE (Section 3.18) ==========
def unique_signature_ratio(generator_id: str, test_cases: List[Dict]) -> float:
    """U(g) = unique signatures / total tests"""
    gen_tests = [tc for tc in test_cases if tc.get('generator') == generator_id]
    if not gen_tests:
        return 0.0
    
    signatures = [get_semantic_signature(tc) for tc in gen_tests]
    unique_sigs = set(tuple(sorted(sig)) for sig in signatures)
    return len(unique_sigs) / len(signatures)

def diversity_score(generator_id: str, test_cases: List[Dict], lambda_param: float = 0.6) -> float:
    """DS(g) = λ(1-IGS) + (1-λ)U"""
    igs = intra_generator_similarity(generator_id, test_cases)
    unique_ratio = unique_signature_ratio(generator_id, test_cases)
    return lambda_param * (1.0 - igs) + (1.0 - lambda_param) * unique_ratio

# ========== HERMITIAN MATRIX (Section 3.19) ==========
def hermitian_comparison_matrix(generators: List[str], test_cases: List[Dict]) -> np.ndarray:
    """H_ab = w_ab ± i*δ_ab (conjugate symmetry)"""
    n = len(generators)
    H = np.zeros((n, n), dtype=complex)
    
    for i, gen_a in enumerate(generators):
        for j, gen_b in enumerate(generators):
            if i == j:
                H[i, j] = 1.0 + 0j
            else:
                pass_rate_a = sum(1 for tc in test_cases if tc.get('generator') == gen_a and tc.get('pass', False))
                pass_rate_b = sum(1 for tc in test_cases if tc.get('generator') == gen_b and tc.get('pass', False))
                len_a = max(1, sum(1 for tc in test_cases if tc.get('generator') == gen_a))
                len_b = max(1, sum(1 for tc in test_cases if tc.get('generator') == gen_b))
                
                w_ab = 0.5
                delta_ab = ((pass_rate_a / len_a) - (pass_rate_b / len_b)) / 2.0
                
                if i < j:
                    H[i, j] = w_ab + 1j * delta_ab
                    H[j, i] = w_ab - 1j * delta_ab
    
    return H

# ========== SPECTRAL RANKING (Section 3.20) ==========
def spectral_ranking(generators: List[str], test_cases: List[Dict], 
                     epsilon: float = 1e-6, max_iterations: int = 100) -> Dict:
    """Power method on Hermitian matrix"""
    H = hermitian_comparison_matrix(generators, test_cases)
    n = len(generators)
    
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
            else:
                delta = H[i, j].imag * 2
                # A higher pass-rate generator should receive stronger incoming
                # preference, so the pairwise edge is oriented toward the
                # generator with the higher score.
                A[i, j] = (1.0 - delta) / 2
    
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    P = A / row_sums
    
    r = np.ones(n) / n
    
    for iteration in range(max_iterations):
        r_new = P.T @ r
        r_new = r_new / (r_new.sum() + 1e-10)
        
        if np.linalg.norm(r_new - r, ord=1) < epsilon:
            break
        r = r_new
    
    scores_dict = dict(zip(generators, r))
    ranking = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'spectral_scores': scores_dict,
        'ranking': ranking,
        'convergence_iterations': iteration + 1,
    }

print("✓ Metrics module loaded")
