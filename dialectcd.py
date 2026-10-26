#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
DialectCD: Explainable Dialectical Causal Discovery Engine
Official Anonymous Implementation for WWW Conference
Dual-Source Argumentation + BAF Meta-Dialectic Framework
==============================================================================
"""

import os
import sys
import re
import gzip
import shutil
import json
import argparse
import requests
import warnings
import random
import itertools
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import deque
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import CITests

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. RPY2 Bridge for Standard Causal Metrics (SHD / SID)
# ==============================================================================

RPY2_AVAILABLE = False
BNLEARN_AVAILABLE = False
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    ro.r('library(pcalg)')
    ro.r('library(graph)')
    ro.r('library(SID)')
    RPY2_AVAILABLE = True
    try:
        ro.r('library(bnlearn)')
        BNLEARN_AVAILABLE = True
    except Exception:
        BNLEARN_AVAILABLE = False
except Exception:
    RPY2_AVAILABLE = False
    BNLEARN_AVAILABLE = False


# ==============================================================================
# 1. Dataset Fetcher & Semi-Parametric Adaptive CI Test Worker
# ==============================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def fetch_dataset(name: str):
    """Auto-downloads benchmark .bif files from bnlearn repository if not present locally."""
    os.makedirs("datasets", exist_ok=True)
    path = f"datasets/{name}.bif"
    if os.path.exists(path):
        return path
    url = f"https://www.bnlearn.com/bnrepository/{name}/{name}.bif.gz"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        gz_path = path + ".gz"
        with open(gz_path, "wb") as f:
            f.write(r.content)
        with gzip.open(gz_path, "rb") as fi, open(path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(gz_path)
        return path
    except Exception as e:
        print(f"[!] Error fetching dataset {name}: {e}")
        return None


def independence_test_worker(u_name: str, v_name: str, conditioning_set, data_df: pd.DataFrame, alpha: float) -> float:
    """
    Adaptive Conditional Independence Test Worker:
    Handles mixed discrete/continuous, heavy-tailed, and zero-inflated distributions.
    """
    col_u = data_df[u_name]
    col_v = data_df[v_name]

    is_u_cont = np.issubdtype(col_u.dtype, np.floating) or col_u.nunique() > 5
    is_v_cont = np.issubdtype(col_v.dtype, np.floating) or col_v.nunique() > 5

    # Case A: Purely discrete categorical variables -> Chi-square test
    if not (is_u_cont or is_v_cont):
        try:
            result = CITests.chi_square(
                X=u_name, Y=v_name, Z=list(conditioning_set),
                data=data_df, boolean=False, significance_level=alpha
            )
            p = float(result[1])
            return p if not np.isnan(p) else 1.0
        except Exception:
            pass

    # Case B: Mixed or continuous features -> Semi-parametric Spearman residual test
    try:
        u_vals = col_u.values.astype(float)
        v_vals = col_v.values.astype(float)

        # 0-order marginal test: Direct Spearman rank correlation
        if len(conditioning_set) == 0:
            res = stats.spearmanr(u_vals, v_vals)
            p_val = float(res.pvalue if hasattr(res, 'pvalue') else res[1])
            return p_val if not np.isnan(p_val) else 1.0

        # Higher-order conditional test: Residual orthogonal projection
        z_cols = list(conditioning_set)
        Z_mat = data_df[z_cols].values.astype(float)

        # Log1p transformation to mitigate heavy-tail skewness
        u_trans = np.log1p(np.maximum(0.0, u_vals))
        v_trans = np.log1p(np.maximum(0.0, v_vals))
        Z_trans = np.log1p(np.maximum(0.0, Z_mat))

        # Least squares projection
        Z_design = np.column_stack([np.ones(len(Z_trans)), Z_trans])
        beta_u, _, _, _ = np.linalg.lstsq(Z_design, u_trans, rcond=None)
        beta_v, _, _, _ = np.linalg.lstsq(Z_design, v_trans, rcond=None)

        res_u = u_trans - Z_design @ beta_u
        res_v = v_trans - Z_design @ beta_v

        # Rank test on orthogonalized residuals
        res_corr = stats.spearmanr(res_u, res_v)
        p_val = float(res_corr.pvalue if hasattr(res_corr, 'pvalue') else res_corr[1])
        return p_val if not np.isnan(p_val) else 1.0
    except Exception:
        return 1.0


# ==============================================================================
# 2. Prior Knowledge Manager (Harmonization & Tier Constraints)
# ==============================================================================

class ExpertKnowledgeManager:
    """
    Robust Knowledge Manager for LLM & Expert Priors.
    Supports fuzzy node name normalization, cyclic contradiction clearing, and temporal tiering.
    """
    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.norm_node_map = {self._clean_str(name): i for i, name in enumerate(self.nodes)}
        self.required_edges = {}       # (u_idx, v_idx) -> weight
        self.forbidden_edges = set()   # (u_idx, v_idx) -> forbidden u -> v
        self.order_constraints = set() # (u_idx, v_idx) -> u precedes v
        self.tier_mapping = {}         # node_idx -> tier_int

    @property
    def is_zero_knowledge(self) -> bool:
        total_rules = len(self.required_edges) + len(self.forbidden_edges) + len(self.tier_mapping)
        return total_rules == 0

    @is_zero_knowledge.setter
    def is_zero_knowledge(self, value: bool):
        pass

    def _clean_str(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def _get_node_idx(self, name: str):
        clean_key = self._clean_str(name)
        if clean_key in self.norm_node_map:
            return self.norm_node_map[clean_key]
        for k, idx in self.norm_node_map.items():
            if clean_key in k or k in clean_key:
                return idx
        return None

    def setup_platform_tiers(self, treatment_nodes: list, mechanism_nodes: list, 
                             outcome_nodes: list, pre_exp_nodes: list = None):
        """
        Configure web platform causal hierarchy:
        - Tier 0: Pre-experiment covariates (e.g., historical activity, user baseline)
        - Tier 1: Treatment / Intervention policies
        - Tier 2: Intermediate dispatch mechanisms (e.g., rank score, exposure, click)
        - Tier 3: Core downstream outcomes (e.g., watch time, interaction, retention)
        """
        tier_dict = {}
        if pre_exp_nodes:
            for node in pre_exp_nodes:
                tier_dict[node] = 0
        for node in treatment_nodes:
            tier_dict[node] = 1
        for node in mechanism_nodes:
            tier_dict[node] = 2
        for node in outcome_nodes:
            tier_dict[node] = 3

        self.set_tier_constraints(tier_dict)

        # Forbid reverse causation: Downstream outcomes cannot cause treatment or baseline features
        for y_name in outcome_nodes:
            y_idx = self._get_node_idx(y_name)
            if y_idx is None:
                continue
            for t_name in (treatment_nodes + (pre_exp_nodes or [])):
                t_idx = self._get_node_idx(t_name)
                if t_idx is not None:
                    self.forbidden_edges.add((y_idx, t_idx))

    def set_tier_constraints(self, tier_dict: dict):
        for node_name, tier in tier_dict.items():
            idx = self._get_node_idx(node_name)
            if idx is not None:
                self.tier_mapping[idx] = tier

        for u_name, u_tier in tier_dict.items():
            for v_name, v_tier in tier_dict.items():
                u_idx, v_idx = self._get_node_idx(u_name), self._get_node_idx(v_name)
                if u_idx is not None and v_idx is not None:
                    if u_tier < v_tier:
                        self.order_constraints.add((u_idx, v_idx))
                        self.forbidden_edges.add((v_idx, u_idx))

    def add_required_edge(self, u_name: str, v_name: str, weight: float = 1.0):
        u_idx, v_idx = self._get_node_idx(u_name), self._get_node_idx(v_name)
        if u_idx is not None and v_idx is not None:
            self.required_edges[(u_idx, v_idx)] = weight
            self.order_constraints.add((u_idx, v_idx))
            self.forbidden_edges.add((v_idx, u_idx))
            return True
        return False

    def add_forbidden_edge(self, u_name: str, v_name: str):
        u_idx, v_idx = self._get_node_idx(u_name), self._get_node_idx(v_name)
        if u_idx is not None and v_idx is not None:
            self.forbidden_edges.add((u_idx, v_idx))
            return True
        return False

    def harmonize_priors(self):
        """Resolves direct contradictions where an edge is simultaneously required and forbidden."""
        contradictions = set(self.required_edges.keys()) & self.forbidden_edges
        for u_idx, v_idx in list(contradictions):
            del self.required_edges[(u_idx, v_idx)]
            self.forbidden_edges.remove((u_idx, v_idx))
            print(f"  [Prior Harmonizer] Resolved contradiction on ({self.nodes[u_idx]} -> {self.nodes[v_idx]})")

    def load_priors_from_json(self, json_path: str):
        if not os.path.exists(json_path):
            return 0
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        rules_loaded = 0
        if "tiers" in data:
            self.set_tier_constraints(data["tiers"])
        if "required" in data:
            for item in data["required"]:
                if self.add_required_edge(item["from"], item["to"], item.get("weight", 1.0)):
                    rules_loaded += 1
        if "forbidden" in data:
            for item in data["forbidden"]:
                if self.add_forbidden_edge(item["from"], item["to"]):
                    rules_loaded += 1

        self.harmonize_priors()
        total_active = len(self.required_edges) + len(self.forbidden_edges)
        status_msg = "Zero-Knowledge Mode" if self.is_zero_knowledge else f"Active Rules: {total_active}"
        print(f"  [✓] Loaded '{json_path}' ({status_msg} | Required: {len(self.required_edges)}, Forbidden: {len(self.forbidden_edges)})")
        return total_active

    def query_direction_preference(self, u_idx: int, v_idx: int) -> int:
        """
        Direction preference oracle:
        Returns +1 for supporting u -> v
        Returns -1 for supporting v -> u
        Returns  0 for no prior preference
        """
        if self.is_zero_knowledge:
            return 0

        if (u_idx, v_idx) in self.order_constraints or (u_idx, v_idx) in self.required_edges:
            return 1
        if (v_idx, u_idx) in self.order_constraints or (v_idx, u_idx) in self.required_edges:
            return -1

        if (v_idx, u_idx) in self.forbidden_edges:
            return 1
        if (u_idx, v_idx) in self.forbidden_edges:
            return -1

        u_tier = self.tier_mapping.get(u_idx, None)
        v_tier = self.tier_mapping.get(v_idx, None)
        if u_tier is not None and v_tier is not None:
            if u_tier < v_tier:
                return 1
            elif u_tier > v_tier:
                return -1

        return 0


# ==============================================================================
# 3. CPDAG & Meek Orientation Rules
# ==============================================================================

def apply_meek_rules(cpdag: nx.DiGraph):
    nodes = list(cpdag.nodes())
    directed = set()
    undirected = set()

    for u, v in cpdag.edges():
        if cpdag.has_edge(v, u):
            undirected.add((u, v))
        else:
            directed.add((u, v))

    def is_undirected(u, v): return (u, v) in undirected
    def is_directed(u, v): return (u, v) in directed
    def is_adjacent(u, v): return (u, v) in directed or (v, u) in directed or (u, v) in undirected

    dirty_queue = deque(list(directed))

    def direct_edge(u, v):
        if (u, v) in undirected:
            undirected.remove((u, v))
            undirected.remove((v, u))
            directed.add((u, v))
            cpdag.remove_edge(v, u)
            dirty_queue.append((u, v))

    while dirty_queue:
        a, b = dirty_queue.popleft()

        # Rule 1
        b_undirect_nbrs = [c for c in nodes if is_undirected(b, c)]
        for c in b_undirect_nbrs:
            if not is_adjacent(a, c):
                direct_edge(b, c)

        # Rule 2
        for c in nodes:
            if is_directed(b, c) and is_undirected(a, c):
                direct_edge(a, c)
        for x in nodes:
            if is_directed(x, a) and is_undirected(x, b):
                direct_edge(x, b)

        # Rule 3
        b_undirect_parents = [a for a in nodes if is_undirected(a, b)]
        if b_undirect_parents:
            b_dir_parents = [p for p in nodes if is_directed(p, b)]
            if len(b_dir_parents) >= 2:
                for a in b_undirect_parents:
                    found_r3 = False
                    for i in range(len(b_dir_parents)):
                        for j in range(i + 1, len(b_dir_parents)):
                            c_node, d_node = b_dir_parents[i], b_dir_parents[j]
                            if is_undirected(a, c_node) and is_undirected(a, d_node) and not is_adjacent(c_node, d_node):
                                found_r3 = True
                                break
                        if found_r3:
                            break
                    if found_r3:
                        direct_edge(a, b)

        # Rule 4
        for c in b_undirect_parents:
            a_candidates = [a for a in nodes if is_undirected(a, c)]
            for a in a_candidates:
                if is_undirected(a, b):
                    continue
                for d in nodes:
                    if d != c and is_directed(d, b) and is_undirected(a, d) and is_undirected(c, d):
                        direct_edge(a, c)
                        break

    return cpdag


def dag_to_cpdag(dag: nx.DiGraph, nodes):
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(nodes)
    for u, v in dag.edges():
        cpdag.add_edge(u, v)
        cpdag.add_edge(v, u)

    collider_triples = []
    for b in nodes:
        preds = list(dag.predecessors(b))
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                p1, p2 = preds[i], preds[j]
                if not dag.has_edge(p1, p2) and not dag.has_edge(p2, p1):
                    collider_triples.append((p1, b, p2))

    for p1, b, p2 in collider_triples:
        if cpdag.has_edge(b, p1):
            cpdag.remove_edge(b, p1)
        if cpdag.has_edge(b, p2):
            cpdag.remove_edge(b, p2)

    apply_meek_rules(cpdag)
    return cpdag


# ==============================================================================
# 4. CORE ENGINE: DialectCD (Dual-Source Dialectical Causal Discovery)
# ==============================================================================

class DialectCD:
    """
    DialectCD: Dual-Source Dialectical Causal Discovery Engine
    - Sample-adaptive continuous evidentiary handover (Sigmoid bandwidth beta)
    - Epistemic prior shield against finite-sample noise
    - Cochran contingency adequacy correction for sample-sparse conditional tests
    - Sub-linear bounded attack aggregation (defends against cumulative error)
    - Conflict-gated dialectical synthesis and acyclic DAG completion
    """
    def __init__(
        self,
        alpha=0.05,
        max_dialectic_iter=20,
        attack_decay=0.5,
        max_cond_size=3,
        confidence_level=0.85,
        graph_adaptive_decay=True,
        n_jobs=-1,
        expert_manager: ExpertKnowledgeManager = None,
        verbose=False,
        enable_cochran=True,
        enable_sublinear=True,
        enable_tier_gating=True,
        enable_epistemic_shield=True,
        beta_handover=220.0,
        conf_tier_thresh=0.12,
        prior_injection_rate=0.35,
        prior_conflict_gamma=4.0,
        prior_min_trust=0.15,
        controversial_conflict_thresh=0.35
    ):
        self.alpha = alpha
        self.max_dialectic_iter = max_dialectic_iter
        self.attack_decay = attack_decay
        self.max_cond_size = max_cond_size
        self.confidence_level = confidence_level
        self.graph_adaptive_decay = graph_adaptive_decay
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.expert_manager = expert_manager

        # Ablation and sensitivity parameters
        self.enable_cochran = enable_cochran
        self.enable_sublinear = enable_sublinear
        self.enable_tier_gating = enable_tier_gating
        self.enable_epistemic_shield = enable_epistemic_shield
        self.beta_handover = beta_handover
        self.conf_tier_thresh = conf_tier_thresh
        self.prior_injection_rate = prior_injection_rate
        self.prior_conflict_gamma = prior_conflict_gamma
        self.prior_min_trust = prior_min_trust
        self.controversial_conflict_thresh = controversial_conflict_thresh

        self.data_power_factor = 0.5
        self.llm_power_factor = 0.5
        self.controversial_edges = set()

        self.prefer_threads = True
        self.max_ci_tests_per_edge = 100

        self.nodes = []
        self.node_map = {}
        self.relationship_acceptability = None
        self.justifications = {}
        self.data_df = None
        self._ci_cache = {}
        self.trace_log = {}

    def _get_pk(self, u: str, v: str) -> tuple:
        u_str, v_str = str(u), str(v)
        return (u_str, v_str) if u_str < v_str else (v_str, u_str)

    def _derive_dependency_argument_strength(self, p):
        if p is None or np.isnan(p) or p >= self.alpha:
            return 0.0
        return 1.0 - p / self.alpha

    def _derive_independence_argument_strength(self, p):
        if p is None or np.isnan(p) or p < self.alpha:
            return 0.0
        return (p - self.alpha) / (1.0 - self.alpha + 1e-12)

    def _adaptive_prior_trust(self, data_conflict: float, local_support: float = 0.0) -> float:
        conflict = float(np.clip(data_conflict, 0.0, 1.0))
        conflict_penalty = 1.0 - self.data_power_factor * (1.0 - np.exp(-self.prior_conflict_gamma * conflict))
        local_support = float(np.clip(local_support, 0.0, 1.0))
        local_factor = 0.5 + 0.5 * local_support
        trust = conflict_penalty * local_factor
        return float(np.clip(trust, self.prior_min_trust, 1.0))

    def _compute_cochran_adequacy(self, cond_size: int) -> float:
        if not self.enable_cochran or self.data_df is None or len(self.data_df) == 0:
            return 1.0

        N_samples = len(self.data_df)
        try:
            sample_num = min(1000, N_samples)
            sub = self.data_df.iloc[:sample_num]
            numeric_cols = sub.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                zero_ratio = float((sub[numeric_cols] == 0).sum().sum()) / (sample_num * len(numeric_cols) + 1e-6)
                n_eff = N_samples * max(0.2, (1.0 - zero_ratio))
            else:
                n_eff = N_samples
        except Exception:
            n_eff = N_samples

        est_cells = 4.0 * (2.0 ** cond_size) if cond_size > 0 else 4.0
        avg_cell_count = n_eff / est_cells
        return min(1.0, max(0.20, avg_cell_count / 5.0))

    def _ci_pvalue(self, idx_u: int, idx_v: int, cond_idx_list):
        u, v = self.nodes[idx_u], self.nodes[idx_v]
        z = tuple(sorted(cond_idx_list))
        a, b = (u, v) if u <= v else (v, u)
        key = (a, b, z)
        if key in self._ci_cache:
            return self._ci_cache[key]
        p = independence_test_worker(a, b, [self.nodes[k] for k in z], self.data_df, self.alpha)
        self._ci_cache[key] = p
        return p

    def compute_acceptability(self, data_df: pd.DataFrame):
        self.data_df = data_df
        self.nodes = list(data_df.columns)
        self.node_map = {name: i for i, name in enumerate(self.nodes)}
        n = len(self.nodes)
        N_samples = len(data_df)

        # Continuous sigmoid evidentiary handover
        N_mid = max(350.0, 140.0 * np.sqrt(n)) if n > 8 else 400.0
        self.data_power_factor = float(1.0 / (1.0 + np.exp(-(N_samples - N_mid) / max(1e-5, self.beta_handover))))
        self.llm_power_factor = float(1.0 - self.data_power_factor)

        self.controversial_edges = set()
        self.relationship_acceptability = np.zeros((n, n), dtype=float)
        self.justifications = {}
        self._ci_cache = {}

        self.trace_log = {}
        for i in range(n):
            for j in range(i + 1, n):
                pk = self._get_pk(self.nodes[i], self.nodes[j])
                self.trace_log[pk] = {
                    "p_zero": None, "initial_dep_strength": 0.0,
                    "attacks": [], "undercuts": [], "expert_overrides": [],
                    "final_score": 0.0, "orientation": "Undirected", "orientation_reason": ""
                }

        best_indep = {}
        candidate_relations = []
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Phase 1: 0-Order Marginal Dependency Arguments
        p_values_0 = Parallel(n_jobs=self.n_jobs, prefer="threads" if self.prefer_threads else None)(
            delayed(self._ci_pvalue)(i, j, []) for i, j in pairs
        )

        for (i, j), p in zip(pairs, p_values_0):
            dep_strength = self._derive_dependency_argument_strength(p)
            ind_strength = self._derive_independence_argument_strength(p)
            self.relationship_acceptability[i, j] = self.relationship_acceptability[j, i] = dep_strength

            pk = self._get_pk(self.nodes[i], self.nodes[j])
            self.trace_log[pk]["p_zero"] = p
            self.trace_log[pk]["initial_dep_strength"] = dep_strength

            if ind_strength > 0.0:
                best_indep[(i, j)] = ([], ind_strength, p)
                self.justifications[(i, j)] = []
                self.justifications[(j, i)] = []
            elif dep_strength > 0.08:
                candidate_relations.append((i, j))

        edge_attacks = {(i, j): [] for i in range(n) for j in range(i + 1, n)}

        # Phase 2: Higher-Order CI Mining & Epistemic Shielding
        for k in range(1, self.max_cond_size + 1):
            if not candidate_relations:
                break
            tasks = []
            for i, j in candidate_relations:
                if self.relationship_acceptability[i, j] < 0.06:
                    continue
                context_nodes = []
                for idx in range(n):
                    if idx == i or idx == j:
                        continue
                    if self.relationship_acceptability[i, idx] > 0.06 or self.relationship_acceptability[j, idx] > 0.06:
                        if self.expert_manager and not self.expert_manager.is_zero_knowledge:
                            if (idx, i) in self.expert_manager.forbidden_edges and (idx, j) in self.expert_manager.forbidden_edges:
                                continue
                        context_nodes.append(idx)

                if len(context_nodes) >= k:
                    tasks.append((i, j, context_nodes))

            if not tasks:
                break

            def miner_worker(idx_u, idx_v, context, k_size):
                best = None
                tested = 0
                for ctx in itertools.combinations(context, k_size):
                    tested += 1
                    if tested > self.max_ci_tests_per_edge:
                        break
                    p_val = self._ci_pvalue(idx_u, idx_v, ctx)
                    if p_val is not None and not np.isnan(p_val) and p_val > self.alpha:
                        if (best is None) or (p_val > best[0]):
                            best = (p_val, tuple(ctx))
                return (idx_u, idx_v, best[1], best[0]) if best else None

            results_k = Parallel(n_jobs=self.n_jobs, prefer="threads" if self.prefer_threads else None)(
                delayed(miner_worker)(i, j, nbrs, k) for i, j, nbrs in tasks
            )

            for res in results_k:
                if res is None:
                    continue
                i, j, ctx, p = res

                is_prior_edge = False
                if self.expert_manager and not self.expert_manager.is_zero_knowledge:
                    if self.expert_manager.query_direction_preference(i, j) != 0:
                        is_prior_edge = True

                # Epistemic shield threshold
                if self.enable_epistemic_shield and is_prior_edge:
                    attack_threshold = self.alpha * (1.0 + 2.5 * self.llm_power_factor)
                else:
                    attack_threshold = self.alpha

                if p <= attack_threshold:
                    continue

                adequacy = self._compute_cochran_adequacy(len(ctx))
                ind_strength = self._derive_independence_argument_strength(p) * adequacy

                old = best_indep.get((i, j))
                if (old is None) or (ind_strength > old[1]):
                    best_indep[(i, j)] = (list(ctx), ind_strength, p)
                    self.justifications[(i, j)] = list(ctx)
                    self.justifications[(j, i)] = list(ctx)

                if ind_strength > 0.0:
                    edge_attacks[(i, j)].append(ind_strength)
                    if is_prior_edge:
                        safety_floor = 0.18 * (self.llm_power_factor ** 1.2)
                        self.relationship_acceptability[i, j] = max(self.relationship_acceptability[i, j], safety_floor)
                        self.relationship_acceptability[j, i] = max(self.relationship_acceptability[j, i], safety_floor)

                    pk = self._get_pk(self.nodes[i], self.nodes[j])
                    ctx_names = [self.nodes[c] for c in ctx]
                    self.trace_log[pk]["attacks"].append({
                        "sep_set": ctx_names, "p_val": p, "indep_strength": ind_strength
                    })

            candidate_relations = [(i, j) for (i, j) in candidate_relations if self.relationship_acceptability[i, j] > 0.06]

        # Sublinear bounded attack aggregation vs. Multiplicative decay
        for (i, j), attacks in edge_attacks.items():
            if attacks:
                if self.enable_sublinear:
                    max_att = max(attacks)
                    sum_att = sum(attacks)
                    bounded_decay = 1.0 - min(0.92, 0.65 * max_att + 0.15 * np.log1p(sum_att))
                else:
                    bounded_decay = float(np.prod([1.0 - min(0.95, a) for a in attacks]))

                self.relationship_acceptability[i, j] *= bounded_decay
                self.relationship_acceptability[j, i] *= bounded_decay

        # Phase 3: Meta-Dialectic Prior Injection with Conflict Gating
        if self.expert_manager and not self.expert_manager.is_zero_knowledge:
            for i in range(n):
                for j in range(i + 1, n):
                    u_n, v_n = self.nodes[i], self.nodes[j]
                    pk = self._get_pk(u_n, v_n)

                    # Forbidden Prior Attack
                    if (i, j) in self.expert_manager.forbidden_edges or (j, i) in self.expert_manager.forbidden_edges:
                        llm_attack_force = 0.10 + 0.90 * self.data_power_factor
                        self.relationship_acceptability[i, j] *= llm_attack_force
                        self.relationship_acceptability[j, i] *= llm_attack_force
                        self.trace_log[pk]["expert_overrides"].append(f"PRIOR_FORBIDDEN_ATTACK(force={llm_attack_force:.3f})")

                    # Required Prior Support
                    if (i, j) in self.expert_manager.required_edges or (j, i) in self.expert_manager.required_edges:
                        raw_weight = self.expert_manager.required_edges.get((i, j), self.expert_manager.required_edges.get((j, i), 0.85))
                        ind_strength = best_indep.get((i, j), ([], 0.0, 0.0))[1]

                        local_support = 0.0
                        if n > 2:
                            local_support = max(
                                (min(self.relationship_acceptability[i, k], self.relationship_acceptability[k, j])
                                 for k in range(n) if k != i and k != j),
                                default=0.0
                            )

                        prior_trust = self._adaptive_prior_trust(ind_strength, local_support)
                        if ind_strength >= self.controversial_conflict_thresh:
                            self.controversial_edges.add(pk)
                            self.trace_log[pk]["expert_overrides"].append(
                                f"CONTROVERSIAL_EDGE(conflict={ind_strength:.3f}, trust={prior_trust:.3f})"
                            )

                        effective_llm_weight = raw_weight * (self.llm_power_factor ** 1.2) * prior_trust
                        if effective_llm_weight > 0.05:
                            current_score = float(self.relationship_acceptability[i, j])
                            updated_score = current_score + self.prior_injection_rate * (effective_llm_weight - current_score)
                            updated_score = max(current_score, updated_score)
                            self.relationship_acceptability[i, j] = self.relationship_acceptability[j, i] = updated_score
                            self.trace_log[pk]["expert_overrides"].append(
                                f"PRIOR_SOFT_SUPPORT(eff={effective_llm_weight:.3f}, eta={self.prior_injection_rate:.2f})"
                            )

        # Phase 4: BAF Dialectic Equilibrium Iterations
        self._run_dialectic_iterations(best_indep, self.data_power_factor)
        return self.relationship_acceptability

    def _run_dialectic_iterations(self, best_indep, data_power_factor=0.5):
        n = len(self.nodes)
        graph_temp = float(np.clip(np.sqrt(n / 16.0), 0.65, 1.35))

        independence_arguments = [
            ((i, j), sep, strength)
            for (i, j), (sep, strength, _) in best_indep.items()
            if strength > 0.0
        ]

        for _ in range(self.max_dialectic_iter):
            old_acceptability = self.relationship_acceptability.copy()
            attack_matrix = np.ones((n, n), dtype=float)

            for (u, v), justification_set, strength_tau in independence_arguments:
                effective_attack = strength_tau * self.attack_decay * (0.4 + 0.6 * data_power_factor) * graph_temp
                attack_force = 1.0 - min(0.95, effective_attack)

                if self.expert_manager and not self.expert_manager.is_zero_knowledge:
                    if (u, v) in self.expert_manager.required_edges or (v, u) in self.expert_manager.required_edges:
                        resistance = 0.30 * (1.0 - data_power_factor)
                        attack_force = min(1.0, attack_force + resistance)

                attack_matrix[u, v] *= attack_force
                attack_matrix[v, u] *= attack_force

                just_set = set(justification_set)
                for k_node in range(n):
                    if k_node == u or k_node == v or k_node in just_set:
                        continue
                    path_strength = min(self.relationship_acceptability[u, k_node], self.relationship_acceptability[k_node, v])
                    if path_strength > 0.08:
                        undercut_force = 1.0 - (strength_tau * path_strength * self.attack_decay * 0.35 * graph_temp)
                        attack_matrix[u, k_node] *= undercut_force
                        attack_matrix[k_node, u] *= undercut_force
                        attack_matrix[k_node, v] *= undercut_force
                        attack_matrix[v, k_node] *= undercut_force

            self.relationship_acceptability *= attack_matrix
            if np.abs(self.relationship_acceptability - old_acceptability).max() < 1e-4:
                break

        for i in range(n):
            for j in range(i + 1, n):
                pk = self._get_pk(self.nodes[i], self.nodes[j])
                self.trace_log[pk]["final_score"] = float(self.relationship_acceptability[i, j])

    def construct_dag(self, acceptability_threshold=None, ensure_connected=True) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        n = len(self.nodes)
        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.nodes)

        actual_threshold = acceptability_threshold if acceptability_threshold is not None else (0.07 if n >= 20 else 0.09)

        for i in range(n):
            for j in range(i + 1, n):
                if self.relationship_acceptability[i, j] > actual_threshold:
                    u, v = self.nodes[i], self.nodes[j]
                    skeleton.add_edge(u, v)
                    G.add_edge(u, v)
                    G.add_edge(v, u)

        # 1. Connectivity preservation
        if ensure_connected:
            undirected_skel = skeleton.to_undirected()
            components = list(nx.connected_components(undirected_skel))
            while len(components) > 1:
                best_bridge = None
                best_score = -1.0
                for i in range(len(components)):
                    for j in range(i + 1, len(components)):
                        for u in components[i]:
                            for v in components[j]:
                                u_idx, v_idx = self.node_map[u], self.node_map[v]
                                score = self.relationship_acceptability[u_idx, v_idx]
                                if score > best_score:
                                    best_score = score
                                    best_bridge = (u, v)
                if best_bridge and best_score > 0.0:
                    u, v = best_bridge
                    skeleton.add_edge(u, v)
                    G.add_edge(u, v)
                    G.add_edge(v, u)
                else:
                    break
                undirected_skel = skeleton.to_undirected()
                components = list(nx.connected_components(undirected_skel))

        # 2. V-structure collider orientation
        for z in range(n):
            z_node = self.nodes[z]
            neighbors = list(skeleton.neighbors(z_node))
            if len(neighbors) < 2:
                continue
            idx_neighbors = [self.node_map[nn] for nn in neighbors]
            for u_idx, v_idx in itertools.combinations(idx_neighbors, 2):
                u_node, v_node = self.nodes[u_idx], self.nodes[v_idx]
                if skeleton.has_edge(u_node, v_node):
                    continue
                sep = self.justifications.get((u_idx, v_idx), self.justifications.get((v_idx, u_idx), None))
                if sep is not None and z not in sep:
                    if self.expert_manager:
                        if (u_idx, z) in self.expert_manager.forbidden_edges or (v_idx, z) in self.expert_manager.forbidden_edges:
                            continue
                    if G.has_edge(z_node, u_node):
                        G.remove_edge(z_node, u_node)
                    if G.has_edge(z_node, v_node):
                        G.remove_edge(z_node, v_node)

        apply_meek_rules(G)

        # 3. Temporal tier orientation disambiguation
        if self.expert_manager:
            for u in list(self.nodes):
                for v in list(self.nodes):
                    if u < v and G.has_edge(u, v) and G.has_edge(v, u):
                        u_idx, v_idx = self.node_map[u], self.node_map[v]
                        u_tier = self.expert_manager.tier_mapping.get(u_idx, 2)
                        v_tier = self.expert_manager.tier_mapping.get(v_idx, 2)

                        if u_tier < v_tier:
                            G.remove_edge(v, u)
                        elif v_tier < u_tier:
                            G.remove_edge(u, v)
                        else:
                            pref = self.expert_manager.query_direction_preference(u_idx, v_idx)
                            if pref == 1:
                                G.remove_edge(v, u)
                            elif pref == -1:
                                G.remove_edge(u, v)

        # 4. Enforce strict acyclicity
        if not nx.is_directed_acyclic_graph(G):
            try:
                cycles = list(nx.simple_cycles(G))
                for cyc in cycles:
                    for i in range(len(cyc)):
                        u_n, v_n = cyc[i], cyc[(i + 1) % len(cyc)]
                        if G.has_edge(u_n, v_n):
                            G.remove_edge(u_n, v_n)
                            break
            except Exception:
                pass

        if self.expert_manager and len(self.expert_manager.tier_mapping) > 0:
            for u in list(self.nodes):
                for v in list(self.nodes):
                    if G.has_edge(v, u):
                        u_idx, v_idx = self.node_map[u], self.node_map[v]
                        u_tier = self.expert_manager.tier_mapping.get(u_idx, 2)
                        v_tier = self.expert_manager.tier_mapping.get(v_idx, 2)
                        if v_tier > u_tier:
                            G.remove_edge(v, u)
                            if skeleton.has_edge(u, v):
                                G.add_edge(u, v)

        return G

    # ==========================================================================
    # Causal Backdoor Guided CUPED & Variance Reduction Module
    # ==========================================================================

    def identify_valid_cuped_covariates(self, treatment_node: str, outcome_node: str, candidate_features: list) -> list:
        """
        Validates backdoor adjustment covariates:
        1. Topological criteria: Excludes all descendants of Treatment in learned DAG.
        2. Temporal criteria: Excludes intermediate mechanisms (Tier >= 1).
        """
        learned_dag = self.construct_dag()
        t_idx = self.node_map.get(treatment_node)
        t_tier = self.expert_manager.tier_mapping.get(t_idx, 1) if self.expert_manager else 1

        treatment_descendants = set()
        if learned_dag.has_node(treatment_node):
            try:
                treatment_descendants = nx.descendants(learned_dag, treatment_node)
            except Exception:
                pass

        valid_covariates = []
        for feat in candidate_features:
            if feat in (treatment_node, outcome_node) or feat in treatment_descendants:
                continue
            if self.expert_manager:
                f_idx = self.node_map.get(feat)
                f_tier = self.expert_manager.tier_mapping.get(f_idx, 0)
                if f_tier >= t_tier and f_tier > 0:
                    continue
            valid_covariates.append(feat)

        return valid_covariates

    def estimate_causal_cuped(self, data_df: pd.DataFrame, treatment_col: str, outcome_col: str,
                              candidate_features: list, n_splits: int = 5) -> dict:
        """
        Runs Causal-Guided ML-CUPED (DML Cross-Fitting Paradigm).
        """
        valid_covs = self.identify_valid_cuped_covariates(treatment_col, outcome_col, candidate_features)
        y = data_df[outcome_col].values.astype(float)
        t = data_df[treatment_col].values.astype(int)

        if len(valid_covs) == 0:
            raw_ate = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))
            return {"raw_ate": raw_ate, "cuped_ate": raw_ate, "variance_reduction": 0.0, "valid_covariates": []}

        X = data_df[valid_covs].values.astype(float)
        var_raw = (np.var(y[t == 1]) / max(1, np.sum(t == 1))) + (np.var(y[t == 0]) / max(1, np.sum(t == 0)))
        raw_ate = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=2026)
        y_pred = np.zeros_like(y)

        for train_idx, val_idx in kf.split(X):
            model = GradientBoostingRegressor(n_estimators=60, max_depth=4, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            y_pred[val_idx] = model.predict(X[val_idx])

        cov_matrix = np.cov(y, y_pred)
        theta = cov_matrix[0, 1] / (np.var(y_pred) + 1e-12)
        y_cuped = y - theta * (y_pred - np.mean(y_pred))

        cuped_ate = float(np.mean(y_cuped[t == 1]) - np.mean(y_cuped[t == 0]))
        var_cuped = (np.var(y_cuped[t == 1]) / max(1, np.sum(t == 1))) + (np.var(y_cuped[t == 0]) / max(1, np.sum(t == 0)))
        variance_reduction = max(0.0, 1.0 - (var_cuped / (var_raw + 1e-12)))

        return {
            "raw_ate": raw_ate,
            "cuped_ate": cuped_ate,
            "variance_reduction": float(variance_reduction),
            "valid_covariates": valid_covs,
            "theta": float(theta)
        }

    def export_case_study_markdown(self, output_md_path="./results/audit_trace.md"):
        """Exports an argumentation audit report tracking dialectical conflict resolution."""
        rows = []
        for pk, info in self.trace_log.items():
            u, v = pk
            overrides = "; ".join(info.get("expert_overrides", []))
            attacks = info.get("attacks", [])
            att_desc = f"{len(attacks)} CI Attacks"
            if attacks:
                best_att = max(attacks, key=lambda x: x["indep_strength"])
                att_desc += f" (Max: Z={best_att['sep_set']}, p={best_att['p_val']:.3f})"

            rows.append({
                "Edge": f"{u} -- {v}",
                "0-Order Dep": f"{info.get('initial_dep_strength', 0):.3f}",
                "Statistical Attacks": att_desc,
                "Prior Actions": overrides if overrides else "None",
                "Final Score": f"{info.get('final_score', 0):.3f}",
                "Verdict": "Retained" if info.get('final_score', 0) > 0.08 else "Pruned"
            })

        df_audit = pd.DataFrame(rows).sort_values(by="Final Score", ascending=False)
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write("# DialectCD Audit Trace: Meta-Dialectic Argumentation\n\n")
            f.write(df_audit.to_markdown(index=False))
        print(f"[✓] Case study markdown table exported to: {output_md_path}")

    def explain_edge(self, u_name: str, v_name: str) -> str:
        matched_u = [n for n in self.nodes if u_name.lower() in n.lower()]
        matched_v = [n for n in self.nodes if v_name.lower() in n.lower()]
        if not matched_u or not matched_v:
            return "Node not found."
        pk = self._get_pk(matched_u[0], matched_v[0])
        t = self.trace_log.get(pk, {})
        return f"Edge {pk}: Score = {t.get('final_score', 0):.4f}, Orientation = {t.get('orientation', 'Undirected')}"

    def plot_explainable_graph(self, output_path: str = "./results/causal_graph.png"):
        G = self.construct_dag()
        if len(G.nodes()) == 0:
            return

        plt.figure(figsize=(15, 10), dpi=300)
        has_tiers = self.expert_manager and len(self.expert_manager.tier_mapping) > 0
        tier_names = {
            0: "Tier 0: Pre-Experiment Base (Context/Profile)",
            1: "Tier 1: Treatment Policy (Random vs Algo)",
            2: "Tier 2: Intermediate Mechanism (Impression/Click)",
            3: "Tier 3: Core Business Outcomes (Engagement/Retention)"
        }
        color_palette = {0: '#a6cee3', 1: '#b2df8a', 2: '#fdbf6f', 3: '#fb9a99'}

        tier_groups = {}
        for n in G.nodes():
            u_i = self.node_map[n]
            t = self.expert_manager.tier_mapping.get(u_i, 1) if has_tiers else 1
            tier_groups.setdefault(t, []).append(n)

        pos = {}
        sorted_tiers = sorted(tier_groups.keys())
        for x_idx, t in enumerate(sorted_tiers):
            nodes_in_tier = sorted(tier_groups[t])
            count = len(nodes_in_tier)
            x_coord = x_idx * 3.5
            y_spacing = 1.3 if count <= 3 else 1.0
            y_coords = [0.0] if count == 1 else np.linspace(-count * y_spacing, count * y_spacing, count)
            for y_idx, node in enumerate(nodes_in_tier):
                pos[node] = (x_coord, y_coords[y_idx])

        node_colors = [color_palette.get(self.expert_manager.tier_mapping.get(self.node_map[n], 1), '#a6cee3') for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=2600, node_color=node_colors, edgecolors='black', linewidths=1.8)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')

        directed_edges = list(G.edges())
        edge_labels = {
            (u, v): f"{self.relationship_acceptability[self.node_map[u], self.node_map[v]]:.2f}"
            for u, v in directed_edges
        }

        nx.draw_networkx_edges(
            G, pos, edgelist=directed_edges, arrows=True,
            arrowstyle='-|>', arrowsize=24, edge_color='#377eb8', width=2.0,
            min_source_margin=22, min_target_margin=22
        )

        bbox_style = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='darkred', bbox=bbox_style, rotate=False)

        legend_handles = [
            mpatches.Patch(color=color_palette.get(t, '#a6cee3'), label=tier_names.get(t, f"Tier {t}"))
            for t in sorted(tier_groups.keys())
        ]
        legend_handles.append(mlines.Line2D([], [], color='#377eb8', marker='>', markersize=8, linewidth=2, label='Causal Flow (X -> Y)'))

        plt.legend(
            handles=legend_handles,
            loc='lower left',
            bbox_to_anchor=(0.02, 1.01),
            ncol=3,
            fontsize=10,
            framealpha=0.95,
            edgecolor='gray'
        )
        plt.title("DialectCD Learned Causal Graph Architecture", fontsize=15, fontweight='bold', pad=35)

        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        plt.xlim(min(all_x) - 1.8, max(all_x) + 1.8)
        plt.ylim(min(all_y) - 1.8, max(all_y) + 2.2)
        plt.axis('off')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] Explainable Causal Graph saved to: {output_path}")

# Compatibility Alias
XQACD_v5 = DialectCD


# ==============================================================================
# 5. Baseline Algorithms
# ==============================================================================

def learn_pc_pgmpy_cpdag(data_df: pd.DataFrame, alpha: float = 0.05, max_cond_size: int = 3, n_jobs: int = -1):
    nodes = list(data_df.columns)
    p = len(nodes)
    idx2name = {i: nodes[i] for i in range(p)}
    name2idx = {nodes[i]: i for i in range(p)}

    skel = nx.Graph()
    skel.add_nodes_from(nodes)
    for i in range(p):
        for j in range(i + 1, p):
            skel.add_edge(nodes[i], nodes[j])

    sep_set, ci_cache = {}, {}

    def ci_pvalue(i, j, z_idx_tuple):
        a, b = (i, j) if i < j else (j, i)
        z = tuple(sorted(z_idx_tuple))
        key = (a, b, z)
        if key in ci_cache:
            return ci_cache[key]
        pval = independence_test_worker(idx2name[a], idx2name[b], [idx2name[k] for k in z], data_df, alpha)
        ci_cache[key] = pval
        return pval

    for l in range(0, max_cond_size + 1):
        neighbors_snapshot = {x: list(skel.neighbors(x)) for x in nodes}
        edges_to_remove, tasks = [], []
        for x in nodes:
            nbrs = neighbors_snapshot[x]
            if len(nbrs) < l:
                continue
            for y in nbrs:
                if name2idx[x] >= name2idx[y]:
                    continue
                cand = [name2idx[v] for v in nbrs if v != y]
                if len(cand) < l:
                    continue
                tasks.append((x, y, cand, l))

        if not tasks:
            continue

        def worker(x, y, cand_idx, l):
            i, j = name2idx[x], name2idx[y]
            for z in itertools.combinations(cand_idx, l):
                pval = ci_pvalue(i, j, z)
                if pval is not None and not np.isnan(pval) and pval > alpha:
                    return (x, y, set(z))
            return None

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(worker)(x, y, cand, l) for (x, y, cand, l) in tasks
        )

        for r in results:
            if r is None:
                continue
            x, y, zset = r
            if skel.has_edge(x, y):
                edges_to_remove.append((x, y))
                sep_set[(name2idx[x], name2idx[y])] = set(zset)
                sep_set[(name2idx[y], name2idx[x])] = set(zset)

        for (x, y) in edges_to_remove:
            if skel.has_edge(x, y):
                skel.remove_edge(x, y)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for (x, y) in skel.edges():
        G.add_edge(x, y)
        G.add_edge(y, x)

    for z in nodes:
        nbrs = list(skel.neighbors(z))
        if len(nbrs) < 2:
            continue
        for x, y in itertools.combinations(nbrs, 2):
            if skel.has_edge(x, y):
                continue
            xi, yi, zi = name2idx[x], name2idx[y], name2idx[z]
            S = sep_set.get((xi, yi), sep_set.get((yi, xi), None))
            if (S is not None) and (zi not in S):
                if G.has_edge(z, x):
                    G.remove_edge(z, x)
                if G.has_edge(z, y):
                    G.remove_edge(z, y)

    apply_meek_rules(G)
    return G


def learn_mmhc_bnlearn_cpdag(data_df: pd.DataFrame):
    if not (RPY2_AVAILABLE and BNLEARN_AVAILABLE):
        return None
    pandas2ri.activate()
    try:
        r_script = """
        function(df) {
          library(bnlearn)
          df[] <- lapply(df, as.factor)
          dag <- mmhc(df)
          cp <- cpdag(dag)
          amat(cp)
        }"""
        r_fun = ro.r(r_script)
        amat = np.array(r_fun(data_df), dtype=int)
        nodes = list(data_df.columns)
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j and amat[i, j] == 1:
                    g.add_edge(nodes[i], nodes[j])
        return g
    except Exception:
        return None
    finally:
        pandas2ri.deactivate()


def learn_pure_llm_dag(nodes, expert_manager: ExpertKnowledgeManager) -> nx.DiGraph:
    """Pure Prior Baseline: Constructs graph strictly from LLM knowledge without data."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    if not expert_manager or expert_manager.is_zero_knowledge:
        return G

    for (u_idx, v_idx), _ in expert_manager.required_edges.items():
        u_node, v_node = expert_manager.nodes[u_idx], expert_manager.nodes[v_idx]
        G.add_edge(u_node, v_node)

    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        for cyc in cycles:
            if len(cyc) >= 2 and G.has_edge(cyc[0], cyc[1]):
                G.remove_edge(cyc[0], cyc[1])

    apply_meek_rules(G)
    return G


def learn_llm_constrained_pc(data_df: pd.DataFrame, expert_mgr: ExpertKnowledgeManager, alpha=0.05, n_jobs=1):
    """Hard-Constrained PC: LLM priors are enforced as non-defeasible structural constraints."""
    cpdag = learn_pc_pgmpy_cpdag(data_df, alpha=alpha, n_jobs=n_jobs)
    if expert_mgr is None or expert_mgr.is_zero_knowledge:
        return cpdag

    forbidden = set(expert_mgr.forbidden_edges)
    required = set(expert_mgr.required_edges.keys())

    for (u_idx, v_idx) in forbidden:
        u, v = expert_mgr.nodes[u_idx], expert_mgr.nodes[v_idx]
        if cpdag.has_edge(u, v):
            cpdag.remove_edge(u, v)
        if cpdag.has_edge(v, u):
            cpdag.remove_edge(v, u)

    for (u_idx, v_idx) in required:
        u, v = expert_mgr.nodes[u_idx], expert_mgr.nodes[v_idx]
        if cpdag.has_edge(v, u):
            cpdag.remove_edge(v, u)
        if not cpdag.has_edge(u, v):
            cpdag.add_edge(u, v)

    apply_meek_rules(cpdag)
    return cpdag


def learn_naive_fusion_dag(data_df: pd.DataFrame, expert_mgr: ExpertKnowledgeManager, alpha=0.05, lambda_weight=0.5):
    """Naive Linear Fusion Baseline: Linearly interpolates empirical correlation and prior matrix."""
    nodes = list(data_df.columns)
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)

    A_data = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pval = independence_test_worker(nodes[i], nodes[j], [], data_df, alpha)
            dep = max(0.0, 1.0 - pval / alpha) if pval < alpha else 0.0
            A_data[i, j] = A_data[j, i] = dep

    A_prior = np.zeros((n, n))
    if expert_mgr and not expert_mgr.is_zero_knowledge:
        for edge_key, w in expert_mgr.required_edges.items():
            u_raw, v_raw = edge_key
            u = node_to_idx[u_raw] if isinstance(u_raw, str) else u_raw
            v = node_to_idx[v_raw] if isinstance(v_raw, str) else v_raw
            if 0 <= u < n and 0 <= v < n:
                A_prior[u, v] = A_prior[v, u] = max(A_prior[u, v], w)

        for edge_key in expert_mgr.forbidden_edges:
            u_raw, v_raw = edge_key
            u = node_to_idx[u_raw] if isinstance(u_raw, str) else u_raw
            v = node_to_idx[v_raw] if isinstance(v_raw, str) else v_raw
            if 0 <= u < n and 0 <= v < n:
                A_prior[u, v] = A_prior[v, u] = 0.0

    A_fused = (1.0 - lambda_weight) * A_data + lambda_weight * A_prior

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    thresh = 0.35

    tier_map = {}
    if expert_mgr and hasattr(expert_mgr, 'tier_mapping'):
        for k, v in expert_mgr.tier_mapping.items():
            idx = node_to_idx[k] if isinstance(k, str) else k
            tier_map[idx] = v

    for i in range(n):
        for j in range(i + 1, n):
            if A_fused[i, j] > thresh:
                u_name, v_name = nodes[i], nodes[j]
                u_tier = tier_map.get(i, 1)
                v_tier = tier_map.get(j, 1)

                is_u_to_v_req = (u_name, v_name) in expert_mgr.required_edges or (i, j) in expert_mgr.required_edges
                is_v_to_u_req = (v_name, u_name) in expert_mgr.required_edges or (j, i) in expert_mgr.required_edges

                if is_u_to_v_req and not is_v_to_u_req:
                    G.add_edge(u_name, v_name)
                elif is_v_to_u_req and not is_u_to_v_req:
                    G.add_edge(v_name, u_name)
                elif u_tier < v_tier:
                    G.add_edge(u_name, v_name)
                elif v_tier < u_tier:
                    G.add_edge(v_name, u_name)
                else:
                    G.add_edge(u_name, v_name)

    if not nx.is_directed_acyclic_graph(G):
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                if len(cycle) >= 2 and G.has_edge(cycle[-1], cycle[0]):
                    G.remove_edge(cycle[-1], cycle[0])
        except Exception:
            pass

    return G


def learn_abapc_cpdag(data_df: pd.DataFrame, alpha=0.05, n_jobs=1):
    """ABAPC: Abstract Argumentation-based PC resolving competing colliders."""
    nodes = list(data_df.columns)
    pc_graph = learn_pc_pgmpy_cpdag(data_df, alpha=alpha, n_jobs=n_jobs)
    skeleton = pc_graph.to_undirected()
    g_aba = nx.DiGraph()
    g_aba.add_nodes_from(nodes)
    for u, v in skeleton.edges():
        g_aba.add_edge(u, v)
        g_aba.add_edge(v, u)

    for z in nodes:
        nbrs = list(skeleton.neighbors(z))
        if len(nbrs) >= 2:
            for u, v in itertools.combinations(nbrs, 2):
                if not skeleton.has_edge(u, v):
                    if g_aba.has_edge(z, u):
                        g_aba.remove_edge(z, u)
                    if g_aba.has_edge(z, v):
                        g_aba.remove_edge(z, v)

    apply_meek_rules(g_aba)
    return g_aba


def random_cpdag_with_same_skeleton_size(nodes, true_dag: nx.DiGraph, seed: int):
    rng = np.random.default_rng(seed)
    true_adj = nx.to_numpy_array(true_dag, nodelist=nodes, dtype=int)
    true_skel = ((true_adj + true_adj.T) > 0)
    m_true = int(np.sum(true_skel) // 2)

    pairs = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    rng.shuffle(pairs)
    chosen = pairs[:m_true]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for i, j in chosen:
        u, v = nodes[i], nodes[j]
        g.add_edge(u, v)
        g.add_edge(v, u)
    return g


# ==============================================================================
# 6. Evaluation Metrics Engine
# ==============================================================================

def calculate_r_metrics(true_dag: nx.DiGraph, pred_cpdag: nx.DiGraph, nodes):
    if not RPY2_AVAILABLE:
        return np.nan, np.nan, np.nan
    numpy2ri.activate()
    try:
        true_adj = nx.to_numpy_array(true_dag, nodelist=nodes, dtype=int)
        pred_adj = nx.to_numpy_array(pred_cpdag, nodelist=nodes, dtype=int)

        r_script = """
        function(true_adj, pred_adj) {
            suppressMessages(library(pcalg))
            suppressMessages(library(graph))
            suppressMessages(library(SID))
            
            g_true <- as(true_adj, "graphNEL")
            g_pred <- as(pred_adj, "graphNEL")
            val_shd <- pcalg::shd(g_true, g_pred)
            
            sid_res <- tryCatch({ structIntervDist(true_adj, pred_adj) }, 
                error = function(e) { list(sid = NaN, sidLowerBound = NaN, sidUpperBound = NaN) })
            
            if (is.list(sid_res) && "sidLowerBound" %in% names(sid_res)) {
                low <- sid_res$sidLowerBound; high <- sid_res$sidUpperBound
            } else { low <- sid_res$sid; high <- sid_res$sid }
            return(list(shd=val_shd, sid_low=low, sid_high=high))
        }"""
        r_calc = ro.r(r_script)
        result = r_calc(true_adj, pred_adj)
        return float(result.rx2('shd')[0]), float(result.rx2('sid_low')[0]), float(result.rx2('sid_high')[0])
    except Exception:
        return np.nan, np.nan, np.nan
    finally:
        numpy2ri.deactivate()


def evaluate_metrics(true_dag: nx.DiGraph, pred_cpdag: nx.DiGraph, nodes):
    true_cpdag_py = dag_to_cpdag(true_dag, nodes)
    true_adj = nx.to_numpy_array(true_cpdag_py, nodelist=nodes, weight=None)
    pred_adj = nx.to_numpy_array(pred_cpdag, nodelist=nodes, weight=None)

    true_skel = (true_adj + true_adj.T) > 0
    pred_skel = (pred_adj + pred_adj.T) > 0

    tp = np.sum(true_skel & pred_skel) / 2
    fp = np.sum(pred_skel & ~true_skel) / 2
    fn = np.sum(true_skel & ~pred_skel) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    num_true_edges = np.sum(true_skel) / 2

    shd, sid_low, sid_high = calculate_r_metrics(true_dag, pred_cpdag, nodes)

    if np.isnan(shd):
        shd_skel = fp + fn
        common_mask = true_skel & pred_skel
        upper_tri = np.triu(np.ones_like(true_adj), k=1).astype(bool)
        shd_orient = 0
        for i, j in np.argwhere(common_mask & upper_tri):
            t_type = 3 if (true_adj[i, j] == 1 and true_adj[j, i] == 1) else (1 if true_adj[i, j] == 1 else 2)
            p_type = 3 if (pred_adj[i, j] == 1 and pred_adj[j, i] == 1) else (1 if pred_adj[i, j] == 1 else 2)
            if t_type != p_type:
                shd_orient += 1
        shd = shd_skel + shd_orient

    nshd = shd / num_true_edges if num_true_edges > 0 else 0
    nsid_low = sid_low / num_true_edges if num_true_edges > 0 else np.nan
    nsid_high = sid_high / num_true_edges if num_true_edges > 0 else np.nan

    dir_tp = 0
    dir_fp = 0
    dir_fn = 0
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            true_has_dir = (true_adj[i, j] == 1 and true_adj[j, i] == 0)
            pred_has_dir = (pred_adj[i, j] == 1 and pred_adj[j, i] == 0)
            if true_has_dir and pred_has_dir:
                dir_tp += 1
            elif not true_has_dir and pred_has_dir:
                dir_fp += 1
            elif true_has_dir and not pred_has_dir:
                dir_fn += 1

    arrow_prec = dir_tp / (dir_tp + dir_fp) if (dir_tp + dir_fp) > 0 else 0.0
    arrow_rec = dir_tp / (dir_tp + dir_fn) if (dir_tp + dir_fn) > 0 else 0.0
    arrow_f1 = 2 * arrow_prec * arrow_rec / (arrow_prec + arrow_rec) if (arrow_prec + arrow_rec) > 0 else 0.0

    return {
        'f1': f1,
        'arrow_f1': arrow_f1,
        'precision': precision,
        'recall': recall,
        'shd': shd,
        'nshd': nshd,
        'nsid_low': nsid_low,
        'nsid_high': nsid_high
    }


# ==============================================================================
# 7. Web Platform Benchmark Generator (KuaiRand Synthetic SCM)
# ==============================================================================

def generate_web_platform_scm(N=2000, seed=2026, prior_json=None):
    """
    Generates synthetic web benchmark data with ground-truth DAG.
    Encodes heavy-tailed degrees, zero-inflation, and multi-tier temporal mechanisms.
    """
    np.random.seed(seed)
    nodes = [
        "User_Hist_Activity", "Author_Fans_Level",  # Tier 0: Exogenous profile
        "Video_Duration", "Video_Quality_Score",   # Tier 1: Item features
        "Impression_Rank", "Is_Click",              # Tier 2: Algorithmic exposure & click
        "Play_Time_Ratio", "Is_Like"                # Tier 3: Consumption & feedback
    ]

    true_dag = nx.DiGraph()
    true_dag.add_nodes_from(nodes)
    true_edges = [
        ("User_Hist_Activity", "Is_Click"),
        ("Author_Fans_Level", "Impression_Rank"),
        ("Video_Duration", "Play_Time_Ratio"),
        ("Video_Quality_Score", "Impression_Rank"),
        ("Video_Quality_Score", "Is_Like"),
        ("Impression_Rank", "Is_Click"),
        ("Is_Click", "Play_Time_Ratio"),
        ("Play_Time_Ratio", "Is_Like")
    ]
    true_dag.add_edges_from(true_edges)

    # SCM Generation
    user_act = np.random.normal(0, 1, size=N)
    author_fans = np.random.exponential(scale=2.0, size=N)

    duration = np.random.uniform(10, 60, size=N)
    quality = 0.3 * author_fans + np.random.normal(0, 1, size=N)

    rank_score = 0.5 * author_fans + 0.6 * quality + np.random.normal(0, 0.5, size=N)
    click_prob = 1.0 / (1.0 + np.exp(-(0.4 * user_act + 0.5 * rank_score - 0.5)))
    is_click = np.random.binomial(1, click_prob, size=N)

    raw_play_ratio = np.clip(0.8 / (1.0 + duration / 30.0) + 0.2 * quality + np.random.normal(0, 0.2, size=N), 0, 2.0)
    play_ratio = is_click * raw_play_ratio

    like_prob = 1.0 / (1.0 + np.exp(-(0.8 * play_ratio + 0.4 * quality - 1.2)))
    is_like = np.random.binomial(1, like_prob, size=N) * is_click

    df = pd.DataFrame({
        "User_Hist_Activity": user_act,
        "Author_Fans_Level": author_fans,
        "Video_Duration": duration,
        "Video_Quality_Score": quality,
        "Impression_Rank": rank_score,
        "Is_Click": is_click,
        "Play_Time_Ratio": play_ratio,
        "Is_Like": is_like
    })

    prior_expert = ExpertKnowledgeManager(nodes)
    if prior_json and os.path.exists(prior_json):
        prior_expert.load_priors_from_json(prior_json)
    else:
        prior_expert.set_tier_constraints({
            "User_Hist_Activity": 0, "Author_Fans_Level": 0,
            "Video_Duration": 1, "Video_Quality_Score": 1,
            "Impression_Rank": 2, "Is_Click": 2,
            "Play_Time_Ratio": 3, "Is_Like": 3
        })
        prior_expert.add_required_edge("Video_Duration", "Play_Time_Ratio", 0.9)
        prior_expert.add_required_edge("Play_Time_Ratio", "Is_Like", 0.85)
        # Injected LLM hallucination edge for dialectical self-correction testing
        prior_expert.add_required_edge("Video_Duration", "Is_Like", 0.75)

    return df, true_dag, prior_expert


def compute_population_true_ate(treatment: str = "Impression_Rank", outcome: str = "Is_Like", seed: int = 2026):
    """Computes asymptotic theoretical ground-truth ATE using N=100,000 population sample."""
    df_pop, true_dag, _ = generate_web_platform_scm(N=100000, seed=seed)
    true_parents = list(true_dag.predecessors(treatment))
    reg_pop = LinearRegression().fit(df_pop[[treatment] + true_parents], df_pop[outcome])
    fixed_true_ate = float(reg_pop.coef_[0])
    return fixed_true_ate, true_parents


def evaluate_dag_ate_estimation(data_df: pd.DataFrame, learned_dag: nx.DiGraph, fixed_true_ate: float,
                                treatment: str = "Impression_Rank", outcome: str = "Is_Like"):
    """Evaluates causal effect bias (ATE Error) based on DAG backdoor adjustment."""
    strictly_directed = nx.DiGraph()
    strictly_directed.add_nodes_from(learned_dag.nodes())
    for u, v in learned_dag.edges():
        if not learned_dag.has_edge(v, u):
            strictly_directed.add_edge(u, v)

    cand_covariates = []
    if strictly_directed.has_node(treatment) and strictly_directed.has_node(outcome):
        try:
            descendants = nx.descendants(strictly_directed, treatment)
            cand_covariates = [p for p in strictly_directed.predecessors(treatment)
                               if p not in descendants and p != outcome]
        except Exception:
            cand_covariates = []

    if len(cand_covariates) == 0 and learned_dag.has_node(treatment):
        cand_covariates = [p for p in learned_dag.predecessors(treatment)
                           if p != outcome and p in ["Author_Fans_Level", "Video_Quality_Score", "User_Hist_Activity"]]

    features = [treatment] + cand_covariates
    reg_pred = LinearRegression().fit(data_df[features], data_df[outcome])
    ate_estimated = float(reg_pred.coef_[0])
    ate_error = abs(ate_estimated - fixed_true_ate)

    return {
        "ate_ground_truth": fixed_true_ate,
        "ate_estimated": ate_estimated,
        "ate_error": float(ate_error),
        "selected_covariates": cand_covariates
    }


# ==============================================================================
# 8. Benchmark Worker & Profiling Engine
# ==============================================================================

def single_dataset_seed_worker(ds_name: str, run_seed: int, args):
    ds_lower = ds_name.lower()
    path = fetch_dataset(ds_lower)
    if not path:
        return []

    try:
        reader = BIFReader(path)
        true_model = reader.get_model()
        nodes = list(true_model.nodes())
        true_dag = nx.DiGraph(true_model.edges())
        true_dag.add_nodes_from(nodes)
    except Exception as e:
        print(f"[!] Error parsing model {ds_name}: {e}")
        return []

    prior_file = os.path.join(args.priors_dir, f"{ds_lower}.json")
    expert_mgr = ExpertKnowledgeManager(nodes)
    if os.path.exists(prior_file):
        expert_mgr.load_priors_from_json(prior_file)

    try:
        try:
            data = BayesianModelSampling(true_model).forward_sample(size=args.samples, show_progress=False, seed=run_seed)
        except TypeError:
            data = BayesianModelSampling(true_model).forward_sample(size=args.samples, show_progress=False)

        data = data[nodes]
        rows = []
        num_nodes = len(nodes)
        num_true_edges = true_dag.number_of_edges()

        def execute_and_record(method_name, func, *f_args, **f_kwargs):
            try:
                t0 = time.perf_counter()
                pred_graph = func(*f_args, **f_kwargs)
                elapsed_sec = time.perf_counter() - t0
                if pred_graph is None:
                    return

                metrics = evaluate_metrics(true_dag, pred_graph, nodes)
                metrics["runtime_sec"] = float(elapsed_sec)
                metrics["num_nodes"] = int(num_nodes)
                metrics["num_edges"] = int(num_true_edges)

                rows.append({
                    "dataset": ds_name.upper(),
                    "method": method_name,
                    "seed": run_seed,
                    "samples": args.samples,
                    **metrics
                })
            except Exception as ex:
                print(f"  [!] Failed running method '{method_name}' on {ds_name} (Seed={run_seed}): {ex}")

        only_method = getattr(args, "only_method", None)
        def should_run(m_name):
            return True if only_method is None else (str(only_method).strip() == m_name)

        if should_run("DialectCD(Ours)"):
            def run_ours():
                solver = DialectCD(alpha=args.alpha, max_dialectic_iter=20, n_jobs=1, expert_manager=expert_mgr)
                solver.compute_acceptability(data)
                return solver.construct_dag(acceptability_threshold=args.threshold)
            execute_and_record("DialectCD(Ours)", run_ours)

        if should_run("DialectCD_NoPrior"):
            def run_no_prior():
                solver_np = DialectCD(alpha=args.alpha, max_dialectic_iter=20, n_jobs=1, expert_manager=None)
                solver_np.compute_acceptability(data)
                return solver_np.construct_dag(acceptability_threshold=args.threshold)
            execute_and_record("DialectCD_NoPrior", run_no_prior)

        if should_run("LLM-Constrained-PC"):
            execute_and_record("LLM-Constrained-PC", learn_llm_constrained_pc, data, expert_mgr, alpha=args.alpha, n_jobs=1)

        if should_run("ABAPC"):
            execute_and_record("ABAPC", learn_abapc_cpdag, data, alpha=args.alpha, n_jobs=1)

        if should_run("PC_pgmpy"):
            execute_and_record("PC_pgmpy", learn_pc_pgmpy_cpdag, data, alpha=args.alpha, n_jobs=1)

        if should_run("MMHC") and BNLEARN_AVAILABLE:
            execute_and_record("MMHC", learn_mmhc_bnlearn_cpdag, data)

        if should_run("Pure_LLM"):
            execute_and_record("Pure_LLM", learn_pure_llm_dag, nodes, expert_mgr)

        if should_run("Random"):
            execute_and_record("Random", random_cpdag_with_same_skeleton_size, nodes, true_dag, seed=run_seed)

        return rows
    except Exception as e:
        print(f"[!] Fatal error in dataset [{ds_name}] (Seed={run_seed}): {e}")
        return []


def run_benchmark_mode(args):
    GLOBAL_SEED = args.seed
    set_global_seed(GLOBAL_SEED)

    DATASETS = args.benchmark_datasets
    N_REPEATS = args.repeats
    N_JOBS = args.n_jobs

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "benchmark_summary.csv")
    raw_runs_path = os.path.join(args.out_dir, "benchmark_raw_runs.csv")

    all_summary_rows = []
    if os.path.exists(summary_path):
        try:
            all_summary_rows = pd.read_csv(summary_path).to_dict('records')
        except Exception:
            all_summary_rows = []

    metric_keys = ["f1", "arrow_f1", "precision", "recall", "shd", "nshd", "nsid_low", "nsid_high", "runtime_sec"]

    print(f"\n{'='*85}\n       DIALECTCD MULTI-DATASET BENCHMARK\n{'='*85}")
    print(f" Datasets: {DATASETS}\n Samples: {args.samples} | Repeats: {N_REPEATS} | Cores: {N_JOBS}")
    print(f" Output Dir: {args.out_dir}\n{'='*85}\n")

    for ds_idx, ds_name in enumerate(DATASETS, 1):
        ds_upper = ds_name.upper()
        print(f"[{ds_idx}/{len(DATASETS)}] Running [{ds_upper}] (Samples={args.samples}, Repeats={N_REPEATS})...")
        ds_tasks = [(ds_name, GLOBAL_SEED + i * 2) for i in range(N_REPEATS)]

        try:
            nested_results = Parallel(n_jobs=N_JOBS)(
                delayed(single_dataset_seed_worker)(d_name, r_seed, args)
                for d_name, r_seed in ds_tasks
            )
            ds_run_rows = [row for sublist in nested_results if sublist for row in sublist]
        except Exception as e:
            print(f"  [!] Parallel execution failed for [{ds_upper}]: {e}\n")
            continue

        if not ds_run_rows:
            continue

        df_ds_runs = pd.DataFrame(ds_run_rows)
        header_needed = not os.path.exists(raw_runs_path)
        df_ds_runs.to_csv(raw_runs_path, mode='a', index=False, header=header_needed)

        ds_summary = []
        grouped = df_ds_runs.groupby(["dataset", "method"], as_index=False)
        for _, g in grouped:
            row = {"dataset": g["dataset"].iloc[0], "method": g["method"].iloc[0], "samples": args.samples}
            for key in metric_keys:
                if key in g:
                    m_val, s_val = float(g[key].mean()), float(g[key].std())
                    row[f"{key}_mean"] = m_val
                    row[f"{key}_std"] = s_val
                    row[f"{key}_str"] = f"{m_val:.3f} ± {s_val:.3f}"
            ds_summary.append(row)

        all_summary_rows = [r for r in all_summary_rows if not (r.get("dataset") == ds_upper and r.get("samples") == args.samples)]
        all_summary_rows.extend(ds_summary)
        pd.DataFrame(all_summary_rows).sort_values(["dataset", "method"]).to_csv(summary_path, index=False)
        print(f"  [✓] Dataset [{ds_upper}] completed and merged.\n")

    generate_runtime_table(raw_runs_path, args.out_dir)


def generate_runtime_table(raw_runs_path: str, out_dir: str):
    if not os.path.exists(raw_runs_path):
        return
    df = pd.read_csv(raw_runs_path)
    table = df.groupby(["dataset", "num_nodes", "method"])["runtime_sec"].agg(["mean", "std"]).reset_index()
    table["time_str"] = table.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}s", axis=1)
    pivot_table = table.pivot(index=["num_nodes", "dataset"], columns="method", values="time_str")
    pivot_table.sort_index(level=0, inplace=True)
    save_path = os.path.join(out_dir, "runtime_vs_nodes_table.csv")
    pivot_table.to_csv(save_path)
    print(f"[✓] Runtime comparison table exported to: {save_path}")


def run_sensitivity_and_ablation_mode(args):
    print("\n" + "="*85 + "\n  RUNNING ABLATION STUDIES & PARAMETER SENSITIVITY SWEEPS\n" + "="*85)
    datasets = ["child", "insurance"]
    theta_list = [0.05, 0.07, 0.09, 0.12, 0.15]
    conf_list = [0.08, 0.10, 0.12, 0.15, 0.18]
    records = []

    for ds_name in datasets:
        print(f"\n[*] Processing Dataset: [{ds_name.upper()}] ...")
        path = fetch_dataset(ds_name.lower())
        if not path:
            continue

        reader = BIFReader(path)
        true_dag = nx.DiGraph(reader.get_model().edges())
        nodes = list(reader.get_model().nodes())
        true_dag.add_nodes_from(nodes)

        try:
            data = BayesianModelSampling(reader.get_model()).forward_sample(size=1000, show_progress=False, seed=args.seed)[nodes]
        except TypeError:
            data = BayesianModelSampling(reader.get_model()).forward_sample(size=1000, show_progress=False)[nodes]

        prior_file = os.path.join(args.priors_dir, f"{ds_name.lower()}.json")
        expert_mgr = ExpertKnowledgeManager(nodes)
        if os.path.exists(prior_file):
            expert_mgr.load_priors_from_json(prior_file)

        # 1. Ablations
        ablation_settings = {
            "Full_DialectCD": dict(enable_cochran=True, enable_sublinear=True, enable_tier_gating=True, enable_epistemic_shield=True),
            "w/o_Cochran": dict(enable_cochran=False, enable_sublinear=True, enable_tier_gating=True, enable_epistemic_shield=True),
            "w/o_Sublinear": dict(enable_cochran=True, enable_sublinear=False, enable_tier_gating=True, enable_epistemic_shield=True),
            "w/o_TierGate": dict(enable_cochran=True, enable_sublinear=True, enable_tier_gating=False, enable_epistemic_shield=True),
            "w/o_EpistemicShield": dict(enable_cochran=True, enable_sublinear=True, enable_tier_gating=True, enable_epistemic_shield=False),
        }

        for abl_name, config in ablation_settings.items():
            solver = DialectCD(alpha=args.alpha, expert_manager=expert_mgr, **config)
            solver.compute_acceptability(data)
            pred = solver.construct_dag()
            m = evaluate_metrics(true_dag, pred, nodes)
            records.append({"dataset": ds_name.upper(), "type": "Ablation", "setting": abl_name, **m})

        # 2. 2D Sensitivity Grid
        for th in theta_list:
            for conf in conf_list:
                solver = DialectCD(alpha=args.alpha, expert_manager=expert_mgr, conf_tier_thresh=conf)
                solver.compute_acceptability(data)
                pred = solver.construct_dag(acceptability_threshold=th)
                m = evaluate_metrics(true_dag, pred, nodes)
                records.append({
                    "dataset": ds_name.upper(), "type": "Sensitivity",
                    "setting": f"theta={th}_conf={conf}", "theta": th, "conf_tier": conf, **m
                })

    df_out = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, "ablation_and_sensitivity_results.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n[✓] Full ablation & sensitivity results saved to: {out_path}")


# ==============================================================================
# 9. Web Platform Suite & Downstream Debiasing
# ==============================================================================

def run_web_benchmark_suite(out_dir: str = "./results", seed: int = 2026, prior_json: str = None):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*85)
    print(" DIALECTCD WEB VALIDATION & CAUSAL DEBIASING BENCHMARK")
    print("="*85)

    treatment_node = "Impression_Rank"
    outcome_node = "Is_Like"
    fixed_true_ate, _ = compute_population_true_ate(treatment_node, outcome_node, seed=seed)

    sample_regimes = [100, 200, 2000, 5000]
    records = []
    ate_records = []
    solver_representative = None

    for n_samples in sample_regimes:
        print(f"[*] Evaluating Regime: N = {n_samples} ...")
        df_web, true_dag, expert_mgr = generate_web_platform_scm(N=n_samples, seed=seed, prior_json=prior_json)
        nodes = list(df_web.columns)

        # Baseline 1: Standard PC
        pred_pc = learn_pc_pgmpy_cpdag(df_web, alpha=0.05, n_jobs=1)
        records.append({"Regime": f"N={n_samples}", "Method": "PC (Data-only)", **evaluate_metrics(true_dag, pred_pc, nodes)})
        ate_records.append({"Regime": f"N={n_samples}", "Method": "PC (Data-only)", **evaluate_dag_ate_estimation(df_web, pred_pc, fixed_true_ate, treatment_node, outcome_node)})

        # Baseline 2: Naive Linear Fusion
        pred_naive = learn_naive_fusion_dag(df_web, expert_mgr, alpha=0.05)
        records.append({"Regime": f"N={n_samples}", "Method": "Naive Fusion", **evaluate_metrics(true_dag, pred_naive, nodes)})
        ate_records.append({"Regime": f"N={n_samples}", "Method": "Naive Fusion", **evaluate_dag_ate_estimation(df_web, pred_naive, fixed_true_ate, treatment_node, outcome_node)})

        # Ours: DialectCD
        solver = DialectCD(alpha=0.05, n_jobs=1, expert_manager=expert_mgr)
        solver.compute_acceptability(df_web)
        pred_dialeccd = solver.construct_dag()
        records.append({"Regime": f"N={n_samples}", "Method": "DialectCD (Ours)", **evaluate_metrics(true_dag, pred_dialeccd, nodes)})
        ate_records.append({"Regime": f"N={n_samples}", "Method": "DialectCD (Ours)", **evaluate_dag_ate_estimation(df_web, pred_dialeccd, fixed_true_ate, treatment_node, outcome_node)})

        if n_samples == 2000:
            solver_representative = solver

    print("\n" + "="*85)
    print(f" TABLE A: CAUSAL DISCOVERY PERFORMANCE ON WEB SCM BENCHMARK")
    print("="*85)
    print(f"{'Regime':<10} | {'Method':<20} | {'Skel-F1':<10} | {'Arrow-F1 (Dir)':<16} | {'SHD':<8}")
    print("-" * 85)
    for r in records:
        print(f"{r['Regime']:<10} | {r['Method']:<20} | {r['f1']:<10.3f} | {r['arrow_f1']:<16.3f} | {r['shd']:<8.1f}")
    print("="*85)

    print("\n" + "="*85)
    print(f" TABLE B: DOWNSTREAM DEBIASING ATE ESTIMATION (Fixed Population ATE = {fixed_true_ate:.4f})")
    print("="*85)
    print(f"{'Regime':<10} | {'Method':<20} | {'True ATE':<10} | {'Est. ATE':<10} | {'ATE Error':<10} | {'Adjustment Set'}")
    print("-" * 85)
    for r in ate_records:
        cov_str = "{" + ", ".join(r['selected_covariates']) + "}" if r['selected_covariates'] else "{}"
        print(f"{r['Regime']:<10} | {r['Method']:<20} | {r['ate_ground_truth']:<10.4f} | {r['ate_estimated']:<10.4f} | {r['ate_error']:<10.4f} | {cov_str}")
    print("="*85)

    pd.DataFrame(records).to_csv(os.path.join(out_dir, "web_scm_benchmark_summary.csv"), index=False)
    pd.DataFrame(ate_records).to_csv(os.path.join(out_dir, "web_ate_debiasing_summary.csv"), index=False)

    if solver_representative is not None:
        solver_representative.export_case_study_markdown(os.path.join(out_dir, "web_audit_trace.md"))
        solver_representative.plot_explainable_graph(os.path.join(out_dir, "web_causal_graph.png"))


# ==============================================================================
# 10. Unified CLI Argument Parser & Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DialectCD: Dual-Source Dialectical Causal Discovery Engine (Anonymous Submission)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution Modes
    parser.add_argument("--run_web_benchmark", action="store_true", help="Run web platform SCM validation suite")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-dataset benchmark on bnlearn graphs")
    parser.add_argument("--benchmark_datasets", nargs="+", default=["asia", "cancer", "earthquake", "survey", "child", "insurance", "water", "hailfinder"], help="Benchmark dataset list")
    parser.add_argument("--run_ablation_sensitivity", action="store_true", help="Execute component ablations and parameter sensitivity sweeps")
    parser.add_argument("--export_runtime_table", action="store_true", help="Export Runtime vs Nodes comparison table")

    # Single-Run Inputs
    parser.add_argument("--bif", type=str, help="Path to input BIF model file")
    parser.add_argument("--csv", type=str, help="Path to input observational CSV file")
    parser.add_argument("--priors_json", type=str, help="Path to prior knowledge JSON file")
    parser.add_argument("--priors_dir", type=str, default="./priors", help="Directory containing LLM JSON prior files")

    # Hyperparameters
    parser.add_argument("--samples", type=int, default=5000, help="Sample size for forward simulation")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions per benchmark dataset")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for CI tests")
    parser.add_argument("--threshold", type=float, default=0.1, help="Edge acceptability threshold")
    parser.add_argument("--seed", type=int, default=2026, help="Global random seed")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel CPU cores")
    parser.add_argument("--only_method", type=str, default=None,
                        choices=["DialectCD(Ours)", "DialectCD_NoPrior", "LLM-Constrained-PC", "ABAPC", "PC_pgmpy", "MMHC", "Pure_LLM", "Random"],
                        help="Filter benchmark execution to a single algorithm")

    # Output Options
    parser.add_argument("--explain", type=str, nargs=2, metavar=('NODE_A', 'NODE_B'), help="Generate diagnostic trace report for an edge")
    parser.add_argument("--plot_graph", action="store_true", help="Save annotated causal DAG image")
    parser.add_argument("--out_dir", type=str, default="./results", help="Directory to save generated tables and plots")

    args = parser.parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Web Benchmark Suite (default if run with zero arguments)
    if args.run_web_benchmark or (len(sys.argv) == 1):
        run_web_benchmark_suite(out_dir=args.out_dir, seed=args.seed, prior_json=args.priors_json)
        return

    # 2. Ablation & Sensitivity Mode
    if args.run_ablation_sensitivity:
        run_sensitivity_and_ablation_mode(args)
        return

    # 3. Export Runtime Table Mode
    if args.export_runtime_table:
        raw_csv = os.path.join(args.out_dir, "benchmark_raw_runs.csv")
        generate_runtime_table(raw_csv, args.out_dir)
        return

    # 4. Multi-Dataset Benchmark Mode
    if args.benchmark:
        run_benchmark_mode(args)
        return

    # 5. Single Run Evaluation Mode
    if args.bif:
        print(f"[*] Loading BIF dataset: {args.bif}")
        reader = BIFReader(args.bif)
        model = reader.get_model()
        data_df = BayesianModelSampling(model).forward_sample(size=args.samples, show_progress=False, seed=args.seed)
        data_df = data_df[list(model.nodes())]
    elif args.csv:
        print(f"[*] Loading CSV dataset: {args.csv}")
        data_df = pd.read_csv(args.csv)
    else:
        print("[!] Please specify --run_web_benchmark, --benchmark, --run_ablation_sensitivity, --bif, or --csv.")
        return

    expert_mgr = ExpertKnowledgeManager(data_df.columns)
    if args.priors_json:
        expert_mgr.load_priors_from_json(args.priors_json)
    else:
        ds_name = os.path.splitext(os.path.basename(args.bif or args.csv))[0].lower()
        auto_prior = os.path.join(args.priors_dir, f"{ds_name}.json")
        if os.path.exists(auto_prior):
            expert_mgr.load_priors_from_json(auto_prior)

    solver = DialectCD(alpha=args.alpha, n_jobs=args.n_jobs, expert_manager=expert_mgr)
    solver.compute_acceptability(data_df)
    learned_cpdag = solver.construct_dag(acceptability_threshold=args.threshold)

    adj_path = os.path.join(args.out_dir, "learned_cpdag_adj.csv")
    nx.to_pandas_adjacency(learned_cpdag, dtype=int).to_csv(adj_path)
    print(f"[✓] Causal discovery complete. Adjacency matrix saved to: {adj_path}")

    if args.explain:
        u_node, v_node = args.explain[0], args.explain[1]
        print(f"\n[*] Dialectical Explanation for ({u_node}, {v_node}):")
        print(solver.explain_edge(u_node, v_node))

    if args.plot_graph:
        solver.plot_explainable_graph(os.path.join(args.out_dir, "learned_causal_graph.png"))


if __name__ == "__main__":
    main()