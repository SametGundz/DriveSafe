#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bilimsel Gaze Zone Değerlendirme Modülü

Bu modül, gaze zone tahminlerinin bilimsel analizi için gelişmiş 
istatistiksel metrikler ve görselleştirmeler sağlar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    classification_report, cohen_kappa_score, 
    matthews_corrcoef, balanced_accuracy_score
)
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter


class ScientificGazeEvaluator:
    """Bilimsel standardlarda gaze zone değerlendirme sınıfı."""
    
    def __init__(self):
        # Zone tanımlamaları - EU regülasyonu C(2023)4523 uyumlu
        self.zone_definitions = {
            0: {"name": "Road Center", "area": "Area 2", "category": "Driving-Critical"},
            1: {"name": "Driving Instruments", "area": "Area 2", "category": "Driving-Related"},
            2: {"name": "Infotainment", "area": "Area 1", "category": "Non-Driving"},
            3: {"name": "Left Side", "area": "Area 2", "category": "Driving-Related"},
            4: {"name": "Right Side", "area": "Area 2", "category": "Driving-Related"},
            5: {"name": "Rear Mirror", "area": "Area 2", "category": "Driving-Critical"}
        }
        
        # Bilimsel metrikler için kategoriler
        self.critical_zones = [0, 5]  # Road Center, Rear Mirror
        self.driving_related_zones = [1, 3, 4]  # Instruments, Sides
        self.non_driving_zones = [2]  # Infotainment
        
    def comprehensive_analysis(self, ground_truth: Dict, predictions: Dict) -> Dict[str, Any]:
        """Kapsamlı bilimsel analiz gerçekleştir."""
        
        # Temel hazırlık
        results = {}
        gt_array, pred_array = self._prepare_data(ground_truth, predictions)
        
        if len(gt_array) == 0:
            return {"error": "No valid data for comparison"}
        
        # 1. Temel Metrikler
        results["basic_metrics"] = self._calculate_basic_metrics(gt_array, pred_array)
        
        # 2. İleri Düzey İstatistiksel Metrikler
        results["advanced_metrics"] = self._calculate_advanced_metrics(gt_array, pred_array)
        
        # 3. Kategori Bazında Analiz
        results["category_analysis"] = self._category_analysis(gt_array, pred_array)
        
        # 4. Temporal Analysis (Zamansal Analiz)
        results["temporal_analysis"] = self._temporal_analysis(ground_truth, predictions)
        
        # 5. Error Analysis (Hata Analizi)
        results["error_analysis"] = self._error_analysis(gt_array, pred_array)
        
        # 6. Reliability Analysis (Güvenilirlik Analizi)
        results["reliability"] = self._reliability_analysis(gt_array, pred_array)
        
        # 7. Statistical Significance Tests
        results["statistical_tests"] = self._statistical_significance_tests(gt_array, pred_array)
        
        return results
    
    def _prepare_data(self, ground_truth: Dict, predictions: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Veri hazırlama ve eşleştirme."""
        common_frames = set(ground_truth.keys()).intersection(set(predictions.keys()))
        
        gt_list = []
        pred_list = []
        
        for frame_id in sorted(common_frames):
            gt_val = ground_truth[frame_id]
            pred_val = predictions[frame_id]
            
            if gt_val is not None and pred_val is not None:
                gt_list.append(gt_val)
                pred_list.append(pred_val)
        
        return np.array(gt_list), np.array(pred_list)
    
    def _calculate_basic_metrics(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, float]:
        """Temel doğruluk metrikleri."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Overall metrics
        accuracy = accuracy_score(gt_array, pred_array) * 100
        balanced_acc = balanced_accuracy_score(gt_array, pred_array) * 100
        
        # Macro ve Micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gt_array, pred_array, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            gt_array, pred_array, average='micro', zero_division=0
        )
        
        return {
            "overall_accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_precision": precision_macro * 100,
            "macro_recall": recall_macro * 100,
            "macro_f1": f1_macro * 100,
            "micro_precision": precision_micro * 100,
            "micro_recall": recall_micro * 100,
            "micro_f1": f1_micro * 100
        }
    
    def _calculate_advanced_metrics(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, float]:
        """İleri düzey istatistiksel metrikler."""
        
        # Cohen's Kappa (inter-rater agreement)
        kappa = cohen_kappa_score(gt_array, pred_array)
        
        # Matthews Correlation Coefficient (MCC) - multiclass için ortalama
        unique_classes = np.unique(np.concatenate([gt_array, pred_array]))
        mcc_scores = []
        
        for class_label in unique_classes:
            gt_binary = (gt_array == class_label).astype(int)
            pred_binary = (pred_array == class_label).astype(int)
            
            if len(np.unique(gt_binary)) > 1 and len(np.unique(pred_binary)) > 1:
                mcc = matthews_corrcoef(gt_binary, pred_binary)
                mcc_scores.append(mcc)
        
        avg_mcc = np.mean(mcc_scores) if mcc_scores else 0.0
        
        # Krippendorff's Alpha approximation
        krippendorff_alpha = self._krippendorff_alpha(gt_array, pred_array)
        
        return {
            "cohens_kappa": kappa,
            "matthews_correlation_coeff": avg_mcc,
            "krippendorff_alpha": krippendorff_alpha,
            "agreement_strength": self._interpret_kappa(kappa)
        }
    
    def _krippendorff_alpha(self, gt_array: np.ndarray, pred_array: np.ndarray) -> float:
        """Krippendorff's Alpha hesaplama (basitleştirilmiş)."""
        # Bu basit bir implementasyon - tam Krippendorff's Alpha için özel kütüphane gerekir
        n = len(gt_array)
        if n == 0:
            return 0.0
        
        # Disagreement calculation
        disagreements = np.sum(gt_array != pred_array)
        total_pairs = n
        
        # Expected disagreement (random baseline)
        unique_vals, counts = np.unique(np.concatenate([gt_array, pred_array]), return_counts=True)
        expected_disagreement = 1.0 - np.sum((counts / (2 * n)) ** 2)
        
        # Alpha calculation
        if expected_disagreement == 0:
            return 1.0 if disagreements == 0 else 0.0
        
        alpha = 1 - (disagreements / n) / expected_disagreement
        return max(0.0, alpha)
    
    def _interpret_kappa(self, kappa: float) -> str:
        """Kappa değerini yorumla (Landis & Koch, 1977)."""
        if kappa < 0:
            return "Poor"
        elif kappa <= 0.20:
            return "Slight"
        elif kappa <= 0.40:
            return "Fair"
        elif kappa <= 0.60:
            return "Moderate"
        elif kappa <= 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"
    
    def _category_analysis(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, Any]:
        """Kategori bazında analiz (Critical, Driving-Related, Non-Driving)."""
        
        categories = {
            "Critical": self.critical_zones,
            "Driving_Related": self.driving_related_zones,
            "Non_Driving": self.non_driving_zones
        }
        
        category_results = {}
        
        for cat_name, zone_list in categories.items():
            # Bu kategorideki frame'leri filtrele
            cat_mask = np.isin(gt_array, zone_list)
            
            if np.sum(cat_mask) == 0:
                continue
            
            cat_gt = gt_array[cat_mask]
            cat_pred = pred_array[cat_mask]
            
            # Bu kategori için metrikler
            accuracy = np.mean(cat_gt == cat_pred) * 100
            
            category_results[cat_name] = {
                "accuracy": accuracy,
                "sample_count": len(cat_gt),
                "zones_included": zone_list
            }
        
        return category_results
    
    def _temporal_analysis(self, ground_truth: Dict, predictions: Dict) -> Dict[str, Any]:
        """Zamansal analiz - frame sekansları üzerinde."""
        
        # Frame'leri sıralı şekilde al
        common_frames = sorted(set(ground_truth.keys()).intersection(set(predictions.keys())))
        
        gt_sequence = []
        pred_sequence = []
        
        for frame_id in common_frames:
            gt_val = ground_truth[frame_id]
            pred_val = predictions[frame_id]
            
            if gt_val is not None and pred_val is not None:
                gt_sequence.append(gt_val)
                pred_sequence.append(pred_val)
        
        if len(gt_sequence) < 2:
            return {"error": "Insufficient data for temporal analysis"}
        
        # Transition analysis
        gt_transitions = self._count_transitions(gt_sequence)
        pred_transitions = self._count_transitions(pred_sequence)
        
        # Sequence stability
        gt_stability = self._calculate_stability(gt_sequence)
        pred_stability = self._calculate_stability(pred_sequence)
        
        # Temporal autocorrelation
        temporal_correlation = self._temporal_correlation(gt_sequence, pred_sequence)
        
        return {
            "gt_transitions": gt_transitions,
            "pred_transitions": pred_transitions,
            "gt_stability": gt_stability,
            "pred_stability": pred_stability,
            "temporal_correlation": temporal_correlation,
            "sequence_length": len(gt_sequence)
        }
    
    def _count_transitions(self, sequence: List[int]) -> Dict[str, int]:
        """Geçiş sayılarını hesapla."""
        transitions = {"total": 0, "within_category": 0, "between_category": 0}
        
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                transitions["total"] += 1
                
                # Kategori içi/arası geçiş kontrolü
                prev_cat = self._get_zone_category(sequence[i-1])
                curr_cat = self._get_zone_category(sequence[i])
                
                if prev_cat == curr_cat:
                    transitions["within_category"] += 1
                else:
                    transitions["between_category"] += 1
        
        return transitions
    
    def _get_zone_category(self, zone_id: int) -> str:
        """Zone'un kategorisini döndür."""
        if zone_id in self.critical_zones:
            return "Critical"
        elif zone_id in self.driving_related_zones:
            return "Driving_Related"
        elif zone_id in self.non_driving_zones:
            return "Non_Driving"
        else:
            return "Unknown"
    
    def _calculate_stability(self, sequence: List[int]) -> float:
        """Sekans stabilitesi hesapla."""
        if len(sequence) <= 1:
            return 1.0
        
        changes = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
        stability = 1.0 - (changes / (len(sequence) - 1))
        return stability
    
    def _temporal_correlation(self, gt_sequence: List[int], pred_sequence: List[int]) -> float:
        """Zamansal korelasyon hesapla."""
        if len(gt_sequence) != len(pred_sequence) or len(gt_sequence) < 2:
            return 0.0
        
        # Pearson korelasyon
        try:
            correlation, p_value = stats.pearsonr(gt_sequence, pred_sequence)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _error_analysis(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, Any]:
        """Detaylı hata analizi."""
        
        # En sık karışan zone çiftleri
        confusion_pairs = defaultdict(int)
        
        for gt, pred in zip(gt_array, pred_array):
            if gt != pred:
                pair = tuple(sorted([gt, pred]))
                confusion_pairs[pair] += 1
        
        # En çok karışan çiftleri sırala
        top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Critical zone hataları
        critical_errors = self._analyze_critical_errors(gt_array, pred_array)
        
        # Sistematik hatalar
        systematic_bias = self._detect_systematic_bias(gt_array, pred_array)
        
        return {
            "top_confusion_pairs": [
                {"zones": pair[0], "count": pair[1], "error_rate": pair[1]/len(gt_array)*100}
                for pair in top_confusions
            ],
            "critical_zone_errors": critical_errors,
            "systematic_bias": systematic_bias
        }
    
    def _analyze_critical_errors(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, Any]:
        """Critical zone hata analizi."""
        
        # Critical zone'lardaki hatalar
        critical_mask = np.isin(gt_array, self.critical_zones)
        
        if np.sum(critical_mask) == 0:
            return {"error": "No critical zone data"}
        
        critical_gt = gt_array[critical_mask]
        critical_pred = pred_array[critical_mask]
        
        critical_accuracy = np.mean(critical_gt == critical_pred) * 100
        critical_errors = np.sum(critical_gt != critical_pred)
        
        # Critical zone'dan non-driving zone'a yanlış tahmin
        critical_to_nondriving = 0
        for gt, pred in zip(critical_gt, critical_pred):
            if gt in self.critical_zones and pred in self.non_driving_zones:
                critical_to_nondriving += 1
        
        return {
            "critical_accuracy": critical_accuracy,
            "critical_errors": critical_errors,
            "critical_to_nondriving_errors": critical_to_nondriving,
            "safety_impact_score": 100 - critical_accuracy  # Güvenlik riski skoru
        }
    
    def _detect_systematic_bias(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, Any]:
        """Sistematik yanlılık tespiti."""
        
        # Zone dağılımlarını karşılaştır
        gt_dist = Counter(gt_array)
        pred_dist = Counter(pred_array)
        
        all_zones = set(list(gt_dist.keys()) + list(pred_dist.keys()))
        
        bias_scores = {}
        for zone in all_zones:
            gt_count = gt_dist.get(zone, 0)
            pred_count = pred_dist.get(zone, 0)
            total_gt = len(gt_array)
            total_pred = len(pred_array)
            
            gt_ratio = gt_count / total_gt if total_gt > 0 else 0
            pred_ratio = pred_count / total_pred if total_pred > 0 else 0
            
            # Bias skoru: pozitif değer over-prediction, negatif under-prediction
            bias_scores[zone] = (pred_ratio - gt_ratio) * 100
        
        return {
            "zone_bias_scores": bias_scores,
            "overall_bias": np.mean(list(bias_scores.values())),
            "max_overpredict_zone": max(bias_scores.items(), key=lambda x: x[1]),
            "max_underpredict_zone": min(bias_scores.items(), key=lambda x: x[1])
        }
    
    def _reliability_analysis(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, float]:
        """Güvenilirlik analizi."""
        
        # Test-retest reliability approximation
        # (Gerçek test-retest için multiple runs gerekir)
        
        # Internal consistency (zone tutarlılığı)
        zone_consistency = {}
        for zone in np.unique(gt_array):
            zone_mask = (gt_array == zone)
            if np.sum(zone_mask) > 0:
                zone_predictions = pred_array[zone_mask]
                consistency = np.mean(zone_predictions == zone) * 100
                zone_consistency[zone] = consistency
        
        avg_consistency = np.mean(list(zone_consistency.values())) if zone_consistency else 0.0
        
        # Split-half reliability
        mid_point = len(gt_array) // 2
        first_half_acc = np.mean(gt_array[:mid_point] == pred_array[:mid_point]) * 100
        second_half_acc = np.mean(gt_array[mid_point:] == pred_array[mid_point:]) * 100
        split_half_reliability = abs(first_half_acc - second_half_acc)
        
        return {
            "average_zone_consistency": avg_consistency,
            "split_half_difference": split_half_reliability,
            "zone_consistency_scores": zone_consistency
        }
    
    def _statistical_significance_tests(self, gt_array: np.ndarray, pred_array: np.ndarray) -> Dict[str, Any]:
        """İstatistiksel anlamlılık testleri."""
        
        # Chi-square test for independence
        from scipy.stats import chi2_contingency
        
        # Contingency table oluştur
        unique_zones = np.unique(np.concatenate([gt_array, pred_array]))
        contingency_table = np.zeros((len(unique_zones), len(unique_zones)))
        
        zone_to_idx = {zone: idx for idx, zone in enumerate(unique_zones)}
        
        for gt, pred in zip(gt_array, pred_array):
            gt_idx = zone_to_idx[gt]
            pred_idx = zone_to_idx[pred]
            contingency_table[gt_idx][pred_idx] += 1
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Effect size (Cramér's V)
            n = np.sum(contingency_table)
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
        except:
            chi2, p_value, cramers_v = 0, 1, 0
        
        # McNemar's test (for paired data) - sadece 2x2 için
        mcnemar_result = None
        if len(unique_zones) == 2:
            try:
                from scipy.stats import mcnemar
                table_2x2 = contingency_table
                mcnemar_stat, mcnemar_p = mcnemar(table_2x2, exact=False, correction=True)
                mcnemar_result = {"statistic": mcnemar_stat, "p_value": mcnemar_p}
            except:
                mcnemar_result = None
        
        return {
            "chi_square_test": {
                "statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "cramers_v": cramers_v,
                "effect_size_interpretation": self._interpret_cramers_v(cramers_v)
            },
            "mcnemar_test": mcnemar_result
        }
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Cramér's V değerini yorumla."""
        if cramers_v < 0.1:
            return "Negligible"
        elif cramers_v < 0.3:
            return "Small"
        elif cramers_v < 0.5:
            return "Medium"
        else:
            return "Large"
    
    def generate_academic_report(self, results: Dict[str, Any], 
                                 gt_file: str, pred_file: str) -> str:
        """Akademik formatta rapor oluştur."""
        
        report_lines = []
        
        # Başlık ve özet
        report_lines.extend([
            "# GAZE ZONE CLASSIFICATION PERFORMANCE EVALUATION",
            "## Scientific Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## EXECUTIVE SUMMARY",
            f"Ground Truth File: {os.path.basename(gt_file)}",
            f"Prediction File: {os.path.basename(pred_file)}",
            "",
        ])
        
        # Temel metrikler
        if "basic_metrics" in results:
            basic = results["basic_metrics"]
            report_lines.extend([
                "## CLASSIFICATION PERFORMANCE METRICS",
                f"Overall Accuracy: {basic['overall_accuracy']:.2f}%",
                f"Balanced Accuracy: {basic['balanced_accuracy']:.2f}%",
                f"Macro-averaged Precision: {basic['macro_precision']:.2f}%",
                f"Macro-averaged Recall: {basic['macro_recall']:.2f}%",
                f"Macro-averaged F1-Score: {basic['macro_f1']:.2f}%",
                ""
            ])
        
        # İleri düzey metrikler
        if "advanced_metrics" in results:
            advanced = results["advanced_metrics"]
            report_lines.extend([
                "## STATISTICAL RELIABILITY MEASURES",
                f"Cohen's Kappa: {advanced['cohens_kappa']:.4f} ({advanced['agreement_strength']})",
                f"Matthews Correlation Coefficient: {advanced['matthews_correlation_coeff']:.4f}",
                f"Krippendorff's Alpha: {advanced['krippendorff_alpha']:.4f}",
                ""
            ])
        
        # Kategori analizi
        if "category_analysis" in results:
            report_lines.append("## CATEGORY-BASED ANALYSIS")
            for category, data in results["category_analysis"].items():
                report_lines.append(f"{category} Zones: {data['accuracy']:.2f}% accuracy ({data['sample_count']} samples)")
            report_lines.append("")
        
        # Hata analizi
        if "error_analysis" in results:
            error_data = results["error_analysis"]
            report_lines.extend([
                "## ERROR ANALYSIS",
                "### Most Frequent Confusion Pairs:"
            ])
            
            for pair_data in error_data.get("top_confusion_pairs", [])[:3]:
                zones = pair_data["zones"]
                count = pair_data["count"]
                rate = pair_data["error_rate"]
                report_lines.append(f"- Zones {zones[0]}↔{zones[1]}: {count} errors ({rate:.2f}%)")
            
            if "critical_zone_errors" in error_data:
                critical = error_data["critical_zone_errors"]
                report_lines.extend([
                    "",
                    "### Critical Zone Performance:",
                    f"- Critical Zone Accuracy: {critical.get('critical_accuracy', 0):.2f}%",
                    f"- Safety Impact Score: {critical.get('safety_impact_score', 0):.2f}%"
                ])
            
            report_lines.append("")
        
        # İstatistiksel testler
        if "statistical_tests" in results:
            stats_data = results["statistical_tests"]
            chi_square = stats_data.get("chi_square_test", {})
            
            report_lines.extend([
                "## STATISTICAL SIGNIFICANCE TESTS",
                f"Chi-square test: χ² = {chi_square.get('statistic', 0):.3f}, p = {chi_square.get('p_value', 1):.6f}",
                f"Effect size (Cramér's V): {chi_square.get('cramers_v', 0):.3f} ({chi_square.get('effect_size_interpretation', 'Unknown')})",
                ""
            ])
        
        # Sonuç ve öneriler
        overall_acc = results.get("basic_metrics", {}).get("overall_accuracy", 0)
        kappa = results.get("advanced_metrics", {}).get("cohens_kappa", 0)
        
        report_lines.extend([
            "## CONCLUSIONS AND RECOMMENDATIONS",
            self._generate_conclusions(overall_acc, kappa, results),
            "",
            "## METHODOLOGY NOTE",
            "This evaluation follows established machine learning evaluation protocols ",
            "and includes multiple statistical measures to ensure robust assessment of ",
            "gaze zone classification performance. Metrics are computed according to ",
            "standards outlined in scientific literature for multi-class classification problems.",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def _generate_conclusions(self, accuracy: float, kappa: float, results: Dict) -> str:
        """Sonuç ve önerileri otomatik oluştur."""
        
        conclusions = []
        
        # Genel performans değerlendirmesi
        if accuracy >= 90:
            conclusions.append("- Excellent classification performance achieved (>90% accuracy)")
        elif accuracy >= 80:
            conclusions.append("- Good classification performance achieved (80-90% accuracy)")
        elif accuracy >= 70:
            conclusions.append("- Moderate classification performance achieved (70-80% accuracy)")
        else:
            conclusions.append("- Classification performance requires improvement (<70% accuracy)")
        
        # Kappa değerlendirmesi
        if kappa >= 0.8:
            conclusions.append("- Almost perfect inter-rater agreement demonstrated")
        elif kappa >= 0.6:
            conclusions.append("- Substantial agreement level achieved")
        elif kappa >= 0.4:
            conclusions.append("- Moderate agreement observed, room for improvement")
        else:
            conclusions.append("- Low agreement detected, significant improvement needed")
        
        # Kategori bazında öneriler
        if "category_analysis" in results:
            cat_analysis = results["category_analysis"]
            critical_acc = cat_analysis.get("Critical", {}).get("accuracy", 0)
            
            if critical_acc < 85:
                conclusions.append("- RECOMMENDATION: Focus on improving critical zone detection accuracy")
        
        # Hata analizi bazında öneriler
        if "error_analysis" in results:
            error_data = results["error_analysis"]
            safety_impact = error_data.get("critical_zone_errors", {}).get("safety_impact_score", 0)
            
            if safety_impact > 20:
                conclusions.append("- SAFETY CONCERN: High error rate in safety-critical zones detected")
        
        return "\n".join(conclusions)


# GUI entegrasyonu için yardımcı fonksiyonlar
def create_publication_figures(results: Dict[str, Any], save_dir: str = None) -> List[str]:
    """Yayın kalitesinde figürler oluştur."""
    
    if save_dir is None:
        save_dir = "evaluation_figures"
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    figure_paths = []
    
    # 1. Performance Metrics Radar Chart
    if "basic_metrics" in results and "advanced_metrics" in results:
        fig_path = os.path.join(save_dir, "performance_radar.png")
        create_performance_radar(results, fig_path)
        figure_paths.append(fig_path)
    
    # 2. Category Analysis Bar Chart
    if "category_analysis" in results:
        fig_path = os.path.join(save_dir, "category_analysis.png")
        create_category_analysis_chart(results["category_analysis"], fig_path)
        figure_paths.append(fig_path)
    
    # 3. Error Distribution Heatmap
    if "error_analysis" in results:
        fig_path = os.path.join(save_dir, "error_heatmap.png")
        create_error_heatmap(results["error_analysis"], fig_path)
        figure_paths.append(fig_path)
    
    return figure_paths


def create_performance_radar(results: Dict, save_path: str):
    """Performans radar grafiği oluştur."""
    
    basic = results["basic_metrics"]
    advanced = results["advanced_metrics"]
    
    # Metrikleri normalize et (0-100 arası)
    metrics = {
        "Accuracy": basic["overall_accuracy"],
        "Balanced Acc": basic["balanced_accuracy"],
        "Precision": basic["macro_precision"],
        "Recall": basic["macro_recall"],
        "F1-Score": basic["macro_f1"],
        "Kappa*100": advanced["cohens_kappa"] * 100,
        "MCC*100": advanced["matthews_correlation_coeff"] * 100
    }
    
    # Radar chart oluştur
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values = list(metrics.values())
    
    # Çemberi kapatmak için ilk değeri sona ekle
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label='Performance', color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    
    # Etiketleri ekle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics.keys(), fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title('Gaze Zone Classification Performance Metrics', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid'i güzelleştir
    ax.grid(True, alpha=0.3)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_category_analysis_chart(category_data: Dict, save_path: str):
    """Kategori analiz grafiği oluştur."""
    
    categories = []
    accuracies = []
    sample_counts = []
    
    for cat_name, data in category_data.items():
        categories.append(cat_name.replace('_', ' '))
        accuracies.append(data['accuracy'])
        sample_counts.append(data['sample_count'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy chart
    bars1 = ax1.bar(categories, accuracies, color=['#2E8B57', '#4682B4', '#CD5C5C'])
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Category-wise Classification Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Değerleri çubukların üzerine yaz
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Sample count chart
    bars2 = ax2.bar(categories, sample_counts, color=['#2E8B57', '#4682B4', '#CD5C5C'])
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Sample Distribution by Category', fontsize=13, fontweight='bold')
    
    # Değerleri çubukların üzerine yaz
    for bar, count in zip(bars2, sample_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_error_heatmap(error_data: Dict, save_path: str):
    """Hata analizi heatmap'i oluştur."""
    
    # En çok karışan zone çiftlerini al
    confusion_pairs = error_data.get("top_confusion_pairs", [])
    
    if not confusion_pairs:
        return
    
    # Zone ID'lerini topla
    all_zones = set()
    for pair_data in confusion_pairs:
        zones = pair_data["zones"]
        all_zones.update(zones)
    
    all_zones = sorted(list(all_zones))
    
    # Confusion matrix oluştur
    matrix = np.zeros((len(all_zones), len(all_zones)))
    zone_to_idx = {zone: idx for idx, zone in enumerate(all_zones)}
    
    for pair_data in confusion_pairs:
        zones = pair_data["zones"]
        count = pair_data["count"]
        i, j = zone_to_idx[zones[0]], zone_to_idx[zones[1]]
        matrix[i][j] = count
        matrix[j][i] = count  # Simetrik
    
    # Zone adlarını al
    zone_names = [f"Zone {z}" for z in all_zones]
    
    # Heatmap oluştur
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='Reds', aspect='auto')
    
    # Renk barı ekle
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error Count', fontsize=12)
    
    # Eksen etiketleri
    ax.set_xticks(range(len(zone_names)))
    ax.set_yticks(range(len(zone_names)))
    ax.set_xticklabels(zone_names, rotation=45, ha='right')
    ax.set_yticklabels(zone_names)
    
    # Değerleri hücrelere yaz
    for i in range(len(all_zones)):
        for j in range(len(all_zones)):
            if matrix[i][j] > 0:
                ax.text(j, i, f'{int(matrix[i][j])}', 
                       ha='center', va='center', fontweight='bold')
    
    ax.set_title('Zone Confusion Error Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def export_detailed_results(results: Dict[str, Any], output_file: str):
    """Detaylı sonuçları Excel formatında export et."""
    
    try:
        import pandas as pd
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Basic Metrics
            if "basic_metrics" in results:
                basic_df = pd.DataFrame([results["basic_metrics"]]).T
                basic_df.columns = ['Value']
                basic_df.to_excel(writer, sheet_name='Basic_Metrics')
            
            # Advanced Metrics
            if "advanced_metrics" in results:
                advanced_df = pd.DataFrame([results["advanced_metrics"]]).T
                advanced_df.columns = ['Value']
                advanced_df.to_excel(writer, sheet_name='Advanced_Metrics')
            
            # Category Analysis
            if "category_analysis" in results:
                cat_data = []
                for cat, data in results["category_analysis"].items():
                    cat_data.append({
                        'Category': cat,
                        'Accuracy': data['accuracy'],
                        'Sample_Count': data['sample_count']
                    })
                cat_df = pd.DataFrame(cat_data)
                cat_df.to_excel(writer, sheet_name='Category_Analysis', index=False)
            
            # Error Analysis
            if "error_analysis" in results:
                error_pairs = results["error_analysis"].get("top_confusion_pairs", [])
                if error_pairs:
                    error_df = pd.DataFrame(error_pairs)
                    error_df.to_excel(writer, sheet_name='Error_Analysis', index=False)
        
        print(f"Detailed results exported to: {output_file}")
        
    except ImportError:
        print("pandas not available for Excel export")


# Ana evaluation fonksiyonu
def evaluate_gaze_predictions(ground_truth_file: str, predictions_file: str, 
                             output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """
    Ana evaluation fonksiyonu - bilimsel analiz ve görselleştirme.
    
    Args:
        ground_truth_file: Ground truth JSON dosyası
        predictions_file: Tahmin sonuçları JSON dosyası
        output_dir: Çıktı klasörü
        
    Returns:
        Dict: Kapsamlı evaluation sonuçları
    """
    
    # Dosyaları yükle
    print("Loading data files...")
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Evaluator oluştur
    evaluator = ScientificGazeEvaluator()
    
    # Kapsamlı analiz yap
    print("Performing comprehensive analysis...")
    results = evaluator.comprehensive_analysis(ground_truth, predictions)
    
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Akademik rapor oluştur
    print("Generating academic report...")
    report_text = evaluator.generate_academic_report(
        results, ground_truth_file, predictions_file
    )
    
    report_file = os.path.join(output_dir, "scientific_evaluation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Görselleştirmeler oluştur
    print("Creating publication figures...")
    figure_paths = create_publication_figures(results, 
                                            os.path.join(output_dir, "figures"))
    
    # Excel raporu oluştur
    excel_file = os.path.join(output_dir, "detailed_metrics.xlsx")
    export_detailed_results(results, excel_file)
    
    # JSON sonuçları kaydet
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Özet istatistikler
    summary = {
        "overall_accuracy": results.get("basic_metrics", {}).get("overall_accuracy", 0),
        "cohens_kappa": results.get("advanced_metrics", {}).get("cohens_kappa", 0),
        "agreement_strength": results.get("advanced_metrics", {}).get("agreement_strength", "Unknown"),
        "report_file": report_file,
        "figures": figure_paths,
        "excel_file": excel_file,
        "results_file": results_file
    }
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
    print(f"Cohen's Kappa: {summary['cohens_kappa']:.4f} ({summary['agreement_strength']})")
    
    return summary


# GUI ile entegrasyon için yardımcı sınıf
class EvaluationGUI:
    """PyQt6 tabanlı evaluation GUI'si."""
    
    def __init__(self):
        self.ground_truth_file = ""
        self.predictions_file = ""
        self.output_dir = "evaluation_results"
    
    def create_evaluation_widget(self, parent_layout):
        """Ana widget'a evaluation paneli ekle."""
        from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, 
                                   QPushButton, QLabel, QFileDialog, QTextEdit)
        
        # Evaluation group box
        eval_group = QGroupBox("Scientific Evaluation")
        eval_layout = QVBoxLayout(eval_group)
        
        # File selection
        file_layout = QHBoxLayout()
        
        self.gt_label = QLabel("Ground Truth: Not selected")
        self.gt_button = QPushButton("Select GT File")
        self.gt_button.clicked.connect(self.select_ground_truth)
        
        file_layout.addWidget(self.gt_label)
        file_layout.addWidget(self.gt_button)
        eval_layout.addLayout(file_layout)
        
        pred_layout = QHBoxLayout()
        self.pred_label = QLabel("Predictions: Not selected")
        self.pred_button = QPushButton("Select Pred File")
        self.pred_button.clicked.connect(self.select_predictions)
        
        pred_layout.addWidget(self.pred_label)
        pred_layout.addWidget(self.pred_button)
        eval_layout.addLayout(pred_layout)
        
        # Evaluate button
        self.eval_button = QPushButton("Run Scientific Evaluation")
        self.eval_button.clicked.connect(self.run_evaluation)
        self.eval_button.setEnabled(False)
        eval_layout.addWidget(self.eval_button)
        
        # Results text
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        eval_layout.addWidget(self.results_text)
        
        parent_layout.addWidget(eval_group)
    
    def select_ground_truth(self):
        """Ground truth dosyası seç."""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select Ground Truth JSON", "", "JSON Files (*.json)")
        
        if file_path:
            self.ground_truth_file = file_path
            self.gt_label.setText(f"GT: {os.path.basename(file_path)}")
            self.check_ready()
    
    def select_predictions(self):
        """Predictions dosyası seç."""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select Predictions JSON", "", "JSON Files (*.json)")
        
        if file_path:
            self.predictions_file = file_path
            self.pred_label.setText(f"Pred: {os.path.basename(file_path)}")
            self.check_ready()
    
    def check_ready(self):
        """Evaluation için hazır mı kontrol et."""
        ready = bool(self.ground_truth_file and self.predictions_file)
        self.eval_button.setEnabled(ready)
    
    def run_evaluation(self):
        """Evaluation'ı çalıştır."""
        try:
            self.results_text.append("Starting scientific evaluation...")
            self.eval_button.setEnabled(False)
            
            # Evaluation çalıştır
            summary = evaluate_gaze_predictions(
                self.ground_truth_file, 
                self.predictions_file,
                self.output_dir
            )
            
            # Sonuçları göster
            self.results_text.append(f"\n✅ Evaluation completed!")
            self.results_text.append(f"📊 Overall Accuracy: {summary['overall_accuracy']:.2f}%")
            self.results_text.append(f"📈 Cohen's Kappa: {summary['cohens_kappa']:.4f}")
            self.results_text.append(f"📝 Agreement: {summary['agreement_strength']}")
            self.results_text.append(f"📁 Results: {self.output_dir}")
            
        except Exception as e:
            self.results_text.append(f"❌ Error: {str(e)}")
        
        finally:
            self.eval_button.setEnabled(True)


# CLI kullanımı için fonksiyon
def run_cli():
    """Komut satırı arayüzünü çalıştır."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scientific Gaze Zone Evaluation")
    parser.add_argument("--ground_truth", required=True, help="Ground truth JSON file")
    parser.add_argument("--predictions", required=True, help="Predictions JSON file")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--gui", action="store_true", help="Launch the enhanced GUI interface")
    
    args = parser.parse_args()
    
    # GUI modunu kontrol et
    if args.gui:
        # GUI modunda evaulation_gui_new.py'yi çalıştır
        try:
            # Yeni GUI modülünü dinamik olarak import et ve çalıştır
            from scripts.evaulation_gui_new import main as run_gui
            run_gui()
            return
        except ImportError:
            print("Enhanced GUI is not available. Please make sure 'scripts/evaulation_gui_new.py' exists.")
            print("Falling back to command line mode...")
    
    # Evaluation çalıştır
    summary = evaluate_gaze_predictions(
        args.ground_truth,
        args.predictions, 
        args.output_dir
    )
    
    print("\n" + "="*50)
    print("SCIENTIFIC EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
    print(f"Cohen's Kappa: {summary['cohens_kappa']:.4f}")
    print(f"Agreement Level: {summary['agreement_strength']}")
    print(f"Results Directory: {args.output_dir}")
    print("="*50)


# Modül olarak kullanımı sağla
if __name__ == "__main__":
    run_cli()