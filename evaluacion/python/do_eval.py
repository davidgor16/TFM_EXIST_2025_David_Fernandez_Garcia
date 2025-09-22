from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
from pyevall.metrics.metricfactory import MetricFactory
import sys

def do_eval(json, eval_type = "HARD"):
    # PROCESO DE EVALUACIÓN---------------------------------------------------------------------------------------------------- 

    if eval_type =="HARD":
        gold = "../../../evaluacion/ground_truths/EXIST2025_training_task3_1_gold_hard.json"
        metrics=[MetricFactory.ICM.value, MetricFactory.ICMNorm.value, MetricFactory.FMeasure.value]
        
    if eval_type == "SOFT":
        gold = "../../../evaluacion/ground_truths/EXIST2025_training_task3_1_gold_soft.json"
        metrics=[MetricFactory.ICMSoft.value, MetricFactory.ICMSoftNorm.value, MetricFactory.CrossEntropy.value]


    test = PyEvALLEvaluation()
    params= dict()
    report = test.evaluate(json, gold, metrics, **params)
    
    if eval_type == "HARD":
    # Extraemos las métricas
        icm = report.__dict__['report']["metrics"]['ICM']["results"]["test_cases"][0]["average"]
        icm_norm = report.__dict__['report']["metrics"]['ICMNorm']["results"]["test_cases"][0]["average"]
        f1 = report.__dict__['report']["metrics"]['FMeasure']["results"]["test_cases"][0]["average"]
        
        return icm, icm_norm, f1
    
    if eval_type =="SOFT":
        
        icm = report.__dict__['report']["metrics"]['ICMSoft']["results"]["test_cases"][0]["average"]
        icm_norm = report.__dict__['report']["metrics"]['ICMSoftNorm']["results"]["test_cases"][0]["average"]
        f1 = report.__dict__['report']["metrics"]['CrossEntropy']["results"]["test_cases"][0]["average"]
        
        return icm, icm_norm, f1
    #-------------------------------------------------------------------------------------------------------------------------
