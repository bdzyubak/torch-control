import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.ui.workspace import Workspace



WORKSPACE = "workspace"
YOUR_PROJECT_NAME = 'Evidently Tutorial California Housing'
YOUR_PROJECT_DESCRIPTION = "Test Evidently"
workspace = Workspace.create(WORKSPACE)
project = workspace.create_project(YOUR_PROJECT_NAME)
project.description = YOUR_PROJECT_DESCRIPTION


def main():
    data = fetch_california_housing(as_frame=True)
    housing_data = data.frame

    housing_data.rename(columns={'MedHouseVal': 'target'}, inplace=True)
    housing_data['prediction'] = housing_data['target'].values + np.random.normal(0, 5, housing_data.shape[0])

    reference = housing_data.sample(n=5000, replace=False)
    current = housing_data.sample(n=5000, replace=False)

    for i in range(3):
        report = create_report(current=current, reference=reference, i=i)
        workspace.add_report(project.id, report)

        test_suite = create_test_suite(current=current, reference=reference, i=i)
        workspace.add_test_suite(project.id, test_suite)


def create_test_suite(current, reference, i):
    if i == 0:
        tests = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
        ])
        tests.run(reference_data=reference, current_data=current)

    elif i == 1:
        suite = TestSuite(tests=[
            NoTargetPerformanceTestPreset(),
        ])
        suite.run(reference_data=reference, current_data=current)

    elif i == 2:
        suite = TestSuite(tests=[
            TestColumnDrift('Population'),
            TestShareOfOutRangeValues('Population'),
            generate_column_tests(TestMeanInNSigmas, columns='num'),

        ])
        suite.run(reference_data=reference, current_data=current)
    return suite


def create_report(current, reference, i=3):
    if i == 0:
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        report.run(reference_data=reference, current_data=current)
    elif i == 1:
        report = Report(metrics=[
            ColumnSummaryMetric(column_name='AveRooms'),
            ColumnQuantileMetric(column_name='AveRooms', quantile=0.25),
            ColumnDriftMetric(column_name='AveRooms')
        ])
        report.run(reference_data=reference, current_data=current)

    elif i == 2:
        report = Report(metrics=[
            generate_column_metrics(ColumnQuantileMetric, parameters={'quantile': 0.25}, columns=['AveRooms', 'AveBedrms']),
        ])
        report.run(reference_data=reference, current_data=current)
        report = Report(metrics=[
            ColumnSummaryMetric(column_name='AveRooms'),
            generate_column_metrics(ColumnQuantileMetric, parameters={'quantile': 0.25}, columns='num'),
            DataDriftPreset()
        ])
        report.run(reference_data=reference, current_data=current)

    # MLFlow log report.as_dict()

    return report


if __name__ == '__main__':
    main()
