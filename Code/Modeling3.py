import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, auc, fbeta_score

#### 1. XGBoost #####
class XGmodeling:

    def __init__(self, _df_train=None, _df_test=None, _x_columns=None):
        self.df_train = _df_train
        self.df_test = _df_test
        self.x_columns = _x_columns

    def my_score(self, x_train, y_train, params):

        ### 학습 데이터에 대한 교차검증을 위해 fold를 4개로 쪼갠다
        fold1 = list((x_train[(x_train.index >= datetime.strptime('2012-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2014-12-01', '%Y-%m-%d'))].index).unique())
        fold2 = list((x_train[(x_train.index >= datetime.strptime('2015-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2016-12-01', '%Y-%m-%d'))].index).unique())
        fold3 = list((x_train[(x_train.index >= datetime.strptime('2017-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2018-12-01', '%Y-%m-%d'))].index).unique())
        fold4 = list((x_train[(x_train.index >= datetime.strptime('2019-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2020-12-01', '%Y-%m-%d'))].index).unique())

        test_fold = np.full(x_train.shape[0], -1)
        test_fold[x_train.index.isin(fold1)] = 0
        test_fold[x_train.index.isin(fold2)] = 1
        test_fold[x_train.index.isin(fold3)] = 2
        test_fold[x_train.index.isin(fold4)] = 3

        from sklearn.model_selection import PredefinedSplit
        ps = PredefinedSplit(test_fold)

        x = x_train.values
        y = y_train.values

        # 모델 검증
        scores = []
        for train_index, test_index in ps.split():
            x_train_cross, x_test_cross = x[train_index], x[test_index]
            y_train_cross, y_test_cross = y[train_index], y[test_index]

            scaler = MinMaxScaler()
            x_train_cross_scaled = scaler.fit_transform(x_train_cross)      # 학습 데이터에 대해서만 fit
            x_test_cross_scaled = scaler.transform(x_test_cross)  

            model = XGBClassifier(**params, random_state=1)
            model.fit(x_train_cross_scaled, y_train_cross)

            probas = model.predict_proba(x_test_cross_scaled)

            # fpr, tpr, thresholds = roc_curve(y_test_cross, y_pred_proba[:, 1])
            precision, recall, thresholds = precision_recall_curve(y_test_cross, probas[:, 1])
            f1_scores = 2*(precision*recall)/(precision+recall)
            f1_scores = f1_scores[~np.isnan(f1_scores)]

            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            y_pred = (probas[:, 1] > optimal_threshold).astype(int)

            score = f1_score(y_test_cross, y_pred)  # F1 스코어로 변경
            scores.append(score)

        score_mean = sum(scores)/len(scores)
        print(f'최적의 threshold : {optimal_threshold}, 최적의 교차검증 f1 score : {score_mean}')
        return score_mean
    
    def objective(self, params):

        x_train = self.df_train[self.x_columns]
        y_train = self.df_train['부실여부']

        score = self.my_score(x_train, y_train, params)
        return {'loss': -score, 'status': STATUS_OK }

    def my_tuning(self):

        # 파라미터 공간을 정의합니다
        space = {
            'max_depth': hp.choice('max_depth', range(1, 10)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'gamma': hp.uniform('gamma', 0, 0.5),
            'min_child_weight': hp.uniform('min_child_weight', 0, 10),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1)
        }

        # fmin 함수를 실행하여 최적의 파라미터를 찾습니다
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
        return best
    
    def my_modeling(self, params):

        x_train = self.df_train[self.x_columns]
        x_test = self.df_test[self.x_columns]

        y_train = self.df_train['부실여부']
        y_test = self.df_test['부실여부']

        # 모델 학습
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)      # 학습 데이터에 대해서만 fit
        x_test_scaled = scaler.transform(x_test)  

        model = XGBClassifier(**params, random_state=1)
        model.fit(x_train_scaled, y_train)

        probas = model.predict_proba(x_test_scaled)

        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        # ROC 커브 그리기
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # f1_scores = 2*(precision*recall)/(precision+recall)
        # f1_scores = f1_scores[~np.isnan(f1_scores)]
        beta = 2
        f2_scores = ((1 + beta**2) * (precision * recall)) / ((beta**2 * precision) + recall)
        f2_scores = f2_scores[~np.isnan(f2_scores)]
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx]

        # thresholds = np.clip(np.linspace(optimal_threshold - 0.1, optimal_threshold + 0.1, 40), 0, 1)
        # for threshold in thresholds:

        predictions = (probas[:, 1] >= optimal_threshold).astype('int')

        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, probas[:, 1])
        test_f1 = f1_score(y_test, predictions)
        test_f2 = fbeta_score(y_test, predictions, beta=2)

        # 테스트 데이터에 대한 예측 성능 출력
        print('Threshold :', optimal_threshold)
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))
        print('f2 스코어 :', round(test_f2,4))
        print('\n')
###############


#### 2. CatBoost #####
class Catmodeling:

    def __init__(self, _df_train=None, _df_test=None, _x_columns=None):
        self.df_train = _df_train
        self.df_test = _df_test
        self.x_columns = _x_columns

    def my_score(self, x_train, y_train, params):

        ### 학습 데이터에 대한 교차검증을 위해 fold를 4개로 쪼갠다
        fold1 = list((x_train[(x_train.index >= datetime.strptime('2012-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2014-12-01', '%Y-%m-%d'))].index).unique())
        fold2 = list((x_train[(x_train.index >= datetime.strptime('2015-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2016-12-01', '%Y-%m-%d'))].index).unique())
        fold3 = list((x_train[(x_train.index >= datetime.strptime('2017-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2018-12-01', '%Y-%m-%d'))].index).unique())
        fold4 = list((x_train[(x_train.index >= datetime.strptime('2019-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2020-12-01', '%Y-%m-%d'))].index).unique())

        test_fold = np.full(x_train.shape[0], -1)
        test_fold[x_train.index.isin(fold1)] = 0
        test_fold[x_train.index.isin(fold2)] = 1
        test_fold[x_train.index.isin(fold3)] = 2
        test_fold[x_train.index.isin(fold4)] = 3

        from sklearn.model_selection import PredefinedSplit
        ps = PredefinedSplit(test_fold)

        x = x_train.values
        y = y_train.values

        # 모델 검증
        scores = []
        for train_index, test_index in ps.split():
            x_train_cross, x_test_cross = x[train_index], x[test_index]
            y_train_cross, y_test_cross = y[train_index], y[test_index]

            scaler = MinMaxScaler()
            x_train_cross_scaled = scaler.fit_transform(x_train_cross)      # 학습 데이터에 대해서만 fit
            x_test_cross_scaled = scaler.transform(x_test_cross)  

            model = CatBoostClassifier(**params, random_state=1, verbose=0)
            model.fit(x_train_cross_scaled, y_train_cross)

            probas = model.predict_proba(x_test_cross_scaled)

            # fpr, tpr, thresholds = roc_curve(y_test_cross, y_pred_proba[:, 1])
            precision, recall, thresholds = precision_recall_curve(y_test_cross, probas[:, 1])
            f1_scores = 2*(precision*recall)/(precision+recall)
            f1_scores = f1_scores[~np.isnan(f1_scores)]

            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            y_pred = (probas[:, 1] > optimal_threshold).astype(int)

            score = f1_score(y_test_cross, y_pred)  # F1 스코어로 변경
            scores.append(score)

        score_mean = sum(scores)/len(scores)
        print(f'최적의 threshold : {optimal_threshold}, 최적의 교차검증 f1 score : {score_mean}')
        return score_mean
    
    def objective(self, params):

        x_train = self.df_train[self.x_columns]
        y_train = self.df_train['부실여부']

        score = self.my_score(x_train, y_train, params)
        return {'loss': -score, 'status': STATUS_OK }

    def my_tuning(self):

        # 파라미터 공간을 정의합니다
        space = {
            'depth': hp.quniform("depth", 6, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 2, 10),
            'border_count': hp.quniform('border_count', 32, 255, 1),
            'iterations': hp.quniform('iterations', 100, 1000, 1)
        }

        # fmin 함수를 실행하여 최적의 파라미터를 찾습니다
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
        return best
    
    def my_modeling(self, params):

        x_train = self.df_train[self.x_columns]
        x_test = self.df_test[self.x_columns]

        y_train = self.df_train['부실여부']
        y_test = self.df_test['부실여부']

        # 모델 학습
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)      # 학습 데이터에 대해서만 fit
        x_test_scaled = scaler.transform(x_test)  

        model = CatBoostClassifier(**params, random_state=1, verbose=0)
        model.fit(x_train_scaled, y_train)

        probas = model.predict_proba(x_test_scaled)

        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        # ROC 커브 그리기
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # f1_scores = 2*(precision*recall)/(precision+recall)
        # f1_scores = f1_scores[~np.isnan(f1_scores)]
        beta = 2
        f2_scores = ((1 + beta**2) * (precision * recall)) / ((beta**2 * precision) + recall)
        f2_scores = f2_scores[~np.isnan(f2_scores)]
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx]

        # thresholds = np.clip(np.linspace(optimal_threshold - 0.1, optimal_threshold + 0.1, 40), 0, 1)
        # for threshold in thresholds:

        predictions = (probas[:, 1] >= optimal_threshold).astype('int')

        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, probas[:, 1])
        test_f1 = f1_score(y_test, predictions)
        test_f2 = fbeta_score(y_test, predictions, beta=2)

        # 테스트 데이터에 대한 예측 성능 출력
        print('Threshold :', optimal_threshold)
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))
        print('f2 스코어 :', round(test_f2,4))
        print('\n')
###############


#### 3. AdaBoost #####
class Adamodeling:

    def __init__(self, _df_train=None, _df_test=None, _x_columns=None):
        self.df_train = _df_train
        self.df_test = _df_test
        self.x_columns = _x_columns

    def my_score(self, x_train, y_train, params):

        ### 학습 데이터에 대한 교차검증을 위해 fold를 4개로 쪼갠다
        fold1 = list((x_train[(x_train.index >= datetime.strptime('2012-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2014-12-01', '%Y-%m-%d'))].index).unique())
        fold2 = list((x_train[(x_train.index >= datetime.strptime('2015-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2016-12-01', '%Y-%m-%d'))].index).unique())
        fold3 = list((x_train[(x_train.index >= datetime.strptime('2017-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2018-12-01', '%Y-%m-%d'))].index).unique())
        fold4 = list((x_train[(x_train.index >= datetime.strptime('2019-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2020-12-01', '%Y-%m-%d'))].index).unique())

        test_fold = np.full(x_train.shape[0], -1)
        test_fold[x_train.index.isin(fold1)] = 0
        test_fold[x_train.index.isin(fold2)] = 1
        test_fold[x_train.index.isin(fold3)] = 2
        test_fold[x_train.index.isin(fold4)] = 3

        from sklearn.model_selection import PredefinedSplit
        ps = PredefinedSplit(test_fold)

        x = x_train.values
        y = y_train.values

        # 모델 검증
        scores = []
        for train_index, test_index in ps.split():
            x_train_cross, x_test_cross = x[train_index], x[test_index]
            y_train_cross, y_test_cross = y[train_index], y[test_index]

            scaler = MinMaxScaler()
            x_train_cross_scaled = scaler.fit_transform(x_train_cross)      # 학습 데이터에 대해서만 fit
            x_test_cross_scaled = scaler.transform(x_test_cross)  

            model = AdaBoostClassifier(**params, random_state=1)
            model.fit(x_train_cross_scaled, y_train_cross)

            probas = model.predict_proba(x_test_cross_scaled)

            # fpr, tpr, thresholds = roc_curve(y_test_cross, y_pred_proba[:, 1])
            precision, recall, thresholds = precision_recall_curve(y_test_cross, probas[:, 1])
            f1_scores = 2*(precision*recall)/(precision+recall)
            f1_scores = f1_scores[~np.isnan(f1_scores)]

            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            y_pred = (probas[:, 1] > optimal_threshold).astype(int)

            score = f1_score(y_test_cross, y_pred)  # F1 스코어로 변경
            scores.append(score)

        score_mean = sum(scores)/len(scores)
        print(f'최적의 threshold : {optimal_threshold}, 최적의 교차검증 f1 score : {score_mean}')
        return score_mean
    
    def objective(self, params):

        x_train = self.df_train[self.x_columns]
        y_train = self.df_train['부실여부']

        score = self.my_score(x_train, y_train, params)
        return {'loss': -score, 'status': STATUS_OK }

    def my_tuning(self):

        # 파라미터 공간을 정의합니다
        space = {
            'n_estimators': hp.choice('n_estimators', range(50, 500)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 1.0)
        }

        # fmin 함수를 실행하여 최적의 파라미터를 찾습니다
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
        return best
    
    def my_modeling(self, params):

        x_train = self.df_train[self.x_columns]
        x_test = self.df_test[self.x_columns]

        y_train = self.df_train['부실여부']
        y_test = self.df_test['부실여부']

        # 모델 학습
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)      # 학습 데이터에 대해서만 fit
        x_test_scaled = scaler.transform(x_test)  

        model = AdaBoostClassifier(**params, random_state=1)
        model.fit(x_train_scaled, y_train)

        probas = model.predict_proba(x_test_scaled)

        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        # ROC 커브 그리기
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # f1_scores = 2*(precision*recall)/(precision+recall)
        # f1_scores = f1_scores[~np.isnan(f1_scores)]
        beta = 2
        f2_scores = ((1 + beta**2) * (precision * recall)) / ((beta**2 * precision) + recall)
        f2_scores = f2_scores[~np.isnan(f2_scores)]
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx]

        # thresholds = np.clip(np.linspace(optimal_threshold - 0.1, optimal_threshold + 0.1, 40), 0, 1)
        # for threshold in thresholds:

        predictions = (probas[:, 1] >= optimal_threshold).astype('int')

        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, probas[:, 1])
        test_f1 = f1_score(y_test, predictions)
        test_f2 = fbeta_score(y_test, predictions, beta=2)

        # 테스트 데이터에 대한 예측 성능 출력
        print('Threshold :', optimal_threshold)
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))
        print('f2 스코어 :', round(test_f2,4))
        print('\n')
###############


#### 4. RandomForest #####
class RFmodeling:

    def __init__(self, _df_train=None, _df_test=None, _x_columns=None):
        self.df_train = _df_train
        self.df_test = _df_test
        self.x_columns = _x_columns

    def my_score(self, x_train, y_train, params):

        ### 학습 데이터에 대한 교차검증을 위해 fold를 4개로 쪼갠다
        fold1 = list((x_train[(x_train.index >= datetime.strptime('2012-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2014-12-01', '%Y-%m-%d'))].index).unique())
        fold2 = list((x_train[(x_train.index >= datetime.strptime('2015-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2016-12-01', '%Y-%m-%d'))].index).unique())
        fold3 = list((x_train[(x_train.index >= datetime.strptime('2017-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2018-12-01', '%Y-%m-%d'))].index).unique())
        fold4 = list((x_train[(x_train.index >= datetime.strptime('2019-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2020-12-01', '%Y-%m-%d'))].index).unique())

        test_fold = np.full(x_train.shape[0], -1)
        test_fold[x_train.index.isin(fold1)] = 0
        test_fold[x_train.index.isin(fold2)] = 1
        test_fold[x_train.index.isin(fold3)] = 2
        test_fold[x_train.index.isin(fold4)] = 3

        from sklearn.model_selection import PredefinedSplit
        ps = PredefinedSplit(test_fold)

        x = x_train.values
        y = y_train.values

        # 모델 검증
        scores = []
        for train_index, test_index in ps.split():
            x_train_cross, x_test_cross = x[train_index], x[test_index]
            y_train_cross, y_test_cross = y[train_index], y[test_index]

            scaler = MinMaxScaler()
            x_train_cross_scaled = scaler.fit_transform(x_train_cross)      # 학습 데이터에 대해서만 fit
            x_test_cross_scaled = scaler.transform(x_test_cross)  

            model = RandomForestClassifier(**params, random_state=1)
            model.fit(x_train_cross_scaled, y_train_cross)

            probas = model.predict_proba(x_test_cross_scaled)

            # fpr, tpr, thresholds = roc_curve(y_test_cross, y_pred_proba[:, 1])
            precision, recall, thresholds = precision_recall_curve(y_test_cross, probas[:, 1])
            f1_scores = 2*(precision*recall)/(precision+recall)
            f1_scores = f1_scores[~np.isnan(f1_scores)]

            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            y_pred = (probas[:, 1] > optimal_threshold).astype(int)

            score = f1_score(y_test_cross, y_pred)  # F1 스코어로 변경
            scores.append(score)

        score_mean = sum(scores)/len(scores)
        print(f'최적의 threshold : {optimal_threshold}, 최적의 교차검증 f1 score : {score_mean}')
        return score_mean
    
    def objective(self, params):

        x_train = self.df_train[self.x_columns]
        y_train = self.df_train['부실여부']

        score = self.my_score(x_train, y_train, params)
        return {'loss': -score, 'status': STATUS_OK }

    def my_tuning(self):

        # 파라미터 공간을 정의합니다
        space = {
            'n_estimators': hp.choice('n_estimators', range(100,501,50)),
            'max_features': hp.choice('max_features', [0.2, 0.4, 0.7, 0.8, 0.9, 1]),
            'max_depth': hp.choice('max_depth', range(1, 31, 5)),
            'min_samples_split': hp.choice ('min_samples_split', range(2, 23, 5)),
            'min_samples_leaf': hp.choice ('min_samples_leaf', range(2, 9, 2)),
        }

        # fmin 함수를 실행하여 최적의 파라미터를 찾습니다
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
        return best
    
    def my_modeling(self, params):

        x_train = self.df_train[self.x_columns]
        x_test = self.df_test[self.x_columns]

        y_train = self.df_train['부실여부']
        y_test = self.df_test['부실여부']

        # 모델 학습
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)      # 학습 데이터에 대해서만 fit
        x_test_scaled = scaler.transform(x_test)  

        model = RandomForestClassifier(**params, random_state=1)
        model.fit(x_train_scaled, y_train)

        probas = model.predict_proba(x_test_scaled)

        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        # ROC 커브 그리기
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # f1_scores = 2*(precision*recall)/(precision+recall)
        # f1_scores = f1_scores[~np.isnan(f1_scores)]
        beta = 2
        f2_scores = ((1 + beta**2) * (precision * recall)) / ((beta**2 * precision) + recall)
        f2_scores = f2_scores[~np.isnan(f2_scores)]
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx]

        # thresholds = np.clip(np.linspace(optimal_threshold - 0.1, optimal_threshold + 0.1, 40), 0, 1)
        # for threshold in thresholds:

        predictions = (probas[:, 1] >= optimal_threshold).astype('int')

        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, probas[:, 1])
        test_f1 = f1_score(y_test, predictions)
        test_f2 = fbeta_score(y_test, predictions, beta=2)

        # 테스트 데이터에 대한 예측 성능 출력
        print('Threshold :', optimal_threshold)
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))
        print('f2 스코어 :', round(test_f2,4))
        print('\n')
###############


#### 5. LogisticRegression #####
class LRmodeling:

    def __init__(self, _df_train=None, _df_test=None, _x_columns=None):
        self.df_train = _df_train
        self.df_test = _df_test
        self.x_columns = _x_columns

    def my_score(self, x_train, y_train, params):

        ### 학습 데이터에 대한 교차검증을 위해 fold를 4개로 쪼갠다
        fold1 = list((x_train[(x_train.index >= datetime.strptime('2012-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2014-12-01', '%Y-%m-%d'))].index).unique())
        fold2 = list((x_train[(x_train.index >= datetime.strptime('2015-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2016-12-01', '%Y-%m-%d'))].index).unique())
        fold3 = list((x_train[(x_train.index >= datetime.strptime('2017-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2018-12-01', '%Y-%m-%d'))].index).unique())
        fold4 = list((x_train[(x_train.index >= datetime.strptime('2019-12-01', '%Y-%m-%d')) & (x_train.index <= datetime.strptime('2020-12-01', '%Y-%m-%d'))].index).unique())

        test_fold = np.full(x_train.shape[0], -1)
        test_fold[x_train.index.isin(fold1)] = 0
        test_fold[x_train.index.isin(fold2)] = 1
        test_fold[x_train.index.isin(fold3)] = 2
        test_fold[x_train.index.isin(fold4)] = 3

        from sklearn.model_selection import PredefinedSplit
        ps = PredefinedSplit(test_fold)

        x = x_train.values
        y = y_train.values

        # 모델 검증
        scores = []
        for train_index, test_index in ps.split():
            x_train_cross, x_test_cross = x[train_index], x[test_index]
            y_train_cross, y_test_cross = y[train_index], y[test_index]

            scaler = MinMaxScaler()
            x_train_cross_scaled = scaler.fit_transform(x_train_cross)      # 학습 데이터에 대해서만 fit
            x_test_cross_scaled = scaler.transform(x_test_cross)  

            model = LogisticRegression(**params, random_state=1)
            model.fit(x_train_cross_scaled, y_train_cross)

            probas = model.predict_proba(x_test_cross_scaled)

            # fpr, tpr, thresholds = roc_curve(y_test_cross, y_pred_proba[:, 1])
            precision, recall, thresholds = precision_recall_curve(y_test_cross, probas[:, 1])
            f1_scores = 2*(precision*recall)/(precision+recall)
            f1_scores = f1_scores[~np.isnan(f1_scores)]

            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            y_pred = (probas[:, 1] > optimal_threshold).astype(int)

            score = f1_score(y_test_cross, y_pred)  # F1 스코어로 변경
            scores.append(score)

        score_mean = sum(scores)/len(scores)
        print(f'최적의 threshold : {optimal_threshold}, 최적의 교차검증 f1 score : {score_mean}')
        return score_mean
    
    def objective(self, params):

        x_train = self.df_train[self.x_columns]
        y_train = self.df_train['부실여부']

        score = self.my_score(x_train, y_train, params)
        return {'loss': -score, 'status': STATUS_OK }

    def my_tuning(self):

        # 파라미터 공간을 정의합니다
        space = {
            'solver': 'liblinear',     
            'C': hp.loguniform('C', -4, 4),                                                     # 0.01에서 100 사이의 규제화 파라미터 'C'를 로그 스케일로 탐색
            'penalty': hp.choice('penalty', ['l1', 'l2'] ),               # 'l1' 규제화와 'l2' 규제화 중 선택
            'fit_intercept': hp.choice('fit_intercept', [True, False]),                         # 절편 포함 여부 선택                                                             # solver를 'liblinear'로 설정
        }

        # fmin 함수를 실행하여 최적의 파라미터를 찾습니다
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
        return best
    
    def my_modeling(self, params):

        x_train = self.df_train[self.x_columns]
        x_test = self.df_test[self.x_columns]

        y_train = self.df_train['부실여부']
        y_test = self.df_test['부실여부']

        # 모델 학습
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)      # 학습 데이터에 대해서만 fit
        x_test_scaled = scaler.transform(x_test)  

        model = LogisticRegression(**params, random_state=1)
        model.fit(x_train_scaled, y_train)

        probas = model.predict_proba(x_test_scaled)

        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        # ROC 커브 그리기
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        # f1_scores = 2*(precision*recall)/(precision+recall)
        # f1_scores = f1_scores[~np.isnan(f1_scores)]
        beta = 2
        f2_scores = ((1 + beta**2) * (precision * recall)) / ((beta**2 * precision) + recall)
        f2_scores = f2_scores[~np.isnan(f2_scores)]
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx]

        # thresholds = np.clip(np.linspace(optimal_threshold - 0.1, optimal_threshold + 0.1, 40), 0, 1)
        # for threshold in thresholds:

        predictions = (probas[:, 1] >= optimal_threshold).astype('int')

        # 테스트 데이터에 대한 예측 성능 지표를 계산
        test_cm = confusion_matrix(y_test, predictions)
        test_acc = accuracy_score(y_test, predictions)
        test_pre = precision_score(y_test, predictions)
        test_rcll = recall_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, probas[:, 1])
        test_f1 = f1_score(y_test, predictions)
        test_f2 = fbeta_score(y_test, predictions, beta=2)

        # 테스트 데이터에 대한 예측 성능 출력
        print('Threshold :', optimal_threshold)
        print('혼돈행렬 :', test_cm)
        print('정확도 :', round(test_acc,4))
        print('정밀도 :', round(test_pre,4))
        print('재현율 :', round(test_rcll,4))
        print('roc_auc 스코어 :', round(test_roc_auc,4))
        print('f1 스코어 :', round(test_f1,4))
        print('f2 스코어 :', round(test_f2,4))
        print('\n')
###############