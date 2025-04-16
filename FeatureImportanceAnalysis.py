import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import sem
from lightgbm import LGBMClassifier
import shap
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
import os
import xgboost as xgb
import re
from datetime import datetime


def load_data_from_json(file_path, required_columns):
    """
    从JSON文件加载数据，并验证所需列是否存在。
    随机抽取70%的数据以减少噪声和过拟合风险。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            df = pd.DataFrame(raw_data)
            df = df.sample(frac=0.7, random_state=42)
            print("✅ 成功从JSON文件加载数据")
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"JSON文件中缺少关键列: {missing_cols}")
        return df[required_columns]
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 - {file_path}")
    except json.JSONDecodeError:
        print(f"❌ 错误: JSON文件格式错误 - {file_path}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")
    return None


def preprocess_data(df, categorical_cols, numerical_cols):
    """
    对数据进行预处理，包括缺失值填充、分类变量编码和数值特征标准化。
    """
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    return df_encoded, scaler


def preprocess_data_extended(df, categorical_cols, numerical_cols,specialCol):
    """
    扩展的预处理函数，处理特殊情况，如字典值和低频分类值。
    """
    df_encoded = df.copy()
    df_encoded = df_encoded.reset_index(drop=True)
    for column in df_encoded.columns:
        if pd.api.types.is_numeric_dtype(df_encoded[column]):
            df_encoded[column] = df_encoded[column].apply(lambda x: 0 if isinstance(x, dict) and not x else x)
        elif pd.api.types.is_object_dtype(df_encoded[column]):
            df_encoded[column] = df_encoded[column].apply(lambda x: False if isinstance(x, dict) and not x else x)
    if specialCol in categorical_cols:
        min_frequency = 0.003
        original_counts = df_encoded[specialCol].value_counts(normalize=True).reset_index()
        original_counts.columns = [specialCol, 'Original_Frequency']
        city_bins = original_counts.copy()
        city_bins.loc[city_bins['Original_Frequency'] < min_frequency, specialCol] = 'Other'
        merged_counts = city_bins.groupby(specialCol)['Original_Frequency'].sum().reset_index()
        merged_counts.columns = [specialCol, 'Merged_Frequency']
        freq_mapping = merged_counts.set_index(specialCol)['Merged_Frequency'].to_dict()
        df_encoded[specialCol] = df_encoded[specialCol].map(freq_mapping)
        print("原始市场名称及频率：")
        print(original_counts)
        print("\n合并后的市场频率分布：")
        sorted_merged_counts = merged_counts.sort_values(by='Merged_Frequency', ascending=False)
        print(sorted_merged_counts)
        categorical_cols.remove(specialCol)
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    df_encoded = clean_feature_names(df_encoded)
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    if df_encoded.isnull().any().any():
        print("处理后的数据仍存在缺失值，进行进一步处理...")
        imputer = SimpleImputer(strategy='mean')
        df_encoded = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
    return df_encoded, scaler, freq_mapping


def split_dataset(df_encoded, target_col):
    """
    将编码后的数据拆分为训练集和测试集。
    """
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def clean_feature_names(X):
    """
    清理特征名称，去除特殊字符。
    """
    pattern = r'[^a-zA-Z0-9_]'
    new_columns = [re.sub(pattern, '', col) for col in X.columns]
    X.columns = new_columns
    return X


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """
    使用LightGBM模型进行训练和评估，通过随机搜索进行超参数调优。
    """
    param_distributions = {
        'n_estimators': np.arange(120, 181, 30),
        'max_depth': np.arange(9, 11, 1),
        'num_leaves': [70, 100],
        'learning_rate': [0.08, 0.12],
        'min_child_samples': [15, 25],
        'subsample': [0.75, 0.85],
        'colsample_bytree': [0.75, 0.85]
    }
    model = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return best_model


def visualize_feature_importance(model, X, categorical_prefixes, fileDir):
    """
    可视化LightGBM模型的特征重要性，并保存结果到文件。
    """
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    def get_original_feature(feature_name, categorical_prefixes):
        for prefix in categorical_prefixes:
            if feature_name.startswith(prefix + "_"):
                return prefix
        return feature_name
    feature_importance["Original_Feature"] = feature_importance["Feature"].apply(
        lambda x: get_original_feature(x, categorical_prefixes)
    )
    aggregated_importance = (
        feature_importance
        .groupby("Original_Feature", as_index=False)
        .agg({"Importance": "sum"})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    plt.figure(figsize=(10, 6))
    top_features = aggregated_importance.head(15).sort_values("Importance", ascending=True)
    plt.barh(top_features["Original_Feature"], top_features["Importance"], color='skyblue')
    plt.title("Lightgbm Aggregated Feature Importance by Original Variable", fontsize=14)
    plt.xlabel("Importance Score", fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
    save_path = os.path.join(fileDir, f'Lightgbm visualize_feature_importance_{timestamp}.jpg')
    save_outPutPath = os.path.join(fileDir, f'Lightgbm visualize_feature_importance_{datetime.now().strftime("%Y-%m-%d")}.csv')
    print(aggregated_importance)
    append_feature_importance(save_outPutPath, aggregated_importance, header=' Feature Importance from Lightgbm Regression:')
    plt.savefig(save_path)
    plt.close()


def predict_new_data(model, scaler, input_data, X_columns, categorical_cols, numerical_cols, freq_mapping):
    """
    对新数据进行预测，并返回预测结果和置信度。
    """
    new_df = pd.DataFrame([input_data])
    new_df_clean = new_df.drop(columns=[
        'StateId', 'StateName',
        'StartTime', 'StartTimeLocal', 'Holiday',
        'HasCovid', 'OnboardingStatusDoneDateTime'
    ], errors='ignore')
    new_df_clean['MarketName'] = new_df_clean['MarketName'].map(lambda x: freq_mapping.get(x, freq_mapping.get('Other')))
    new_encoded = pd.get_dummies(new_df_clean, columns=categorical_cols, drop_first=True)
    missing_cols = set(X_columns) - set(new_encoded.columns)
    missing_df = pd.DataFrame({col: [0] for col in missing_cols})
    new_encoded = pd.concat([new_encoded, missing_df], axis=1)[X_columns]
    new_encoded[numerical_cols] = scaler.transform(new_encoded[numerical_cols])
    prob = model.predict_proba(new_encoded)[0][1]
    prediction = model.predict(new_encoded)
    if prediction[0]:
        probability_str = f"{prob * 100:.1f}%"
    else:
        probability_str = f"{(1 - prob) * 100:.1f}%"
    return {
        'prediction': 'Will be filled' if prediction[0] else 'Will NOT be filled',
        'probability': probability_str
    }


def run_predictions(model, scaler, test_cases, X_columns, categorical_cols, numerical_cols, freq_mapping):
    """
    批量运行预测，并格式化输出结果。
    """
    print("\n=== 开始预测 ===")
    for i, case in enumerate(test_cases, 1):
        result = predict_new_data(model, scaler, case["data"], X_columns, categorical_cols, numerical_cols,
                                  freq_mapping)
        print(f"\n预测结果 {i}: {case['name']}")
        print(f"- 结果: {result['prediction']}")
        confidence_str = result['probability'].rstrip('%')
        try:
            confidence = float(confidence_str) / 100
            print(f"- 置信度: {confidence:.2%}")
        except ValueError:
            print("- 置信度: 无有效数据")
    print("\n=== 预测完成 ===")


def logistic_regression_feature_importance(X_train, y_train, X, categorical_prefixes, fileDir):
    """
    计算逻辑回归模型的特征重要性，并可视化和保存结果。
    """
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    feature_importance = model.coef_[0]
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = merge_categorical_features(importance_df, categorical_prefixes)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Name')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Importance from Logistic Regression')
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.35)
    plt.xticks(fontsize=8)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
    save_path = os.path.join(fileDir, f'logistic_regression_feature_importance_{timestamp}.jpg')
    save_outPutPath = os.path.join(fileDir, f'logistic_regression_feature_importance_{datetime.now().strftime("%Y-%m-%d")}.csv')
    print(importance_df)
    append_feature_importance(save_outPutPath, importance_df, header='Feature Importance from Logistic Regression:')
    plt.savefig(save_path)
    plt.close()
    return importance_df


def random_forest_feature_importance(X_train, y_train, X, categorical_prefixes, fileDir):
    """
    计算随机森林模型的特征重要性，并可视化和保存结果。
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = merge_categorical_features(importance_df, categorical_prefixes)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Name')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance from Random Forest')
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.35)
    plt.xticks(fontsize=8)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
    save_path = os.path.join(fileDir, f'random_forest_feature_importance_{timestamp}.jpg')
    save_outPutPath = os.path.join(fileDir, f'random_forest_feature_importance_{datetime.now().strftime("%Y-%m-%d")}.csv')
    print(importance_df)
    append_feature_importance(save_outPutPath, importance_df, header='Feature Importance from Random Forest:')
    plt.savefig(save_path)
    plt.close()
    return importance_df


def permutation_feature_importance_analysis2(model, X, y, categorical_prefixes, fileDir):
    """
    计算基于置换的特征重要性，并可视化和保存结果。
    """
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    feature_importance = result.importances_mean
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = merge_categorical_features(importance_df, categorical_prefixes)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Name')
    plt.ylabel('Permutation Importance Score')
    plt.title('Feature Importance from Permutation Importance')
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.35)
    plt.xticks(fontsize=8)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
    save_path = os.path.join(fileDir, f'permutation_feature_importance_{timestamp}.jpg')
    save_outPutPath = os.path.join(fileDir, f'permutation_feature_importance_{datetime.now().strftime("%Y-%m-%d")}.csv')
    print(importance_df)
    append_feature_importance(save_outPutPath, importance_df, header='Feature Importance from Permutation Importance:')
    plt.savefig(save_path)
    plt.close()
    return importance_df

def permutation_feature_importance_analysis(model, X, y, categorical_prefixes, fileDir):
    """
    计算基于置换的特征重要性，并可视化和保存结果。
    """
    # 使用多线程并行计算，n_jobs=-1表示使用所有可用CPU核心
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importance = result.importances_mean
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = merge_categorical_features(importance_df, categorical_prefixes)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Name')
    plt.ylabel('Permutation Importance Score')
    plt.title('Feature Importance from Permutation Importance')
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.35)
    plt.xticks(fontsize=8)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
    save_path = os.path.join(fileDir, f'permutation_feature_importance_{timestamp}.jpg')
    save_outPutPath = os.path.join(fileDir, f'permutation_feature_importance_{datetime.now().strftime("%Y-%m-%d")}.csv')
    print(importance_df)
    append_feature_importance(save_outPutPath, importance_df, header='Feature Importance from Permutation Importance:')
    plt.savefig(save_path)
    plt.close()
    return importance_df

def xgboost_feature_importance(X_train, y_train, X, categorical_prefixes, fileDir):
    """
    计算XGBoost模型的特征重要性，并可视化和保存结果。
    """
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = merge_categorical_features(importance_df, categorical_prefixes)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Name')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance from XGBoost')
    plt.xticks(rotation=75)
    plt.subplots_adjust(bottom=0.35)
    plt.xticks(fontsize=8)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
    save_path = os.path.join(fileDir, f'xgboost_feature_importance_{timestamp}.jpg')
    save_outPutPath = os.path.join(fileDir, f'xgboost_feature_importance_{datetime.now().strftime("%Y-%m-%d")}.csv')
    print(importance_df)
    append_feature_importance(save_outPutPath, importance_df, header='Feature Importance from XGBoost:')
    plt.savefig(save_path)
    plt.close()
    return importance_df


def append_feature_importance(file_path, feature_importance, header=None):
    """
    将特征重要性结果追加到CSV文件中。
    """
    try:
        print(feature_importance)
        if header is not None:
            with open(file_path, 'a') as f:
                f.write(f'{header}\n\n')
        feature_importance.to_csv(file_path, sep=',', na_rep='nan', mode='w', index=True, header=header)
        print(f"特征重要性已成功追加到 {file_path}")
    except Exception as e:
        print(f"写入文件时出错: {e}")


def merge_categorical_features(importance_df, categorical_prefixes):
    """
    合并类别特征的重要性。
    """
    def get_original_feature(feature_name, categorical_prefixes):
        for prefix in categorical_prefixes:
            if feature_name.startswith(prefix + "_"):
                return prefix
        return feature_name
    importance_df["Original_Feature"] = importance_df["Feature"].apply(
        lambda x: get_original_feature(x, categorical_prefixes)
    )
    aggregated_importance = (
        importance_df
        .groupby("Original_Feature", as_index=False)
        .agg({"Importance": "sum"})
    )
    aggregated_importance.rename(columns={"Original_Feature": "Feature"}, inplace=True)
    return aggregated_importance


if __name__ == "__main__":
    required_columns = [
        'MarketName', 'LicenseTypeName', 'FallInPromotion',
        'Duration', 'PayRate', 'IsPromoted',
        'DayPeriod', 'DescriptionLength',
        'DayOfWeek', 'PostDateTime2StartTimeInHour',
        'CommunityType', 'HowNewTheCommunityInMonth', 'Conditions', 'IsFilled'
    ]
    file_path = r'D:\Project\kare-forecast-shift\shift.json'
    fileDir =r'E:\Project\features' 
    
    df = load_data_from_json(file_path, required_columns)
    if df is not None:
        print(f"加载数据维度：{df.shape}")
        categorical_cols = ['MarketName', 'DayOfWeek', 'LicenseTypeName', 'CommunityType', 'DayPeriod', 'Conditions']
        numerical_cols = ['Duration', 'PayRate', 'PostDateTime2StartTimeInHour', 'HowNewTheCommunityInMonth', 'IsPromoted',
                          'FallInPromotion']
        df_true_total = df[df['IsFilled'] == True]
        df_false_total = df[df['IsFilled'] == False]
        df_trueData = df_true_total.sample(min(250000, len(df_true_total)), random_state=42)
        df_falseData = df_false_total.sample(min(250000, len(df_false_total)), random_state=42)
        df_sampled = pd.concat([df_trueData, df_falseData])
        df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        df_encoded, scaler, freq_mapping = preprocess_data_extended(df_sampled, categorical_cols, numerical_cols,'MarketName')
        X_train, X_test, y_train, y_test = split_dataset(df_encoded, 'IsFilled')
        model = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        categorical_prefixes = ['MarketName', 'DayOfWeek', 'LicenseTypeName', 'CommunityType', 'DayPeriod', 'Conditions']
        visualize_feature_importance(model, df_encoded.drop(columns=['IsFilled']), categorical_prefixes, fileDir)
        features = ['MarketName', 'DayOfWeek', 'LicenseTypeName', 'CommunityType', 'DayPeriod', 'Conditions', 'Duration',
                    'PayRate', 'PostDateTime2StartTimeInHour', 'HowNewTheCommunityInMonth', 'IsPromoted',
                    'FallInPromotion']
        print("xgboost_feature_importance：")
        xgboost_feature_importance(X_train, y_train, df_encoded.drop(columns=['IsFilled']), categorical_prefixes, fileDir)
        print("逻辑回归特征重要性分析：")
        logistic_regression_feature_importance(X_train, y_train, df_encoded.drop(columns=['IsFilled']), categorical_prefixes, fileDir)
        print("随机森林特征重要性分析：")
        random_forest_feature_importance(X_train, y_train, df_encoded.drop(columns=['IsFilled']), categorical_prefixes, fileDir)
        print("基于置换的特征重要性分析：")
        permutation_feature_importance_analysis(model, df_encoded.drop(columns=['IsFilled']), df_encoded['IsFilled'], categorical_prefixes, fileDir)
   