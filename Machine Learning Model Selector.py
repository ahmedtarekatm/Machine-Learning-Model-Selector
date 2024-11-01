import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.preprocessing import normalize

def load_data(file_path):
    return pd.read_csv(file_path)


def prepare_data(data, target_column, test_size):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def determine_task(y):
    return 'classification' if len(np.unique(y)) <= 10 else 'regression'


def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def label_train_and_evaluate_models(X_train, X_test, y_train, y_test, task):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if task == 'classification':
        models = [
            ('Logistic Regression', LogisticRegression(), {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }),
            ('Decision Tree', DecisionTreeClassifier(), {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            ('Random Forest', RandomForestClassifier(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            ('SVM', SVC(), {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }),
            ('KNN', KNeighborsClassifier(), {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }),
            ('Naive Bayes', GaussianNB(), {}),
            ('Gradient Boosting', GradientBoostingClassifier(), {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10]
            })
        ]
    else:  # regression
        models = [
            ('Linear Regression', LinearRegression(), {}),
            ('Ridge Regression', Ridge(), {
                'alpha': [0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }),
            ('Lasso Regression', Lasso(), {
                'alpha': [0.1, 1.0, 10.0],
                'selection': ['cyclic', 'random']
            }),
            ('Decision Tree Regressor', DecisionTreeRegressor(), {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            ('Random Forest Regressor', RandomForestRegressor(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            ('SVR', SVR(), {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }),
            ('Gradient Boosting Regressor', GradientBoostingRegressor(), {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10]
            })
        ]

    best_model = None
    best_score = float('-inf') if task == 'classification' else float('inf')
    best_params = None

    for name, model, params in models:
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, params, cv=5,
                                   scoring='accuracy' if task == 'classification' else 'neg_mean_squared_error',
                                   n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        if task == 'classification':
            score = evaluate_classification(grid_search.best_estimator_, X_test_scaled, y_test)
            print(f"{name} Accuracy: {score}")
            if score > best_score:
                best_score = score
                best_model = name
                best_params = grid_search.best_params_
        else:
            mse, r2 = evaluate_regression(grid_search.best_estimator_, X_test_scaled, y_test)
            print(f"{name} MSE: {mse}, R2: {r2}")
            if mse < best_score:
                best_score = mse
                best_model = name
                best_params = grid_search.best_params_

    return best_model, best_params, best_score


def get_test_size():
    while True:
        try:
            test_size = float(input("Enter the percentage of test data (10-50): "))
            if 10 <= test_size <= 50:
                return test_size / 100  # Convert percentage to decimal
            else:
                print("Please enter a number between 10 and 50.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def analyze_columns(data):
    analysis = {}
    for column in data.columns:
        column_type = data[column].dtype
        unique_values = data[column].nunique()
        missing_values = data[column].isnull().sum()

        if pd.api.types.is_numeric_dtype(column_type):
            if unique_values <= 10:
                col_type = "Categorical (Numeric)"
            else:
                col_type = "Numeric"
        elif pd.api.types.is_string_dtype(column_type):
            if unique_values <= 10:
                col_type = "Categorical (String)"
            else:
                col_type = "Text"
        else:
            col_type = "Other"

        preprocessing_options = []
        if missing_values > 0:
            preprocessing_options.append("Impute missing values")
        if col_type == "Numeric":
            preprocessing_options.extend(["StandardScaler", "MinMaxScaler"])
        if col_type in ["Categorical (Numeric)", "Categorical (String)"]:
            preprocessing_options.extend(["One-Hot Encoding", "Label Encoding"])
        if col_type == "Text":
            preprocessing_options.append("TF-IDF Vectorization")

        analysis[column] = {
            "type": col_type,
            "unique_values": unique_values,
            "missing_values": missing_values,
            "preprocessing_options": preprocessing_options
        }

    return analysis


def display_column_analysis(analysis):
    print("\nColumn Analysis:")
    for column, info in analysis.items():
        print(f"\n{column}:")
        print(f"  Type: {info['type']}")
        print(f"  Unique Values: {info['unique_values']}")
        print(f"  Missing Values: {info['missing_values']}")
        print("  Preprocessing Options:")
        for i, option in enumerate(info['preprocessing_options'], 1):
            print(f"    {i}. {option}")


def get_preprocessing_choices(analysis):
    choices = {}
    for column, info in analysis.items():
        if info['preprocessing_options']:
            print(f"\nChoose preprocessing options for {column}:")
            for i, option in enumerate(info['preprocessing_options'], 1):
                print(f"  {i}. {option}")
            while True:
                choice = input("Your choices (comma-separated numbers, or 0 for none): ")
                if choice == '0':
                    break
                try:
                    selected_options = [int(c.strip()) for c in choice.split(',') if c.strip()]
                    if all(1 <= opt <= len(info['preprocessing_options']) for opt in selected_options):
                        choices[column] = [info['preprocessing_options'][i - 1] for i in selected_options]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter comma-separated numbers.")
        else:
            print(f"\nNo preprocessing options available for {column}")
    return choices


def create_preprocessing_pipeline(choices):
    transformers = []

    for column, chosen_options in choices.items():
        column_pipeline = []
        for option in chosen_options:
            if option == "Impute missing values":
                column_pipeline.append(('imputer', SimpleImputer(strategy='mean')))
            elif option == "StandardScaler":
                column_pipeline.append(('scaler', StandardScaler()))
            elif option == "MinMaxScaler":
                column_pipeline.append(('minmax', MinMaxScaler()))
            elif option == "One-Hot Encoding":
                column_pipeline.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
            elif option == "Label Encoding":
                column_pipeline.append(('label', LabelEncoder()))
            elif option == "TF-IDF Vectorization":
                column_pipeline.append(('tfidf', TfidfVectorizer()))

        if column_pipeline:
            transformers.append((f'pipeline_{column}', Pipeline(column_pipeline), [column]))

    return ColumnTransformer(transformers, remainder='passthrough')


def prepare_data_with_preprocessing(data, target_column, test_size, preprocessing_pipeline):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


def display_column_info(df, column):
    print(f"\nColumn: {column}")
    col_type = df[column].dtype
    print(f"Type: {col_type}")

    num_unique = df[column].nunique()
    print(f"Number of unique values: {num_unique}")

    num_missing = df[column].isnull().sum()
    print(f"Number of missing values: {num_missing}")

    return col_type, num_missing


def get_numeric_options(df, column, num_missing):
    options = ( ["0. No preprocessing"] if num_missing == 0 else []
    ) + [
        "1. Handling Outliers",
        "2. Log transformation",
        "3. Standard scaling",
        "4. Min-max scaling"
    ]
    if num_missing > 0:
        options.extend([
            "5. Impute missing values with mean",
            "6. Impute missing values with median"
        ])

    print(f"Mean: {df[column].mean():.2f}")
    print(f"Median: {df[column].median():.2f}")
    print(f"Standard deviation: {df[column].std():.2f}")
    print(f"min: {df[column].min():.2f}")
    print(f"max: {df[column].max():.2f}")
    print(f"25th_percentile: {df[column].quantile(0.25):.2f}")
    print(f"75th_percentile: {df[column].quantile(0.75):.2f}")


    skewness = stats.skew(df[column].dropna())
    print(f"Skewness: {skewness:.2f}")

    return options


def get_categorical_options(df, column, num_missing):
    options = [
        "1. One-hot encoding",
        "2. Label encoding"
    ]
    if num_missing > 0:
        options.extend([
            "3. Impute missing values with mode",
            "4. Create 'Missing' category for missing values"
        ])

    print(f"Most common value: {df[column].mode().values[0]}")
    return options


def get_boolean_options(num_missing):
    options = ( ["0. No preprocessing"] if num_missing == 0 else []
    ) + [
        "1. Convert to integer (0 and 1)"
    ]
    if num_missing > 0:
        options.append("2. Impute missing values with mode")
    return options


def get_datetime_options():
    return [
        "0. No preprocessing",
        "1. Extract year",
        "2. Extract month",
        "3. Extract day",
        "4. Extract day of week",
        "5. Calculate days since a reference date"
    ]


def apply_numeric_preprocessing(df, column, choice, num_missing):
    if choice == 1:
        Q1 = df[column].quantile(0.5)
        Q2 = df[column].quantile(0.95)
        IQR = Q2 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q2 + 1.5 * IQR)))]
    elif choice == 2:
        df[column] = np.log1p(df[column])
    elif choice == 3:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
    elif choice == 4:
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[[column]])

    elif num_missing > 0:
        if choice == 5:
            imputer = SimpleImputer(strategy='mean')
            df[column] = imputer.fit_transform(df[[column]])
        elif choice == 6:
            imputer = SimpleImputer(strategy='median')
            df[column] = imputer.fit_transform(df[[column]])
    return df


def apply_categorical_preprocessing(df, column, choice, num_missing):
    if choice == 1:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df, encoded_df], axis=1)
    elif choice == 2:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
    elif num_missing > 0:
        if choice == 3:
            df[column] = df[column].fillna(df[column].mode()[0])
        elif choice == 4:
            df[column] = df[column].fillna('Missing')
    return df


def apply_boolean_preprocessing(df, column, choice, num_missing):
    if choice == 1:
        df[f'{column}_int'] = df[column].astype(int)
    elif num_missing > 0 and choice == 2:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df


def apply_datetime_preprocessing(df, column, choice):
    if choice == 1:
        df[f'{column}_year'] = df[column].dt.year
    elif choice == 2:
        df[f'{column}_month'] = df[column].dt.month
    elif choice == 3:
        df[f'{column}_day'] = df[column].dt.day
    elif choice == 4:
        df[f'{column}_dayofweek'] = df[column].dt.dayofweek
    elif choice == 5:
        reference_date = df[column].min()
        df[f'{column}_days_since_ref'] = (df[column] - reference_date).dt.days
    return df


def preprocess_column(df, column):
    col_type, num_missing = display_column_info(df, column)

    if np.issubdtype(col_type, np.number):
        options = get_numeric_options(df, column, num_missing)
    elif col_type == 'object':
        options = get_categorical_options(df, column, num_missing)
    elif col_type == 'bool':
        options = get_boolean_options(num_missing)
    elif col_type == 'datetime64':
        options = get_datetime_options()
    else:
        print("Unsupported column type")
        return df

    print("\nPreprocessing options:")
    for option in options:
        print(option)

    choices = input("Enter the numbers of the preprocessing options you want to apply (comma-separated): ").split(',')

    for choice in choices:
        choice = int(choice.strip())
        if choice == 0:
            continue  # No preprocessing

        if np.issubdtype(col_type, np.number):
            df = apply_numeric_preprocessing(df, column, choice, num_missing)
        elif col_type == 'object':
            df = apply_categorical_preprocessing(df, column, choice, num_missing)
        elif col_type == 'bool':
            df = apply_boolean_preprocessing(df, column, choice, num_missing)
        elif col_type == 'datetime64':
            df = apply_datetime_preprocessing(df, column, choice)

    return df


def analyze_and_preprocess_dataframe(df):
    print_data(get_df_summary(df))
    for column in df.columns:
        df = preprocess_column(df, column)
        print("\n" + "-" * 50)

    return df
def get_df_summary(df):

    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'column_types': df.dtypes.apply(str).to_dict(),
        'missing_values': detect_missing_values(df),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # in MB
    }


    return summary
def detect_missing_values(df):

    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0].to_dict()


def get_target_column(data, df):
    print("\nAvailable columns:")
    for i, col in enumerate(data.columns, 1):
        print(f"{i}. {col}")

    while True:
        choice = input("\nEnter the number of the target column, or type 'clustering' if this is a clustering task: ")
        if choice.lower() == 'clustering':
            return None, 'clustering'
        try:
            choice = int(choice)
            if 1 <= choice <= len(data.columns):
                print("the target column info")
                display_column_info(data, data.columns[choice - 1])
                if df[df.columns[choice - 1]].dtype == 'object':
                    # Classification task
                    print("Classification task")
                    task = 'classification'
                elif df[df.columns[choice - 1]].dtype in ['int64', 'float64']:
                    # Regression task
                    print("Regression task")
                    task = 'regression'

                return data.columns[choice - 1], task

        except ValueError:
            pass
        print("Invalid input. Please try again.")


def split_data(df, target_column, test_size):
    if target_column is None:
        # For clustering tasks, we don't need to separate features and target
        X = df
        return train_test_split(X, test_size=test_size, random_state=42)
    else:
        # For supervised learning tasks
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)

def print_data(data):
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for kk, vv in v.items():
                        print(f"    {kk}: {vv}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key} : {value}")

def main():
    file_path = "data/Data.csv"
    df = load_data(file_path)
    data = analyze_and_preprocess_dataframe(df)
    nan_counts = df.isnull().sum()
    if nan_counts.sum() != 0:
        print("NaN values found in the following columns:")
        for column, count in nan_counts.items():
            if count > 0:
                print(f"{column}: {count} NaN values")
        print("drop all non data")
        data.dropna(inplace=True)
    duplicate_count = df.duplicated(keep=False).sum()
    if duplicate_count != 0:
        print(f"drop all duplicates:  {duplicate_count}")
        data.drop_duplicates(inplace=True)
    task = None
    while task is None:
        target_column, task = get_target_column(data, df)

    # Get the test set size
    while True:
        test_size_input = input("\nEnter the desired test set size (0.0 to 1.0): ")
        try:
            test_size = float(test_size_input)
            if 0.0 < test_size < 1.0:
                break
            else:
                print("Test size must be between 0.0 and 1.0. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0.")

    # Split the data
    print(data.head())
    if target_column is None:
        X_train, X_test = split_data(data, target_column, test_size)
        print("\nData split for clustering task:")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
    else:
        X_train, X_test, y_train, y_test = split_data(data, target_column, test_size)
        print("\nData split for supervised learning task:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        best_model, best_params, best_score = label_train_and_evaluate_models(X_train, X_test, y_train, y_test, task)
        print("\nBest Model:", best_model)
        print("Best Parameters:", best_params)
        if task == 'classification':
            print(f"Best Accuracy: {best_score:.2%}")
        else:
            print(f"Best MSE: {best_score:.4f}")






if __name__ == "__main__":
    main()