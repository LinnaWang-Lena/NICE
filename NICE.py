import pandas as pd
import numpy as np
import matplotlib as mpl

import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus']=False
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# One-hot encoding
def onehot_coder(rundf, cate_v):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_columm = []
    for col in cate_v:
        
        encoded_data = encoder.fit_transform(rundf[[col]])
        
        encoded_columns = encoder.get_feature_names_out([col])
        onehot_columm.extend(encoded_columns)
        
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=rundf.index)
        
        rundf = pd.concat([rundf.drop(columns=[col]), encoded_df], axis=1)
    
    return rundf, onehot_columm

# Calculate the Propensity Score (GPS)
def calculate_gps(Train, treatments, label):
    treatments_ps = pd.DataFrame(index=Train.index)
    
    # Identify covariates (excluding labels)
    Train_gps = Train.drop(columns = label)
    
    for treatment in treatments:
        
        covariates = Train_gps.columns.difference(treatments).tolist()

        if '_' in treatment:
            original_var = treatment.split('_')[0] + '_'  
        else:
            original_var = treatment  
    
        
        covariates = [col for col in covariates if not col.startswith(original_var)]

        X = Train_gps[covariates]
        T = Train_gps[treatment]
        
        unique_values = T.nunique()
            
        if unique_values == 2:
            
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty="l1", C=0.1, solver="liblinear", class_weight="balanced")
            )
            model.fit(X, T)
            probas = model.predict_proba(X)
            gps = probas[:, 1]  
            treatments_ps[f'{treatment}_GPS'] = gps
            
        else:
            # Continuous variables (gradient boosting regression is used by default)
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X, T)
            mu = model.predict(X)
            residuals = T - mu
            sigma = np.std(residuals)
            

            sigma = sigma if sigma > 0 else 1e-6
            
            # Calculate the normal density as GPS
            gps = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (residuals / sigma)**2)
            treatments_ps[f'{treatment}_GPS'] = gps
        
    
    return treatments_ps

# Customize k-nearest neighbors (knn) to only match Propensity Scores (PS)
class SingleFeatureKNN:
    def __init__(self, k=3, target_feature=0):
        self.k = k
        self.target_feature = target_feature  

    def fit(self, X, y):
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X_train = X[:, self.target_feature].reshape(-1, 1)  
        self.y_train = np.array(y)  

    def predict(self, X):
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_test = X[:, self.target_feature].reshape(-1, 1)
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        
        distances = np.abs(self.X_train - x).flatten()
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = self.y_train[k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Get the IDs after multiple-few Propensity Score Matching (PSM)
def get_PSM_id(treatments_ps, treatments, label):
    
    ps_majority = treatments_ps[treatments_ps[label] == 0].drop(label, axis=1).reset_index(drop=True)
    ps_minority = treatments_ps[treatments_ps[label] == 1].drop(label, axis=1).reset_index(drop=True)

    MatchID = pd.DataFrame({'majority_id': ps_majority.index.values})

    for i in range(len(treatments)):
        knn = SingleFeatureKNN(k=1, target_feature=i)
        knn.fit(ps_minority, ps_minority.index)  

        min_id = 'PS_minority_id_' + ps_majority.columns[i]
        MatchID[min_id] = 0

        for i in ps_majority.index.values:
            indxs = knn.predict(ps_majority.iloc[i].values.reshape(1, -1))
            MatchID[min_id].iloc[i] = indxs[0]

    return MatchID

# Calculate tolerance
def get_tolerance(nums_col, Train):
    
    #计算df中num_col的Tolerance
    mean_values = Train[nums_col].apply(np.mean).tolist() 
    std_values = Train[nums_col].apply(np.std).tolist()   
    mean_column = pd.DataFrame(mean_values, columns=['Mean'])
    std_column = pd.DataFrame(std_values, columns=['STD'])

    mean_std = pd.concat([mean_column,std_column], axis=1)
    mean_std['numcols'] = nums_col
    mean_std["tolerance"] = 0

    
    for ind, row in mean_std.iterrows():
        mean_std.loc[ind, "std_value"] = row['Mean'] + (1*row['STD'])
    for ind, row in mean_std.iterrows():
        mean_std.loc[ind, "tolerance"] = 0.1 * (row['std_value'] - (row['Mean']))
    
    tolerance_dict = dict(zip(mean_std['numcols'], mean_std['tolerance']))
    return tolerance_dict

# Process the XXXX-1, retaining only those with differences.
def process_row(row, majority_class, minority_class, num_col, cate_col, tolerance):
    majority_id = row['majority_id']
    features = row['high_ps_columns']
    minority_ids = row['matched_minority_ids']
    
    
    if majority_id not in majority_class.index:
        return pd.Series({
            'majority_id': majority_id,
            'high_ps_columns': [],
            'matched_minority_ids': []
        })
    majority_sample = majority_class.loc[majority_id]
    
    new_features = []
    new_minority_ids = []
    
    for feature, m_id in zip(features, minority_ids):
    
        if m_id not in minority_class.index:
            continue
        minority_sample = minority_class.loc[m_id]
        
        if feature not in majority_sample or feature not in minority_sample:
            continue
        
        if feature in num_col:
            try:
                majority_val = float(majority_sample[feature])
                minority_val = float(minority_sample[feature])
            except (ValueError, TypeError):
                continue  
            
            diff = abs(majority_val - minority_val)
            
            tol = tolerance.get(feature, None)
            if tol is not None and diff >= tol:
                new_features.append(feature)
                new_minority_ids.append(m_id)
        elif feature in cate_col:
        
            majority_val = str(majority_sample[feature])
            minority_val = str(minority_sample[feature])
            if majority_val != minority_val:
                new_features.append(feature)
                new_minority_ids.append(m_id)
    
    return pd.Series({
        'majority_id': majority_id,
        'high_ps_columns': new_features,
        'matched_minority_ids': new_minority_ids
    })


# Improve ENN XXXX-1 cleaning: only remove newly generated samples
def new_samples_enn_clean(XXXX-1, target_name, n_train, 
                          k_ir=5, k_vote=13, 
                          max_removal_ratio=0.1, exponent=1):
    
    X = XXXX-1.drop(columns=[target_name]).values
    y = XXXX-1[target_name].values
    n_samples = len(X)
    
    
    # Calculate the global imbalance ratio IR_all
    train_y = y[:n_train]
    class_counts = pd.Series(train_y).value_counts()
    majority_label = class_counts.idxmax()
    minority_label = class_counts.idxmin()
    IR_all = class_counts[majority_label] / class_counts[minority_label]

    nn_ir = NearestNeighbors(n_neighbors=k_ir+1, algorithm='auto', n_jobs=-1)
    nn_ir.fit(X)
    
    to_remove = set()
    new_samples_count = n_samples - n_train
    
    for i in range(n_train, n_samples):

        x_i = X[i].reshape(1, -1)
        current_label = y[i]
        
        disadvantage_class = minority_label if current_label == majority_label else majority_label
        
        _, init_indices = nn_ir.kneighbors(x_i)
        init_indices = init_indices[0]
        
        neighbor_indices = init_indices[init_indices != i][:k_ir]
        neighbor_labels = y[neighbor_indices]
        
        # Calculate the local imbalance ratio IR_nn
        maj_count = np.sum(neighbor_labels == majority_label)
        min_count = np.sum(neighbor_labels == minority_label)
        min_count = max(min_count, 1)  
        IR_nn = maj_count / min_count
        
        # Calculate the weight coefficient
        IW = IR_nn / IR_all
        weight = np.exp(IW ** exponent)
        
        distances = pairwise_distances(x_i, X, metric='euclidean')[0]
        
        adjusted_distances = distances.copy()
        disadvantage_mask = (y == disadvantage_class)
        adjusted_distances[disadvantage_mask] *= weight

        sorted_indices = np.argsort(adjusted_distances)
        neighbor_indices_adj = [idx for idx in sorted_indices if idx != i][:k_vote]
        neighbor_labels_adj = y[neighbor_indices_adj]
        
        majority_label_adj = np.bincount(neighbor_labels_adj).argmax()
        
        if current_label != majority_label_adj:
            to_remove.add(i)
    
    max_to_remove = int(new_samples_count * max_removal_ratio)
    removal_count = len(to_remove)
    
    if removal_count > max_to_remove:
        to_remove = list(to_remove)[:max_to_remove]
        removal_count = max_to_remove
    
    keep_indices = [i for i in range(n_samples) if i not in to_remove]
    X_clean = X[keep_indices]
    y_clean = y[keep_indices]
    
    cleaned_data = pd.DataFrame(X_clean, columns=XXXX-1.drop(columns=[target_name]).columns)
    cleaned_data[target_name] = y_clean
    
    return cleaned_data


# Inverse one-hot encoding, Decode only the one-hot encoded columns whose column name prefix is "cate_v", 
# and keep other columns unchanged.
def reverse_onehot(encoded_df, cate_v):

    import pandas as pd
    from collections import defaultdict

    original_df = pd.DataFrame(index=encoded_df.index)

    onehot_cols = [col for col in encoded_df.columns if '_' in col and col.rsplit('_', 1)[0] in cate_v]

    col_groups = defaultdict(list)
    for col in onehot_cols:
        orig_col, category = col.rsplit('_', 1)
        col_groups[orig_col].append((col, category))

    for orig_col, cols in col_groups.items():
        sub_df = encoded_df[[c for c, _ in cols]]
        categories = sub_df.idxmax(axis=1).apply(lambda x: x.rsplit('_', 1)[1] if pd.notnull(x) else None)
        original_df[orig_col] = categories

    decoded_cols = set(sum([[c for c, _ in v] for v in col_groups.values()], []))
    other_cols = [col for col in encoded_df.columns if col not in decoded_cols]
    original_df = pd.concat([original_df, encoded_df[other_cols]], axis=1)

    return original_df



# Main function
def run(df,label, n, x, k_ir, k_vote):
    ############################################### Preprocessing #####################################################################
    cate_col = []
    num_col = []

    for col in df.columns:
        if col == label:
            continue 
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            cate_col.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]): 
            unique_values = df[col].nunique()
            if unique_values <= 10: 
                cate_col.append(col)
            else:
                num_col.append(col)
        else:
            cate_col.append(col)

    Fset = num_col + cate_col
    onehot_col = cate_col

    # Divide the training set and test set
    Train, Test = train_test_split(df, test_size=0.4, random_state=42)
    Train = Train.reset_index(drop=True)
    Test = Test.reset_index(drop=True)
  
    Test.to_csv('Test_Data/NICE_test2.csv')

    
    ############################################################# One-hot encoding processing #############################################
    Train, onehot_columm = onehot_coder(Train, cate_col)
    Fset_onehot = [col for col in Train.columns if any(f in col for f in Fset)]
    
    # The second processing: identify the numerical and categorical features after one-hot encoding.
    cate_col = []
    num_col = []

    for col in Fset_onehot:
        if col == label:
            continue 
        if Train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(Train[col]):
            cate_col.append(col)
        elif pd.api.types.is_numeric_dtype(Train[col]): 
            unique_values = Train[col].nunique()
            if unique_values <= 10: 
                cate_col.append(col)
            else:
                num_col.append(col)
        else:
            cate_col.append(col)

    Fset = num_col + cate_col

    # Divide the majority class and the minority class
    majority_class = Train[Train[label]  == 0].reset_index(drop=True)
    minority_class = Train[Train[label]  == 1].reset_index(drop=True)

    ############################################################# 特征选择 #############################################################
    
    
    X = Train.drop(label, axis=1)  
    y = Train[label]               
    
    n_features = X.shape[1]
    scores_matrix = np.zeros((n_features, 20))

    for i in range(10):
        mi_selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=i), k='all')
        mi_selector.fit(X, y)
        scores_matrix[:, i] = mi_selector.scores_

    # Calculate the average score
    avg_scores = np.mean(scores_matrix, axis=1)

    mi_scores = pd.DataFrame({'Feature': X.columns, 'MI_Score': avg_scores})
    mi_scores = mi_scores.sort_values('MI_Score', ascending=False)
    treatments = mi_scores.head(n)['Feature'].tolist() 
    
    ############################################################### Calculate the Propensity Score (GPS) #############################################################
    treatments_ps = calculate_gps(Train, treatments, label)

    labels = Train[label]  
    treatments_ps = pd.concat([labels, treatments_ps], axis=1)

    PSM_MatchID = get_PSM_id(treatments_ps, treatments, label)

    ps_majority = treatments_ps[treatments_ps[label] == 0].drop(label, axis=1).reset_index(drop=True)

    # Calculate the dynamic threshold
    thresholds = {col: ps_majority[col].quantile(x) for col in ps_majority.columns if col.endswith('_GPS')}

    result = []

    for idx, row in PSM_MatchID.iterrows():
        majority_id = row['majority_id']
        
        ps_scores = ps_majority.loc[majority_id]
        
        high_ps_cols = [col for col in thresholds if ps_scores[col] >= thresholds[col]]
        
        psm_cols = ['PS_minority_id_' + col for col in high_ps_cols]
        
        minority_ids = [row[col] for col in psm_cols if col in row]
        
        result.append({
            'majority_id': majority_id,
            'high_ps_columns': [col.replace('_GPS', '') for col in high_ps_cols],  # 可选：去除_GPS后缀展示
            'matched_minority_ids': minority_ids
        })

    result_df = pd.DataFrame(result)

    # Filter to obtain the filtered results (retaining only those with differing treatments)
    screen_result = result_df[result_df["high_ps_columns"].apply(len) > 0]
    screen_result = screen_result.reset_index(drop=True)

    tolerance = get_tolerance(num_col, Train)

    processed_screen_result = screen_result.apply(
    lambda row: process_row(row, majority_class, minority_class, num_col, cate_col, tolerance),
    axis=1
    )

    # Remove records where high_ps_columns is empty.
    processed_screen_result = processed_screen_result[processed_screen_result['high_ps_columns'].apply(len) > 0]
    processed_screen_result = processed_screen_result.reset_index(drop=True)


    majority_class = majority_class.reset_index(drop=True)
    screen_result_indexed = processed_screen_result.set_index('majority_id')

    # Merge the matched majority class samples (good_x_sets)
    good_x_sets = majority_class.join(
        screen_result_indexed[['high_ps_columns', 'matched_minority_ids']],
        how='inner' 
    )

    # Extract unmatched majority class samples (unpaired_x_sets)
    unpaired_x_sets = majority_class.loc[
        majority_class.index.difference(screen_result_indexed.index)
    ]
    x_sets_info = good_x_sets[["high_ps_columns", "matched_minority_ids"]].copy()
    good_x_sets = good_x_sets.drop(columns=["high_ps_columns", "matched_minority_ids"])
    good_x_sets = good_x_sets.reset_index(drop=True)
    unpaired_x_sets = unpaired_x_sets.reset_index(drop=True)
    x_sets_info = x_sets_info.reset_index(drop=True)


    ############################################################# Generate new XXXX-2 pairs #############################################################

    minority_class = minority_class.reset_index(drop=True)

    knn = KNeighborsClassifier(n_neighbors=1, algorithm='auto', metric='euclidean') 
    knn.fit(good_x_sets[Fset], good_x_sets[label])
    ux_x = pd.DataFrame(columns = ['unpaied_x_id','good_x_id'])
        
    for i in unpaired_x_sets.index.values:
        indxs = knn.kneighbors(unpaired_x_sets[unpaired_x_sets.index == i][Fset], return_distance=False)
        pdict = {'unpaied_x_id':[i],'good_x_id':[indxs[0][0]]}
        pdict = pd.DataFrame(pdict)
        ux_x = pd.concat([ux_x,pdict])

    ux_x = ux_x.reset_index(drop = True)

    new_ps_list = []

    for idx, row in ux_x.iterrows():
        unpaied_x_id = row["unpaied_x_id"]
        good_x_id = row["good_x_id"]
        
        high_ps_cols = x_sets_info.loc[good_x_id, "high_ps_columns"]
        minority_ids = x_sets_info.loc[good_x_id, "matched_minority_ids"]
        
        if len(high_ps_cols) != len(minority_ids):
            raise ValueError(f"Inconsistent lengths between high_ps_columns and matched_minority_ids  (good_x_id={good_x_id})")
        
        other_cols = unpaired_x_sets.columns.difference(high_ps_cols)
        matchF = unpaired_x_sets.loc[[unpaied_x_id], other_cols].squeeze().to_dict()  
        
        umatchF = {}
        for col, mid in zip(high_ps_cols, minority_ids):
            value = minority_class.loc[mid, col]
            umatchF[col] = value  
        
        new_p = {**matchF, **umatchF}
        new_ps_list.append(new_p)

    new_ps = pd.DataFrame(new_ps_list)
    new_ps[label]  = 1
 

    ################################################################# XXXX-1 cleaning #################################################################


    after_data = pd.concat([Train, new_ps])

    n_train = len(Train)
    
    cleaned_df = new_samples_enn_clean(
        after_data, 
        label, 
        n_train=n_train,
        k_ir=k_ir,        # Calculate the small k value of IR_nn
        k_vote=k_vote,     # The large k value for voting
        max_removal_ratio=0.4,
        exponent=num_features
    )
    
    after_data = cleaned_df.copy()

    after_data = reverse_onehot(after_data, onehot_col)
    print("Processing completed")

    return after_data

if __name__ == "__main__":
    
    XXXX-1 = pd.read_csv('Initial_Data/glass_3VR.csv')
    
    label = 'Type' 

    # Remove irrelevant columns
    # XXXX-1 = XXXX-1.drop(columns=['Sequence'])
    
    # Number of features
    num_features = XXXX-1.shape[1] - 1 

    y =  run(XXXX-1,label, 2, 0.85, 3, 8)

















