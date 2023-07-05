import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data_clinical = pd.read_csv("input/data_clinical_input.csv")
    data_tissue = pd.read_csv("input/data_tissue_first_line_treatment_input.csv")

    ##### preprocessing
    # data_clinical
    columns = ['patientID', 'tissue_ID', 'age', 'Gender', 'patientID', 'stage', 'smoking', "sample_type", 'Medication',
               'PFS', 'BTR', "Progresses"]
    data_clinical = data_clinical[data_clinical.columns.intersection(columns)]
    data_clinical = data_clinical.dropna(subset=['patientID', 'tissue_ID', 'PFS', 'BTR', 'Progresses'])
    data_clinical = data_clinical[~data_clinical['PFS'].isin(['lost', 'lost,withdrew'])]
    data_clinical = data_clinical.drop('sample_type', axis=1)
    data_clinical['Progresses'] = data_clinical['Progresses'].replace({'yes': 1, 'no': 0})
    data_clinical['BTR'] = data_clinical['BTR'].replace({'PR': 0, 'SD': 1, 'PD': 2})
    data_clinical['Medication'], _ = pd.factorize(data_clinical['Medication'])
    data_clinical['Gender'], _ = pd.factorize(data_clinical['Gender'])
    data_clinical['stage'], _ = pd.factorize(data_clinical['stage'])
    data_clinical['smoking'], _ = pd.factorize(data_clinical['smoking'])

    data_clinical = data_clinical.drop_duplicates(subset='patientID', keep='first')
    data_clinical.reset_index(drop=True)
    data_clinical = data_clinical.astype(float)

    #  data_tissue_first_line_tratment
    columns_ = ['tissue_ID', 'gene', 'mutation_abundance', 'medication_alert_status', 'mutation_type', "caseDep",
                'caseAltDep']
    data_tissue = data_tissue[data_tissue.columns.intersection(columns_)]
    data_tissue = data_tissue.dropna(subset=['tissue_ID'])
    data_tissue = data_tissue[data_tissue['gene'] == 'EGFR']
    data_tissue = data_tissue.drop('gene', axis=1)

    data_tissue['mutation_abundance'] = data_tissue['mutation_abundance'].str.replace('%', '')
    data_tissue['mutation_abundance'] = data_tissue['mutation_abundance'].astype(float)
    # data_tissue['mutation_abundance'] = np.log(data_tissue['mutation_abundance'])

    data_tissue['medication_alert_status'] = data_tissue['medication_alert_status'].replace({'yes': 1, 'no': 0})
    data_tissue['mutation_type'], _ = pd.factorize(data_tissue['mutation_type'])
    data_tissue.reset_index(drop=True)

    data_tissue = data_tissue.drop_duplicates(subset='tissue_ID', keep='first')
    data_tissue = data_tissue.astype(float)

    common_tissue_ids = data_clinical[data_clinical['tissue_ID'].isin(data_tissue['tissue_ID'])]['tissue_ID']
    data_clinical = data_clinical[data_clinical['tissue_ID'].isin(common_tissue_ids)]
    print(data_clinical.shape)
    data_tissue = data_tissue[data_tissue['tissue_ID'].isin(common_tissue_ids)]
    print(data_tissue.shape)

    df_merged = pd.merge(data_clinical, data_tissue, on='tissue_ID', how='inner')

    #####  Generate label
    df_merged['label'] = float('nan')

    for i in range(len(df_merged)):
        if df_merged.loc[i, 'Progresses'] == 0:
            df_merged.loc[i, 'label'] = 1
        elif df_merged.loc[i, 'Progresses'] == 1:
            if df_merged.loc[i, 'BTR'] == 0:
                if df_merged.loc[i, 'PFS'] >= 12:
                    df_merged.loc[i, 'label'] = 0.66
                else:
                    df_merged.loc[i, 'label'] = 0.33
            else:
                df_merged.loc[i, 'label'] = 0

    # df_merged.to_csv('df_merged.csv')
    ##### feature selection
    contingency_tables = []
    df = df_merged
    features = ['age', 'Gender', 'stage', 'smoking',  'mutation_abundance', 'medication_alert_status',
                'mutation_type', 'caseDep', 'caseAltDep']
    for feature in features:
        unique_values = df[feature].nunique()
        if unique_values <= 5:
            contingency_table = pd.crosstab(df[feature], df['label'])
        else:
            df['feature_category'] = pd.qcut(df[feature], 4, labels=False)
            contingency_table = pd.crosstab(df['feature_category'], df['label'])

        contingency_tables.append(contingency_table)

    results = []
    for contingency_table in contingency_tables:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        results.append({'chi2': chi2, 'p_value': p_value, 'dof': dof})

    for i, feature in enumerate(features):
        if results[i]['p_value'] < 0.05:
            print(f"feature: {feature}  √")
        else:
            print(f"feature: {feature}  ×")

    ##### cluster
    data = df_merged[['age', 'mutation_abundance', 'medication_alert_status',  'label', 'PFS', 'BTR', 'Progresses']]
    cols_to_normalize = ['age', 'mutation_abundance']

    min_values = data[cols_to_normalize].min()
    max_values = data[cols_to_normalize].max()
    print('max_values',max_values)
    print('min_values', min_values)
    df_normalized = (data[cols_to_normalize] - min_values) / (max_values - min_values)
    data[cols_to_normalize] = df_normalized

    features = data[['age', 'mutation_abundance', 'medication_alert_status']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    cluster_labels = kmeans.labels_
    data['cluster_label'] = cluster_labels

    #####  Generate bags
    MIL_600 = pd.DataFrame()
    bag_num = 1
    for j in range(0, 4):
        cluster_data = data[data['cluster_label'] == j]

        for i in range(1, 151):
            bag_name = 'bag' + str(bag_num)

            bag_samples = random.randint(1, 10)
            samples = cluster_data.sample(n=bag_samples, replace=True)
            bag_ma = samples['mutation_abundance'].mean()
            bag_label = samples['label'].mean()

            bag_data = pd.DataFrame(samples.values, columns=samples.columns)
            bag_data['bag_ma'] = bag_ma
            bag_data['bag_names'] = bag_name
            bag_data['bag_labels'] = bag_label

            MIL_600 = pd.concat([MIL_600, bag_data], ignore_index=True)
            bag_num += 1

    pfs_avg = MIL_600.groupby('bag_names')['PFS'].mean().reset_index()
    ma_avg = MIL_600.groupby('bag_names')['bag_ma'].mean().reset_index()

    MIL_600 = MIL_600[['age', 'mutation_abundance', 'medication_alert_status', 'label', 'bag_names', 'bag_labels']]
    MIL_600.reset_index(drop=True, inplace=True)

    output_file1 = "output/MIL_600.csv"
    MIL_600.to_csv(output_file1)

    output_file2 = "output/pfs_avg.csv"
    pfs_avg.to_csv(output_file2)

    output_file3 = "output/ma_avg.csv"
    ma_avg.to_csv(output_file3)

    MIL_1000 = pd.DataFrame()
    bag_num = 1
    for j in range(0, 4):
        cluster_data = data[data['cluster_label'] == j]

        for i in range(1, 251):
            bag_name = 'bag' + str(bag_num)
            bag_samples = random.randint(1, 10)
            samples = cluster_data.sample(n=bag_samples, replace=True)
            bag_ma = samples['mutation_abundance'].mean()
            bag_label = samples['label'].mean()

            bag_data = pd.DataFrame(samples.values, columns=samples.columns)
            bag_data['bag_ma'] = bag_ma
            bag_data['bag_names'] = bag_name
            bag_data['bag_labels'] = bag_label

            MIL_1000 = pd.concat([MIL_1000, bag_data], ignore_index=True)
            bag_num += 1

    MIL_1000 = MIL_1000[['age', 'mutation_abundance', 'medication_alert_status', 'label', 'bag_names', 'bag_labels']]
    MIL_1000.reset_index(drop=True, inplace=True)

    output_file4 = "output/MIL_1000.csv"
    MIL_1000.to_csv(output_file4)











































